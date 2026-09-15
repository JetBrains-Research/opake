# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
r"""Per-example clipping for MoE models with the load-balancing loss.

The Switch-style load-balancing loss
:math:`E \sum_e f_e(B)\, P_e(B)` couples the examples of a batch through the
load vector :math:`f(B)`, the fraction of the batch's tokens routed to each
expert.  That vector is argmax-derived and has zero gradient almost
everywhere, so the batch gradient of the loss is exactly the sum of
per-example gradients of the surrogate
:math:`E\, w_x \langle \tilde f - k/E,\, P(x) \rangle` evaluated at the
constant :math:`\tilde f = f(B)`, with :math:`P(x)` the example's mean router
probabilities and :math:`w_x = T_x / \bar T` its public token weight.  The
only object a per-example pipeline lacks is that constant.

:func:`moe_clipped_grad` supplies it the way adaptive clipping supplies its
clipping threshold: as a side release computed inside the clipper and
consumed one step later.  Each example's centred, token-weighted load
:math:`s(x) = w_x (h(x) - k/E)` has the structural bound
:math:`\|s(x)\|_2 \le \Delta_L = (T_{\max}/\bar T)\sqrt{k L (1 - k/E)}`
for any input, so its batch mean is released with Gaussian noise of standard
deviation ``noise_multiplier / sqrt(ratio) * Delta_L / normalize_by`` drawn
from the clipper's own key stream, filtered by a bias-corrected exponential
moving average, and carried in :class:`MoeClipState` as the ``f_tilde`` the
next step's surrogate uses.  The gradient stream is the ordinary clipped
pytree at bound ``clipping_norm``; the noise function and the optimizer see
nothing new.

Privacy: the two releases of a step are one Gaussian on the concatenation of
the clipped gradient and the load, with whitened per-record sensitivity
``1/nm² + ratio/nm²``.  The accountant therefore prices the step as a
Gaussian mechanism at the joint multiplier ``nm / sqrt(1 + ratio)``, which
:func:`opaque.dpsgd.accounting.moe_aux` computes from the same ``ratio``.
Everything computed from private examples stays inside the gradient
transform: the per-example router statistics never leave it, and the
per-example load is stripped from the returned diagnostics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.engine.clipping._clipped_fun import (
    ClippingStats,
    _compute_clipping_stats,
)
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.clipping._helpers import normalize_to_tuple
from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.pytree import tree_leaves
from opaque.api.engine.random import fold_in, generator_from_key
from opaque.api.engine.types import ClipState, PerGroup
from opaque.exceptions import ConfigurationError, OperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opaque.api.engine.random.types import RngKey

log = logging.getLogger(__name__)

#: Fold-in tag of the load-release noise stream, beneath the clipper's key
#: and before the step index, so the stream is disjoint from every other
#: derivation of the same key.
MOE_LOAD_STREAM_FOLD = "opaque.clipping.moe_load"

_MIN_EXPERTS = 2


# ---------------------------------------------------------------------------
# Per-example router statistics (vmap-safe)
# ---------------------------------------------------------------------------


def _binary_mask(
    attention_mask: torch.Tensor | None,
    num_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Flat fp32 0/1 token mask (``attention_mask != 0``; all ones when absent)."""
    if attention_mask is None:
        return torch.ones(num_tokens, dtype=torch.float32, device=device)
    mask = (attention_mask != 0).reshape(-1).to(torch.float32)
    if mask.shape[0] != num_tokens:
        raise ConfigurationError(
            *(
                f"attention_mask covers {mask.shape[0]} tokens but the router "
                f"logits cover {num_tokens}.",
            )
        )
    return mask


def router_load_and_probs(
    router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None,
    *,
    top_k: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example load fractions, mean router probabilities and token count.

    Written to run inside ``vmap(grad(...))`` on one example: reductions are
    out-of-place, the executed route set is recovered with ``torch.topk`` on
    the same fp32 softmax the router used, and the per-expert indicator is a
    broadcast compare.  The statistics follow the Switch Transformer
    load-balancing loss (Fedus, Zoph, Shazeer, 2022,
    https://arxiv.org/abs/2101.03961, eqs. 4-6) as pooled by Hugging Face:
    load and router probability are averaged over all layers and all valid
    tokens with one common denominator.

    Args:
        router_logits: One ``(T, E)`` (or ``(..., E)``, flattened) logits
            tensor per routed layer, in layer order.
        attention_mask: Token validity for the same example (``None`` counts
            every position).  Non-binary masks are read as
            ``attention_mask != 0``.
        top_k: Experts executed per token.
        num_layers: Expected number of captured layers; a mismatch raises so
            a duplicated capture cannot silently double the released load.

    Returns:
        ``(h_layers, P, T_x)``: the ``(L, E)`` fp32 executed load fraction
        per layer (``sum_e h[l] = top_k`` on a non-empty row), the ``(E,)``
        fp32 mean router probability pooled over layers and valid tokens
        (``sum_e P = 1``), and the fp32 valid-token count.  A fully masked
        row returns zeros rather than ``0 / 0``.
    """
    if len(router_logits) != num_layers:
        raise ConfigurationError(
            *(
                f"expected router logits for {num_layers} layers, got "
                f"{len(router_logits)}; a duplicated capture would double the load.",
            )
        )
    first = router_logits[0]
    num_experts = first.shape[-1]
    num_tokens = first.reshape(-1, num_experts).shape[0]
    mask = _binary_mask(attention_mask, num_tokens, device=first.device)
    valid_tokens = mask.sum()
    denominator = valid_tokens.clamp(min=1.0)
    expert_ids = torch.arange(num_experts, device=first.device)

    loads = []
    prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=first.device)
    for logits in router_logits:
        z = logits.reshape(-1, num_experts)
        if z.shape[0] != num_tokens:
            raise ConfigurationError(
                *("router logits disagree on the token count across layers.",)
            )
        probs = torch.softmax(z.float(), dim=-1)
        indices = torch.topk(probs, top_k, dim=-1).indices
        one_hot = (indices[..., None] == expert_ids).sum(dim=-2).to(torch.float32)
        loads.append((one_hot * mask[:, None]).sum(dim=0) / denominator)
        prob_sum = prob_sum + (probs * mask[:, None]).sum(dim=0)

    h_layers = torch.stack(loads)
    mean_probs = prob_sum / (len(router_logits) * denominator)
    has_tokens = valid_tokens > 0
    h_layers = torch.where(has_tokens, h_layers, torch.zeros_like(h_layers))
    mean_probs = torch.where(has_tokens, mean_probs, torch.zeros_like(mean_probs))
    return h_layers, mean_probs, valid_tokens


def centred_load(
    h_layers: torch.Tensor,
    *,
    top_k: int,
    valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``h - top_k / E`` per layer; sums to zero over experts on a non-empty row.

    With ``valid_tokens`` (the ``T_x`` of :func:`router_load_and_probs`) a
    fully masked row returns an explicit zero instead of ``-top_k / E``.
    """
    num_experts = h_layers.shape[-1]
    centred = h_layers - top_k / num_experts
    if valid_tokens is None:
        return centred
    return torch.where(valid_tokens > 0, centred, torch.zeros_like(centred))


def load_balancing_surrogate(
    probs: torch.Tensor,
    f_tilde: torch.Tensor,
    token_weight: torch.Tensor | float,
    *,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Per-example surrogate ``E * w_x * <f_tilde - top_k / E, P(x)>``.

    Its gradient with respect to the model equals the gradient of the
    Switch load-balancing loss evaluated at the constant load ``f_tilde``;
    since ``sum_e P = 1`` the centring leaves the gradient unchanged and
    makes the value vanish at balance.
    """
    centred = f_tilde.to(probs.dtype) - top_k / num_experts
    return num_experts * token_weight * (centred * probs).sum()


def load_bound(
    *,
    top_k: int,
    num_experts: int,
    num_layers: int,
    mean_tokens: float,
    max_tokens: float,
) -> float:
    """Structural L2 bound ``Delta_L`` of one example's token-weighted load.

    For one example ``0 <= h^l_e <= 1`` and ``sum_e h^l_e = k`` on every
    layer, so ``||h^l - k/E||^2 <= k (1 - k/E)``; over ``L`` layers and with
    the token weight ``w_x <= max_tokens / mean_tokens``::

        Delta_L = (max_tokens / mean_tokens) * sqrt(k * L * (1 - k / E))

    The bound is attained when every token of every layer is routed to the
    same ``k`` experts, and it holds for adversarial inputs.
    """
    if num_experts < _MIN_EXPERTS or not 1 <= top_k < num_experts:
        raise ConfigurationError(
            *(
                "the router-load release requires 1 <= top_k < num_experts, "
                f"got top_k={top_k}, num_experts={num_experts}.",
            )
        )
    if num_layers < 1:
        raise ConfigurationError(*(f"num_layers must be >= 1, got {num_layers}.",))
    if mean_tokens <= 0 or max_tokens <= 0:
        raise ConfigurationError(
            *(
                "mean_tokens and max_tokens must be positive, got "
                f"mean_tokens={mean_tokens}, max_tokens={max_tokens}.",
            )
        )
    return (max_tokens / mean_tokens) * math.sqrt(
        top_k * num_layers * (1.0 - top_k / num_experts)
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeClipState(ClipState):
    """State of :func:`moe_clipped_grad`.

    ``f_tilde`` is the public load estimate the next step's surrogate uses;
    everything else is the filter, the release stream and the public
    constants the release needs, threaded explicitly like every other Opaque
    state.  The dataclass round-trips through
    :func:`opaque.serialization.state_dict`.

    Attributes:
        f_tilde: Public load estimate ``(E,)`` in force for the next step.
    """

    f_tilde: torch.Tensor
    _m: torch.Tensor
    _step: int
    _rng_key: RngKey
    _local_load: torch.Tensor
    _pending: bool
    _noise_multiplier: float
    _ratio: float
    _beta: float
    _top_k: int
    _num_experts: int
    _num_layers: int
    _load_bound: float
    _normalize_by: float

    @property
    def step(self) -> int:
        """Number of releases consumed."""
        return self._step

    @property
    def load_noise_std(self) -> float:
        """Per-entry noise std of one released per-layer load mean."""
        if self._noise_multiplier == 0.0:
            return 0.0
        return (
            self._noise_multiplier
            / math.sqrt(self._ratio)
            * self._load_bound
            / self._normalize_by
        )

    @property
    def filtered_noise_std(self) -> float:
        """Known per-entry noise std of the current bias-corrected estimate.

        The pooled, sum-zero projected release has per-entry variance
        ``load_noise_std² / L · (E - 1) / E``; the bias-corrected EMA of
        ``t`` such releases has noise factor
        ``sqrt((1 - β)² (1 - β^{2t}) / (1 - β²)) / (1 - β^t)``.  Zero before
        the first release.
        """
        t = self._step
        if t == 0 or self.load_noise_std == 0.0:
            return 0.0
        beta = self._beta
        base = (
            self.load_noise_std
            / math.sqrt(self._num_layers)
            * math.sqrt((self._num_experts - 1) / self._num_experts)
        )
        phi = math.sqrt((1 - beta) ** 2 * (1 - beta ** (2 * t)) / (1 - beta**2))
        return base * phi / (1 - beta**t)

    @property
    def imbalance(self) -> float:
        """Public monitor ``max_e |f_tilde_e - k/E| / (k/E)``."""
        share = self._top_k / self._num_experts
        return float((self.f_tilde - share).abs().max() / share)


def _release(state: MoeClipState, load_mean: torch.Tensor) -> MoeClipState:
    """Noise one ``(L, E)`` load mean, filter it and advance the state."""
    step = state._step
    load_mean = load_mean.detach().to(dtype=torch.float32, device="cpu")
    sigma = state.load_noise_std
    if sigma > 0.0:
        generator = generator_from_key(
            fold_in(state._rng_key, MOE_LOAD_STREAM_FOLD, step)
        )
        load_mean = load_mean + sigma * torch.randn(
            load_mean.shape, generator=generator, dtype=torch.float32
        )
    d_hat = load_mean.mean(0)
    d_hat = d_hat - d_hat.mean()
    t = step + 1
    m = state._beta * state._m + (1.0 - state._beta) * d_hat
    d_tilde = m / (1.0 - state._beta**t)
    share = state._top_k / state._num_experts
    f_tilde = torch.clamp(share + d_tilde, 0.0, 1.0)
    return replace(
        state,
        f_tilde=f_tilde,
        _m=m,
        _step=t,
        _local_load=torch.zeros_like(state._local_load),
        _pending=False,
    )


def _stationary_filtered_noise(state: MoeClipState) -> float:
    beta = state._beta
    return (
        state.load_noise_std
        / math.sqrt(state._num_layers)
        * math.sqrt((state._num_experts - 1) / state._num_experts)
        * math.sqrt((1 - beta) / (1 + beta))
    )


def _first_device(params: Any) -> torch.device:
    for leaf in tree_leaves(params):
        if isinstance(leaf, torch.Tensor):
            return leaf.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def moe_clipped_grad(  # noqa: PLR0913 - the fixed factory contract
    loss_fn: Callable,
    *,
    clipping_norm: float | PerGroup,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    noise_multiplier: float,
    ratio: float = 0.02,
    key: RngKey,
    top_k: int,
    num_experts: int,
    num_layers: int,
    max_tokens: float,
    mean_tokens: float | None = None,
    alpha: float,
    filter_beta: float = 0.99,
    return_aux: bool = False,
    return_stats: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    microbatch_size: int | None = None,
    dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
    _chunk_compiler: Callable | None = None,
) -> tuple[Callable, MoeClipState]:
    r"""Create a per-example clipper for a MoE loss with the load-balancing term.

    ``loss_fn(params, *batch)`` is evaluated on one example and returns
    ``(loss, router_logits, attention_mask)``: the example's scalar loss, one
    ``(T, E)`` router-logits tensor per routed layer in layer order (the
    logits the router executed), and the example's token mask (``None``
    counts every position).  The returned function differentiates
    ``loss + alpha * E * w_x * <f_tilde - k/E, P(x)>`` with respect to
    ``params`` (the surrogate enters value-neutrally, so the reported loss
    values are the plain ``loss``), clips and sums the per-example gradients
    exactly like :func:`opaque.dpsgd.clipping.clipped_grad`, and releases the
    batch mean of the per-example token-weighted centred load with Gaussian
    noise inside the state, from which the next step's ``f_tilde`` is
    filtered.

    Privacy accounting is ``moe_aux(gaussian(noise_multiplier), ratio=ratio)``
    with the same ``noise_multiplier`` handed to ``gaussian_noise`` and the
    same ``ratio`` given here: the joint release is one Gaussian mechanism at
    multiplier ``noise_multiplier / sqrt(1 + ratio)``.  The load noise is drawn
    from ``key`` folded with :data:`MOE_LOAD_STREAM_FOLD` and the step index,
    so it is disjoint from any other stream derived from the same key; under
    DDP pass the same ``key`` on every rank and synchronize the state after
    every step (:func:`opaque.distributed.sync`), which all-reduces the
    rank-local load means and adds the noise once.

    Args:
        loss_fn: Per-example function returning
            ``(loss, router_logits, attention_mask)``.  ``params`` is its
            first positional argument.
        clipping_norm: Per-example gradient bound ``C_g`` (float or
            :class:`~opaque.types.PerGroup`).
        normalize_by: Divisor of the summed gradients and of the released
            load mean; set to the expected batch size.
        batch_argnums: Which arguments after ``params`` carry the batch
            dimension, counted as in :func:`clipped_grad` (``1`` is the first
            argument after ``params``).
        noise_multiplier: The gradient noise multiplier, the same value
            handed to ``gaussian_noise``.
        ratio: Share of the whitened sensitivity given to the load release,
            ``ρ = nm_g² / nm_h²``; the load noise std is
            ``noise_multiplier / sqrt(ratio) * Delta_L / normalize_by`` and the
            accountant prices the step at ``noise_multiplier / sqrt(1 + ratio)``.
        key: RNG key of the load-release noise stream.
        top_k: Experts executed per token ``k``.
        num_experts: Number of experts ``E``.
        num_layers: Number of routed layers ``L`` (router logits per example).
        max_tokens: Public bound on the valid tokens of one example
            ``T_max``; the load of a longer row is clipped to the bound.
        mean_tokens: Public token constant ``T̄`` of the token weight
            ``w_x = T_x / T̄`` (``None``: ``max_tokens``).
        alpha: Coefficient of the surrogate (the model's router aux-loss
            coefficient).
        filter_beta: Coefficient of the bias-corrected exponential moving
            average that turns the released loads into ``f_tilde``.
        return_aux: If True, also return per-example diagnostics as
            :class:`~opaque.api.engine.clipping.ClippedGradAux` with
            ``loss_aux`` removed (the per-example load is private-internal).
        return_stats: If True, return aggregate clipping statistics instead.
        pre_clipping_transform: Applied to each per-example gradient pytree
            before clipping, as in :func:`clipped_grad`.
        microbatch_size: Process the batch in chunks of this size.
        dtype: Output dtype of the summed gradient.
        compute_dtype: Accumulation dtype of the reductions.

    Returns:
        ``(grad_fn, state)``.  ``grad_fn(params, *batch, state=state, **kwargs)``
        returns ``(grads, new_state)`` with ``grads`` a plain
        :class:`~opaque.types.ClippedPytree` at bound
        ``clipping_norm / normalize_by``; ``((grads, aux), new_state)`` with
        ``return_aux``; ``((grads, stats), new_state)`` with ``return_stats``.

    Formal guarantee:
        Under add/remove adjacency one record moves the clipped gradient sum
        by at most ``clipping_norm`` and the released load sum by at most
        ``Delta_L`` (:func:`load_bound`), the latter by a rescale that only
        ever removes floating-point round-off.  With the noise scales above,
        the pair is a sensitivity-one Gaussian mechanism at multiplier
        ``noise_multiplier / sqrt(1 + ratio)`` (whitened sensitivities add in
        quadrature: ``1/nm² + ratio/nm²``).  ``f_tilde`` is post-processing of
        earlier releases and costs nothing further.

    References:
        Fedus, Zoph, Shazeer. "Switch Transformers." JMLR 2022.
        https://arxiv.org/abs/2101.03961.  Andrew et al. "Differentially
        Private Learning with Adaptive Clipping." NeurIPS 2021.
        https://arxiv.org/abs/1905.03871 (the joint-release allocation).
    """
    if noise_multiplier < 0:
        raise ConfigurationError(
            *(f"noise_multiplier must be non-negative, got {noise_multiplier}.",)
        )
    if not ratio > 0:
        raise ConfigurationError(*(f"ratio must be positive, got {ratio}.",))
    if not 0.0 < filter_beta < 1.0:
        raise ConfigurationError(
            *(f"filter_beta must be in (0, 1), got {filter_beta}.",)
        )
    if not normalize_by > 0:
        raise ConfigurationError(
            *(f"normalize_by must be positive, got {normalize_by}.",)
        )
    if not math.isfinite(float(alpha)):
        raise ConfigurationError(*(f"alpha must be finite, got {alpha}.",))
    if return_aux and return_stats:
        raise ConfigurationError(*("return_aux and return_stats cannot both be set.",))
    mean_tokens = float(max_tokens) if mean_tokens is None else float(mean_tokens)
    bound = load_bound(
        top_k=top_k,
        num_experts=num_experts,
        num_layers=num_layers,
        mean_tokens=mean_tokens,
        max_tokens=float(max_tokens),
    )
    alpha = float(alpha)
    batch_positions = normalize_to_tuple(batch_argnums)
    if any(i < 1 for i in batch_positions):
        raise ConfigurationError(
            *("batch_argnums must index arguments after params (>= 1).",)
        )
    # ``f_tilde`` is inserted as the second positional argument of the
    # wrapped function, so the caller's batch positions shift by one.
    shifted = tuple(i + 1 for i in batch_positions)

    def wrapped(params, f_tilde, *rest, **kwargs):
        loss, router_logits, attention_mask = loss_fn(params, *rest, **kwargs)
        h_layers, probs, n_tokens = router_load_and_probs(
            router_logits, attention_mask, top_k=top_k, num_layers=num_layers
        )
        weight = n_tokens / mean_tokens
        surrogate = load_balancing_surrogate(
            probs, f_tilde, weight, num_experts=num_experts, top_k=top_k
        )
        augmented = loss + alpha * (surrogate - surrogate.detach())
        load = weight * centred_load(h_layers, top_k=top_k, valid_tokens=n_tokens)
        return augmented, load.detach()

    inner_fn, _ = clipped_grad(
        wrapped,
        argnums=0,
        has_aux=True,
        clipping_norm=clipping_norm,
        normalize_by=normalize_by,
        batch_argnums=shifted,
        return_aux=True,
        pre_clipping_transform=pre_clipping_transform,
        microbatch_size=microbatch_size,
        dtype=dtype,
        compute_dtype=compute_dtype,
        _chunk_compiler=_chunk_compiler,
    )

    share = top_k / num_experts
    state = MoeClipState(
        f_tilde=torch.full((num_experts,), share, dtype=torch.float32),
        _m=torch.zeros(num_experts, dtype=torch.float32),
        _step=0,
        _rng_key=key,
        _local_load=torch.zeros(num_layers, num_experts, dtype=torch.float32),
        _pending=False,
        _noise_multiplier=float(noise_multiplier),
        _ratio=float(ratio),
        _beta=float(filter_beta),
        _top_k=int(top_k),
        _num_experts=int(num_experts),
        _num_layers=int(num_layers),
        _load_bound=float(bound),
        _normalize_by=float(normalize_by),
    )
    log.info(
        "moe_clipped_grad: E=%d k=%d L=%d bound=%.4g ratio=%g; per-release load "
        "noise %.3g per entry, filtered stationary noise %.3g of k/E "
        "(beta=%g)",
        num_experts,
        top_k,
        num_layers,
        bound,
        ratio,
        state.load_noise_std,
        _stationary_filtered_noise(state) / share,
        filter_beta,
    )

    def _local_load_mean(per_example: torch.Tensor | None) -> torch.Tensor:
        if per_example is None or per_example.numel() == 0:
            return torch.zeros(num_layers, num_experts, dtype=torch.float32)
        flat = per_example.detach().reshape(per_example.shape[0], -1).float()
        norms = flat.norm(dim=1)
        # The structural bound holds for every input; the rescale only ever
        # removes floating-point round-off above it.
        scale = torch.clamp(bound / norms.clamp_min(1e-30), max=1.0)
        total = (flat * scale[:, None]).sum(0) / normalize_by
        return total.reshape(num_layers, num_experts).cpu()

    def grad_fn(params, *rest, state: MoeClipState, **kwargs):
        if state._pending:
            raise OperationError(
                *(
                    "moe_clipped_grad: the previous step's load release is still "
                    "pending; under DDP call opaque.distributed.sync(state) after "
                    "every step before the next call.",
                )
            )
        f_tilde = state.f_tilde.to(_first_device(params))
        (grads, aux), _ = inner_fn(params, f_tilde, *rest, state=None, **kwargs)
        local = _local_load_mean(aux.loss_aux)
        if is_distributed():
            new_state = replace(state, _local_load=local, _pending=True)
        else:
            new_state = _release(state, local)
        if return_aux:
            return (grads, replace(aux, loss_aux=None)), new_state
        if return_stats:
            stats: ClippingStats = _compute_clipping_stats(
                aux.grad_norms,
                clipping_norm=clipping_norm,
                group_norms_dict=aux.group_norms,
            )
            return (grads, stats), new_state
        return grads, new_state

    return grad_fn, state


__all__ = [
    "MOE_LOAD_STREAM_FOLD",
    "MoeClipState",
    "centred_load",
    "load_balancing_surrogate",
    "load_bound",
    "moe_clipped_grad",
    "router_load_and_probs",
]
