# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
r"""Per-example clipping for MoE models with the load-balancing loss (DP-SGD).

The Switch load-balancing loss :math:`E \sum_e f_e(B)\, P_e(B)` couples the
examples of a batch only through the load vector :math:`f(B)`, which has
zero gradient almost everywhere, and the token total :math:`T_B` inside
:math:`P_e(B)`.  Replacing :math:`T_B` by the public constant
``normalize_by * mean_tokens`` and :math:`f(B)` by a lagged private estimate
:math:`\tilde f` turns the batch gradient into the sum of per-example
gradients of :math:`E\, w_x \langle \tilde f - k/E,\, P(x) \rangle`, a
public-normalized, one-step-lagged, top-k, layer-pooled generalization of
the Switch objective (exactly its frozen-load gradient when the realized
token total matches the public normalization).  :func:`moe_clipped_grad`
releases :math:`\tilde f` the way adaptive clipping releases its threshold:
noised inside the clipper, carried in :class:`MoeClipState`, consumed one
step later.  The joint release is one Gaussian at multiplier
``nm / sqrt(1 + ratio)``, priced by :func:`opaque.dpsgd.accounting.moe_aux`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from opaque.api.engine.clipping._clipped_fun import (
    ClippingStats,
    _compute_clipping_stats,
)
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.clipping._helpers import normalize_to_tuple
from opaque.api.engine.clipping._moe import (
    centred_load,
    load_balancing_surrogate,
    load_bound,
    router_load_and_probs,
)
from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.distributed._state import (
    assert_string_equal,
    register_sync_type,
)
from opaque.api.engine.pytree import tree_leaves
from opaque.api.engine.random import fold_in, generator_from_key
from opaque.api.engine.types import ClipState, PerGroup
from opaque.exceptions import ConfigurationError, InputTypeError, OperationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random.types import RngKey

log = logging.getLogger(__name__)

#: Fold-in tag of the load-release noise stream.
MOE_LOAD_STREAM_FOLD = "opaque.clipping.moe_load"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeClipState(ClipState):
    """State of :func:`moe_clipped_grad`.

    ``f_tilde`` is the public load estimate the next step's surrogate uses;
    the private fields are the filter, the release stream and its constants.
    """

    f_tilde: torch.Tensor
    _m: torch.Tensor
    _step: int
    _rng_key: RngKey
    _local_load: torch.Tensor
    _pending: bool
    #: Noise variance of the filter accumulator ``_m`` and the product of the
    #: betas applied so far, tracked recursively so the telemetry stays exact
    #: when the noise scale or ``filter_beta`` changes between steps.
    _noise_var: float
    _decay: float
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
    def release_noise_std(self) -> float:
        """Per-entry noise std of one pooled, sum-zero projected release."""
        return (
            self.load_noise_std
            / math.sqrt(self._num_layers)
            * math.sqrt((self._num_experts - 1) / self._num_experts)
        )

    @property
    def filtered_noise_std(self) -> float:
        """Per-entry noise std of the latent estimate ``k/E + d_tilde`` before the clamp.

        Tracked recursively through the filter: each release adds
        ``(1 - β)²`` of its own noise variance to ``β²`` of the accumulated
        one, and the bias correction divides by ``1 - Πβ``, so the value is
        exact even when the noise scale or ``filter_beta`` changed between
        steps.  Zero before the first release.
        """
        if self._step == 0 or self._noise_var <= 0.0:
            return 0.0
        return math.sqrt(self._noise_var) / (1.0 - self._decay)

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
    beta = state._beta
    m = beta * state._m + (1.0 - beta) * d_hat
    noise_var = beta**2 * state._noise_var + (1.0 - beta) ** 2 * (
        state.release_noise_std**2
    )
    decay = beta * state._decay
    d_tilde = m / (1.0 - decay)
    share = state._top_k / state._num_experts
    f_tilde = torch.clamp(share + d_tilde, 0.0, 1.0)
    return replace(
        state,
        f_tilde=f_tilde,
        _m=m,
        _step=step + 1,
        _local_load=torch.zeros_like(state._local_load),
        _pending=False,
        _noise_var=noise_var,
        _decay=decay,
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
    ``(loss, router_logits, attention_mask)``: the scalar loss, one ``(T, E)``
    router-logits tensor per routed layer, and the token mask (``None``
    counts every position).  The returned function differentiates
    ``loss + alpha * E * w_x * <f_tilde - k/E, P(x)>`` (value-neutrally, so
    reported losses stay the plain ``loss``), clips and sums per-example
    gradients like :func:`clipped_grad`, and releases the batch mean of the
    token-weighted centred load with Gaussian noise inside the state, from
    which the next step's ``f_tilde`` is filtered.

    Accounting is ``moe_aux(gaussian(noise_multiplier), ratio=ratio)`` with
    the same two values given here.  Under DDP pass the same ``key`` on every
    rank and call :func:`opaque.distributed.sync` on the state after every
    step.

    Args:
        loss_fn: Per-example function returning
            ``(loss, router_logits, attention_mask)``.
        clipping_norm: Per-example gradient bound (float or
            :class:`~opaque.types.PerGroup`).
        normalize_by: Divisor of the summed gradients and of the released
            load mean; set to the expected batch size.
        batch_argnums: Which arguments after ``params`` carry the batch
            dimension, as in :func:`clipped_grad`.
        noise_multiplier: The gradient noise multiplier handed to
            ``gaussian_noise``.
        ratio: Share of the whitened sensitivity given to the load release;
            the load noise std is
            ``noise_multiplier / sqrt(ratio) * Delta_L / normalize_by``.
        key: RNG key of the load-release noise stream.
        top_k: Experts executed per token.
        num_experts: Number of experts.
        num_layers: Number of routed layers.
        max_tokens: Public bound on the valid tokens of one example.
        mean_tokens: Public token constant of the weight ``w_x = T_x / mean``
            (``None``: ``max_tokens``).
        alpha: Coefficient of the surrogate (the model's router aux-loss
            coefficient).
        filter_beta: Bias-corrected EMA coefficient of the load estimate.
        return_aux: Also return per-example diagnostics with ``loss_aux``
            removed (the per-example load is private).
        return_stats: Return aggregate clipping statistics instead.
        pre_clipping_transform: As in :func:`clipped_grad`.
        microbatch_size: As in :func:`clipped_grad`.
        dtype: As in :func:`clipped_grad`.
        compute_dtype: As in :func:`clipped_grad`.

    Returns:
        ``(grad_fn, state)``; ``grad_fn(params, *batch, state=state)`` returns
        ``(grads, new_state)`` with ``grads`` a plain
        :class:`~opaque.types.ClippedPytree`.

    References:
        Fedus, Zoph, Shazeer (2022), https://arxiv.org/abs/2101.03961;
        Andrew et al. (2021), https://arxiv.org/abs/1905.03871.
    """
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ConfigurationError(
            *(
                f"noise_multiplier must be finite and non-negative, got {noise_multiplier}.",
            )
        )
    if not math.isfinite(ratio) or not ratio > 0:
        raise ConfigurationError(*(f"ratio must be finite and positive, got {ratio}.",))
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
        _noise_var=0.0,
        _decay=1.0,
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


# ---------------------------------------------------------------------------
# DDP synchronisation
# ---------------------------------------------------------------------------


def _reduce_device() -> torch.device:
    backend = dist.get_backend() if dist.is_initialized() else None
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise OperationError(
                *(
                    "Distributed backend is 'nccl' but CUDA is not available; "
                    "cannot all-reduce the MoE load release.",
                )
            )
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _shared_release_state(state: MoeClipState) -> str:
    """Everything the ranks must agree on before one shared release."""
    return repr(
        (
            bool(state._pending),
            int(state._step),
            int(state._rng_key.seed),
            str(state._rng_key.impl),
            float(state._noise_multiplier),
            float(state._ratio),
            float(state._beta),
            float(state._load_bound),
            float(state._normalize_by),
            int(state._top_k),
            int(state._num_experts),
            int(state._num_layers),
        )
    )


def sync_moe_clip_state(state: MoeClipState) -> MoeClipState:
    """Finish the pending load release of :func:`moe_clipped_grad` under DDP.

    Every rank first agrees, collectively, on the pending flag, the step, the
    release key and the release constants; a disagreement raises on every
    rank instead of leaving some ranks blocked in the reduction or noising
    the estimate differently per rank.  The rank-local load means are then
    all-reduced, the noise is added once from the shared ``(key, step)`` and
    filtered, so every rank lands on the same ``f_tilde``.  Not distributed:
    returned unchanged.
    """
    if not is_distributed():
        return state
    if not isinstance(state, MoeClipState):
        raise InputTypeError(*(f"Expected a MoeClipState, got {type(state).__name__}",))
    assert_string_equal(_shared_release_state(state), name="MoeClipState release state")
    if not state._pending:
        return state
    device = _reduce_device()
    total = state._local_load.detach().to(device=device, dtype=torch.float32).clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return _release(state, total.cpu())


register_sync_type(MoeClipState, sync_moe_clip_state)


__all__ = [
    "MOE_LOAD_STREAM_FOLD",
    "MoeClipState",
    "moe_clipped_grad",
    "sync_moe_clip_state",
]
