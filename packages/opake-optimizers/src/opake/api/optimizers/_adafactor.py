"""Adafactor optimizer (Shazeer & Stern, 2018).

Memory-efficient Adam variant: the second moment is **factored** for
tensors of rank ≥ 2 (one ``v_row`` of size ``rows`` and one ``v_col``
of size ``cols`` instead of a full ``rows × cols`` ``v``).  For
tensors of rank < 2 (vectors, scalars) Adafactor keeps a full
per-element second moment ``v``.

Reference:
    Shazeer & Stern, "Adafactor: Adaptive Learning Rates with Sublinear
    Memory Cost", arXiv:1804.04235.

The factored estimator approximates the full ``v_t`` matrix as the
outer product ``v_row · v_col / mean(v_row)``.  This saves
``rows·cols − rows − cols`` floats of state per matrix parameter, which
matters at LM scale.

DP noise-variance bias correction.  Under additive Gaussian noise
``g_ij → g_ij + ξ_ij`` with ``ξ_ij ~ N(0, σ²)``,
``E[(g_ij + ξ_ij)²] = g_ij² + σ²``.  The bias is a uniform ``+σ²``
per element, so it propagates cleanly through the row and column
*means*::

    E[mean_j (g_ij + ξ_ij)²] = mean_j g_ij² + σ²
    E[mean_i (g_ij + ξ_ij)²] = mean_i g_ij² + σ²

A single scalar φ-EMA tracking ``β₂_t`` matches the v_row/v_col EMAs;
subtracting it from each factor before the ``v̂`` approximation recovers
the clean second-moment estimate wherever the result stays positive and
falls back to the uncorrected factor elsewhere (see
``_bias_correction.py``).  ``noise_bias_correction=True`` activates
this path; it defaults to ``False``.

The private second-moment substitution path (``noisy_squared_grads``)
is **not** offered for Adafactor.  Substituting a privately-estimated
``g²`` stream for ``(g+ξ)²`` does not preserve the rank-1 ``r ⊗ c``
factorisation in any obvious way; deriving a sound factored variant
is left as future work.

Skipped (orthogonal knobs, can be added later):

- ``relative_step``: LR derived from ``min(1e-2, 1/√t)``.  Compose
  with ``opake.scheduling`` instead.
- ``scale_parameter``: LR scaled by ``RMS(params)``.  Independent
  enhancement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from opake.exceptions import ConfigurationError

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "torchopt is required for opake.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

import optree

from opake.api.optimizers._bias_correction import resolve_noise_variance
from opake.api.optimizers._chain import _clip_by_global_rms, make_optimizer_chain

if TYPE_CHECKING:
    from opake.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class AdafactorState:
    """State for Adafactor moment scaling.

    Attributes:
        m: Optional first-moment EMA (pytree matching params, or None
            when β₁ == 0 — most Adafactor configs).
        v_flat: Per-leaf second-moment state.  For each leaf, a tuple:

            - ``(v_row, v_col)`` for tensors of rank ≥ 2 (factored),
            - ``(v,)`` for tensors of rank < 2 (scalar).
        phi_flat: Per-leaf noise-variance EMA.  Tracks ``β₂_t``-weighted
            ``σ²`` for DP bias correction; stays at ``0.0`` per leaf
            unless ``NoisedPytree`` updates supply realized σ metadata
            and ``noise_bias_correction`` is enabled.
        treespec: Frozen tree spec from ``optree`` so updates can be
            re-packed in the same shape.
        paths: Optree ParamPath per leaf, aligned with ``v_flat`` /
            ``phi_flat``.  Used to look up per-group ``noise_stddev``
            values during BC.
        step: Number of completed updates.
    """

    m: TensorPytree | None
    v_flat: tuple[tuple[torch.Tensor, ...], ...]
    phi_flat: tuple[float, ...]
    treespec: Any  # optree.PyTreeSpec — vendor-specific, kept opake
    paths: tuple
    step: int


def _init_v_for_leaf(leaf: torch.Tensor) -> tuple:
    """Allocate row/col or scalar second-moment state for one leaf."""
    if leaf.dim() >= 2:  # noqa: PLR2004 - factoring requires matrix-shaped leaves
        # Factored: one v_row per "row" axis (-2) and one v_col per
        # "col" axis (-1).  Higher-rank tensors collapse the leading
        # dims into rows: shape ``(*leading, rows, cols)`` → row state
        # of shape ``(*leading, rows)``, col state of shape
        # ``(*leading, cols)``.
        v_row = torch.zeros(leaf.shape[:-1], dtype=leaf.dtype, device=leaf.device)
        v_col = torch.zeros(
            (*leaf.shape[:-2], leaf.shape[-1]),
            dtype=leaf.dtype,
            device=leaf.device,
        )
        return (v_row, v_col)
    return (torch.zeros_like(leaf),)


def _approx_v_hat(
    v_row: torch.Tensor, v_col: torch.Tensor, eps_root: float
) -> torch.Tensor:
    """Factored ``v̂`` approximation::

        v̂ ≈ v_row[..., :, None] · v_col[..., None, :] / mean(v_row, dim=-1)

    Following Shazeer & Stern (2018, Algorithm 4).  The mean (rather
    than sum) keeps the factored estimate scale-invariant so the
    approximation matches the full second moment on rank-1 inputs.

    The ``r_mean`` floor is **scale-relative**: ``eps_root`` is applied as a
    fraction of ``mean(v_row)`` (the global scale of the row factor, which at
    steady state tracks the gradient second moment).  This keeps the floor in
    the same units as ``r_mean`` itself and prevents it from firing spuriously
    when gradients are small in absolute terms (e.g. after DP clipping) while
    still guarding against genuine numerical underflow.
    """
    r_mean = v_row.mean(dim=-1, keepdim=True)
    # Scale-relative floor: eps_root * E[v_row].  At steady state
    # E[v_row] ≈ E[g²], so the floor tracks the gradient second-moment scale.
    # The inner clamp(1e-30) guards the initial step where v_row is all zeros.
    v_row_scale = v_row.mean().clamp(min=1e-30)
    r_mean = r_mean.clamp(min=eps_root * v_row_scale)
    return (v_row / r_mean).unsqueeze(-1) * v_col.unsqueeze(-2)


def _scale_by_adafactor(
    b1: float,
    b2_decay: float,
    eps_grad: float,
    eps_root: float,
    update_rms_clip: float,
    noise_bias_correction: bool,
) -> GradientTransformation:
    """Adafactor moment scaling.

    ``b2_decay`` is the paper's ``c`` exponent in the time-varying
    β₂_t = 1 − t^c (default c = −0.8).
    """
    use_first_moment = b1 > 0.0

    def init_fn(params: Any) -> AdafactorState:
        from opake.api.engine.pytree import tree_flatten_with_paths

        path_list, flat, treespec = tree_flatten_with_paths(params)
        paths = tuple(path_list)
        v_flat = tuple(_init_v_for_leaf(leaf) for leaf in flat)
        phi_flat = tuple(0.0 for _ in flat)
        m = None
        if use_first_moment:
            m_flat = tuple(torch.zeros_like(leaf) for leaf in flat)
            m = optree.tree_unflatten(treespec, list(m_flat))
        return AdafactorState(
            m=m,
            v_flat=v_flat,
            phi_flat=phi_flat,
            treespec=treespec,
            paths=paths,
            step=0,
        )

    def update_fn(
        updates: Any,
        state: AdafactorState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
    ) -> tuple[Any, AdafactorState]:
        from opake.api.engine.pytree import tree_flatten_with_paths

        t = state.step + 1
        # Time-varying β₂_t per the paper: β₂_t = 1 − t^c.
        beta2_t = 1.0 - (float(t) ** b2_decay)

        flat_paths, flat_grads, _ = tree_flatten_with_paths(updates)
        if flat_paths != list(state.paths):
            raise ConfigurationError(
                *(
                    f"updates ParamPath mismatch: expected {list(state.paths)!r}, "
                    f"got {flat_paths!r}.",
                )
            )
        if len(flat_grads) != len(state.v_flat):
            raise ConfigurationError(
                *(
                    f"updates pytree has {len(flat_grads)} leaves, "
                    f"but state has {len(state.v_flat)} — params/grads "
                    "shape mismatch.",
                )
            )

        bc_active = noise_bias_correction and noise_stddev is not None

        new_v_flat: list[tuple] = []
        new_phi_flat: list[float] = []
        new_grads: list[torch.Tensor] = []

        for i, (g, v_state) in enumerate(zip(flat_grads, state.v_flat, strict=True)):
            g_sq = g.pow(2) + eps_grad

            if bc_active:
                nv = resolve_noise_variance(noise_stddev, state.paths[i])
                old_phi = state.phi_flat[i]
                new_phi = beta2_t * old_phi + (1.0 - beta2_t) * nv
            else:
                new_phi = state.phi_flat[i]

            new_phi_flat.append(new_phi)

            if len(v_state) == 2:  # noqa: PLR2004 - factored state is (row, col)
                v_row, v_col = v_state
                # Factored update.
                new_v_row = beta2_t * v_row + (1.0 - beta2_t) * g_sq.mean(dim=-1)
                new_v_col = beta2_t * v_col + (1.0 - beta2_t) * g_sq.mean(dim=-2)
                if bc_active and new_phi > 0.0:
                    corr_row = new_v_row - new_phi
                    corr_col = new_v_col - new_phi
                    v_row_eff = torch.where(corr_row > 0, corr_row, new_v_row)
                    v_col_eff = torch.where(corr_col > 0, corr_col, new_v_col)
                else:
                    v_row_eff = new_v_row
                    v_col_eff = new_v_col
                v_hat = _approx_v_hat(v_row_eff, v_col_eff, eps_root)
                # Scale-relative floor: eps_root as a fraction of √mean(v̂).
                # Fires only when individual elements are eps_root² times smaller
                # than the mean — genuine numerical underflow, not a scale mismatch.
                v_hat_scale = v_hat.mean().clamp(min=1e-30).sqrt()
                update = g / v_hat.sqrt().clamp(min=eps_root * v_hat_scale)
                new_v_flat.append((new_v_row, new_v_col))
            else:
                (v,) = v_state
                new_v = beta2_t * v + (1.0 - beta2_t) * g_sq
                if bc_active and new_phi > 0.0:
                    corr = new_v - new_phi
                    v_eff = torch.where(corr > 0, corr, new_v)
                else:
                    v_eff = new_v
                # Scale-relative floor: eps_root as a fraction of √mean(v_eff).
                v_eff_scale = v_eff.mean().clamp(min=1e-30).sqrt()
                update = g / v_eff.sqrt().clamp(min=eps_root * v_eff_scale)
                new_v_flat.append((new_v,))

            new_grads.append(update)

        new_grads = list(_clip_by_global_rms(tuple(new_grads), update_rms_clip))

        # Optional first moment β₁.
        if use_first_moment:
            assert state.m is not None
            _, flat_old_m, _ = tree_flatten_with_paths(state.m)
            new_flat_m = [
                b1 * old_m + (1.0 - b1) * g
                for old_m, g in zip(flat_old_m, new_grads, strict=True)
            ]
            updates_unflat = optree.tree_unflatten(state.treespec, new_flat_m)
            new_m = optree.tree_unflatten(state.treespec, new_flat_m)
        else:
            updates_unflat = optree.tree_unflatten(state.treespec, new_grads)
            new_m = None

        return updates_unflat, AdafactorState(
            m=new_m,
            v_flat=tuple(new_v_flat),
            phi_flat=tuple(new_phi_flat),
            treespec=state.treespec,
            paths=state.paths,
            step=t,
        )

    return GradientTransformation(init_fn, update_fn)


def adafactor(
    lr: _LR = 1e-3,
    beta1: float = 0.0,
    decay_rate: float = -0.8,
    eps_grad: float = 1e-30,
    eps_root: float = 1e-3,
    weight_decay: float = 0.0,
    update_rms_clip: float = 1.0,
    *,
    decoupled_weight_decay: bool = True,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an Adafactor optimizer.

    Args:
        lr: Learning rate, scalar or schedule.  In the paper this is
            often left at 1.0 with ``relative_step``; we don't ship
            ``relative_step`` yet, so set an explicit LR.
        beta1: First-moment EMA coefficient.  ``0.0`` (default) disables
            the first moment, matching the paper's most memory-efficient
            configuration.
        decay_rate: Exponent ``c`` in β₂_t = 1 − t^c.  ``-0.8`` is the
            paper's default; smaller magnitudes (closer to 0) make β₂_t
            decay slower with step.
        eps_grad: Stability constant added to ``g²`` before factoring;
            paper default 1e-30.
        eps_root: Stability constant for the ``√v̂`` denominator floor; paper
            default 1e-3.  The floor is **scale-relative**: it is applied as
            ``eps_root × √mean(v̂)`` rather than as an absolute constant.  This
            ensures the floor tracks the gradient scale and does not fire
            spuriously when gradients are small in absolute magnitude (e.g.
            after DP per-record clipping), which would otherwise reduce the
            optimizer to RMS-normalised SGD.
        weight_decay: Decoupled weight-decay coefficient by default.
        update_rms_clip: Model-wide RMS clip threshold on the moment-scaled
            update; the RMS covers all tensor leaves in the update pytree.
            Paper default 1.0.
        decoupled_weight_decay: Same semantics as
            :func:`opake.optimizers._adamw`.
        noise_bias_correction: If ``True``, subtract a β₂_t-EMA of the
            realized noise variance from each factor (``v_row``,
            ``v_col``, or scalar ``v``) when ``NoisedPytree`` updates
            are passed.  Defaults to ``False``; Adafactor's relative-step
            per-tensor normalization already plays BC's role for this
            optimizer.  No effect when ``SecondMomentNoiseOutput`` is
            passed — Adafactor does not consume the privatised ``g²``
            stream (deriving a sound factored variant is future work).

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if decay_rate >= 0:
        raise ConfigurationError(*(f"decay_rate must be negative, got {decay_rate}",))
    if eps_grad <= 0 or eps_root <= 0:
        raise ConfigurationError(
            *(f"eps_grad and eps_root must be positive, got {eps_grad}, {eps_root}",)
        )
    if not 0 <= beta1 < 1:
        raise ConfigurationError(*(f"beta1 must satisfy 0 <= beta1 < 1, got {beta1}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if update_rms_clip <= 0:
        raise ConfigurationError(
            *(f"update_rms_clip must be positive, got {update_rms_clip}",)
        )

    moment = _scale_by_adafactor(
        b1=beta1,
        b2_decay=decay_rate,
        eps_grad=eps_grad,
        eps_root=eps_root,
        update_rms_clip=update_rms_clip,
        noise_bias_correction=noise_bias_correction,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=None,
    )


__all__ = ["AdafactorState", "adafactor"]
