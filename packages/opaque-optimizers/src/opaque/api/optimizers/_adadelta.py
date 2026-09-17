"""Adadelta optimizer with two-EMA DP bias correction.

Vanilla Adadelta (Zeiler 2012, arXiv:1212.5701) maintains two EMAs::

    E[g²]_t  = ρ E[g²]_{t-1}  + (1−ρ) g_t²
    Δx_t     = -(√(E[Δx²]_{t-1} + ε) / √(E[g²]_t + ε)) · g_t
    E[Δx²]_t = ρ E[Δx²]_{t-1} + (1−ρ) (Δx_t)²

Adadelta is famously learning-rate-free; the per-element ratio of update
RMS to gradient RMS *is* the adaptive step size.  The opaque chain still
threads a global ``lr`` factor (default 1.0) so users can scale the
update if they need to.

DP behaviour.  Under noised gradients ``g̃_t = g_clean + ξ_t`` with
``ξ_t ~ N(0, σ_t² I)``, both EMAs accumulate noise:

- ``E[g²]_t`` inherits the same per-step ``σ²`` offset as Adam's ``v_t``.
  Subtract a ρ-EMA ``φ_g`` of ``σ²`` to recover the unbiased estimate.
- ``E[Δx²]_t`` is more subtle.  ``Δx_t = -coef_t · g̃_t`` is linear in
  ``g̃_t``, so the noise variance injected into ``Δx_t`` at element ``i``
  is ``coef_t,i² · σ_t²`` — a *known* per-element scalar (the coef is
  computed before noise is added).  Maintain a second EMA
  ``φ_dx`` per element::

      φ_dx,t = ρ φ_dx,{t-1} + (1−ρ) (coef_t · σ_t)²
      E[Δx²]_corrected,t = E[Δx²]_t − φ_dx,t   where that is positive
                           E[Δx²]_t            elsewhere

  Use ``E[Δx²]_corrected,{t-1}`` in the next step's coef.

The two φ-EMAs decay at the same rate ``ρ`` as their respective second
moments, so subtraction at any step is consistent with the EMA history.

Memory cost: ``2 · |params|`` tensors for the second moments (same as
vanilla).  Optional bias-correction state (``φ_g``, ``φ_dx``) is
allocated only when ``noise_bias_correction=True``, adding
~``|params|`` elements total (``φ_dx`` is a per-element tensor,
``φ_g`` is a scalar or dict per leaf).

The derivation is a straightforward propagation of Gaussian variance
through the linear scaling step; no published prior, but the math is
tight enough to drop in.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from opaque.exceptions import ConfigurationError

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.api.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    resolve_noise_variance,
    update_phi_ema,
    walk_dict_leaves,
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.pytree import tree_map

if TYPE_CHECKING:
    from opaque.pytree import ParamPath
    from opaque.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdadeltaState:
    """Immutable state for Adadelta with optional DP bias correction.

    Attributes:
        v_g: Squared-gradient EMA ``E[g²]`` (pytree matching params).
        v_dx: Squared-update EMA ``E[Δx²]`` (pytree matching params).
        phi_g: Gradient-noise-variance EMA — allocated only when
            ``noise_bias_correction=True``. When BC is enabled, always
            initialized as ``dict[ParamPath, float]`` (per-leaf storage).
            Stays ``None`` when BC is disabled.
        phi_dx: Update-noise-variance EMA.  Per-element pytree matching
            params (allocated only when ``noise_bias_correction=True``).
            The per-step variance ``coef_t² · σ²`` is element-wise even
            when σ is scalar. Stays ``None`` when BC is disabled.
        step: Number of completed updates.
    """

    v_g: TensorPytree
    v_dx: TensorPytree
    phi_g: float | dict[ParamPath, float] | None
    phi_dx: TensorPytree | None
    step: int


# ---------------------------------------------------------------------------
# Moment scaler
# ---------------------------------------------------------------------------


def _rms(tensor_or_zero: torch.Tensor, eps: float) -> torch.Tensor:
    """``√(x + ε)`` — Adadelta's smoothed RMS."""
    return (tensor_or_zero + eps).sqrt()


def _scale_by_adadelta(
    rho: float,
    eps: float,
    noise_bias_correction: bool,
) -> GradientTransformation:
    """Adadelta moment scaling with two-EMA DP bias correction.

    Returns the un-negated update (``coef · g``); the chain's
    ``scale_by_neg_lr`` applies the sign and global lr scaling at the end.
    """

    def init_fn(params: Any) -> AdadeltaState:
        v_g = tree_map(torch.zeros_like, params)
        v_dx = tree_map(torch.zeros_like, params)
        if noise_bias_correction:
            phi_dx = tree_map(torch.zeros_like, params)
            phi_g: float | dict = init_per_group_phi(params)
        else:
            phi_dx = None
            phi_g = None
        state = AdadeltaState(v_g=v_g, v_dx=v_dx, phi_g=phi_g, phi_dx=phi_dx, step=0)
        # Validate state consistency: BC enabled but state has None phi fields.
        if noise_bias_correction and state.phi_dx is None:
            raise ConfigurationError(
                *(
                    "Attempted to initialize Adadelta with noise_bias_correction=True "
                    "but state.phi_dx is None. This indicates a configuration mismatch "
                    "or a corrupted checkpoint. Re-initialize state or disable BC.",
                )
            )
        return state

    def update_fn(
        updates: Any,
        state: AdadeltaState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdadeltaState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ConfigurationError(
                *(
                    "adadelta.update() received both noisy_squared_grads and "
                    "noise_stddev (DP-BC); pass exactly one (or neither).",
                )
            )

        t = state.step + 1
        effective = noise_stddev if noise_stddev is not None else 0.0

        # ---- E[g²] update -------------------------------------------
        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).
            # phi_g is left at its current value (post-processing already
            # debiased v_g for this step).  phi_dx similarly stays put —
            # we don't have σ in this branch, so we can't advance the
            # update-noise EMA without divergence; document the
            # trade-off below.
            new_v_g = tree_map(
                lambda v, g2: rho * v + (1 - rho) * g2,
                state.v_g,
                noisy_squared_grads,
            )
            new_phi_g: Any = state.phi_g
            new_phi_dx: Any = state.phi_dx
        else:
            new_v_g = tree_map(
                lambda v, g: rho * v + (1 - rho) * g * g, state.v_g, updates
            )

            if noise_bias_correction:
                if is_per_group(effective) or isinstance(state.phi_g, dict):
                    new_phi_g_dict: dict[str, float] = {}
                    for path, _leaf in walk_dict_leaves(new_v_g):
                        nv = resolve_noise_variance(effective, path)
                        old = (
                            state.phi_g.get(path, 0.0)
                            if isinstance(state.phi_g, dict)
                            else state.phi_g
                        )
                        new_phi_g_dict[path] = rho * old + (1 - rho) * nv
                    new_phi_g = new_phi_g_dict
                else:
                    new_phi_g = update_phi_ema(state.phi_g, float(effective) ** 2, rho)
            else:
                new_phi_g = state.phi_g
            new_phi_dx = state.phi_dx  # advanced below alongside v_dx

        # ---- Per-leaf Δx, v_dx, and (optional) phi_dx ----------------
        # Walk leaves so we can resolve per-group σ and read/write the
        # per-leaf phi_dx tensor.  Single walk handles both paths.
        import optree

        from opaque.api.engine.pytree import tree_flatten_with_paths

        def _phi_g_for(path) -> float:
            if isinstance(new_phi_g, dict):
                return float(new_phi_g.get(path, 0.0))
            elif new_phi_g is None:
                return 0.0
            return float(new_phi_g)

        u_paths, u_leaves, u_def = tree_flatten_with_paths(updates)
        _, vg_leaves, _ = tree_flatten_with_paths(new_v_g)
        _, vdx_leaves, vdx_def = tree_flatten_with_paths(state.v_dx)

        # Only flatten phi_dx if it's not None (i.e., BC is active)
        if new_phi_dx is not None:
            _, phidx_leaves, phidx_def = tree_flatten_with_paths(new_phi_dx)
        else:
            phidx_leaves = [None] * len(u_leaves)
            phidx_def = None

        dx_out: list[Any] = []
        vdx_out: list[Any] = []
        phidx_out: list[Any] = []

        for path, updates_node, v_g_t, v_dx_node, phi_dx_node in zip(
            u_paths, u_leaves, vg_leaves, vdx_leaves, phidx_leaves, strict=True
        ):
            if noise_bias_correction and noisy_squared_grads is None:
                phi_g_path = _phi_g_for(path)
            else:
                phi_g_path = 0.0

            if phi_g_path > 0:
                corrected_g = v_g_t - phi_g_path
                v_g_corrected = torch.where(corrected_g > 0, corrected_g, v_g_t)
            else:
                v_g_corrected = v_g_t
            if noisy_squared_grads is not None:
                v_g_corrected = torch.where(
                    v_g_corrected > 0,
                    v_g_corrected,
                    updates_node * updates_node,
                )

            if (
                noise_bias_correction
                and noisy_squared_grads is None
                and phi_dx_node is not None
            ):
                corrected_dx = v_dx_node - phi_dx_node
                v_dx_corrected_prev = torch.where(
                    corrected_dx > 0, corrected_dx, v_dx_node
                )
            else:
                v_dx_corrected_prev = v_dx_node

            rms_g = _rms(v_g_corrected, eps)
            rms_dx_prev = _rms(v_dx_corrected_prev, eps)

            coef = rms_dx_prev / rms_g
            dx = coef * updates_node

            v_dx_new_t = rho * v_dx_node + (1 - rho) * dx * dx

            if (
                noise_bias_correction
                and noisy_squared_grads is None
                and phi_dx_node is not None
            ):
                sigma_sq = (
                    resolve_noise_variance(effective, path)
                    if is_per_group(effective)
                    else float(effective) ** 2
                )
                noise_var_dx = coef * coef * sigma_sq
                phi_dx_new_t = rho * phi_dx_node + (1 - rho) * noise_var_dx
            else:
                phi_dx_new_t = phi_dx_node

            dx_out.append(dx)
            vdx_out.append(v_dx_new_t)
            phidx_out.append(phi_dx_new_t)

        result_tree = optree.tree_unflatten(u_def, dx_out)
        new_v_dx_tree = optree.tree_unflatten(vdx_def, vdx_out)

        if new_phi_dx is not None:
            new_phi_dx_tree = optree.tree_unflatten(phidx_def, phidx_out)
        else:
            new_phi_dx_tree = None

        return result_tree, AdadeltaState(
            v_g=new_v_g,
            v_dx=new_v_dx_tree,
            phi_g=new_phi_g,
            phi_dx=new_phi_dx_tree,
            step=t,
        )

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adadelta(
    lr: _LR = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an Adadelta optimizer with two-EMA DP bias correction.

    Args:
        lr: Global multiplier on the update.  Adadelta is classically
            learning-rate-free (paper default 1.0); a non-unit value
            scales the per-element adaptive step.
        rho: EMA decay for both squared-gradient and squared-update
            accumulators (paper default 0.9, torchopt default 0.9).
        eps: Numerical floor inside the RMS computations.
        weight_decay: Weight-decay coefficient.
        decoupled_weight_decay: ``True`` selects decoupled WD (post
            moment scaling); ``False`` folds ``wd · params`` into the
            gradient before moment scaling (L2 regularisation).
        update_rms_clip: Optional StableAdamW-style model-wide RMS clip on
            the moment-scaled update.
        noise_bias_correction: If ``True``, maintain and subtract two
            parallel ρ-EMAs of the realized noise variance:

            - ``φ_g``: a ρ-EMA of σ² from ``E[g²]`` (recovers unbiased
              squared-gradient estimate).
            - ``φ_dx``: a ρ-EMA of ``coef² · σ²`` from ``E[Δx²]``
              (recovers unbiased squared-update estimate).

            Both EMAs decay at the same rate ρ as their respective
            second moments, so subtraction is consistent with the EMA
            history.  **Memory cost**: when enabled, allocates one
            scalar/dict ``φ_g`` and one per-element tensor ``φ_dx``
            per leaf.  Total overhead: ~``|params|`` extra elements.
            Defaults to ``False``; flip on to ablate against vanilla
            Adadelta under noise.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    DP usage notes:

        - ``NoisedPytree`` updates with ``noise_bias_correction=True``
          activate both BC EMAs.  σ travels on the wrapper.
        - ``SecondMomentNoiseOutput`` updates substitute the privatised
          ``g²`` stream into ``E[g²]`` directly.  In this branch we
          freeze ``φ_dx`` because σ isn't carried alongside the second-
          moment stream — the update-noise variance EMA cannot advance
          without it.  Use the ``NoisedPytree`` path for full BC.
    """
    if eps <= 0:
        raise ConfigurationError(*(f"eps must be positive, got {eps}",))
    if not 0 <= rho < 1:
        raise ConfigurationError(*(f"rho must satisfy 0 <= rho < 1, got {rho}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ConfigurationError(
            *(f"update_rms_clip must be positive when set, got {update_rms_clip}",)
        )

    moment = _scale_by_adadelta(
        rho=rho,
        eps=eps,
        noise_bias_correction=noise_bias_correction,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["AdadeltaState", "adadelta"]
