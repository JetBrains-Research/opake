"""RMSprop optimizer with optional DP-aware bias correction.

RMSprop was introduced in Tieleman and Hinton's 2012 Coursera lecture notes,
`Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent
magnitude <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

Hinton's RMSprop: pure second-moment EMA, no first moment::

    nu_t = α nu_{t-1} + (1 − α) g_t²
    update = g_t / (√nu_t + eps)

DP behaviour.  Under noised gradients ``g̃ = g + ξ`` with
``ξ ~ N(0, σ²)``, ``E[g̃²] = g² + σ²``, so ``nu`` is biased upward by
``σ²`` in steady state.  The vanilla optimizer survives this (the
denominator is clipped, unlike Adagrad's runaway), but the inflated
denominator shrinks the effective LR.

``NoisedPytree`` updates activate a φ-EMA correction using the realized σ
carried by the wrapper.  Unlike Adam, RMSprop does *not* divide ``nu``
by ``1 − α^t`` for bias correction, so ``φ`` and ``nu`` accumulate the
noise contribution at exactly the same rate.  Subtracting one from the
other directly yields the unbiased estimate::

    φ_t = α φ_{t-1} + (1 − α) σ_t²
    nu_corrected = max(nu_t − φ_t, floor)
    update = g_t / (√nu_corrected + eps)

``SecondMomentNoiseOutput`` substitutes a private squared-gradient ``g²``
directly into the ``nu`` update (post-processing); no φ-EMA correction is
applied in that branch.
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

from opake.api.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    resolve_noise_variance,
    update_phi_ema,
)
from opake.api.optimizers._chain import make_optimizer_chain
from opake.pytree import tree_map

if TYPE_CHECKING:
    from opake.pytree import ParamPath
    from opake.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class RMSpropState:
    """State for RMSprop moment scaling.

    Attributes:
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar, or ``dict[ParamPath, float]``
            when BC is enabled).  Stays at zero unless a ``NoisedPytree``
            update supplies realized σ metadata.  Same accumulation rate
            as ``nu`` (no bias-correction division), so subtracting
            directly yields the unbiased estimate.
        step: Number of completed updates.
    """

    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


def _scale_by_rmsprop(
    alpha: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    def init_fn(params: Any) -> RMSpropState:
        nu = tree_map(torch.zeros_like, params)
        phi: float | dict = init_per_group_phi(params) if noise_bias_correction else 0.0
        return RMSpropState(nu=nu, phi=phi, step=0)

    def update_fn(
        updates: Any,
        state: RMSpropState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, RMSpropState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ConfigurationError(
                *(
                    "rmsprop.update() received both noisy_squared_grads and "
                    "noise_stddev (DP-BC); pass exactly one (or neither).",
                )
            )

        t = state.step + 1

        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).
            new_nu = tree_map(
                lambda v, g2: alpha * v + (1 - alpha) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi

            def _compute_sm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                # Noisy g² can be negative when second-stream noise dominates the
                # signal; fall back to g² so the update magnitude stays ≈ 1.
                v_eff = torch.where(v > 0, v, g * g).clamp(min=bc_floor)
                return g / (v_eff.sqrt() + eps)

            result = tree_map(_compute_sm, updates, new_nu)
            return result, RMSpropState(nu=new_nu, phi=new_phi, step=t)

        new_nu = tree_map(
            lambda v, g: alpha * v + (1 - alpha) * g * g,
            state.nu,
            updates,
        )

        effective = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:
            result = tree_map(
                lambda g, v: g / (v.sqrt() + eps),
                updates,
                new_nu,
            )
            return result, RMSpropState(nu=new_nu, phi=state.phi, step=t)

        per_group = is_per_group(effective) or isinstance(state.phi, dict)

        if per_group:
            from opake.api.optimizers._bias_correction import map_leaves_with_path

            new_phi: dict = {}

            def _bc_leaf(path, g_node, v_node):
                nv = resolve_noise_variance(effective, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = alpha * old_phi_k + (1 - alpha) * nv
                new_phi[path] = new_phi_k
                if new_phi_k > 0:
                    corrected = v_node - new_phi_k
                    v_corrected = torch.where(corrected > 0, corrected, v_node)
                else:
                    v_corrected = v_node
                return g_node / (v_corrected.sqrt() + eps)

            result = map_leaves_with_path(_bc_leaf, updates, new_nu)
        else:
            scalar_var = float(effective) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, alpha)

            if new_phi > 0:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    corrected = v - new_phi
                    v_corrected = torch.where(corrected > 0, corrected, v)
                    return g / (v_corrected.sqrt() + eps)
            else:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return g / (v.sqrt() + eps)

            result = tree_map(_compute, updates, new_nu)

        return result, RMSpropState(nu=new_nu, phi=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


def rmsprop(
    lr: _LR = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an RMSprop optimizer with optional DP-aware bias correction.

    Args:
        lr: Learning rate, scalar or schedule.
        alpha: Second-moment EMA decay (paper default 0.99).
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient.
        decoupled_weight_decay: ``True`` selects decoupled WD (applied
            after moment scaling); ``False`` folds ``wd·params`` into
            the gradient before moment scaling (L2 regularisation).
        update_rms_clip: Optional StableAdamW-style model-wide RMS clip on
            the moment-scaled update.
        noise_bias_correction: If ``True``, subtract an ``alpha``-EMA of
            the realized noise variance from the second moment when
            ``NoisedPytree`` updates are passed.  Defaults to ``False``;
            see ``docs/user-guide/optimizers.md`` for when to flip it on.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if eps <= 0:
        raise ConfigurationError(*(f"eps must be positive, got {eps}",))
    if not 0 <= alpha < 1:
        raise ConfigurationError(*(f"alpha must satisfy 0 <= alpha < 1, got {alpha}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ConfigurationError(
            *(f"update_rms_clip must be positive when set, got {update_rms_clip}",)
        )
    bc_floor = eps * eps
    moment = _scale_by_rmsprop(
        alpha=alpha,
        eps=eps,
        noise_bias_correction=noise_bias_correction,
        bc_floor=bc_floor,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["RMSpropState", "rmsprop"]
