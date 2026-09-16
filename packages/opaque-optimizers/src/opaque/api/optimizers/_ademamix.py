"""AdEMAMix optimizer (Pagliardini et al., 2024).

Adam variant with **two** first-moment EMAs (a "fast" β₁ and a "slow"
β₃) blended by ``α``::

    m_fast_t = β₁ m_fast_{t-1} + (1 − β₁) g_t
    m_slow_t = β₃ m_slow_{t-1} + (1 − β₃) g_t        # no bias correction (β₃ ≈ 0.9999)
    v_t      = β₂ v_{t-1}      + (1 − β₂) g_t²

    m̂_fast = m_fast_t / (1 − β₁^t)
    v̂      = v_t      / (1 − β₂^t)

    update  = (m̂_fast + α · m_slow_t) / (√v̂ + ε)

Reference:
    Pagliardini, Ablin, Grangier, "The AdEMAMix Optimizer: Better,
    Faster, Older", arXiv:2409.03137.

The slow EMA is intentionally **not** bias-corrected (the paper notes
that with β₃ on the order of 0.9999 the bias is negligible after a
brief warm-up, and applying ``1 − β₃^t`` makes early steps numerically
unstable).

DP behavior.  The second moment EMA is structurally identical to
Adam's, so:

- ``NoisedPytree`` carries realized per-step σ and drives the φ-EMA bias
    correction on ``v̂`` exactly as in :func:`opaque.optimizers._adamw`.
- ``SecondMomentNoiseOutput`` substitutes the separately noised JME
  aggregate square by post-processing.
- The two first-moment EMAs are unaffected — they remain unbiased
  estimates of E[g] regardless of the noise injected into g.
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
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.pytree import tree_map

if TYPE_CHECKING:
    from opaque.pytree import ParamPath
    from opaque.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class AdEMAMixState:
    """State for AdEMAMix moment scaling."""

    m_fast: TensorPytree
    m_slow: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


def _scale_by_ademamix(
    b1: float,
    b2: float,
    b3: float,
    alpha: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    def init_fn(params: Any) -> AdEMAMixState:
        zeros = lambda p: torch.zeros_like(p)  # noqa: E731
        phi: float | dict = init_per_group_phi(params) if noise_bias_correction else 0.0
        return AdEMAMixState(
            m_fast=tree_map(zeros, params),
            m_slow=tree_map(zeros, params),
            nu=tree_map(zeros, params),
            phi=phi,
            step=0,
        )

    def update_fn(
        updates: Any,
        state: AdEMAMixState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdEMAMixState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ConfigurationError(
                *(
                    "ademamix.update() received both noisy_squared_grads and "
                    "noise_stddev (DP-BC); pass exactly one (or neither).",
                )
            )

        t = state.step + 1
        new_mf = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.m_fast, updates)
        new_ms = tree_map(lambda m, g: b3 * m + (1 - b3) * g, state.m_slow, updates)

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        # ---- v-update ----------------------------------------------------
        if noisy_squared_grads is not None:
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi

            def _compute_sm(
                mf: torch.Tensor, ms: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                combined = (mf / bc1) + alpha * ms
                v_hat = v / bc2
                v_eff = torch.where(v_hat > 0, v_hat, combined * combined).clamp(
                    min=bc_floor
                )
                return combined / (v_eff.sqrt() + eps)

            result = tree_map(_compute_sm, new_mf, new_ms, new_nu)
            return result, AdEMAMixState(
                m_fast=new_mf, m_slow=new_ms, nu=new_nu, phi=new_phi, step=t
            )

        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)
        effective = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:
            result = tree_map(
                lambda mf, ms, v: ((mf / bc1) + alpha * ms) / ((v / bc2).sqrt() + eps),
                new_mf,
                new_ms,
                new_nu,
            )
            return result, AdEMAMixState(
                m_fast=new_mf,
                m_slow=new_ms,
                nu=new_nu,
                phi=state.phi,
                step=t,
            )

        per_group = is_per_group(effective) or isinstance(state.phi, dict)

        if per_group:
            from opaque.api.optimizers._bias_correction import map_leaves_with_path

            new_phi: dict = {}

            def _bc_leaf(path, mf_node, ms_node, v_node):
                nv = resolve_noise_variance(effective, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = b2 * old_phi_k + (1 - b2) * nv
                new_phi[path] = new_phi_k
                phi_hat = new_phi_k / bc2
                v_raw = v_node / bc2
                if phi_hat > 0:
                    corrected = v_raw - phi_hat
                    v_hat = torch.where(corrected > 0, corrected, v_raw)
                else:
                    v_hat = v_raw
                return ((mf_node / bc1) + alpha * ms_node) / (v_hat.sqrt() + eps)

            result = map_leaves_with_path(_bc_leaf, new_mf, new_ms, new_nu)
        else:
            scalar_var = float(effective) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, b2)
            phi_hat = new_phi / bc2

            if phi_hat > 0:

                def _compute(mf, ms, v):
                    v_raw = v / bc2
                    corrected = v_raw - phi_hat
                    v_hat = torch.where(corrected > 0, corrected, v_raw)
                    return ((mf / bc1) + alpha * ms) / (v_hat.sqrt() + eps)
            else:

                def _compute(mf, ms, v):
                    return ((mf / bc1) + alpha * ms) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mf, new_ms, new_nu)

        return result, AdEMAMixState(
            m_fast=new_mf, m_slow=new_ms, nu=new_nu, phi=new_phi, step=t
        )

    return GradientTransformation(init_fn, update_fn)


def ademamix(
    lr: _LR = 1e-3,
    betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an AdEMAMix optimizer.

    Args:
        lr: Learning rate, scalar or schedule.
        betas: ``(β₁, β₂, β₃)``.  ``β₁`` is the fast first-moment EMA,
            ``β₂`` is the second moment, ``β₃`` is the slow first-moment
            EMA (paper default 0.9999).
        alpha: Weight on the slow EMA in the update mix.
        eps: Denominator stability constant.
        weight_decay: Decoupled WD coefficient by default.
        decoupled_weight_decay: ``False`` switches to L2-style.
        update_rms_clip: Optional StableAdamW-style model-wide RMS clip on
            the moment-scaled update.
        noise_bias_correction: If ``True``, subtract a β₂-EMA of the
            realized noise variance from the second moment when
            ``NoisedPytree`` updates are passed.  Defaults to ``False``;
            see ``docs/user-guide/optimizers.md`` for when to flip it on.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if len(betas) != 3:  # noqa: PLR2004 - AdEMAMix exposes its documented beta triple
        raise ConfigurationError(
            *(f"betas must contain exactly three values, got {betas}",)
        )
    b1, b2, b3 = betas
    for name, b in (("β₁", b1), ("β₂", b2), ("β₃", b3)):
        if not 0 <= b < 1:
            raise ConfigurationError(*(f"{name} must satisfy 0 <= b < 1, got {b}",))
    if alpha < 0:
        raise ConfigurationError(*(f"alpha must be non-negative, got {alpha}",))
    if eps <= 0:
        raise ConfigurationError(*(f"eps must be positive, got {eps}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ConfigurationError(
            *(f"update_rms_clip must be positive when set, got {update_rms_clip}",)
        )
    bc_floor = eps * eps
    moment = _scale_by_ademamix(
        b1=b1,
        b2=b2,
        b3=b3,
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


__all__ = ["AdEMAMixState", "ademamix"]
