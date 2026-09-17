"""Rectified Adam (RAdam) with DP noise-variance bias correction.

RAdam is Adam plus a variance-rectification factor ``r_t`` that
depends only on ``β₂`` and ``t`` — not on the gradient — so the
DP-BC machinery from :mod:`opaque.optimizers._adam` ports across
verbatim.  When the rectification term is undefined (early
training, ``ρ_t ≤ 5``) RAdam degenerates to SGD-of-momentum, which
needs no second-moment bias correction in the first place.

Reference:
    Liu et al., "On the Variance of the Adaptive Learning Rate and
    Beyond", arXiv:1908.03265.

The optimizer follows torchopt's ``GradientTransformation`` protocol::

    opt = radam(lr=1e-3, weight_decay=0.0)
    state = opt.init(params)
    updates, state = opt.update(noisy_grads, state, params=p)

DP-aware modes mirror :func:`opaque.optimizers._adamw`:

- ``noise_bias_correction=True`` activates the φ-EMA correction on
  the second moment when ``NoisedPytree`` updates are passed.
- ``SecondMomentNoiseOutput`` updates substitute an externally
  privatised ``g²`` stream in place of squaring the noised gradient.
  Both BC and the second-moment substitution target the same source
  of bias; pick one per training run.

The rectification factor::

    ρ_∞ = 2 / (1 − β₂) − 1
    ρ_t = ρ_∞ − 2 t β₂^t / (1 − β₂^t)
    r_t = sqrt(((ρ_t − 4)(ρ_t − 2)ρ_∞) / ((ρ_∞ − 4)(ρ_∞ − 2)ρ_t))

is applied to ``m̂_t / sqrt(v̂_corrected)`` only when ``ρ_t > 5``.
For ``ρ_t ≤ 5`` the update is ``m̂_t`` (no v scaling), which is
identical to vanilla SGD with momentum and so naturally robust to
DP noise on the second moment.
"""

from __future__ import annotations

import dataclasses
import math
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

# RAdam's rectification gate.  When ``ρ_t ≤ ρ_THRESHOLD`` the variance
# rectification term is undefined / non-real and the optimizer falls
# back to SGD-of-momentum.  Liu et al. fix this at 4 in the paper but
# implementations universally use 5 to avoid the boundary case where
# ``r_t`` is zero or imaginary.
_RHO_THRESHOLD: float = 5.0


# ---------------------------------------------------------------------------
# State (shape-compatible with AdamState; re-using AdamState would cross-
# couple checkpoint formats, so keep distinct).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RAdamState:
    """Immutable state for RAdam moment scaling.

    Same shape as :class:`opaque.optimizers.AdamState` — ``phi`` rides
    along regardless of whether DP bias correction is active.

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar, or ``dict[ParamPath, float]``
            when BC is enabled).
        step: Number of completed updates.
    """

    mu: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


# ---------------------------------------------------------------------------
# Rectification factor — depends only on (β₂, t)
# ---------------------------------------------------------------------------


def _rho_t(b2: float, t: int) -> float:
    """RAdam's per-step degrees-of-freedom estimate."""
    rho_inf = 2.0 / (1.0 - b2) - 1.0
    return rho_inf - 2.0 * t * (b2**t) / (1.0 - b2**t)


def _rectification(b2: float, t: int) -> float | None:
    """Return ``r_t`` when defined, else ``None`` (SGD-of-momentum branch)."""
    rho_inf = 2.0 / (1.0 - b2) - 1.0
    rho_t = _rho_t(b2, t)
    if rho_t <= _RHO_THRESHOLD:
        return None
    numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
    denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
    return math.sqrt(numerator / denominator)


# ---------------------------------------------------------------------------
# Moment scaler — branches on rho_t > 5 (Adam-shaped) vs early SGD-of-momentum
# ---------------------------------------------------------------------------


def _scale_by_radam(
    b1: float,
    b2: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    """RAdam moment scaling with optional DP bias correction.

    Two execution branches selected per step by ``ρ_t``:

    - ``ρ_t > 5``: apply the variance-rectification factor ``r_t`` to
      ``m̂ / √v̂_corrected``.  ``v̂_corrected = max(v̂ − φ̂, floor)``
      under BC; otherwise just ``v̂``.
    - ``ρ_t ≤ 5``: skip the second moment entirely; update is ``m̂``
      (SGD with momentum on noisy gradients).  Naturally DP-robust.
    """

    def init_fn(params: Any) -> RAdamState:
        mu = tree_map(torch.zeros_like, params)
        nu = tree_map(torch.zeros_like, params)
        phi: float | dict = init_per_group_phi(params) if noise_bias_correction else 0.0
        return RAdamState(mu=mu, nu=nu, phi=phi, step=0)

    def update_fn(
        updates: Any,
        state: RAdamState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, RAdamState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ConfigurationError(
                *(
                    "radam.update() received both noisy_squared_grads and "
                    "noise_stddev; these select mutually exclusive v-update "
                    "branches.  Pass exactly one (or neither, for vanilla "
                    "RAdam).",
                )
            )

        t = state.step + 1

        # First moment is identical across branches.
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        bc1 = 1 - b1**t

        # Determine the rectification factor for this step.
        r_t = _rectification(b2, t)

        # Update the second moment regardless of which output branch we
        # take.  v_t accumulates noise variance every step; the φ-EMA
        # needs to track that accumulation so the correction at the
        # first rectified step reflects all prior noise contributions,
        # not just the current step.
        if noisy_squared_grads is not None:
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
            # External second-moment branch: post-processing already
            # debiased v; leave φ alone.
            new_phi: Any = state.phi
        else:
            new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)
            # Advance φ through both branches when BC is active so the
            # EMA tracks v's noise history (see comment above).  When
            # BC is off, φ stays at its initial zero.
            if noise_bias_correction:
                effective_stddev = noise_stddev if noise_stddev is not None else 0.0
                if is_per_group(effective_stddev) or isinstance(state.phi, dict):
                    new_phi_dict: dict[str, float] = {}
                    for path, _leaf in walk_dict_leaves(new_mu):
                        nv = resolve_noise_variance(effective_stddev, path)
                        old_phi_k = (
                            state.phi.get(path, 0.0)
                            if isinstance(state.phi, dict)
                            else state.phi
                        )
                        new_phi_dict[path] = b2 * old_phi_k + (1 - b2) * nv
                    new_phi = new_phi_dict
                else:
                    scalar_var = float(effective_stddev) ** 2
                    new_phi = update_phi_ema(state.phi, scalar_var, b2)
            else:
                new_phi = state.phi

        # Early branch — no v scaling, plain SGD-of-momentum on m̂.
        # The φ-EMA already advanced above to keep parity with v.
        if r_t is None:
            result = tree_map(lambda m: m / bc1, new_mu)
            return result, RAdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        bc2 = 1 - b2**t

        # ---- noisy_squared_grads branch (no φ correction needed) ----
        if noisy_squared_grads is not None:

            def _compute_sm(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                m_hat = m / bc1
                v_hat = v / bc2
                v_eff = torch.where(v_hat > 0, v_hat, m_hat * m_hat).clamp(min=bc_floor)
                return r_t * m_hat / (v_eff.sqrt() + eps)

            result = tree_map(_compute_sm, new_mu, new_nu)
            return result, RAdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        if not noise_bias_correction:
            result = tree_map(
                lambda m, v: r_t * (m / bc1) / ((v / bc2).sqrt() + eps),
                new_mu,
                new_nu,
            )
            return result, RAdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        # ---- BC branch — φ already advanced above; subtract here. ---
        if isinstance(new_phi, dict):
            from opaque.api.optimizers._bias_correction import map_leaves_with_path

            def _bc_leaf(path, mu_node, nu_node):
                phi_hat = new_phi[path] / bc2
                m_hat = mu_node / bc1
                v_raw = nu_node / bc2
                if phi_hat > 0:
                    corrected = v_raw - phi_hat
                    v_hat = torch.where(corrected > 0, corrected, v_raw)
                else:
                    v_hat = v_raw
                return r_t * m_hat / (v_hat.sqrt() + eps)

            result = map_leaves_with_path(_bc_leaf, new_mu, new_nu)
        else:
            phi_hat = new_phi / bc2

            if phi_hat > 0:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    v_raw = v / bc2
                    corrected = v_raw - phi_hat
                    v_hat = torch.where(corrected > 0, corrected, v_raw)
                    return r_t * (m / bc1) / (v_hat.sqrt() + eps)
            else:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return r_t * (m / bc1) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mu, new_nu)

        return result, RAdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def radam(
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = False,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create a Rectified Adam optimizer with the wrapper-aware update API.

    Args:
        lr: Learning rate, scalar or ``step → float`` callable schedule.
        betas: ``(β₁, β₂)`` coefficients.
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient.
        decoupled_weight_decay: ``True`` selects RAdamW (decoupled WD,
            applied post moment scaling).  ``False`` (default) folds
            ``weight_decay * params`` into the gradient before moment
            scaling — matching the original RAdam paper.
        update_rms_clip: When not ``None``, divides the moment-scaled
            update by ``max(1, rms / threshold)``, with ``rms`` computed
            model-wide across all tensor leaves. Applies on every step
            (clipping happens in the chain stage that follows the moment
            scaler), including the SGD-of-momentum early branch.
        noise_bias_correction: If ``True``, advance a β₂-EMA of the
            realized noise variance every step when ``NoisedPytree``
            updates are passed; subtract it from the second moment
            before the sqrt only on rectified-branch steps
            (``ρ_t > 5``).  The early branch advances ``φ`` to keep
            it consistent with ``v``'s noise history but does not
            apply it (``v`` is not consumed there).  Defaults to
            ``False``; see ``docs/user-guide/optimizers.md`` for when
            to flip it on.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    DP usage notes:

        - The early ``ρ_t ≤ 5`` branch is naturally DP-robust because
          it does not consume the second moment — ``E[m̂_t]`` is
          unbiased (additive noise has zero mean).
        - When ``noise_bias_correction=True`` is set but ``r_t`` is
          undefined for the current step, the BC EMA still advances —
          it just isn't applied this step.  Once ``ρ_t > 5`` the
          accumulated φ correctly reflects the noise history.
    """
    _validate(eps, betas, weight_decay, update_rms_clip)
    bc_floor = eps * eps
    moment = _scale_by_radam(
        b1=betas[0],
        b2=betas[1],
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


def _validate(
    eps: float,
    betas: tuple[float, float],
    weight_decay: float,
    update_rms_clip: float | None,
) -> None:
    if eps <= 0:
        raise ConfigurationError(*(f"eps must be positive, got {eps}",))
    if len(betas) != 2:  # noqa: PLR2004 - RAdam exposes the documented beta pair
        raise ConfigurationError(
            *(f"betas must contain exactly two values, got {betas}",)
        )
    b1, b2 = betas
    if not 0 <= b1 < 1:
        raise ConfigurationError(*(f"beta_1 must satisfy 0 <= beta_1 < 1, got {b1}",))
    if not 0 <= b2 < 1:
        raise ConfigurationError(*(f"beta_2 must satisfy 0 <= beta_2 < 1, got {b2}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ConfigurationError(
            *(f"update_rms_clip must be positive when set, got {update_rms_clip}",)
        )


__all__ = ["RAdamState", "radam"]
