"""Universal Adam / AdamW factory.

Single entry point for the Adam family.  Handles four orthogonal
behaviors selected at factory time and by the update value type:

1. **L2 vs decoupled weight decay** — constructor flag
   ``decoupled_weight_decay``.  ``True`` is AdamW (Loshchilov &
   Hutter, `Decoupled Weight Decay Regularization
   <https://arxiv.org/abs/1711.05101>`_); ``False`` is the original Adam
   (Kingma & Ba, `Adam: A Method for Stochastic Optimization
   <https://arxiv.org/abs/1412.6980>`_) with L2 regularisation folded
   into the gradient.

2. **StableAdamW RMS clip** — constructor knob ``update_rms_clip``.
   ``None`` disables the clip; a positive float divides the moment-scaled
   update by ``max(1, rms / threshold)``.  See Wortsman et al.,
   `Stable and low-precision training for large-scale vision-language
   models <https://arxiv.org/abs/2304.13013>`_ (2023).

3. **DP noise-variance bias correction** — pass ``NoisedPytree`` updates
    from a DP noise mechanism with ``noise_bias_correction=True``.  The
    wrapper carries the realized per-step σ; the second moment is then
    corrected by a β₂-EMA of the noise variance.  Chooi et al.,
    "DP-AdamW", arXiv:2511.07843.

4. **Private second-moment stream** — pass ``SecondMomentNoiseOutput``
    to ``update()`` to bypass squaring the noised gradient and use a
    privately-estimated ``g²`` instead.  Kalinin, Upadhyay, Lampert,
    "Continual Release Moment Estimation with Differential Privacy",
    arXiv:2502.06597.

Modes (3) and (4) target the same source of bias — that ``E[(g+noise)²]``
is not ``E[g²]`` — by different means.  They are alternatives, not stack
on top of each other; pick one per training run.  They are also mutually
exclusive per step by construction, because a single ``update()`` value
is either ``NoisedPytree`` or ``SecondMomentNoiseOutput``.

The optimizer follows torchopt's ``GradientTransformation`` protocol::

    opt = adamw(lr=1e-3, weight_decay=0.01)
    state = opt.init(params)

    # Vanilla AdamW:
    updates, state = opt.update(grads, state, params=p)

    # DP-AdamW-BC (σ travels with the noised gradients):
    updates, state = opt.update(noisy_grads, state, params=p)

    # DP-AdamW with a private second-moment stream:
    updates, state = opt.update(second_moment_output, state, params=p)
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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdamState:
    """Immutable state for Adam-family moment scaling.

    Carries the noise-variance EMA ``phi`` regardless of whether DP
    bias correction is actively in use — this keeps the state shape
    constant across calls so checkpoints don't depend on call history.
    With ``noise_bias_correction=True``, ``phi`` is a path-keyed dict
    from init (stable for ``state_dict`` / ``from_state_dict``).

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar, or ``dict[ParamPath, float]``
            when BC is enabled).  Stays at zero unless ``NoisedPytree``
            updates supply realized σ metadata.
        step: Number of completed updates.
    """

    mu: TensorPytree
    nu: TensorPytree
    phi: float | dict[ParamPath, float]
    step: int


# ---------------------------------------------------------------------------
# Moment scaler — handles vanilla / BC / external second-moment branches
# ---------------------------------------------------------------------------


def _scale_by_adam(
    b1: float,
    b2: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    """Adam moment scaling with optional DP bias correction or private second moments.

    This is an **internal** moment primitive: :func:`make_optimizer_chain` calls
    ``update()`` and injects DP routing only from ``NoisedPytree`` /
    ``SecondMomentNoiseOutput`` wrappers on ``updates`` — not from public
    per-step kwargs on the returned optimizer.

    Branches (selected by injected ``noise_stddev`` / ``noisy_squared_grads``):

    - Injected privatised second moment: v-update consumes that stream;
      no φ-EMA (post-processed stream).  Injected ``noise_stddev`` ignored.
    - Injected non-zero ``noise_stddev``: square the (possibly noised) gradient
      and apply BC::

          v̂_corrected = max(v̂ − φ̂, floor)

    - Neither: vanilla Adam.
    """

    def init_fn(params: Any) -> AdamState:
        mu = tree_map(torch.zeros_like, params)
        nu = tree_map(torch.zeros_like, params)
        phi: float | dict = init_per_group_phi(params) if noise_bias_correction else 0.0
        return AdamState(mu=mu, nu=nu, phi=phi, step=0)

    def update_fn(
        updates: Any,
        state: AdamState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdamState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ConfigurationError(
                *(
                    "adamw.update() received both noisy_squared_grads and "
                    "noise_stddev (DP-BC); these select mutually exclusive v-update "
                    "branches.  Pass exactly one (or neither, for vanilla AdamW).",
                )
            )

        t = state.step + 1

        # First moment is the same in all branches.
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)

        # ---- v-update ----------------------------------------------------
        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).  No φ-EMA
            # correction (post-processing already gave us an unbiased v).
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi  # unchanged
            bc1 = 1 - b1**t
            bc2 = 1 - b2**t

            def _compute_sm(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                m_hat = m / bc1
                v_hat = v / bc2
                # Noisy g² can be negative when second-stream noise dominates the
                # signal (g² ≪ σ_second).  Falling back to the squared first moment
                # keeps the update magnitude ≈ 1 (sign-gradient-like) for those
                # elements instead of collapsing the denominator to bc_floor → ∞ update.
                v_eff = torch.where(v_hat > 0, v_hat, m_hat * m_hat).clamp(min=bc_floor)
                return m_hat / (v_eff.sqrt() + eps)

            result = tree_map(_compute_sm, new_mu, new_nu)
            return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        # Standard / BC branch: square the (possibly noised) gradient.
        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t
        effective_stddev = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:
            result = tree_map(
                lambda m, v: (m / bc1) / ((v / bc2).sqrt() + eps),
                new_mu,
                new_nu,
            )
            return result, AdamState(mu=new_mu, nu=new_nu, phi=state.phi, step=t)

        per_group = is_per_group(effective_stddev) or isinstance(state.phi, dict)

        if per_group:
            # Per-leaf path: walk moments by optree ParamPath so nested
            # param pytrees match :class:`PerGroup.groups` keys.
            from opake.api.optimizers._bias_correction import map_leaves_with_path

            new_phi: dict = {}

            def _bc_leaf(path, mu_node, nu_node):
                nv = resolve_noise_variance(effective_stddev, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = b2 * old_phi_k + (1 - b2) * nv
                new_phi[path] = new_phi_k
                m_hat = mu_node / bc1
                phi_hat = new_phi_k / bc2
                v_raw = nu_node / bc2
                if phi_hat > 0:
                    corrected = v_raw - phi_hat
                    v_hat = torch.where(corrected > 0, corrected, v_raw)
                else:
                    v_hat = v_raw
                return m_hat / (v_hat.sqrt() + eps)

            result = map_leaves_with_path(_bc_leaf, new_mu, new_nu)
        else:
            scalar_var = float(effective_stddev) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, b2)
            phi_hat = new_phi / bc2

            if phi_hat > 0:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    v_hat = v / bc2
                    corrected = v_hat - phi_hat
                    denom = torch.where(corrected > 0, corrected, v_hat).sqrt() + eps
                    return (m / bc1) / denom
            else:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return (m / bc1) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mu, new_nu)

        return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adam(
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an Adam optimizer with Opake's wrapper-aware update API.

    This is the original Adam/L2 weight-decay variant of :func:`adamw`.
    ``NoisedPytree`` and ``SecondMomentNoiseOutput`` updates are routed the
    same way as AdamW, so callers do not need an optimizer-specific branch.
    """
    return adamw(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        decoupled_weight_decay=False,
        update_rms_clip=update_rms_clip,
        noise_bias_correction=noise_bias_correction,
    )


def adamw(
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Universal Adam / AdamW factory.

    Args:
        lr: Learning rate, scalar or ``step → float`` callable schedule.
        betas: ``(β₁, β₂)`` coefficients for first / second moment EMAs.
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient (decoupled by default).
        decoupled_weight_decay: ``True`` selects AdamW (decoupled WD,
            modern default).  ``False`` selects the original Adam, where
            ``weight_decay * params`` is added to the gradient before
            moment scaling — i.e. L2 regularisation enters the EMAs.
        update_rms_clip: When not ``None``, divides the moment-scaled
            update by ``max(1, rms / threshold)`` (StableAdamW).  ``rms``
            is the global root-mean-square over all tensor leaves.
        noise_bias_correction: If ``True``, subtract a β₂-EMA of the
            realized noise variance from the second moment when
            ``NoisedPytree`` updates are passed (DP-AdamW-BC, Chooi et al.).
            Defaults to ``False``; see ``docs/user-guide/optimizers.md``
            for when to flip it on.  Has no effect on steps where the
            update is a ``SecondMomentNoiseOutput``, since the privatised
            ``g²`` stream is an alternative answer to the same v-update
            bias.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    DP usage notes:

        - Call ``update(updates, state, ...)`` with ``updates`` as ``NoisedPytree``
          (first-moment noise metadata) or ``SecondMomentNoiseOutput`` (paired
          streams). Do not pass ``noise_stddev=`` / ``noisy_squared_grads=`` —
          those are internal to the composer chain only.
        - ``NoisedPytree``: realized σ feeds bias correction when enabled.
        - ``SecondMomentNoiseOutput``: privatised ``g²`` stream substitutes the
          usual v-update; mutually exclusive per step with the BC-from-σ path.
    """
    _validate(eps, betas, weight_decay, update_rms_clip)
    bc_floor = eps * eps  # see module docstring on the rationale.
    moment = _scale_by_adam(
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
    if len(betas) != 2:  # noqa: PLR2004 - Adam exposes the documented beta pair
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


__all__ = ["AdamState", "adam", "adamw"]
