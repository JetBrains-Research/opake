"""Projected joint moment estimation for DP-SGD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.autograd.profiler import record_function

from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.pytree import tree_leaves, tree_map
from opaque.api.engine.random import fold_in as rng_fold_in
from opaque.api.engine.random import generator_from_key
from opaque.api.engine.random.types import RngKey
from opaque.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    NoiseState,
    PerGroup,
    SecondMomentNoiseOutput,
)
from opaque.exceptions import ConfigurationError, InputTypeError, OperationError

if TYPE_CHECKING:
    from collections.abc import Callable


JME_FIRST_STREAM_FOLD = "opaque.dpsgd.jme.first"
JME_SECOND_STREAM_FOLD = "opaque.dpsgd.jme.second"

_PAPER_D1_ETA = (11.0 + 5.0 * math.sqrt(5.0)) / 8.0
_PAPER_D_GE_2_ETA = 0.5
_D1_STATIONARY_THRESHOLD_ETA = 2.0
_CAP_BISECTION_STEPS = 128
_FLOAT_GUARD_ULPS = 32


@dataclass(frozen=True)
class JmeAllocation:
    """Dimensionless first/second-stream noise allocation policy.

    Use :meth:`paper_reference` to reproduce the identity-strategy allocation
    from Kalinin, Upadhyay, and Lampert, or :meth:`first_variance_cap` to
    minimize second-stream variance subject to an explicit cap on first-stream
    variance inflation.
    """

    kind: Literal["paper_reference", "first_variance_cap"]
    max_first_variance_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "paper_reference":
            if self.max_first_variance_ratio is not None:
                raise ConfigurationError(
                    *(
                        "paper_reference allocation does not accept "
                        "max_first_variance_ratio.",
                    )
                )
            return
        if self.kind != "first_variance_cap":
            raise ConfigurationError(*(f"Unknown JME allocation kind {self.kind!r}.",))
        ratio = self.max_first_variance_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, Real)
            or not math.isfinite(float(ratio))
            or float(ratio) <= 1.0
        ):
            raise ConfigurationError(
                *(
                    "first_variance_cap requires a finite max_first_variance_ratio "
                    f"> 1, got {ratio!r}.",
                )
            )

    @classmethod
    def paper_reference(cls) -> JmeAllocation:
        """Use the paper's identity-strategy JME allocation."""
        return cls(kind="paper_reference")

    @classmethod
    def first_variance_cap(cls, max_ratio: float) -> JmeAllocation:
        """Minimize second-stream variance under a first-variance cap.

        ``max_ratio`` is relative to the variance of releasing only the
        projected first moment with the same ``noise_multiplier``.
        """
        return cls(
            kind="first_variance_cap",
            max_first_variance_ratio=float(max_ratio),
        )


@dataclass(frozen=True)
class JmeNoiseState(NoiseState):
    """Immutable RNG position and privacy-critical JME configuration."""

    _step_counter: int
    _rng_key: RngKey
    _aggregate_norm: float
    _allocation: JmeAllocation
    _noise_multiplier: float


@dataclass(frozen=True)
class _JmeParameters:
    lambda_: float
    joint_sensitivity: float
    first_stddev: float
    second_stddev: float
    first_max_norm: float
    second_max_norm: float


def _round_up(value: float) -> float:
    if value == 0.0:
        return 0.0
    if not math.isfinite(value):
        raise ConfigurationError(*(f"JME calibration produced non-finite {value}.",))
    for _ in range(_FLOAT_GUARD_ULPS):
        value = math.nextafter(value, math.inf)
    return value


def _effective_distance(delta: float, aggregate_norm: float) -> float:
    return min(delta, 2.0 * aggregate_norm)


def _effective_distance_ratio(delta: float, aggregate_norm: float) -> float:
    if delta >= 2.0 * aggregate_norm:
        return 2.0
    return delta / aggregate_norm


def _fixed_distance_square_factor(distance_ratio: float, dimension: int) -> float:
    """Dimensionless square displacement, equal to ``M_R(r) / R^4``."""
    t = distance_ratio
    collinear = t * t * (2.0 - t) ** 2
    if dimension == 1:
        return collinear
    boundary = 0.5 * t * t * (2.0 - t) * (2.0 + t)
    return max(collinear, boundary)


def _square_factor_per_distance_sq(
    distance_ratio: float,
    dimension: int,
) -> float:
    """Dimensionless ``M_R(r) / (R² r²)`` with a stable zero limit."""
    t = distance_ratio
    collinear = (2.0 - t) ** 2
    if dimension == 1:
        return collinear
    boundary = 0.5 * (2.0 - t) * (2.0 + t)
    return max(collinear, boundary)


def _joint_variance_ratio_raw(
    *,
    distance_ratio: float,
    eta: float,
    dimension: int,
) -> float:
    """Exact ``S² / D²`` without materializing a potentially tiny ``D/R``."""
    endpoint = 1.0 + eta * _square_factor_per_distance_sq(
        distance_ratio,
        dimension,
    )
    candidates = [endpoint]
    if dimension == 1:
        if eta > _D1_STATIONARY_THRESHOLD_ETA:
            discriminant = max(0.0, 1.0 - 2.0 / eta)
            local_max = 0.5 * (3.0 - math.sqrt(discriminant))
            if local_max <= distance_ratio:
                candidates.append(
                    (
                        local_max * local_max
                        + eta * _fixed_distance_square_factor(local_max, dimension)
                    )
                    / (distance_ratio * distance_ratio)
                )
    elif eta > 0.0:
        interior = math.sqrt(2.0 + 1.0 / eta)
        if interior <= distance_ratio:
            candidates.append(
                (
                    interior * interior
                    + eta * _fixed_distance_square_factor(interior, dimension)
                )
                / (distance_ratio * distance_ratio)
            )
    return max(candidates)


def _fixed_distance_square_sensitivity_sq(
    distance: float,
    aggregate_norm: float,
    dimension: int,
) -> float:
    """Exact ``sup ||x²-y²||²`` at a fixed distance."""
    ratio = distance / aggregate_norm
    factor = _fixed_distance_square_factor(ratio, dimension)
    scale_sq = aggregate_norm * aggregate_norm
    return scale_sq * scale_sq * factor


def _joint_sensitivity_factor_sq_raw(
    *,
    distance_ratio: float,
    eta: float,
    dimension: int,
) -> float:
    """Dimensionless joint sensitivity squared, equal to ``S² / R²``."""
    if distance_ratio == 0.0:
        return 0.0
    return (
        distance_ratio
        * distance_ratio
        * _joint_variance_ratio_raw(
            distance_ratio=distance_ratio,
            eta=eta,
            dimension=dimension,
        )
    )


def _joint_sensitivity_sq_raw(
    *,
    delta: float,
    aggregate_norm: float,
    lambda_: float,
    dimension: int,
) -> float:
    """Exact constrained sensitivity squared before conservative rounding."""
    distance_ratio = _effective_distance_ratio(delta, aggregate_norm)
    eta = lambda_ * aggregate_norm * aggregate_norm
    distance = _effective_distance(delta, aggregate_norm)
    variance_ratio = _joint_variance_ratio_raw(
        distance_ratio=distance_ratio,
        eta=eta,
        dimension=dimension,
    )
    return distance * (distance * variance_ratio)


def _paper_eta(*, dimension: int) -> float:
    return _PAPER_D1_ETA if dimension == 1 else _PAPER_D_GE_2_ETA


def _lambda_from_eta(*, eta: float, aggregate_norm: float) -> float:
    lambda_ = eta / aggregate_norm / aggregate_norm
    if not math.isfinite(lambda_) or lambda_ <= 0.0:
        raise ConfigurationError(
            *(
                "aggregate_norm is outside the numerically supported range for "
                "a finite positive JME lambda.",
            )
        )
    return lambda_


def _paper_lambda(*, aggregate_norm: float, dimension: int) -> float:
    return _lambda_from_eta(
        eta=_paper_eta(dimension=dimension),
        aggregate_norm=aggregate_norm,
    )


def _guarded_joint_sensitivity(
    *,
    distance: float,
    distance_ratio: float,
    eta: float,
    dimension: int,
) -> float:
    variance_ratio = _round_up(
        _joint_variance_ratio_raw(
            distance_ratio=distance_ratio,
            eta=eta,
            dimension=dimension,
        )
    )
    return _round_up(distance * math.sqrt(variance_ratio))


def _guarded_first_stddev(
    *,
    noise_multiplier: float,
    distance: float,
    distance_ratio: float,
    eta: float,
    dimension: int,
) -> float:
    if noise_multiplier == 0.0:
        return 0.0
    sensitivity = _guarded_joint_sensitivity(
        distance=distance,
        distance_ratio=distance_ratio,
        eta=eta,
        dimension=dimension,
    )
    return _round_up(noise_multiplier * sensitivity)


def _variance_cap_eta(
    *,
    noise_multiplier: float,
    delta: float,
    aggregate_norm: float,
    dimension: int,
    max_ratio: float,
) -> float:
    """Largest dimensionless eta satisfying the analytical variance cap."""
    distance = _effective_distance(delta, aggregate_norm)
    distance_ratio = _effective_distance_ratio(delta, aggregate_norm)
    if distance == 0.0:
        return _paper_eta(dimension=dimension)
    effective_multiplier = noise_multiplier if noise_multiplier > 0.0 else 1.0
    baseline_stddev = effective_multiplier * distance
    target_stddev = math.sqrt(max_ratio) * baseline_stddev
    for _ in range(_FLOAT_GUARD_ULPS):
        target_stddev = math.nextafter(target_stddev, 0.0)
    if not math.isfinite(target_stddev) or target_stddev <= 0.0:
        raise ConfigurationError(
            *("JME first-variance cap is non-finite at this scale.",)
        )

    low = 0.0
    high = 1.0
    for _ in range(1024):
        if (
            _guarded_first_stddev(
                noise_multiplier=effective_multiplier,
                distance=distance,
                distance_ratio=distance_ratio,
                eta=high,
                dimension=dimension,
            )
            > target_stddev
        ):
            break
        high *= 2.0
        if not math.isfinite(high):
            raise ConfigurationError(
                *("Could not bracket the JME first-variance-cap allocation.",)
            )
    else:
        raise OperationError(*("Could not bracket JME allocation after 1024 steps.",))

    for _ in range(_CAP_BISECTION_STEPS):
        middle = low + 0.5 * (high - low)
        first_stddev = _guarded_first_stddev(
            noise_multiplier=effective_multiplier,
            distance=distance,
            distance_ratio=distance_ratio,
            eta=middle,
            dimension=dimension,
        )
        if first_stddev <= target_stddev:
            low = middle
        else:
            high = middle

    resolved = math.nextafter(low, 0.0)
    if resolved <= 0.0 or not math.isfinite(resolved):
        raise ConfigurationError(
            *(
                "max_first_variance_ratio is too close to 1 for a finite "
                "positive JME allocation at this scale.",
            )
        )
    return resolved


def _resolve_eta(
    *,
    allocation: JmeAllocation,
    noise_multiplier: float,
    delta: float,
    aggregate_norm: float,
    dimension: int,
) -> float:
    if allocation.kind == "paper_reference":
        return _paper_eta(dimension=dimension)
    assert allocation.max_first_variance_ratio is not None
    return _variance_cap_eta(
        noise_multiplier=noise_multiplier,
        delta=delta,
        aggregate_norm=aggregate_norm,
        dimension=dimension,
        max_ratio=allocation.max_first_variance_ratio,
    )


def _marginal_square_sensitivity(
    *,
    delta: float,
    aggregate_norm: float,
    dimension: int,
) -> float:
    distance = _effective_distance(delta, aggregate_norm)
    if dimension == 1:
        maximizing_distance = min(distance, aggregate_norm)
    else:
        maximizing_distance = min(distance, math.sqrt(2.0) * aggregate_norm)
    collinear = maximizing_distance * (2.0 * aggregate_norm - maximizing_distance)
    if dimension == 1 or maximizing_distance <= 2.0 * aggregate_norm / 3.0:
        return _round_up(collinear)
    ratio = maximizing_distance / aggregate_norm
    boundary = maximizing_distance * (
        aggregate_norm * math.sqrt(0.5 * (2.0 - ratio) * (2.0 + ratio))
    )
    return _round_up(boundary)


def _resolve_jme_parameters(
    *,
    noise_multiplier: float,
    delta: float,
    aggregate_norm: float,
    allocation: JmeAllocation,
    dimension: int,
) -> _JmeParameters:
    eta = _resolve_eta(
        allocation=allocation,
        noise_multiplier=noise_multiplier,
        delta=delta,
        aggregate_norm=aggregate_norm,
        dimension=dimension,
    )
    lambda_ = _lambda_from_eta(eta=eta, aggregate_norm=aggregate_norm)
    distance = _effective_distance(delta, aggregate_norm)
    distance_ratio = _effective_distance_ratio(delta, aggregate_norm)
    sensitivity = _guarded_joint_sensitivity(
        distance=distance,
        distance_ratio=distance_ratio,
        eta=eta,
        dimension=dimension,
    )
    first_stddev = _guarded_first_stddev(
        noise_multiplier=noise_multiplier,
        distance=distance,
        distance_ratio=distance_ratio,
        eta=eta,
        dimension=dimension,
    )
    second_stddev = (
        0.0
        if noise_multiplier == 0.0
        else _round_up(first_stddev * aggregate_norm / math.sqrt(eta))
    )
    if noise_multiplier > 0.0 and (first_stddev == 0.0 or second_stddev == 0.0):
        raise ConfigurationError(
            *("JME noise standard deviation underflowed at this scale.",)
        )
    second_max_norm = _marginal_square_sensitivity(
        delta=delta,
        aggregate_norm=aggregate_norm,
        dimension=dimension,
    )
    return _JmeParameters(
        lambda_=lambda_,
        joint_sensitivity=sensitivity,
        first_stddev=first_stddev,
        second_stddev=second_stddev,
        first_max_norm=_round_up(distance),
        second_max_norm=second_max_norm,
    )


def _validate_factory_args(
    *,
    noise_multiplier: float,
    aggregate_norm: float,
    allocation: JmeAllocation,
    key: RngKey,
    compute_dtype: torch.dtype,
) -> tuple[float, float]:
    if isinstance(noise_multiplier, bool) or not isinstance(noise_multiplier, Real):
        raise InputTypeError(
            *(f"noise_multiplier must be a real number, got {noise_multiplier!r}.",)
        )
    resolved_multiplier = float(noise_multiplier)
    if not math.isfinite(resolved_multiplier) or resolved_multiplier < 0.0:
        raise ConfigurationError(
            *(
                "noise_multiplier must be finite and non-negative, "
                f"got {noise_multiplier!r}.",
            )
        )
    if isinstance(aggregate_norm, bool) or not isinstance(aggregate_norm, Real):
        raise InputTypeError(
            *(f"aggregate_norm must be a real number, got {aggregate_norm!r}.",)
        )
    resolved_norm = float(aggregate_norm)
    if not math.isfinite(resolved_norm) or resolved_norm <= 0.0:
        raise ConfigurationError(
            *(f"aggregate_norm must be finite and positive, got {aggregate_norm!r}.",)
        )
    if not isinstance(allocation, JmeAllocation):
        raise InputTypeError(
            *(
                "allocation must be a JmeAllocation from "
                "JmeAllocation.paper_reference() or "
                "JmeAllocation.first_variance_cap(...).",
            )
        )
    if not isinstance(key, RngKey):
        raise InputTypeError(*(f"key must be RngKey, got {type(key).__name__}.",))
    if (
        not isinstance(compute_dtype, torch.dtype)
        or not compute_dtype.is_floating_point
    ):
        raise InputTypeError(
            *(
                "compute_dtype must be a real floating-point torch.dtype, "
                f"got {compute_dtype!r}.",
            )
        )
    return resolved_multiplier, resolved_norm


def _validate_input(
    value: Any,
    *,
    aggregate_norm: float,
    compute_dtype: torch.dtype,
) -> tuple[ClippedPytree, int, torch.dtype]:
    if isinstance(value, NoisedPytree):
        raise InputTypeError(
            *("jme_noise expects a ClippedPytree that has not already been noised.",)
        )
    if not isinstance(value, ClippedPytree):
        raise InputTypeError(
            *(
                "jme_noise expects an already aggregated ClippedPytree. "
                "Apply clipping and any distributed sum before JME.",
            )
        )
    if isinstance(value.max_norm, PerGroup):
        raise InputTypeError(
            *(
                "jme_noise does not support PerGroup clipping; a group-coupled "
                "joint sensitivity has not been derived.",
            )
        )
    if isinstance(value.max_norm, bool) or not isinstance(value.max_norm, Real):
        raise InputTypeError(
            *(f"ClippedPytree.max_norm must be scalar, got {value.max_norm!r}.",)
        )
    delta = float(value.max_norm)
    if not math.isfinite(delta) or delta < 0.0:
        raise ConfigurationError(
            *(
                "ClippedPytree.max_norm must be finite and non-negative for JME, "
                f"got {value.max_norm!r}.",
            )
        )

    leaves = tree_leaves(value.pytree)
    if not leaves:
        raise InputTypeError(*("jme_noise requires at least one tensor leaf.",))
    dimension = 0
    projection_dtype = compute_dtype
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            raise InputTypeError(
                *(f"jme_noise expects tensor-only pytrees, got {type(leaf).__name__}.",)
            )
        if not torch.is_floating_point(leaf) or torch.is_complex(leaf):
            raise InputTypeError(
                *(
                    "jme_noise supports real floating-point tensor leaves, "
                    f"got {leaf.dtype}.",
                )
            )
        if aggregate_norm > math.sqrt(torch.finfo(leaf.dtype).max):
            raise ConfigurationError(
                *(
                    f"aggregate_norm={aggregate_norm} can overflow when squared "
                    f"in output dtype {leaf.dtype}.",
                )
            )
        dimension += leaf.numel()
        projection_dtype = torch.promote_types(projection_dtype, leaf.dtype)
    return value, dimension, projection_dtype


def _add_gaussian_tree(
    pytree: Any,
    *,
    stddev: float,
    rng_key: RngKey,
    compute_dtype: torch.dtype,
) -> Any:
    if stddev == 0.0:
        return pytree
    generator = generator_from_key(rng_key)

    def add_noise(leaf: torch.Tensor) -> torch.Tensor:
        work_dtype = torch.promote_types(leaf.dtype, compute_dtype)
        noise = torch.randn(
            leaf.shape,
            dtype=work_dtype,
            generator=generator,
        ).to(device=leaf.device)
        return (leaf.to(work_dtype) + stddev * noise).to(leaf.dtype)

    return tree_map(add_noise, pytree)


def jme_noise(
    *,
    noise_multiplier: float,
    aggregate_norm: float,
    allocation: JmeAllocation,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[
        [ClippedPytree, JmeNoiseState], tuple[SecondMomentNoiseOutput, JmeNoiseState]
    ],
    JmeNoiseState,
]:
    r"""Create a projected joint-moment Gaussian mechanism for DP-SGD.

    The input must be the normalized, globally aggregated output of scalar
    clipping. Each call projects that aggregate to the public L2 radius
    ``aggregate_norm``, forms its clean element-wise square, and releases both
    values under one exact constrained Mahalanobis sensitivity.

    If the input carries add/remove sensitivity ``Delta`` and the projection
    radius is ``R``, neighboring projected aggregates satisfy
    ``||x-y|| <= min(Delta, 2R)``. For the lambda selected by ``allocation``,
    the mechanism computes the exact sensitivity of
    ``(x, sqrt(lambda) * x.square())``. The realized standard deviations are

    .. math::

        \sigma_1 = \mathrm{noise\_multiplier}\,S_\lambda,\qquad
        \sigma_2 =
        \mathrm{noise\_multiplier}\,S_\lambda/\sqrt{\lambda}.

    Whitening by these scales gives a sensitivity-one Gaussian query. The
    mechanism is therefore dominated by
    ``dpsgd_acc.gaussian(noise_multiplier)`` before ordinary independent
    Poisson amplification. This is a domination statement; the nonlinear
    mechanism's actual PLD need not equal the canonical sampled-Gaussian PLD.

    Args:
        noise_multiplier: Effective Gaussian multiplier used by the privacy
            accountant. Must be finite and non-negative.
        aggregate_norm: Fixed public L2 projection radius for the normalized
            aggregate. Must be finite and positive.
        allocation: Explicit dimensionless first/second-stream allocation from
            :class:`opaque.dpsgd.noise.types.JmeAllocation`.
        key: Explicit RNG key. The first and second streams use independent
            namespaced derivations.
        compute_dtype: Internal projection, square, and Gaussian sampling
            dtype. Output tensor leaves retain the input dtype.

    Returns:
        ``(noise_fn, state)``. ``noise_fn(clipped_grads, state)`` returns a
        :class:`opaque.types.SecondMomentNoiseOutput` and a new immutable
        :class:`opaque.dpsgd.noise.types.JmeNoiseState`.

    Notes:
        Apply distributed summation before ``noise_fn``. The documented privacy
        path requires scalar clipping, fixed public normalization, standard
        unbounded Gaussian noise, and plain independent Poisson sampling.
        ``PerGroup``, bounded Gaussian, truncated/parallel sampling, horizon
        allocation, and matrix-factorization noise require separate analyses.

    References:
        Kalinin, Upadhyay, and Lampert. "Continual Release Moment Estimation
        with Differential Privacy." 2025. https://arxiv.org/abs/2502.06597
    """
    resolved_multiplier, resolved_norm = _validate_factory_args(
        noise_multiplier=noise_multiplier,
        aggregate_norm=aggregate_norm,
        allocation=allocation,
        key=key,
        compute_dtype=compute_dtype,
    )
    initial_state = JmeNoiseState(
        _step_counter=0,
        _rng_key=key,
        _aggregate_norm=resolved_norm,
        _allocation=allocation,
        _noise_multiplier=resolved_multiplier,
    )

    def noise_fn(
        grads: ClippedPytree,
        state: JmeNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, JmeNoiseState]:
        with record_function("opaque::jme_noise"):
            return _noise_fn_impl(grads, state)

    def _noise_fn_impl(
        grads: ClippedPytree,
        state: JmeNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, JmeNoiseState]:
        if not isinstance(state, JmeNoiseState):
            raise InputTypeError(
                *(
                    f"jme_noise state must be JmeNoiseState, got {type(state).__name__}.",
                )
            )
        if (
            state._aggregate_norm != resolved_norm
            or state._allocation != allocation
            or state._noise_multiplier != resolved_multiplier
        ):
            raise ConfigurationError(
                *("JmeNoiseState belongs to a different JME configuration.",)
            )
        clipped_input, dimension, projection_dtype = _validate_input(
            grads,
            aggregate_norm=resolved_norm,
            compute_dtype=compute_dtype,
        )
        delta = float(clipped_input.max_norm)
        parameters = _resolve_jme_parameters(
            noise_multiplier=resolved_multiplier,
            delta=delta,
            aggregate_norm=resolved_norm,
            allocation=allocation,
            dimension=dimension,
        )
        projected, _ = clip_pytree(
            clipped_input.pytree,
            clipping_norm=resolved_norm,
            compute_dtype=projection_dtype,
        )

        def square(leaf: torch.Tensor) -> torch.Tensor:
            work_dtype = torch.promote_types(leaf.dtype, projection_dtype)
            return leaf.to(work_dtype).square().to(leaf.dtype)

        squared = tree_map(square, projected)
        first_step_key = rng_fold_in(
            state._rng_key,
            JME_FIRST_STREAM_FOLD,
            state._step_counter,
        )
        second_step_key = rng_fold_in(
            state._rng_key,
            JME_SECOND_STREAM_FOLD,
            state._step_counter,
        )
        noisy_first = _add_gaussian_tree(
            projected,
            stddev=parameters.first_stddev,
            rng_key=first_step_key,
            compute_dtype=compute_dtype,
        )
        noisy_second = _add_gaussian_tree(
            squared,
            stddev=parameters.second_stddev,
            rng_key=second_step_key,
            compute_dtype=compute_dtype,
        )
        next_step = state._step_counter + 1
        return (
            SecondMomentNoiseOutput(
                noisy_grads=NoisedPytree(
                    pytree=noisy_first,
                    max_norm=parameters.first_max_norm,
                    noise_stddev=parameters.first_stddev,
                ),
                noisy_squared_grads=NoisedPytree(
                    pytree=noisy_second,
                    max_norm=parameters.second_max_norm,
                    noise_stddev=parameters.second_stddev,
                ),
            ),
            JmeNoiseState(
                _step_counter=next_step,
                _rng_key=state._rng_key,
                _aggregate_norm=state._aggregate_norm,
                _allocation=state._allocation,
                _noise_multiplier=state._noise_multiplier,
            ),
        )

    return noise_fn, initial_state


__all__ = [
    "JME_FIRST_STREAM_FOLD",
    "JME_SECOND_STREAM_FOLD",
    "JmeAllocation",
    "JmeNoiseState",
    "jme_noise",
]
