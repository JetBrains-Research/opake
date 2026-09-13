"""DP-lambda-CGD strategy and noise — correlated noise via PRNG replay.

The DP-lambda-CGD mechanism (Kalinin et al., 2026) uses a lower-triangular
Toeplitz strategy matrix :math:`C_\\lambda` whose inverse is bidiagonal: 1 on the
diagonal, :math:`-\\lambda` on the subdiagonal.  The correlated noise at step t is::

    n_t = z_t - lambda * z_{t-1}              (unnormalized)
    n_t = d_t * (z_t - lambda * z_{t-1})      (column-normalized, default)

where :math:`z_t \\sim N(0, \\sigma^2 I)` are i.i.d. Gaussians, and :math:`d_t`
is the column norm of :math:`C_\\lambda` at step t.  Instead of storing
:math:`z_{t-1}`, we regenerate it from the previous step's PRNG seed —
zero additional memory overhead compared to DP-SGD.

References:
    - Kalinin et al. (2026) "DP-λCGD: Efficient Noise Correlation for
      Differentially Private Model Training" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.exceptions import ConfigurationError
from opaque.pytree import tree_map
from opaque.random import generator_from_key

from ._engine import (
    MFNoiseState,
    _check_mf_horizon,
    _iid_normal_noise,
    _mf_gaussian_column_key,
    _require_positive_int_horizon,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.scheduling.types import Schedule
    from opaque.random.types import RngKey
    from opaque.types import PerGroup

    from ._streaming_matrix import StreamingMatrix


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


_COLUMN_NORM_TAIL_CUTOFF = 1e-30


@lru_cache(maxsize=256)
def _lambda_cgd_gram_matrix_cached(
    lambda_: float,
    normalized: bool,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
) -> tuple[float, ...]:
    """Gram sequence for λ-CGD; cached across repeated σ / PLD probes."""
    return tuple(
        _native().lambda_cgd_gram_matrix(
            lambda_, n_steps, min_sep, max_participations, normalized
        )
    )


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm :math:`d_t` of :math:`C_\\lambda` at 0-indexed step t.

    ``step`` must be in ``[0, n_steps)``.  At ``step == n_steps`` the
    closed form collapses to 0 (and beyond it is undefined), which would
    zero out the released noise under ``normalized=True``.
    """
    if step < 0 or step >= n_steps:
        raise ConfigurationError(
            *(
                f"column-norm step {step} is outside the calibrated horizon [0, {n_steps}).",
            )
        )
    if lambda_ == 0.0:
        return 1.0
    remaining = n_steps - step
    lambda2 = lambda_ * lambda_
    lambda2r = lambda2**remaining
    if lambda2r < _COLUMN_NORM_TAIL_CUTOFF:
        return math.sqrt(1.0 / (1.0 - lambda2))
    return math.sqrt((1.0 - lambda2r) / (1.0 - lambda2))


@register_strategy
@dataclass(frozen=True, slots=True)
class LambdaCgdStrategy:
    """DP-lambda-CGD strategy — recipe only (PRNG-replay noise)."""

    lambda_: float
    normalized: bool = True
    # Compatibility tombstone for legacy state dictionaries. Non-None values
    # are rejected because optimizer LR schedules are not part of this encoder.
    lr_schedule: Schedule | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.lr_schedule is not None:
            raise ConfigurationError(
                *(
                    "LambdaCgdStrategy does not support lr_schedule. Learning-rate "
                    "schedules are optimizer post-processing and cannot weight its "
                    "Balls-in-Bins privacy accounting. Remove lr_schedule from the "
                    "strategy, pass it only to the optimizer, and recalibrate privacy "
                    "and noise for any result previously computed with this option.",
                )
            )
        if not math.isfinite(self.lambda_) or not 0.0 <= self.lambda_ < 1.0:
            raise ConfigurationError(
                *(f"lambda_ must be finite and in [0, 1), got {self.lambda_}",)
            )

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        # [1, λ, λ², ..., λ^{n_steps-1}].
        return torch.tensor(
            [self.lambda_**i for i in range(n_steps)], dtype=torch.float64
        )

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _lambda_cgd_gram_matrix_cached(
            self.lambda_,
            self.normalized,
            n_steps,
            min_sep,
            max_participations,
        )

    def streaming_matrix(self, **_) -> StreamingMatrix:
        # Lambda-CGD never materializes a streaming matrix — it uses
        # PRNG replay via :func:`_make_lambda_cgd_noise` instead.  The
        # mf_gaussian_noise dispatcher special-cases this strategy.
        raise NotImplementedError(
            "LambdaCgdStrategy uses PRNG-replay noise; the noise factory "
            "dispatches to _make_lambda_cgd_noise directly."
        )

    def raw_noise_factory(
        self,
        grad_template: Any,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
        key: RngKey,
        compute_dtype: torch.dtype,
    ):
        del min_sep, max_participations
        return _make_lambda_cgd_noise(
            grad_template,
            self,
            n_steps=n_steps,
            key=key,
            compute_dtype=compute_dtype,
        )

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        if self.normalized:
            sens_sq = _native().lambda_cgd_normalized_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        else:
            sens_sq = _native().lambda_cgd_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        return float(sens_sq**0.5)

    def max_column_norm(self, *, n_steps: int) -> float:
        """Max L2 column norm of the strategy matrix at this horizon."""
        if self.normalized:
            return 1.0
        return float(_native().lambda_cgd_max_column_norm(self.lambda_, n_steps))


def lambda_cgd_strategy(
    *,
    lambda_: float,
    normalized: bool = True,
    lr_schedule: Schedule | None = None,
) -> LambdaCgdStrategy:
    """Create a DP-lambda-CGD strategy recipe (bandwidth=2, PRNG-replay noise).

    Args:
        lambda_: Correlation coefficient in [0, 1).
        normalized: Use column-normalized matrix (default True).
        lr_schedule: Deprecated compatibility argument. Only ``None`` is
            accepted. Pass learning-rate schedules to the optimizer instead.

    Returns:
        A :class:`LambdaCgdStrategy` recipe.
    """
    return LambdaCgdStrategy(
        lambda_=lambda_,
        normalized=normalized,
        lr_schedule=lr_schedule,
    )


# ---------------------------------------------------------------------------
# Internal noise builder (called by mf_gaussian_noise() dispatcher)
# ---------------------------------------------------------------------------


def _lambda_cgd_row_l2(strategy: LambdaCgdStrategy, n_steps: int, step: int) -> float:
    """Per-step row L2 norm of the λ-CGD effective C^{-1}.

    The realized per-coordinate noise at step ``t`` is
    ``base_σ · row_l2(t)`` (post-normalization when ``normalized=True``).
    At step 0 there is no previous-step term so the factor is just the
    optional column-norm multiplier; at step t≥1 the unnormalized factor
    is ``sqrt(1 + λ²)`` from ``z_t − λ z_{t−1}``.

    ``step`` must be in ``[0, n_steps)`` — the noise function raises on
    past-horizon calls, so this lookup is never asked to invent a factor
    outside the calibrated matrix.
    """
    lam = strategy.lambda_
    col = _column_norm(lam, n_steps, step) if strategy.normalized else 1.0
    if step == 0 or lam == 0.0:
        return col
    return col * math.sqrt(1.0 + lam * lam)


def _make_lambda_cgd_noise(
    grad_template: Any,
    strategy: LambdaCgdStrategy,
    *,
    n_steps: int,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
    Callable[[int], float],
]:
    """DP-lambda-CGD noise via PRNG replay (zero extra memory).

    Returns ``(noise_fn, state, row_l2_at)`` where ``row_l2_at(step)``
    gives ``‖row_t(C^{-1})‖`` so the wrapping :func:`mf_gaussian_noise`
    factory can publish the realized per-step σ on
    :class:`NoisedPytree.noise_stddev` (= ``base_σ · row_l2_at(step)``).
    Adam-family bias correction reads that realized σ.
    """
    n_steps = _require_positive_int_horizon(n_steps)

    lambda_ = strategy.lambda_
    normalized = strategy.normalized

    state = MFNoiseState(
        _inner_state=None,
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
        *,
        stddev: float | PerGroup,
    ) -> tuple[Any, MFNoiseState]:
        step = st._step_counter
        _check_mf_horizon(step, n_steps)

        current_key = _mf_gaussian_column_key(st._rng_key, step)
        g_current = generator_from_key(current_key)
        z_t = _iid_normal_noise(
            clipped_grads,
            stddev,
            generator=g_current,
            compute_dtype=compute_dtype,
        )

        if step == 0 or lambda_ == 0.0:
            corr_noise = z_t
        else:
            prev_key = _mf_gaussian_column_key(st._rng_key, step - 1)
            g_prev = generator_from_key(prev_key)
            z_prev = _iid_normal_noise(
                clipped_grads,
                stddev,
                generator=g_prev,
                compute_dtype=compute_dtype,
            )
            corr_noise = tree_map(
                lambda zt, zp: zt - lambda_ * zp,
                z_t,
                z_prev,
            )

        if normalized:
            d_t = _column_norm(lambda_, n_steps, step)
            corr_noise = tree_map(lambda n: n * d_t, corr_noise)

        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )

        new_state = MFNoiseState(
            _inner_state=None,
            _step_counter=step + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    def row_l2_at(step: int) -> float:
        return _lambda_cgd_row_l2(strategy, n_steps, step)

    return noise_fn, state, row_l2_at


__all__ = ["LambdaCgdStrategy", "lambda_cgd_strategy"]
