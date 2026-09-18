r"""BSR strategy — banded square root MF (Kalinin & Lampert, NeurIPS 2024).

Closed-form lower-triangular Toeplitz coefficients for the paper workload
:math:`A_{\alpha,\beta}` (multiplicative decay :math:`\alpha`, Polyak momentum
:math:`\beta`).  These are **not** the same names as PyTorch ``weight_decay`` or
generic ``momentum`` on other strategies — here ``alpha`` and ``beta`` match the
paper.  No numerical optimization.

References:
    - BSR: https://arxiv.org/abs/2405.13763 (Theorem 1, banded :math:`C^{|p|}`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from opake.api.dpftrl.noise._strategy_codec import register_strategy
from opake.exceptions import ConfigurationError

from ._sensitivity import minsep_true_max_participations
from ._toeplitz import (
    inverse_as_streaming_matrix,
)
from ._toeplitz import (
    minsep_sensitivity_squared as _toeplitz_minsep_sensitivity_squared,
)
from ._toeplitz import (
    sensitivity_squared as _toeplitz_col_norm_sq,
)

_COEFFICIENT_TOLERANCE = 1e-12

if TYPE_CHECKING:
    from ._streaming_matrix import StreamingMatrix


def _native():
    from opake.api.accounting.core import _native as _n

    return _n


def _r_sequence(length: int) -> list[float]:
    """Binomial-type coefficients r_k = |(-1/2 choose k)| (paper Theorem 1)."""
    if length < 1:
        return []
    r = [1.0]
    for k in range(1, length):
        r.append(r[-1] * (k - 0.5) / k)
    return r


@lru_cache(maxsize=32)
def _bsr_band_coefficients_cached(
    bandwidth: int, alpha: float, beta: float
) -> tuple[float, ...]:
    r"""First-column coefficients of :math:`C^{|p|}_{\alpha,\beta}` (Theorem 1)."""
    if bandwidth < 1:
        raise ConfigurationError(*(f"bandwidth must be >= 1, got {bandwidth}",))
    r = _r_sequence(bandwidth)
    c = [1.0]
    for j in range(1, bandwidth):
        s = 0.0
        for i in range(j + 1):
            s += (alpha ** (j - i)) * r[j - i] * r[i] * (beta**i)
        c.append(s)
    return tuple(c)


def _validate_bsr_hyperparams(bandwidth: int, alpha: float, beta: float) -> None:
    """Raise ValueError if hyperparameters are outside the supported v1 regime."""
    if bandwidth < 1:
        raise ConfigurationError(*(f"bandwidth must be >= 1, got {bandwidth}",))
    if not (0.0 <= beta < 1.0):
        raise ConfigurationError(
            *(
                f"BSR v1 requires β in [0, 1), got {beta}. "
                "Use band_mf_strategy for other workloads.",
            )
        )
    if not (0.0 < alpha <= 1.0):
        raise ConfigurationError(
            *(
                f"BSR v1 requires α in (0, 1] (paper), got {alpha}. "
                "Use band_mf_strategy for other workloads.",
            )
        )
    if alpha <= beta:
        raise ConfigurationError(
            *(
                f"BSR v1 requires α > β (paper regime); got α={alpha}, β={beta}. "
                "Reduce β or increase α, or use band_mf_strategy.",
            )
        )
    coefs = _bsr_band_coefficients_cached(bandwidth, alpha, beta)
    for i in range(1, len(coefs)):
        if coefs[i] > coefs[i - 1] + _COEFFICIENT_TOLERANCE:
            raise ConfigurationError(
                *(
                    "BSR coefficients are not non-increasing for these hyperparameters; "
                    "this configuration is not supported by toeplitz_minsep_sensitivity. "
                    "Try different α, β, or bandwidth, or use band_mf_strategy.",
                )
            )
        if coefs[i] < -_COEFFICIENT_TOLERANCE:
            raise ConfigurationError(
                *(
                    "BSR produced a negative coefficient; unsupported for this Rust path. "
                    "Use band_mf_strategy.",
                )
            )


@lru_cache(maxsize=256)
def _bsr_gram_matrix_cached(
    bandwidth: int,
    alpha: float,
    beta: float,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
) -> tuple[float, ...]:
    """Gram sequence for BSR; cached across repeated σ / PLD probes."""
    band = list(_bsr_band_coefficients_cached(bandwidth, alpha, beta))
    return tuple(
        _native().toeplitz_gram_matrix(
            band, n_steps, min_sep, max_participations, False
        )
    )


def _bsr_full_coefficients(
    bandwidth: int, alpha: float, beta: float, n_steps: int
) -> torch.Tensor:
    """Pad the band coefficients out to ``n_steps`` with zeros."""
    band = _bsr_band_coefficients_cached(bandwidth, alpha, beta)
    out = torch.zeros(n_steps, dtype=torch.float64)
    copy_len = min(bandwidth, n_steps)
    out[:copy_len] = torch.tensor(band[:copy_len], dtype=torch.float64)
    return out


@register_strategy
@dataclass(frozen=True, slots=True)
class BsrStrategy:
    r"""BSR (banded square root) strategy — recipe only.

    Carries the workload knobs ``bandwidth``, ``alpha``, ``beta``; all
    derived quantities are computed on demand via the strategy methods.
    """

    bandwidth: int
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        _validate_bsr_hyperparams(self.bandwidth, self.alpha, self.beta)

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        return _bsr_full_coefficients(self.bandwidth, self.alpha, self.beta, n_steps)

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _bsr_gram_matrix_cached(
            self.bandwidth,
            self.alpha,
            self.beta,
            n_steps,
            min_sep,
            max_participations,
        )

    def streaming_matrix(self, **_) -> StreamingMatrix:
        band = _bsr_band_coefficients_cached(self.bandwidth, self.alpha, self.beta)
        coef_tensor = torch.tensor(list(band), dtype=torch.float64)
        return inverse_as_streaming_matrix(coef_tensor, column_normalize_for_n=None)

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        coef_tensor = self.coefficients(n_steps=n_steps)
        k = minsep_true_max_participations(
            n=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        if k == 1:
            return float(_toeplitz_col_norm_sq(coef_tensor, n_steps).sqrt())
        sens_sq = _toeplitz_minsep_sensitivity_squared(
            strategy_coef=coef_tensor,
            min_sep=min_sep,
            max_participations=max_participations,
        )
        return float(sens_sq.sqrt())


def bsr_strategy(
    *,
    bandwidth: int,
    alpha: float,
    beta: float,
) -> BsrStrategy:
    r"""Create a BSR strategy recipe (closed-form Toeplitz coefficients).

    Args:
        bandwidth: Bandwidth p (>= 1).  Only c_0..c_{p-1} are non-zero.
        alpha: Paper workload decay :math:`\alpha \in (0, 1]`.
        beta: Paper Polyak momentum :math:`\beta \in [0, 1)`.

    Returns:
        A :class:`BsrStrategy` recipe.
    """
    # Hyperparameters are validated in ``BsrStrategy.__post_init__``.
    return BsrStrategy(bandwidth=bandwidth, alpha=alpha, beta=beta)


__all__ = ["BsrStrategy", "bsr_strategy"]
