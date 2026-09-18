"""Shared statistical helpers for one-run audit methods.

Argument validation and the closed-form ε ceiling used as the binary-search
upper bracket.
"""

from __future__ import annotations

import math

from opake.exceptions import ConfigurationError

_MAX_SIGNIFICANCE = 0.5


def validate_significance(significance: float) -> None:
    if not 0 < significance < _MAX_SIGNIFICANCE:
        raise ConfigurationError(
            *(f"significance must be in (0, 0.5), got {significance}",)
        )


def validate_delta(delta: float) -> None:
    if not 0 <= delta <= 1:
        raise ConfigurationError(*(f"delta must be in [0, 1], got {delta}",))


def search_ceiling(m: int, delta: float, significance: float) -> float:
    """Maximum ε any one-run audit at ``(m, delta, significance)`` can return.

    Derived from the perfect-attack p-value
    ``sigmoid(ε) ** n_eff = significance`` where
    ``n_eff = m - round(m * delta)``.

    A safety pad of 1.1× and a floor of 1.0 keep the binary search well-posed
    when ``n_eff`` is too small to certify any positive ε.
    """
    n_eff = max(m - round(m * delta), 1)
    eps_exact = -math.log(significance ** (-1.0 / n_eff) - 1.0)
    return max(eps_exact * 1.1, 1.0)
