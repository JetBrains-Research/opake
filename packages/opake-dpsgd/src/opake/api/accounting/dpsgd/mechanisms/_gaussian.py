"""Gaussian mechanism — base noise for DP-SGD."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from opake.api.accounting.core import _native
from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization
from opake.exceptions import ConfigurationError

_VERY_SMALL_NOISE_MULTIPLIER = 0.1


@dataclass(frozen=True, slots=True)
class Gaussian(DpProcess):
    """Gaussian mechanism — stores noise_multiplier, computes PLD on demand."""

    noise_multiplier: float

    def __post_init__(self) -> None:
        # Reject negative σ on construction (including deserialization, which
        # rebuilds via ``cls(**kwargs)``) instead of failing later inside the
        # native PLD call.  ``0.0`` stays valid: it is the documented
        # non-private value, short-circuited before the native call.
        if self.noise_multiplier < 0:
            raise ConfigurationError(
                *(f"noise_multiplier must be >= 0, got {self.noise_multiplier}",)
            )

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        if self.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        return _native.gaussian_pld(self.noise_multiplier, config.to_native())


def gaussian(noise_multiplier: float) -> Gaussian:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.  ``0`` is accepted
            (non-private: ε=∞).

    Returns:
        A :class:`Gaussian` process.

    Example::

        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if 0 < noise_multiplier < _VERY_SMALL_NOISE_MULTIPLIER:
        warnings.warn(
            f"noise_multiplier={noise_multiplier} is very small: epsilon bounds "
            f"may explode and discretization grids may grow unboundedly, "
            f"leading to slow or inaccurate PLD computation.",
            stacklevel=2,
        )
    return Gaussian(noise_multiplier)
