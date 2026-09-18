"""Fixed (ε, δ)-DP mechanism."""

from __future__ import annotations

from dataclasses import dataclass

from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization
from opake.exceptions import ConfigurationError

from .. import _native


@dataclass(frozen=True, slots=True)
class EpsDelta(DpProcess):
    """Fixed (ε, δ) mechanism."""

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        # Mirrors native ``eps_delta_pld`` bounds; bounds stay inclusive
        # because ``eps_delta(0, 1)`` is the documented non-private equivalent.
        if self.epsilon < 0:
            raise ConfigurationError(
                *(f"epsilon must be non-negative, got {self.epsilon}",)
            )
        if not (0.0 <= self.delta <= 1.0):
            raise ConfigurationError(*(f"delta must be in [0, 1], got {self.delta}",))

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
        return _native.eps_delta_pld(self.epsilon, self.delta, config.to_native())


def eps_delta(epsilon: float, delta: float = 0.0) -> DpProcess:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).

    Returns:
        A :class:`DpProcess` wrapping an (ε, δ) PLD.

    Example::

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    return EpsDelta(epsilon, delta)
