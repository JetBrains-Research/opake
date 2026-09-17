"""Non-private mechanism — infinite privacy loss, composition annihilator."""

from __future__ import annotations

from dataclasses import dataclass

from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization

from .. import _native


@dataclass(frozen=True, slots=True)
class NonPrivate(DpProcess):
    """Non-private mechanism — infinite privacy loss.

    Represents a mechanism with no privacy guarantee (ε=∞ for any δ<1).
    Backed by a PLD with all mass at +∞ (``infinity_mass=1``), equivalent
    to ``eps_delta(0, 1)``.

    Composes correctly through all combinators: subsampling or clipping
    cannot recover privacy from infinite loss, so the result remains
    non-private.
    """

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
        return _native.non_private_pld(config.to_native())


def nonprivate() -> DpProcess:
    """Non-private mechanism (infinite privacy loss).

    Useful for ``noise_multiplier=0`` runs where no noise is added.
    Fits into the compositional API — can be passed to :func:`poisson`,
    :func:`adaclip`, etc. — and produces correct metrics (ε=∞, δ=1)
    without special-case ``if`` guards.

    Returns:
        A :class:`NonPrivate` process (ε=∞ for any δ<1).

    Example::

        # Non-private baseline — composes like any other mechanism
        step = acc.poisson(acc.nonprivate(), sample_rate=0.01)
        eps = step.epsilon_at(1e-5)  # inf
    """
    return NonPrivate()
