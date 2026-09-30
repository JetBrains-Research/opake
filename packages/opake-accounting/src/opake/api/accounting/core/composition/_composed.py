"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

from dataclasses import dataclass

from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache


@dataclass(frozen=True, slots=True, eq=False)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    def __hash__(self) -> int:
        # Iterative tree walk — depth bounded by heap, not stack.
        # See ``_iter_hash``.
        from ._iter_hash import iter_hash

        return iter_hash(self)

    def __eq__(self, other: object) -> bool:
        # Iterative tree walk (dataclass semantics preserved) — depth
        # bounded by heap, not stack.  See ``_iter_eq``.
        if not isinstance(other, DpProcess):
            return NotImplemented
        from ._iter_eq import iter_eq

        return iter_eq(self, other)

    def __repr__(self) -> str:
        # Iterative tree walk, string-identical to the dataclass repr —
        # deep chains (either spine) would otherwise overflow the stack.
        from ._iter_repr import iter_repr

        return iter_repr(self)

    def _pld_cache_key(self) -> tuple[object, ...]:
        from ._iter_cache_key import iter_cache_key

        return iter_cache_key(self)

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
        from ._iter_pld import iter_pld

        return iter_pld(
            self,
            {
                "discretization": discretization,
                "log_x_mass_truncation_bound": log_x_mass_truncation_bound,
                "max_grid_size": max_grid_size,
                "max_conv_grid": max_conv_grid,
                "seed": seed,
                "mc_resolution": mc_resolution,
                "mc_failure_probability": mc_failure_probability,
            },
        )
