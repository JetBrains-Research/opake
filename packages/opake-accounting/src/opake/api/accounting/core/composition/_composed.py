"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization


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
        # Walk the left spine iteratively (mirrors __hash__) to avoid
        # O(depth) Python call stack on left-skewed trees.
        rights: list[DpProcess] = []
        node: DpProcess = self
        while isinstance(node, Composed):
            rights.append(node.right)
            node = node.left
        # The configured probability is an overall bound for this composed
        # PLD, not a fresh allowance for every Monte Carlo leaf. Split it over
        # the top-level groups; nested composed groups recursively split their
        # share. Analytic leaves report zero and merely leave slack unused.
        resolved = get_discretization(
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        group_count = len(rights) + 1
        group_failure = resolved.mc_failure_probability / group_count
        group_resolution = -math.expm1(
            math.log1p(-resolved.mc_resolution) / group_count
        )
        kw = {
            "discretization": discretization,
            "log_x_mass_truncation_bound": log_x_mass_truncation_bound,
            "max_grid_size": max_grid_size,
            "max_conv_grid": max_conv_grid,
            "seed": seed,
            "mc_resolution": group_resolution,
            "mc_failure_probability": group_failure,
        }
        result = node.pld(**kw)
        for right in reversed(rights):
            result = result.compose(right.pld(**kw))
        return result
