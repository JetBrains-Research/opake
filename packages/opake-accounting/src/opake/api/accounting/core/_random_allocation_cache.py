"""Memoised wrappers for the random-allocation native PLD primitives.

Contributor-internal: not re-exported from the ``opake.accounting`` façade.

The random-allocation transform is O(G²) in the convolution grid. The
transform is deterministic, so complete-horizon calls are memoised on every
input that affects the output. The cache is a bounded LRU over immutable
Python :class:`Pld` objects (no native handles), so it is self-limiting and
needs no byte budget or destructor.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from . import _native

if TYPE_CHECKING:
    from ._base import Pld
    from .discretization import DiscretizationConfig

#: Cached entries are one PLD with at most ``max_conv_grid`` bins per PMF
#: (~0.5 MB per entry at the default config), so eight entries per cache are
#: a few MB — comparable to one process PLD cache slice.
_MAXSIZE = 8


@functools.lru_cache(maxsize=_MAXSIZE)
def epoch_pld(
    noise_multiplier: float,
    t: int,
    k: int,
    config: DiscretizationConfig,
) -> Pld:
    """Cached ``_native.random_allocation_gaussian_pld``.

    ``config`` must be the resolved Python-side :class:`DiscretizationConfig`
    (frozen and hashable); the native conversion runs only on a cache miss.
    """
    return _native.random_allocation_gaussian_pld(
        noise_multiplier,
        t,
        k,
        config.to_native(),
    )


def clear_random_allocation_caches() -> None:
    """Drop all memoised random-allocation PLDs."""
    epoch_pld.cache_clear()
