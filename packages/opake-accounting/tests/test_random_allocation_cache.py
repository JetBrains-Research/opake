"""Memoisation contract for the random-allocation native primitive (#802)."""

from __future__ import annotations

import pytest

from opake.api.accounting.core import _native
from opake.api.accounting.core._random_allocation_cache import (
    clear_random_allocation_caches,
    epoch_pld,
)
from opake.api.accounting.core.discretization import DiscretizationConfig

_CFG = DiscretizationConfig(
    discretization=0.01,
    max_grid_size=100_000,
    max_conv_grid=128,
)
_DELTA = 1e-5


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_random_allocation_caches()
    yield
    clear_random_allocation_caches()


def test_epoch_pld_is_memoised_and_matches_native():
    first = epoch_pld(1.0, 8, 1, _CFG)
    direct = _native.random_allocation_gaussian_pld(1.0, 8, 1, _CFG.to_native())
    assert first.epsilon_at(_DELTA) == direct.epsilon_at(_DELTA)

    again = epoch_pld(1.0, 8, 1, _CFG)
    assert again is first
    info = epoch_pld.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_epoch_pld_separates_keys():
    epoch_pld(1.0, 8, 1, _CFG)
    epoch_pld(2.0, 8, 1, _CFG)
    epoch_pld(1.0, 9, 1, _CFG)
    epoch_pld(1.0, 8, 2, _CFG)
    epoch_pld(1.0, 8, 1, DiscretizationConfig(discretization=0.02, max_conv_grid=128))
    assert epoch_pld.cache_info().misses == 5


def test_errors_are_not_cached():
    for _ in range(2):
        with pytest.raises(ValueError, match="noise_multiplier"):
            epoch_pld(0.0, 8, 1, _CFG)
    info = epoch_pld.cache_info()
    assert info.misses == 2  # recomputed, never stored
    assert info.currsize == 0
