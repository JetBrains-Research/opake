"""Whole-horizon queries reuse the memoised random-allocation transform."""

from __future__ import annotations

import pytest

import opake.dpsgd.accounting as dpsgd_acc
from opake.api.accounting.core._random_allocation_cache import (
    clear_random_allocation_caches,
    epoch_pld,
)
from opake.api.accounting.dpsgd.amplification._k_out_of_t import KOutOfT

_CFG = {"discretization": 0.01, "max_grid_size": 100_000, "max_conv_grid": 256}
_DELTA = 1e-5


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_random_allocation_caches()
    KOutOfT.pld.cache_clear()
    yield
    clear_random_allocation_caches()
    KOutOfT.pld.cache_clear()


def test_full_horizon_does_not_rebuild_epoch_transform():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        k=4,
        t=100,
        allocation="block",
    )

    first = process.pld(**_CFG).epsilon_at(_DELTA)
    assert epoch_pld.cache_info().misses == 1

    assert process.pld(**_CFG).epsilon_at(_DELTA) == first
    assert epoch_pld.cache_info().misses == 1
