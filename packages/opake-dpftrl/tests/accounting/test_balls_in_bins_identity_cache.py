"""Identity balls-in-bins reuses the memoised full-horizon transform."""

from __future__ import annotations

import pytest

import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.core._random_allocation_cache import (
    clear_random_allocation_caches,
    epoch_pld,
)
from opake.api.accounting.dpftrl.amplification._balls_in_bins import BallsInBins
from opake.dpftrl.noise import identity_strategy

_CFG = {"discretization": 0.01, "max_grid_size": 100_000, "max_conv_grid": 128}
_DELTA = 1e-5


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_random_allocation_caches()
    BallsInBins.pld.cache_clear()
    yield
    clear_random_allocation_caches()
    BallsInBins.pld.cache_clear()


def test_equivalent_horizons_share_one_epoch_transform():
    process = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        num_bins=10,
        n_steps=100,
    )

    first = process.pld(**_CFG).epsilon_at(_DELTA)
    assert epoch_pld.cache_info().misses == 1
    equivalent = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        num_bins=10,
        n_steps=100,
    )
    second = equivalent.pld(**_CFG).epsilon_at(_DELTA)
    assert first == second
    assert epoch_pld.cache_info().misses == 1
