"""CPU/Gloo contract for DP-SGD's backend-neutral distributed path."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from dpsgd_ddp_helpers import (
    _spawn_gloo,
    _worker_cpu_gloo_training_contract,
    _worker_noise_seed_out_of_int64_range_gloo,
    _worker_per_group_adaptive_one_rank_empty_gloo,
    _worker_per_group_adaptive_state_gloo,
    _worker_per_group_adaptive_training_gloo,
    _worker_summed_noise_scaling_gloo,
)

pytestmark = pytest.mark.distributed


@pytest.mark.slow
def test_adaptive_clipping_and_gaussian_noise_handle_empty_ranks() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_cpu_gloo_training_contract)


def test_per_group_adaptive_clipping_state_syncs_uneven_ranks() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_per_group_adaptive_state_gloo)


def test_summed_noise_scaling_matches_the_key_regime() -> None:
    """Independent keys realize the advertised sqrt(W); a shared key scales by W."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_summed_noise_scaling_gloo)


def test_noise_state_sync_accepts_a_seed_past_int64_max() -> None:
    """Half of all `fold_in`-derived seeds set the top bit; sync must survive."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_noise_seed_out_of_int64_range_gloo)


@pytest.mark.slow
def test_per_group_adaptive_clipping_runs_through_noise() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_per_group_adaptive_training_gloo)


def test_per_group_adaptive_clipping_one_rank_empty_syncs() -> None:
    """Empty-rank count ordering follows the per-group schema (#798)."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_per_group_adaptive_one_rank_empty_gloo)
