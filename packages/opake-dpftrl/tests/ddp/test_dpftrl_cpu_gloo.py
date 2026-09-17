"""CPU/Gloo contract for DP-FTRL's backend-neutral distributed path."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from dpftrl_ddp_helpers import (
    _spawn_gloo,
    _worker_auto_band_mf_gloo,
    _worker_bisr_bounded_state_gloo,
    _worker_cpu_gloo_training_contract,
    _worker_per_group_mf_state_gloo,
    _worker_second_moment_mf_state_gloo,
)

pytestmark = pytest.mark.distributed


@pytest.mark.slow
def test_gradient_reduction_and_mf_noise_handle_empty_ranks() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_cpu_gloo_training_contract)


def test_per_group_mf_noise_state_rejects_cross_rank_bound_mismatch() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_per_group_mf_state_gloo)


def test_private_second_moment_noise_state_syncs_across_ranks() -> None:
    """The paired state has its own handler, and unsigned seeds survive sync."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_second_moment_mf_state_gloo)


def test_bisr_bounded_history_stays_synchronized_across_ranks() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_bisr_bounded_state_gloo)


@pytest.mark.slow
def test_auto_clipping_and_band_mf_noise_stay_synchronized() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_auto_band_mf_gloo)
