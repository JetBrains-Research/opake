"""MF Gaussian noise under NCCL (identity + cross-rank determinism)."""

from __future__ import annotations

import pytest
import torch
from dpftrl_ddp_helpers import (
    _spawn,
    _worker_identity_mf_three_steps,
    _worker_mf_shared_noise,
)

pytestmark = pytest.mark.cuda


class TestIdentityMFMultiStepDistributed:
    def test_identity_mf_three_steps_cross_rank(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_identity_mf_three_steps)


class TestDistributedMFNoiseSpawn:
    def test_mf_noise_shared_seed_byte_identical(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_mf_shared_noise)
