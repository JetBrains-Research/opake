"""AUTO-S (second moment) + MF Gaussian noise under NCCL."""

from __future__ import annotations

import pytest
import torch
from dpftrl_ddp_helpers import _spawn, _worker_auto_mf

pytestmark = pytest.mark.cuda


class TestAutoClippedMFDistributed:
    def test_auto_clipped_band_mf_multi_step(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_auto_mf)
