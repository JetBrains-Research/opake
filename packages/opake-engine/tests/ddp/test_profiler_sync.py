"""PerfState sync under NCCL."""

from __future__ import annotations

import pytest
import torch
from engine_ddp_helpers import _spawn, _worker_sync_profiler

pytestmark = pytest.mark.cuda


class TestProfilerSyncDistributed:
    def test_sync_perf_state(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_profiler)
