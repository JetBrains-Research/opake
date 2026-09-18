"""Distributed adaptive clipping (NCCL)."""

from __future__ import annotations

import pytest
import torch
from dpsgd_ddp_helpers import (
    _spawn,
    _worker_adaptive_clipping,
    _worker_adaptive_clipping_uneven_batches,
    _worker_sync_adaptive_clip_state,
    _worker_sync_aux_adaptive_clipping,
)

pytestmark = pytest.mark.cuda


class TestAdaptiveClippingDistributed:
    def test_sync_adaptive_clip_state(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_adaptive_clip_state)

    def test_adaptive_clipping_with_sync(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping)

    def test_adaptive_clipping_with_uneven_batches(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping_uneven_batches)

    def test_sync_aux_adaptive_clipping(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_aux_adaptive_clipping)
