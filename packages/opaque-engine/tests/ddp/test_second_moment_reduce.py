"""Multi-rank reduce for the JME optimizer handoff (gloo/CPU)."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from engine_ddp_helpers import (
    _spawn,
    _worker_second_moment_noise_gloo,
)


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


class TestSecondMomentReduceGloo:
    @pytest.mark.slow
    @pytest.mark.distributed
    def test_second_moment_noise_sum(self) -> None:
        _require_gloo()
        _spawn(2, _worker_second_moment_noise_gloo)
