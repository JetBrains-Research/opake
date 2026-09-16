"""Distributed ordering and state checks for projected JME."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from dpsgd_ddp_helpers import _spawn_gloo, _worker_jme_after_global_sum_gloo


@pytest.mark.slow
@pytest.mark.distributed
def test_jme_runs_after_global_sum() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    _spawn_gloo(2, _worker_jme_after_global_sum_gloo)
