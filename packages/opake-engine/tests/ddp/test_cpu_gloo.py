"""Fast CPU/Gloo coverage for core distributed engine primitives."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from engine_ddp_helpers import _spawn_gloo, _worker_core_collectives_gloo

pytestmark = pytest.mark.distributed


def test_core_collectives_and_state_sync() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_core_collectives_gloo)
