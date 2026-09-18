"""Fast CPU/Gloo regressions for functional optimizer-state synchronization."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from optimizer_ddp_helpers import (
    _spawn_gloo,
    _worker_optimizer_state_audit_gloo,
    _worker_optimizer_state_micro_drift_gloo,
)

pytestmark = pytest.mark.distributed


def test_optimizer_state_audit_detects_cross_rank_drift() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_optimizer_state_audit_gloo)


def test_optimizer_state_audit_detects_micro_scale_drift() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_optimizer_state_micro_drift_gloo)
