"""Gloo regression coverage for distributed reference-logprob precomputation."""

from __future__ import annotations

import tempfile

import pytest
import torch.distributed as dist
from alignment_ddp_helpers import _spawn_gloo, _worker_precompute_contract

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


def test_precompute_preserves_cross_rank_contract() -> None:
    """Check sharding, cache consensus, and validation in one live process group."""
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(3, _worker_precompute_contract, tmp)
