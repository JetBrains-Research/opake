# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank load release of ``moe_clipped_grad`` (gloo/CPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from dpsgd_ddp_helpers import (
    _moe_factory,
    _moe_fixture,
    _spawn,
    _worker_moe_sync_gloo,
    _worker_moe_sync_pending_disagreement_gloo,
    _worker_moe_sync_ratio_mismatch_gloo,
)


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


@pytest.mark.slow
@pytest.mark.distributed
def test_synced_release_matches_single_process_full_batch() -> None:
    """Two ranks with half a batch each land on the single-process result."""
    _require_gloo()
    params, x, mask, y = _moe_fixture()
    grad_fn, state = _moe_factory()
    reference, ref_state = grad_fn(params, x, mask, y, state=state)
    assert ref_state.step == 1

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "moe.pt")
        _spawn(2, _worker_moe_sync_gloo, out_path)
        distributed = torch.load(out_path, map_location="cpu")

    torch.testing.assert_close(
        distributed["f_tilde"], ref_state.f_tilde, atol=1e-6, rtol=1e-5
    )
    for name, value in reference.pytree.items():
        torch.testing.assert_close(
            distributed["grads"][name], value, atol=1e-6, rtol=1e-5
        )


@pytest.mark.distributed
def test_ratio_mismatch_is_rejected_on_every_rank() -> None:
    _require_gloo()
    _spawn(2, _worker_moe_sync_ratio_mismatch_gloo)


@pytest.mark.distributed
def test_pending_disagreement_is_rejected_instead_of_blocking() -> None:
    _require_gloo()
    _spawn(2, _worker_moe_sync_pending_disagreement_gloo)
