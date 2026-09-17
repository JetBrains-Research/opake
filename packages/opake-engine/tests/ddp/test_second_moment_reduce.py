"""Multi-rank reduce for SecondMoment* wrappers (gloo/CPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from engine_ddp_helpers import (
    _paired_clipping_fixture,
    _paired_clipping_loss,
    _spawn,
    _worker_second_moment_clip_gloo,
    _worker_second_moment_clipping_parity_gloo,
    _worker_second_moment_noise_gloo,
)

from opake.api.engine.clipping import clipped_grad
from opake.pytree import tree_map
from opake.types import SecondMomentClippingOutput


def _paired_clipping_reference() -> SecondMomentClippingOutput:
    params, x, y = _paired_clipping_fixture("cpu")
    grad_fn, clip_state = clipped_grad(
        _paired_clipping_loss,
        clipping_norm=0.7,
        batch_argnums=(1, 2),
        normalize_by=len(x),
        second_moment=True,
    )
    paired, _ = grad_fn(params, x, y, state=clip_state)
    assert isinstance(paired, SecondMomentClippingOutput)
    return paired


def _assert_tree_close(actual: dict, expected: dict) -> None:
    tree_map(torch.testing.assert_close, actual, expected)


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


class TestSecondMomentReduceGloo:
    @pytest.mark.slow
    @pytest.mark.distributed
    def test_second_moment_clipping_sum(self) -> None:
        _require_gloo()
        _spawn(2, _worker_second_moment_clip_gloo)

    @pytest.mark.slow
    @pytest.mark.distributed
    def test_second_moment_noise_sum(self) -> None:
        _require_gloo()
        _spawn(2, _worker_second_moment_noise_gloo)

    @pytest.mark.slow
    @pytest.mark.distributed
    def test_paired_clipping_matches_single_process_full_batch(self) -> None:
        _require_gloo()

        reference = _paired_clipping_reference()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "paired.pt")
            _spawn(2, _worker_second_moment_clipping_parity_gloo, out_path)
            distributed = torch.load(out_path, map_location="cpu")

        _assert_tree_close(distributed["grads"], reference.grads.pytree)
        _assert_tree_close(
            distributed["squared_grads"],
            reference.squared_grads.pytree,
        )
        assert distributed["max_norm"] == pytest.approx(reference.grads.max_norm)
        assert distributed["squared_max_norm"] == pytest.approx(
            reference.squared_grads.max_norm
        )
