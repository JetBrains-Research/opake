"""Distributed collectives on NCCL (engine substrate)."""

from __future__ import annotations

import pytest
import torch
from engine_ddp_helpers import (
    _spawn,
    _worker_all_reduce_values,
    _worker_reduce_pytree,
    _worker_reduce_pytree_nested,
    _worker_reduce_scalar,
)

pytestmark = pytest.mark.cuda


class TestDistributedCollectives:
    def test_reduce_scalar(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_scalar)

    def test_all_reduce_values(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_all_reduce_values)

    def test_reduce_pytree(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree)

    def test_reduce_pytree_nested(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree_nested)
