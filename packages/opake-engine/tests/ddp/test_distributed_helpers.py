"""CPU (non-distributed) behavior of the rank/gather helper extensions.

These wrap ``torch.distributed`` but must degrade gracefully when no
process group is initialized: rank 0, world size 1, no-op barrier, and
``gather_for_metrics`` returning its input unchanged. Multi-rank collective behavior is exercised here with Gloo and by the CUDA
tests in ``test_collectives.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from engine_ddp_helpers import (
    _spawn_gloo,
    _worker_gather_optional_ragged,
    _worker_scalar_exactness_gloo,
)

from opake.distributed import (
    gather_for_metrics,
    is_main_process,
    num_processes,
    process_index,
    wait_for_everyone,
)


def test_is_main_process_non_distributed() -> None:
    assert is_main_process() is True


def test_num_processes_non_distributed() -> None:
    assert num_processes() == 1


def test_process_index_non_distributed() -> None:
    assert process_index() == 0


def test_wait_for_everyone_is_noop_non_distributed() -> None:
    # Must not raise when there is no process group.
    assert wait_for_everyone() is None


def test_gather_for_metrics_returns_input_non_distributed() -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    out = gather_for_metrics(x)
    assert out is x
    assert torch.equal(out, x)


def test_gather_for_metrics_scalar_non_distributed() -> None:
    # A 0-dim per-rank scalar metric (e.g. KTO's detached KL) must pass through
    # unchanged when there is no process group. The distributed path promotes
    # the scalar to 1-D before all_gather/cat so torch.cat does not choke on a
    # 0-dim tensor (exercised by the CUDA collective tests).
    s = torch.tensor(3.5)
    out = gather_for_metrics(s)
    assert out is s
    assert out.dim() == 0


@pytest.mark.slow
@pytest.mark.distributed
def test_gather_optional_and_ragged_payloads() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_gather_optional_ragged)


@pytest.mark.slow
@pytest.mark.distributed
def test_scalar_reductions_preserve_integer_and_float64_exactness() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_scalar_exactness_gloo)
