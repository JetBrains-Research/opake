"""Distributed synchronization helpers for profiling components."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import torch
import torch.distributed as dist

from opake.api.engine.distributed import get_world_size, is_distributed
from opake.api.engine.distributed._state import (
    assert_scalar_equal,
    reduce_scalar,
    register_sync_type,
)
from opake.exceptions import OperationError

from ._memory import (
    PerfState,
    PerfTracker,
    StepPerf,
)

__all__ = ["sync_perf_state", "sync_perf_tracker"]


def _sync_step_perf(last: StepPerf | None, device: torch.device) -> StepPerf | None:
    """Synchronize an optional step record with a fixed presence schedule."""
    local_presence = int(last is not None)
    min_presence = reduce_scalar(local_presence, op="min", device=device)
    max_presence = reduce_scalar(local_presence, op="max", device=device)
    if min_presence != max_presence:
        raise OperationError(*("StepPerf presence mismatch across ranks.",))
    if not local_presence:
        return None

    assert last is not None
    last_time = reduce_scalar(float(last.step_time_sec), op="max", device=device)
    last_samples = int(reduce_scalar(float(last.batch_size), op="sum", device=device))
    last_peak = reduce_scalar(float(last.memory_peak_gb), op="max", device=device)
    return StepPerf(
        step_time_sec=last_time,
        samples_per_second=last_samples / last_time if last_time > 0 else 0.0,
        memory_peak_gb=last_peak,
        memory_allocated_gb=0.0,
        memory_reserved_gb=0.0,
        batch_size=last_samples,
        marks=MappingProxyType({}),
    )


def _synchronized_stage_names(tracker: PerfTracker) -> tuple[str, ...]:
    """Return the shared stage schema or raise before stage reductions begin."""
    local_names = tuple(sorted(tracker._stages))
    gathered: list[tuple[str, ...] | None] = [None] * get_world_size()
    dist.all_gather_object(gathered, local_names)
    mismatched = [rank for rank, names in enumerate(gathered) if names != local_names]
    if mismatched:
        raise OperationError(
            *(
                "PerfTracker stage schema mismatch across ranks: "
                f"mismatched ranks={mismatched}.",
            )
        )
    return local_names


def sync_perf_state(state: PerfState) -> PerfState:
    """Synchronize PerfState across distributed ranks.

    - step time uses max across ranks (critical path)
    - batch size / samples uses sum across ranks (global samples)
    - peak memory uses max across ranks

    Safe to call multiple times or on non-distributed runs (returns
    the input unchanged).
    """
    if not is_distributed():
        return state

    device = (
        state.device
        if isinstance(state.device, torch.device)
        else torch.device(state.device)
    )

    assert_scalar_equal(
        state.num_steps,
        name="PerfState.num_steps",
        device=device,
    )

    total_time = reduce_scalar(float(state.total_time), op="max", device=device)
    total_samples = int(reduce_scalar(state.total_samples, op="sum", device=device))
    max_peak = reduce_scalar(float(state.max_peak_memory_gb), op="max", device=device)

    total_time_stable = reduce_scalar(
        float(state.total_time_stable), op="max", device=device
    )
    total_samples_stable = int(
        reduce_scalar(state.total_samples_stable, op="sum", device=device)
    )

    last_step = _sync_step_perf(state.last_step, device)

    return replace(
        state,
        total_time=total_time,
        total_samples=total_samples,
        max_peak_memory_gb=max_peak,
        total_time_stable=total_time_stable,
        total_samples_stable=total_samples_stable,
        last_step=last_step,
    )


def sync_perf_tracker(tracker: PerfTracker) -> PerfTracker:
    """Synchronize PerfTracker across distributed ranks.

    Returns a new :class:`PerfTracker` with reduced per-stage values.
    The original tracker is not modified.
    """
    if not is_distributed():
        return tracker

    device = tracker.device
    synced = PerfTracker(device, tracker._warmup_steps)

    for name in _synchronized_stage_names(tracker):
        stage = tracker._stages[name]
        assert_scalar_equal(
            stage.num_steps,
            name=f"PerfStage({name!r}).num_steps",
            device=device,
        )

        s = synced._get_stage(name)
        s.num_steps = stage.num_steps
        s._warmup_steps = stage._warmup_steps
        s.total_time = reduce_scalar(float(stage.total_time), op="max", device=device)
        s.total_samples = int(
            reduce_scalar(stage.total_samples, op="sum", device=device)
        )
        s.max_peak_memory_gb = reduce_scalar(
            float(stage.max_peak_memory_gb), op="max", device=device
        )
        s.last = _sync_step_perf(stage.last, device)

    return synced


register_sync_type(PerfState, sync_perf_state)
register_sync_type(PerfTracker, sync_perf_tracker)
