"""Performance profiling for DP training.

Provides :func:`step_perf` to measure individual training steps and
:class:`PerfState` to accumulate throughput statistics across a run.
Memory utilities (:func:`get_memory_stats`, :func:`print_memory`,
:func:`reset_peak_memory`, :func:`empty_cache`) remain available as
standalone helpers.

Example:
    >>> from opake.api.engine.profiling import step_perf, PerfState, print_memory
    >>>
    >>> print_memory(device, "After model load")
    >>> perf_state = PerfState(device=device)
    >>> for batch in dataloader:
    ...     with step_perf(device, batch_size=len(batch)) as perf:
    ...         train_step(batch)
    ...         perf.mark("clip")
    ...     perf_state = perf_state.add(perf.result)
    ...     wandb.log(perf.result.to_dict(prefix="train/"))
"""

from opake.api.engine.profiling._memory import (
    PerfStage,
    PerfState,
    PerfTracker,
    StepPerf,
    empty_cache,
    get_memory_stats,
    perf_tracker,
    print_memory,
    reset_peak_memory,
    step_perf,
)

__all__ = [
    "PerfStage",
    "PerfState",
    "PerfTracker",
    "StepPerf",
    "empty_cache",
    "get_memory_stats",
    "perf_tracker",
    "print_memory",
    "reset_peak_memory",
    "step_perf",
]
