"""Memory and timing profiling tools for DP training.

This module provides lightweight tools for tracking memory usage and timing
during differentially private training.

Key Components:
    - StepPerf: Frozen record of a single step's performance
    - step_perf: Context manager that measures one training step
    - PerfStage / PerfTracker: Mutable multi-stage accumulator
    - perf_tracker: Factory for PerfTracker
    - Utility functions: get_memory_stats, print_memory, reset_peak_memory

Example - PerfTracker in a training loop:
    >>> from opake.profiling import perf_tracker
    >>>
    >>> tracker = perf_tracker(device)
    >>> for batch in dataloader:
    ...     with tracker.train(batch_size=len(batch)) as sp:
    ...         grads = compute_gradients(batch)
    ...         sp.mark("clip")
    ...         update_params(grads)
    ...     wandb.log(tracker.train.last.to_dict(prefix="train/"))

Example - Simple memory tracking:
    >>> from opake.profiling import get_memory_stats, print_memory
    >>>
    >>> print_memory(device, "After model load")
    >>> stats = get_memory_stats(device)
    >>> print(f"Peak: {stats.peak_gb:.2f} GB")
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from opake.api.engine.device import device_capabilities
from opake.exceptions import OperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager


@dataclass(frozen=True)
class MemoryStats:
    """GPU memory statistics at a point in time.

    All values are in GB for easy reading. Use to_dict() for logging.

    Attributes:
        allocated_gb: Currently allocated memory.
        reserved_gb: Reserved by the allocator on CUDA. On MPS, total current
            Metal-driver allocation for the process, including allocator
            caches and MPS/MPSGraph allocations.
        peak_gb: Peak allocated tensor memory since the last
            :func:`reset_peak_memory` call. On MPS without allocator peak
            statistics, this is the current Metal-driver allocation with
            ``exact_peak=False``. ``0.0`` with ``exact_peak=False`` on CPU.
        free_gb: Estimated free memory (total - reserved).
        total_gb: Total memory budget (CUDA total VRAM;
            ``recommended_max_memory`` on MPS).
    """

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_gb: float = 0.0
    free_gb: float = 0.0
    total_gb: float = 0.0
    exact_peak: bool = False
    exact_reserved: bool = False
    known_free: bool = False
    known_total: bool = False

    @property
    def utilization(self) -> float:
        """Memory utilization as fraction (0.0-1.0)."""
        if self.total_gb > 0:
            return self.peak_gb / self.total_gb
        return 0.0

    def to_dict(self, prefix: str = "memory/") -> dict[str, float | bool]:
        """Convert to dict for logging.

        Args:
            prefix: Prefix for all keys (default: "memory/")

        Returns:
            Dict with keys like "memory/allocated_gb", "memory/peak_gb", etc.
        """
        return {
            f"{prefix}allocated_gb": self.allocated_gb,
            f"{prefix}reserved_gb": self.reserved_gb,
            f"{prefix}peak_gb": self.peak_gb,
            f"{prefix}free_gb": self.free_gb,
            f"{prefix}total_gb": self.total_gb,
            f"{prefix}utilization": self.utilization,
            f"{prefix}peak_exact": self.exact_peak,
            f"{prefix}reserved_exact": self.exact_reserved,
            f"{prefix}free_known": self.known_free,
            f"{prefix}total_known": self.known_total,
        }

    def __str__(self) -> str:
        return (
            f"Memory: alloc={self.allocated_gb:.2f}GB, "
            f"peak={self.peak_gb:.2f}GB, "
            f"free={self.free_gb:.2f}GB "
            f"({self.utilization:.1%} used)"
        )


def get_memory_stats(device: torch.device | str) -> MemoryStats:
    """Get current GPU memory statistics.

    Args:
        device: PyTorch device (cuda, mps, or cpu)

    Returns:
        MemoryStats with current memory usage. All zeros for CPU.

    Example:
        >>> stats = get_memory_stats("cuda")
        >>> print(f"Peak memory: {stats.peak_gb:.2f} GB")
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        free = total - reserved
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_gb=peak,
            free_gb=free,
            total_gb=total,
            exact_peak=True,
            exact_reserved=True,
            known_free=True,
            known_total=True,
        )
    elif device.type == "mps":
        peak_trackable = device_capabilities(device).peak_memory_trackable
        allocated = torch.mps.current_allocated_memory() / (1024**3)
        reserved = torch.mps.driver_allocated_memory() / (1024**3)
        peak = (
            torch.accelerator.memory.max_memory_allocated(device) / (1024**3)
            if peak_trackable
            else reserved
        )
        total = torch.mps.recommended_max_memory() / (1024**3)
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_gb=peak,
            free_gb=max(total - reserved, 0.0),
            total_gb=total,
            exact_peak=peak_trackable,
            exact_reserved=True,
            known_free=True,
            known_total=True,
        )
    else:
        return MemoryStats(
            exact_peak=False,
            exact_reserved=False,
            known_free=False,
            known_total=False,
        )


def reset_peak_memory(device: torch.device | str) -> None:
    """Reset the peak-memory high-water mark for accurate per-phase tracking.

    - **CUDA**: resets the exact peak-allocated counter (cheap).
    - **MPS with allocator peak statistics**: resets the exact counter using
      ``torch.accelerator.memory.reset_peak_memory_stats`` (cheap).
    - **MPS without allocator peak statistics**: releases cached blocks with
      ``torch.mps.empty_cache`` to re-baseline the driver-allocation reading.
    - **CPU**: no-op.

    Args:
        device: PyTorch device

    Example:
        >>> reset_peak_memory(device)  # clean baseline before a benchmark
        >>> # ... do training step ...
        >>> stats = get_memory_stats(device)
        >>> print(f"Step peak: {stats.peak_gb:.2f} GB")
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        if device_capabilities(device).peak_memory_trackable:
            torch.accelerator.memory.reset_peak_memory_stats(device)
        else:
            torch.mps.empty_cache()


def print_memory(device: torch.device | str, label: str = "") -> MemoryStats:
    """Print memory stats with optional label. Returns stats for further use.

    Args:
        device: PyTorch device
        label: Optional label to prefix the output

    Returns:
        MemoryStats for the current state

    Example:
        >>> print_memory("cuda", "After model load")
        [After model load] Memory: alloc=7.50GB, peak=7.50GB, free=72.50GB (9.4% used)
    """
    stats = get_memory_stats(device)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{stats}")
    return stats


def empty_cache(device: torch.device | str) -> None:
    """Clear GPU cache to free reserved memory.

    Args:
        device: PyTorch device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass(frozen=True)
class StepPerf:
    """Immutable record of a single step's performance.

    Produced by :func:`step_perf`. Contains wall-clock timing, throughput,
    memory statistics, and optional sub-step marks. All fields are frozen
    at the end of the ``step_perf`` context manager — no live hardware
    queries on read.

    Attributes:
        step_time_sec: Total wall-clock time for the step.
        samples_per_second: Throughput (batch_size / step_time_sec).
        memory_peak_gb: Best available peak allocated GPU tensor memory during
            the step. The precision is backend-dependent:

            - **CUDA and MPS with allocator peak statistics**: exact,
              resettable allocated-memory peak.
            - **MPS without allocator peak statistics**: maximum current
              allocation observed at step entry, marks, and exit.
            - **CPU**: ``0.0`` (no peak counter).

        memory_allocated_gb: Allocated memory at end of step.
        memory_reserved_gb: Reserved memory at end of step. On MPS this is the
            current total Metal-driver allocation, including allocator caches
            and MPS/MPSGraph allocations.
        batch_size: Number of samples processed.
        marks: Sub-step timing marks as ``{name: elapsed_sec}``.
    """

    step_time_sec: float = 0.0
    samples_per_second: float = 0.0
    memory_peak_gb: float = 0.0
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    batch_size: int = 0
    marks: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to flat dict for logging.

        Returns bare keys by default — caller adds the prefix to control
        namespace (``train/``, ``eval/``, ``perf/``, etc.).

        Args:
            prefix: Optional prefix for all keys (e.g. ``"train/"``).

        Returns:
            Dict with performance metrics.
        """
        d: dict[str, float] = {
            f"{prefix}step_time_sec": self.step_time_sec,
            f"{prefix}samples_per_second": self.samples_per_second,
            f"{prefix}memory_peak_gb": self.memory_peak_gb,
            f"{prefix}memory_allocated_gb": self.memory_allocated_gb,
            f"{prefix}memory_reserved_gb": self.memory_reserved_gb,
        }
        for name, elapsed in self.marks.items():
            d[f"{prefix}{name}_sec"] = elapsed
        return d


class _StepPerfBuilder:
    """Mutable builder used inside the ``step_perf`` context manager.

    Collects sub-step ``.mark()`` calls and is finalized into an
    immutable :class:`StepPerf` when the context exits.
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        *,
        sample_memory_peak: bool = False,
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._sample_memory_peak = sample_memory_peak
        self._sampled_memory_peak_gb = 0.0
        self._marks: dict[str, float] = {}
        self._last_mark_time: float = 0.0
        self._perf: StepPerf | None = None

    def _sample_peak(self) -> None:
        if self._sample_memory_peak:
            allocated_gb = torch.mps.current_allocated_memory() / (1024**3)
            self._sampled_memory_peak_gb = max(
                self._sampled_memory_peak_gb,
                allocated_gb,
            )

    def mark(self, name: str) -> None:
        """Record a sub-step timing mark (device-synchronized).

        Records elapsed time since the previous mark (or since the step
        started, for the first mark). The device is synchronized first so the
        mark reflects real *execution* time: on CUDA/MPS kernel launches are
        asynchronous, so an un-synchronized sub-step mark would record only
        kernel-launch time — microseconds for hundreds of ms of GPU work. The
        sync is a no-op on CPU. Marks therefore partition the step, and their
        sum approximately equals the whole-step time. On MPS without allocator
        peak statistics, marks also sample current allocated memory.

        Args:
            name: Label for this sub-step (e.g. ``"clip"``, ``"noise"``,
                ``"optimizer"``).
        """
        _sync_device(self._device)
        self._sample_peak()
        now = time.perf_counter()
        self._marks[name] = now - self._last_mark_time
        self._last_mark_time = now

    @property
    def perf(self) -> StepPerf:
        """The finalized :class:`StepPerf` record.

        Only available after the ``step_perf`` context manager exits.

        Raises:
            RuntimeError: If accessed before the context manager exits.
        """
        if self._perf is None:
            raise OperationError(
                *(
                    "StepPerf is not available until the step_perf context manager exits.",
                )
            )
        return self._perf

    @property
    def result(self) -> StepPerf:
        """Deprecated — use :attr:`perf` instead."""
        warnings.warn(
            "Use .perf instead of .result",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.perf


@contextmanager
def step_perf(
    device: torch.device | str,
    *,
    batch_size: int = 0,
    track_memory: bool = True,
) -> Iterator[_StepPerfBuilder]:
    """Time a training step and produce an immutable :class:`StepPerf` record.

    Handles GPU synchronization for accurate timing and optional peak
    memory tracking. Supports sub-step ``.mark()`` calls for
    fine-grained breakdowns.

    Args:
        device: PyTorch device (for GPU sync and memory reads).
        batch_size: Number of samples in this step (for throughput).
        track_memory: Whether to record peak memory (default True).

    Yields:
        A :class:`_StepPerfBuilder` that supports ``.mark(name)``
        during the step body. After the context exits, access the
        frozen record via ``.perf``.

    Example:
        >>> with step_perf("cuda", batch_size=32) as sp:
        ...     grads = compute_gradients(batch)
        ...     sp.mark("clip")
        ...     update_params(grads)
        ...     sp.mark("update")
        >>> wandb.log(sp.perf.to_dict(prefix="train/"))
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Drain work queued before this context before resetting the peak, so
    # earlier allocations cannot contaminate this step's measurement window.
    _sync_device(device)

    peak_trackable = device_capabilities(device).peak_memory_trackable
    sample_memory_peak = track_memory and device.type == "mps" and not peak_trackable
    builder = _StepPerfBuilder(
        device,
        batch_size,
        sample_memory_peak=sample_memory_peak,
    )

    if track_memory and peak_trackable:
        reset_peak_memory(device)
    builder._sample_peak()

    t0 = time.perf_counter()
    builder._last_mark_time = t0

    yield builder

    _sync_device(device)
    elapsed = time.perf_counter() - t0
    builder._sample_peak()

    mem = get_memory_stats(device) if track_memory else MemoryStats()
    memory_peak_gb = (
        builder._sampled_memory_peak_gb if sample_memory_peak else mem.peak_gb
    )

    builder._perf = StepPerf(
        step_time_sec=elapsed,
        samples_per_second=batch_size / elapsed if elapsed > 0 else 0.0,
        memory_peak_gb=memory_peak_gb,
        memory_allocated_gb=mem.allocated_gb,
        memory_reserved_gb=mem.reserved_gb,
        batch_size=batch_size,
        marks=MappingProxyType(dict(builder._marks)),
    )


@dataclass(frozen=True)
class PerfState:
    """Functional accumulator for step performance across a training run.

    Tracks cumulative timing and throughput statistics with
    warmup-excluded stable averages. Designed for DDP
    :func:`~opake.distributed.sync` reduction.

    All fields are frozen — call :meth:`add` to produce a new state
    with updated counters.

    Attributes:
        device: PyTorch device (carried for DDP sync).
        num_steps: Total number of recorded steps.
        total_time: Cumulative wall-clock time across all steps.
        total_samples: Cumulative sample count.
        num_steps_stable: Steps after warmup (excludes first step).
        total_time_stable: Cumulative time for stable steps.
        total_samples_stable: Cumulative samples for stable steps.
        max_peak_memory_gb: High-water mark of peak memory.
        last_step: Most recent :class:`StepPerf` record.
    """

    device: torch.device
    num_steps: int = 0
    total_time: float = 0.0
    total_samples: int = 0
    num_steps_stable: int = 0
    total_time_stable: float = 0.0
    total_samples_stable: int = 0
    max_peak_memory_gb: float = 0.0
    last_step: StepPerf | None = None

    def add(self, perf: StepPerf) -> PerfState:
        """Return a new state incorporating a completed step.

        The first step is treated as warmup and excluded from the
        ``*_stable`` counters.

        Args:
            perf: Completed step record.

        Returns:
            Updated PerfState.
        """
        num_steps = self.num_steps + 1
        total_time = self.total_time + perf.step_time_sec
        total_samples = self.total_samples + perf.batch_size
        max_peak = max(self.max_peak_memory_gb, perf.memory_peak_gb)

        num_steps_stable = self.num_steps_stable
        total_time_stable = self.total_time_stable
        total_samples_stable = self.total_samples_stable
        if self.num_steps >= 1:
            num_steps_stable += 1
            total_time_stable += perf.step_time_sec
            total_samples_stable += perf.batch_size

        return replace(
            self,
            num_steps=num_steps,
            total_time=total_time,
            total_samples=total_samples,
            max_peak_memory_gb=max_peak,
            num_steps_stable=num_steps_stable,
            total_time_stable=total_time_stable,
            total_samples_stable=total_samples_stable,
            last_step=perf,
        )

    @property
    def avg_step_time(self) -> float:
        """Average step time in seconds (all steps)."""
        if self.num_steps == 0:
            return 0.0
        return self.total_time / self.num_steps

    @property
    def avg_step_time_stable(self) -> float:
        """Average step time excluding first step (warmup)."""
        if self.num_steps_stable == 0:
            return self.avg_step_time
        return self.total_time_stable / self.num_steps_stable

    @property
    def avg_samples_per_second(self) -> float:
        """Average throughput in samples/second (all steps)."""
        if self.total_time > 0:
            return self.total_samples / self.total_time
        return 0.0

    @property
    def avg_samples_per_second_stable(self) -> float:
        """Average throughput in samples/second (stable, excluding warmup)."""
        if self.total_time_stable > 0:
            return self.total_samples_stable / self.total_time_stable
        return self.avg_samples_per_second

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Accumulated metrics as flat dict for logging.

        Args:
            prefix: Optional prefix for all keys.

        Returns:
            Dict with running averages and peak memory.
        """
        return {
            f"{prefix}avg_step_time_sec": self.avg_step_time_stable,
            f"{prefix}avg_samples_per_second": self.avg_samples_per_second_stable,
            f"{prefix}max_peak_memory_gb": self.max_peak_memory_gb,
        }


_STAGE_SHORTCUTS = frozenset({"train", "eval", "test"})


class PerfStage:
    """Mutable accumulator for a single named profiling stage.

    Tracks step count, wall-clock totals (post-warmup), peak memory,
    and the most recent :class:`StepPerf`.  Callable as a context
    manager that wraps :func:`step_perf` and auto-absorbs the result.

    Attributes:
        name: Stage label (e.g. ``"train"``, ``"eval"``).
        num_steps: Total steps recorded (including warmup).
        total_time: Cumulative wall-clock time (post-warmup).
        total_samples: Cumulative sample count (post-warmup).
        max_peak_memory_gb: High-water mark of peak memory.
        last: Most recent :class:`StepPerf`, or ``None``.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        warmup_steps: int = 1,
    ) -> None:
        self.name = name
        self.num_steps: int = 0
        self.total_time: float = 0.0
        self.total_samples: int = 0
        self.max_peak_memory_gb: float = 0.0
        self.last: StepPerf | None = None
        self._device = device
        self._warmup_steps = warmup_steps

    @property
    def samples_per_second(self) -> float:
        if self.total_time > 0:
            return self.total_samples / self.total_time
        return 0.0

    @property
    def steps_per_second(self) -> float:
        post = max(0, self.num_steps - self._warmup_steps)
        if self.total_time > 0 and post > 0:
            return post / self.total_time
        return 0.0

    def _absorb(self, perf: StepPerf) -> None:
        self.num_steps += 1
        self.max_peak_memory_gb = max(self.max_peak_memory_gb, perf.memory_peak_gb)
        self.last = perf
        if self.num_steps > self._warmup_steps:
            self.total_time += perf.step_time_sec
            self.total_samples += perf.batch_size

    def __ior__(self, perf: StepPerf) -> PerfStage:
        self._absorb(perf)
        return self

    def __call__(
        self,
        *,
        batch_size: int = 0,
        track_memory: bool = True,
    ) -> AbstractContextManager[_StepPerfBuilder]:
        @contextmanager
        def _ctx() -> Iterator[_StepPerfBuilder]:
            with step_perf(
                self._device,
                batch_size=batch_size,
                track_memory=track_memory,
            ) as sp:
                yield sp
            self._absorb(sp.perf)

        return _ctx()

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}num_steps": self.num_steps,
            f"{prefix}total_time_sec": self.total_time,
            f"{prefix}samples_per_second": self.samples_per_second,
            f"{prefix}steps_per_second": self.steps_per_second,
            f"{prefix}max_peak_memory_gb": self.max_peak_memory_gb,
        }


class PerfTracker:
    """Mutable multi-stage performance tracker.

    Provides named stages (``train``, ``eval``, ``test`` as attribute
    shortcuts; arbitrary names via ``tracker["name"]``).  Each stage
    is a :class:`PerfStage` created lazily on first access.

    Created via :func:`perf_tracker`.

    Example:
        >>> tracker = perf_tracker(device)
        >>> with tracker.train(batch_size=32) as sp:
        ...     train_step(batch)
        ...     sp.mark("clip")
        >>> wandb.log(tracker.train.last.to_dict("train/"))
    """

    def __init__(self, device: torch.device, warmup_steps: int = 1) -> None:
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "_warmup_steps", warmup_steps)
        object.__setattr__(self, "_stages", {})

    def _get_stage(self, name: str) -> PerfStage:
        stages: dict[str, PerfStage] = self._stages
        if name not in stages:
            stages[name] = PerfStage(name, self.device, self._warmup_steps)
        return stages[name]

    def __getattr__(self, name: str) -> PerfStage:
        if name in _STAGE_SHORTCUTS:
            return self._get_stage(name)
        raise AttributeError(  # noqa: TRY003 - preserve standard Python error contract
            f"{type(self).__name__!r} has no attribute {name!r}. "
            f"Use tracker[{name!r}] for custom stages."
        )

    def __setattr__(self, name: str, value: object) -> None:
        if name in _STAGE_SHORTCUTS:
            return
        object.__setattr__(self, name, value)

    def __getitem__(self, name: str) -> PerfStage:
        return self._get_stage(name)

    @property
    def stages(self) -> dict[str, PerfStage]:
        return dict(self._stages)


def perf_tracker(
    device: torch.device | str,
    warmup_steps: int = 1,
) -> PerfTracker:
    """Create a multi-stage performance tracker.

    Args:
        device: PyTorch device (for GPU sync and memory reads).
        warmup_steps: Steps to exclude from time/sample totals per
            stage (default ``1``).

    Returns:
        A new :class:`PerfTracker`.

    Example:
        >>> tracker = perf_tracker("cuda")
        >>> with tracker.train(batch_size=64) as sp:
        ...     train_step(batch)
        >>> print(tracker.train.samples_per_second)
    """
    if isinstance(device, str):
        device = torch.device(device)
    return PerfTracker(device, warmup_steps)


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
