"""Tests for memory profiling tools.

Tests the profiling API (MemoryStats, step_perf, PerfState)
across different devices (CUDA, MPS, CPU).
"""

import pytest
import torch

from opake.profiling import PerfState, StepPerf, get_memory_stats, step_perf
from opake.profiling.types import MemoryStats

# Test on available devices
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.insert(0, "cuda")
if torch.backends.mps.is_available():
    DEVICES.insert(0, "mps")


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_default_values(self):
        """Should have zero default values."""
        stats = MemoryStats()
        assert stats.allocated_gb == 0.0
        assert stats.reserved_gb == 0.0
        assert stats.peak_gb == 0.0

    def test_utilization_with_zero_total(self):
        """Should return 0 utilization when total is 0."""
        stats = MemoryStats(peak_gb=1.0, total_gb=0.0)
        assert stats.utilization == 0.0

    def test_utilization_calculation(self):
        """Should calculate utilization correctly."""
        stats = MemoryStats(peak_gb=40.0, total_gb=80.0)
        assert stats.utilization == pytest.approx(0.5)

    def test_to_dict(self):
        """Should convert to dict for WANDB."""
        stats = MemoryStats(
            allocated_gb=10.0,
            reserved_gb=15.0,
            peak_gb=12.0,
            free_gb=65.0,
            total_gb=80.0,
        )
        d = stats.to_dict()
        assert d["memory/allocated_gb"] == 10.0
        assert d["memory/peak_gb"] == 12.0


class TestGetMemoryStats:
    """Tests for get_memory_stats function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_returns_memory_stats(self, device):
        """Should return MemoryStats object."""
        stats = get_memory_stats(device)
        assert isinstance(stats, MemoryStats)

    def test_cpu_returns_zeros(self):
        """CPU should return all zeros."""
        stats = get_memory_stats("cpu")
        assert stats.allocated_gb == 0.0
        assert stats.peak_gb == 0.0
        assert stats.exact_peak is False
        assert stats.known_total is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_real_values(self):
        """CUDA should return real memory values."""
        x = torch.randn(1000, 1000, device="cuda")
        stats = get_memory_stats("cuda")
        assert stats.total_gb > 0
        assert stats.allocated_gb > 0
        assert stats.exact_peak is True
        assert stats.exact_reserved is True
        assert stats.known_total is True
        assert stats.known_free is True
        del x
        torch.cuda.empty_cache()


class TestStepPerf:
    """Tests for step_perf context manager and StepPerf record."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_measures_time(self, device):
        """Should measure elapsed time."""
        with step_perf(device, batch_size=32) as perf:
            x = torch.randn(100, 100)
            _ = x @ x.T
        result = perf.perf
        assert isinstance(result, StepPerf)
        assert result.step_time_sec > 0

    def test_batch_size_and_throughput(self):
        """Should compute throughput from batch size."""
        with step_perf("cpu", batch_size=64) as perf:
            pass
        result = perf.perf
        assert result.batch_size == 64
        assert result.samples_per_second > 0

    def test_marks(self):
        """Should record sub-step marks."""
        with step_perf("cpu", batch_size=16) as perf:
            _ = torch.randn(50, 50)
            perf.mark("clip")
            _ = torch.randn(50, 50)
            perf.mark("noise")
        result = perf.perf
        assert "clip" in result.marks
        assert "noise" in result.marks
        assert result.marks["clip"] > 0
        assert result.marks["noise"] > 0

    def test_to_dict(self):
        """Should convert to flat dict for logging."""
        with step_perf("cpu", batch_size=8) as perf:
            pass
        d = perf.perf.to_dict(prefix="train/")
        assert "train/step_time_sec" in d
        assert "train/samples_per_second" in d
        assert "train/memory_peak_gb" in d

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_synchronizes_before_timing(self, monkeypatch):
        """MPS timer should synchronize before measuring elapsed time."""
        calls = {"count": 0}
        original_sync = torch.mps.synchronize

        def _sync() -> None:
            calls["count"] += 1
            original_sync()

        monkeypatch.setattr(torch.mps, "synchronize", _sync)

        with step_perf("mps", batch_size=16):
            x = torch.randn(16, 16, device="mps")
            _ = x @ x.T

        assert calls["count"] >= 1


class TestPerfState:
    """Tests for PerfState accumulator."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize with device."""
        state = PerfState(device=torch.device(device))
        assert state.num_steps == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_add_step(self, device):
        """Should record completed step metrics."""
        state = PerfState(device=torch.device(device))
        with step_perf(device, batch_size=32) as perf:
            x = torch.randn(100, 100)
            _ = x @ x.T
        state = state.add(perf.perf)
        assert state.num_steps == 1
        assert state.last_step is not None
        assert state.last_step.batch_size == 32

    def test_avg_step_time_stable(self):
        """Should exclude first step for stable average."""
        state = PerfState(device=torch.device("cpu"))
        # Simulate steps with known times
        step1 = StepPerf(
            step_time_sec=10.0,
            batch_size=1,
            samples_per_second=0.1,
        )
        step2 = StepPerf(
            step_time_sec=2.0,
            batch_size=1,
            samples_per_second=0.5,
        )
        step3 = StepPerf(
            step_time_sec=2.0,
            batch_size=1,
            samples_per_second=0.5,
        )
        step4 = StepPerf(
            step_time_sec=2.0,
            batch_size=1,
            samples_per_second=0.5,
        )

        state = state.add(step1)
        state = state.add(step2)
        state = state.add(step3)
        state = state.add(step4)

        assert state.avg_step_time_stable == pytest.approx(2.0)

    def test_to_dict(self):
        """Should return accumulated metrics as dict."""
        state = PerfState(device=torch.device("cpu"))
        with step_perf("cpu", batch_size=16) as perf:
            pass
        state = state.add(perf.perf)
        d = state.to_dict(prefix="train/")
        assert "train/avg_step_time_sec" in d
        assert "train/max_peak_memory_gb" in d
