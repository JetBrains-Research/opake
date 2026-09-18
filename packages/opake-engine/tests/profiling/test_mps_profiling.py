"""MPS profiling behavior on Apple Silicon."""

import pytest
import torch

from opake.device import device_capabilities
from opake.profiling import (
    get_memory_stats,
    reset_peak_memory,
    step_perf,
)

_GiB_FLOATS = 256 * 1024 * 1024  # 256M float32 = 1 GiB


@pytest.mark.mps
class TestMpsMemoryStats:
    def test_total_and_reserved_are_real(self):
        stats = get_memory_stats("mps")
        # recommended_max_memory() → a real multi-GB budget, not the old 0.0.
        assert stats.total_gb > 1.0
        assert stats.known_total is True
        assert stats.reserved_gb >= stats.allocated_gb
        assert stats.exact_reserved is True
        trackable = device_capabilities("mps").peak_memory_trackable
        assert stats.exact_peak is trackable
        if not trackable:
            assert stats.peak_gb == stats.reserved_gb

    def test_allocated_reflects_a_known_tensor(self):
        before = get_memory_stats("mps").allocated_gb
        t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")  # 1 GiB
        torch.mps.synchronize()
        after = get_memory_stats("mps").allocated_gb
        assert after - before == pytest.approx(1.0, abs=0.05)
        del t


@pytest.mark.mps
class TestMpsPeakTracking:
    def test_peak_captures_a_freed_transient(self):
        """The regression guard: peak must reflect a transient freed mid-step.

        Current allocation reports ~0 after the free; the allocator peak keeps
        the 1 GiB. On older PyTorch releases, the explicit mark samples it.
        """
        reset_peak_memory("mps")  # clean high-water baseline
        with step_perf("mps", batch_size=1) as sp:
            t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")  # 1 GiB
            torch.mps.synchronize()
            sp.mark("allocated")
            del t  # freed before the step ends
            torch.mps.synchronize()
        perf = sp.perf
        assert perf.memory_peak_gb >= 0.9  # captured the ~1 GiB transient
        # ... and is well above the (near-zero) end-of-step allocation.
        assert perf.memory_peak_gb > perf.memory_allocated_gb + 0.5

    def test_reset_peak_memory_lowers_peak(self):
        t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")
        del t
        torch.mps.synchronize()
        high = get_memory_stats("mps").peak_gb
        reset_peak_memory("mps")
        low = get_memory_stats("mps").peak_gb
        assert low < high

    def test_later_small_step_is_not_inflated_by_earlier_spike(self):
        if not device_capabilities("mps").peak_memory_trackable:
            pytest.skip("MPS allocator peak statistics require PyTorch 2.13+")

        with step_perf("mps", batch_size=1) as first:
            t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")
            del t
        with step_perf("mps", batch_size=1) as second:
            t = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device="mps")
            del t

        assert first.perf.memory_peak_gb >= 0.9
        assert second.perf.memory_peak_gb == pytest.approx(0.0625, abs=0.01)
        assert second.perf.memory_peak_gb < first.perf.memory_peak_gb

    def test_step_perf_does_not_empty_cache_per_step(self, monkeypatch):
        """empty_cache every step would cripple a training loop — guard it."""
        calls: list[int] = []
        monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
        with step_perf("mps", batch_size=1, track_memory=True):
            a = torch.randn(64, 64, device="mps")
            _ = a @ a
        assert calls == []


@pytest.mark.mps
class TestMpsMarks:
    def test_marks_capture_real_gpu_time_by_default(self):
        """Marks sync by default, so they reflect execution, not launch time.

        A non-synced mark over 200 large matmuls would report microseconds
        (kernel-launch only); a synced one reports the real tens+ of ms.
        """
        a = torch.randn(2048, 2048, device="mps")
        with step_perf("mps", batch_size=1) as sp:
            x = torch.randn(2048, 2048, device="mps")
            for _ in range(200):
                x = x @ a
            sp.mark("compute")
        assert sp.perf.marks["compute"] > 0.01

    def test_marks_partition_the_step(self):
        """Because every mark syncs, the marks sum to ~ the whole step time."""
        a = torch.randn(1024, 1024, device="mps")
        with step_perf("mps", batch_size=1) as sp:
            x = torch.randn(1024, 1024, device="mps")
            for _ in range(100):
                x = x @ a
            sp.mark("first")
            for _ in range(100):
                x = x @ a
            sp.mark("second")
        perf = sp.perf
        marks_sum = perf.marks["first"] + perf.marks["second"]
        assert marks_sum <= perf.step_time_sec + 1e-3
        assert marks_sum >= 0.7 * perf.step_time_sec
