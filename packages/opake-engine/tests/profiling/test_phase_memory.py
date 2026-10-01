"""Per-phase memory accounting (``step_perf(..., phase_memory=True)``).

The whole point of a per-phase peak is attribution: each mark must report the
high-water mark of *its own* segment. A cumulative reading would show the same
number at every mark once the peak has been set, which makes the breakdown
unreadable — that regression is what the CUDA tests here pin.
"""

import pytest
import torch

from opake.profiling import (
    StepPerf,
    finish_snapshot,
    start_snapshot,
    step_perf,
)


class TestPhaseMemory:
    def test_shape_without_a_peak_counter(self):
        """phase_table/to_dict must work on hand-built records (no device)."""
        perf = StepPerf(
            marks={"clip": 1.0, "noise": 2.0},
            mem_marks={"clip": 3.0, "noise": 1.0},
        )
        assert perf.phase_table() == [
            ("clip", 1.0, 3.0),
            ("noise", 2.0, 1.0),
        ]
        assert perf.to_dict(prefix="train/")["train/noise_peak_gb"] == 1.0

    def test_off_by_default(self):
        with step_perf("cpu", batch_size=1) as sp:
            sp.mark("a")
        assert sp.perf.mem_marks == {}

    def test_cpu_has_no_fake_peaks(self):
        """CPU has no peak counter: report nothing rather than zeros."""
        with step_perf("cpu", batch_size=1, phase_memory=True) as sp:
            sp.mark("a")
            sp.mark("b")
        assert sp.perf.mem_marks == {}
        assert [row[0] for row in sp.perf.phase_table()] == [
            "a",
            "b",
        ], "timing marks must still be recorded without memory"

    @pytest.mark.cuda
    def test_segment_peak_is_not_cumulative(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with step_perf("cuda", batch_size=1, phase_memory=True) as sp:
            big = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
            del big
            sp.mark("big")
            small = torch.empty(1024**2, device="cuda", dtype=torch.uint8)
            sp.mark("small")
            del small

        peaks = sp.perf.mem_marks
        assert peaks["big"] > 0.2
        assert peaks["small"] < peaks["big"] / 10, (
            "second phase looks cumulative: per-phase peaks must measure "
            "independent segments or the profile cannot attribute memory"
        )
        assert sp.perf.memory_peak_gb >= peaks["big"]

    @pytest.mark.cuda
    def test_snapshot_roundtrip(self, tmp_path):
        assert start_snapshot("cuda") is True
        scratch = torch.randn(4096, 4096, device="cuda")
        del scratch
        out = finish_snapshot(tmp_path / "snap.pickle", "cuda")
        assert out is not None
        assert out.exists() and out.stat().st_size > 0


class TestSnapshotOffCuda:
    def test_start_is_noop(self):
        assert start_snapshot("cpu") is False

    def test_finish_is_noop(self, tmp_path):
        assert finish_snapshot(tmp_path / "unused.pickle", "cpu") is None
        assert not (tmp_path / "unused.pickle").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
