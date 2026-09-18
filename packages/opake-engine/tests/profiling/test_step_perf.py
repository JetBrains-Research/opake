"""Tests for the profiling API (StepPerf, step_perf, PerfStage, PerfTracker)."""

import time
import warnings
from dataclasses import replace

import pytest
import torch

from opake.api.engine.device import device_capabilities
from opake.profiling import (
    PerfStage,
    PerfState,
    PerfTracker,
    StepPerf,
    perf_tracker,
    step_perf,
)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.insert(0, "cuda")
if torch.backends.mps.is_available():
    DEVICES.insert(0, "mps")


class TestStepPerf:
    """Tests for StepPerf frozen dataclass."""

    def test_default_values(self):
        perf = StepPerf()
        assert perf.step_time_sec == 0.0
        assert perf.samples_per_second == 0.0
        assert perf.memory_peak_gb == 0.0
        assert perf.batch_size == 0
        assert perf.marks == {}

    def test_to_dict_bare_keys(self):
        perf = StepPerf(step_time_sec=1.5, samples_per_second=20.0, memory_peak_gb=3.0)
        d = perf.to_dict()
        assert "step_time_sec" in d
        assert "samples_per_second" in d
        assert "memory_peak_gb" in d
        assert d["step_time_sec"] == 1.5
        assert d["samples_per_second"] == 20.0

    def test_to_dict_with_prefix(self):
        perf = StepPerf(step_time_sec=1.0)
        d = perf.to_dict(prefix="train/")
        assert "train/step_time_sec" in d
        assert "train/samples_per_second" in d
        assert "step_time_sec" not in d

    def test_to_dict_includes_marks(self):
        perf = StepPerf(step_time_sec=2.0, marks={"clip": 1.2, "noise": 0.5})
        d = perf.to_dict(prefix="perf/")
        assert d["perf/clip_sec"] == 1.2
        assert d["perf/noise_sec"] == 0.5

    def test_frozen(self):
        perf = StepPerf(step_time_sec=1.0)
        with pytest.raises(AttributeError):
            perf.step_time_sec = 2.0


class TestStepPerfContextManager:
    """Tests for step_perf() context manager."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_measures_time(self, device):
        with step_perf(device, batch_size=32) as sp:
            x = torch.randn(100, 100)
            _ = x @ x.T
        result = sp.perf
        assert result.step_time_sec > 0
        assert result.batch_size == 32
        assert result.samples_per_second > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_perf_not_available_inside_context(self, device):
        with (
            step_perf(device) as sp,
            pytest.raises(RuntimeError, match="not available"),
        ):
            _ = sp.perf

    def test_result_deprecation_warning(self):
        with step_perf("cpu") as sp:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = sp.result
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert ".perf" in str(w[0].message)

    def test_marks_basic(self):
        with step_perf("cpu", batch_size=10) as sp:
            time.sleep(0.01)
            sp.mark("phase_a")
            time.sleep(0.01)
            sp.mark("phase_b")
        result = sp.perf
        assert "phase_a" in result.marks
        assert "phase_b" in result.marks
        assert result.marks["phase_a"] > 0
        assert result.marks["phase_b"] > 0

    def test_marks_appear_in_to_dict(self):
        with step_perf("cpu") as sp:
            sp.mark("clip")
            sp.mark("noise")
        d = sp.perf.to_dict(prefix="train/")
        assert "train/clip_sec" in d
        assert "train/noise_sec" in d

    def test_no_memory_tracking(self):
        with step_perf("cpu", track_memory=False) as sp:
            pass
        result = sp.perf
        assert result.memory_peak_gb == 0.0
        assert result.memory_allocated_gb == 0.0

    def test_legacy_mps_peak_is_sampled_per_step(self, monkeypatch):
        from opake.api.engine.profiling import _memory

        samples = iter([1 * 1024**3, 3 * 1024**3, 2 * 1024**3])
        monkeypatch.setattr(
            _memory,
            "device_capabilities",
            lambda device: replace(
                device_capabilities("cpu"),
                device_type="mps",
                peak_memory_trackable=False,
            ),
        )
        monkeypatch.setattr(_memory, "_sync_device", lambda device: None)
        monkeypatch.setattr(
            torch.mps,
            "current_allocated_memory",
            lambda: next(samples),
        )
        monkeypatch.setattr(
            _memory,
            "get_memory_stats",
            lambda device: _memory.MemoryStats(
                allocated_gb=2.0,
                reserved_gb=4.0,
                peak_gb=4.0,
                exact_peak=False,
            ),
        )

        with step_perf("mps", batch_size=8) as sp:
            sp.mark("middle")

        assert sp.perf.memory_peak_gb == 3.0
        assert sp.perf.memory_allocated_gb == 2.0
        assert sp.perf.memory_reserved_gb == 4.0

    def test_synchronizes_before_resetting_peak(self, monkeypatch):
        from opake.api.engine.profiling import _memory

        events: list[str] = []
        monkeypatch.setattr(
            _memory,
            "device_capabilities",
            lambda device: replace(
                device_capabilities(device), peak_memory_trackable=True
            ),
        )
        monkeypatch.setattr(
            _memory, "_sync_device", lambda device: events.append("sync")
        )
        monkeypatch.setattr(
            _memory, "reset_peak_memory", lambda device: events.append("reset")
        )

        with step_perf("cpu"):
            pass

        assert events[:2] == ["sync", "reset"]

    def test_string_device(self):
        with step_perf("cpu", batch_size=16) as sp:
            pass
        assert sp.perf.batch_size == 16
        assert sp.perf.step_time_sec > 0

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_synchronizes(self, monkeypatch):
        calls = {"count": 0}

        def _sync():
            calls["count"] += 1

        monkeypatch.setattr(torch.mps, "synchronize", _sync)

        with step_perf("mps") as _sp:
            pass
        # sync called twice: once before timing, once after
        assert calls["count"] == 2


class TestPerfState:
    """Tests for PerfState functional accumulator."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        state = PerfState(device=torch.device(device))
        assert state.num_steps == 0
        assert state.total_time == 0.0
        assert state.total_samples == 0
        assert state.max_peak_memory_gb == 0.0
        assert state.last_step is None

    def test_add_single_step(self):
        state = PerfState(device=torch.device("cpu"))
        perf = StepPerf(step_time_sec=1.0, batch_size=32, memory_peak_gb=2.0)
        state = state.add(perf)
        assert state.num_steps == 1
        assert state.total_time == 1.0
        assert state.total_samples == 32
        assert state.max_peak_memory_gb == 2.0
        assert state.last_step is perf

    def test_warmup_exclusion(self):
        state = PerfState(device=torch.device("cpu"))
        # First step (warmup) — excluded from stable averages
        warmup = StepPerf(step_time_sec=10.0, batch_size=32)
        state = state.add(warmup)
        assert state.num_steps_stable == 0
        assert state.avg_step_time_stable == 10.0  # falls back to avg_step_time

        # Second step — included in stable averages
        step2 = StepPerf(step_time_sec=2.0, batch_size=32)
        state = state.add(step2)
        assert state.num_steps_stable == 1
        assert state.avg_step_time_stable == pytest.approx(2.0)

    def test_avg_step_time_stable_multi(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=10.0, batch_size=1))  # warmup
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        assert state.avg_step_time_stable == pytest.approx(2.0)

    def test_avg_samples_per_second_stable(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=10.0, batch_size=32))  # warmup
        state = state.add(StepPerf(step_time_sec=1.0, batch_size=32))
        state = state.add(StepPerf(step_time_sec=1.0, batch_size=32))
        assert state.avg_samples_per_second_stable == pytest.approx(32.0)

    def test_max_peak_memory_tracking(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(memory_peak_gb=2.0))
        state = state.add(StepPerf(memory_peak_gb=5.0))
        state = state.add(StepPerf(memory_peak_gb=3.0))
        assert state.max_peak_memory_gb == 5.0

    def test_to_dict(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(
            StepPerf(step_time_sec=1.0, batch_size=32, memory_peak_gb=4.0)
        )
        d = state.to_dict(prefix="perf/")
        assert "perf/avg_step_time_sec" in d
        assert "perf/avg_samples_per_second" in d
        assert "perf/max_peak_memory_gb" in d
        assert d["perf/max_peak_memory_gb"] == 4.0

    def test_to_dict_bare_keys(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=1.0))
        d = state.to_dict()
        assert "avg_step_time_sec" in d

    def test_immutable(self):
        state = PerfState(device=torch.device("cpu"))
        new_state = state.add(StepPerf(step_time_sec=1.0))
        assert state.num_steps == 0  # original unchanged
        assert new_state.num_steps == 1

    def test_empty_state_properties(self):
        state = PerfState(device=torch.device("cpu"))
        assert state.avg_step_time == 0.0
        assert state.avg_step_time_stable == 0.0
        assert state.avg_samples_per_second == 0.0
        assert state.avg_samples_per_second_stable == 0.0


class TestStepPerfIntegration:
    """Integration test: step_perf + PerfState together."""

    def test_full_loop(self):
        state = PerfState(device=torch.device("cpu"))
        for _i in range(3):
            with step_perf("cpu", batch_size=16) as sp:
                x = torch.randn(50, 50)
                _ = x @ x.T
                sp.mark("compute")
            state = state.add(sp.perf)

        assert state.num_steps == 3
        assert state.total_samples == 48
        assert state.avg_step_time > 0
        assert state.last_step is not None
        assert "compute" in state.last_step.marks


class TestPerfStage:
    """Tests for PerfStage mutable accumulator."""

    def test_init_defaults(self):
        stage = PerfStage("train", torch.device("cpu"))
        assert stage.name == "train"
        assert stage.num_steps == 0
        assert stage.total_time == 0.0
        assert stage.total_samples == 0
        assert stage.max_peak_memory_gb == 0.0
        assert stage.last is None
        assert stage.samples_per_second == 0.0
        assert stage.steps_per_second == 0.0

    def test_absorb_warmup_excluded(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=1)
        warmup = StepPerf(step_time_sec=10.0, batch_size=32, memory_peak_gb=5.0)
        stage._absorb(warmup)
        assert stage.num_steps == 1
        assert stage.total_time == 0.0
        assert stage.total_samples == 0
        assert stage.max_peak_memory_gb == 5.0
        assert stage.last is warmup

    def test_absorb_post_warmup(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=1)
        stage._absorb(StepPerf(step_time_sec=10.0, batch_size=32))
        step2 = StepPerf(step_time_sec=2.0, batch_size=64)
        stage._absorb(step2)
        assert stage.num_steps == 2
        assert stage.total_time == pytest.approx(2.0)
        assert stage.total_samples == 64
        assert stage.last is step2

    def test_warmup_configurable(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=2)
        stage._absorb(StepPerf(step_time_sec=10.0, batch_size=32))
        stage._absorb(StepPerf(step_time_sec=8.0, batch_size=32))
        assert stage.total_time == 0.0
        step3 = StepPerf(step_time_sec=2.0, batch_size=32)
        stage._absorb(step3)
        assert stage.total_time == pytest.approx(2.0)

    def test_ior_operator(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=0)
        perf = StepPerf(step_time_sec=1.0, batch_size=16)
        returned = (stage := stage.__ior__(perf))
        assert returned is stage
        assert stage.num_steps == 1
        assert stage.total_time == pytest.approx(1.0)

    def test_samples_per_second(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=0)
        stage._absorb(StepPerf(step_time_sec=2.0, batch_size=100))
        assert stage.samples_per_second == pytest.approx(50.0)

    def test_steps_per_second(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=0)
        stage._absorb(StepPerf(step_time_sec=0.5, batch_size=1))
        stage._absorb(StepPerf(step_time_sec=0.5, batch_size=1))
        assert stage.steps_per_second == pytest.approx(2.0)

    def test_context_manager_auto_absorbs(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=0)
        with stage(batch_size=32):
            _ = torch.randn(10, 10)
        assert stage.num_steps == 1
        assert stage.last is not None
        assert stage.last.batch_size == 32

    def test_to_dict(self):
        stage = PerfStage("eval", torch.device("cpu"), warmup_steps=0)
        stage._absorb(StepPerf(step_time_sec=1.0, batch_size=32, memory_peak_gb=4.0))
        d = stage.to_dict(prefix="eval/")
        assert d["eval/num_steps"] == 1
        assert d["eval/samples_per_second"] == pytest.approx(32.0)
        assert d["eval/max_peak_memory_gb"] == 4.0

    def test_max_peak_memory_tracks_across_steps(self):
        stage = PerfStage("train", torch.device("cpu"), warmup_steps=0)
        stage._absorb(StepPerf(memory_peak_gb=2.0))
        stage._absorb(StepPerf(memory_peak_gb=5.0))
        stage._absorb(StepPerf(memory_peak_gb=3.0))
        assert stage.max_peak_memory_gb == 5.0


class TestPerfTracker:
    """Tests for PerfTracker multi-stage tracker."""

    def test_factory_function(self):
        tracker = perf_tracker("cpu")
        assert isinstance(tracker, PerfTracker)
        assert tracker.device == torch.device("cpu")

    def test_getattr_train(self):
        tracker = perf_tracker("cpu")
        stage = tracker.train
        assert isinstance(stage, PerfStage)
        assert stage.name == "train"

    def test_getattr_eval_test(self):
        tracker = perf_tracker("cpu")
        assert tracker.eval.name == "eval"
        assert tracker.test.name == "test"

    def test_getitem_custom(self):
        tracker = perf_tracker("cpu")
        stage = tracker["generate"]
        assert isinstance(stage, PerfStage)
        assert stage.name == "generate"

    def test_lazy_creation_same_object(self):
        tracker = perf_tracker("cpu")
        a = tracker.train
        b = tracker.train
        assert a is b

    def test_getitem_same_object(self):
        tracker = perf_tracker("cpu")
        a = tracker["custom"]
        b = tracker["custom"]
        assert a is b

    def test_getattr_unknown_raises(self):
        tracker = perf_tracker("cpu")
        with pytest.raises(AttributeError):
            _ = tracker.nonexistent

    def test_setattr_shortcut_ignored(self):
        tracker = perf_tracker("cpu")
        _ = tracker.train
        tracker.train = "something"
        assert isinstance(tracker.train, PerfStage)

    def test_stages_property(self):
        tracker = perf_tracker("cpu")
        _ = tracker.train
        _ = tracker["custom"]
        stages = tracker.stages
        assert "train" in stages
        assert "custom" in stages
        assert isinstance(stages, dict)

    def test_warmup_propagated(self):
        tracker = perf_tracker("cpu", warmup_steps=3)
        assert tracker.train._warmup_steps == 3
        assert tracker["custom"]._warmup_steps == 3

    def test_context_manager_integration(self):
        tracker = perf_tracker("cpu", warmup_steps=0)
        for _ in range(3):
            with tracker.train(batch_size=16) as sp:
                x = torch.randn(10, 10)
                _ = x @ x.T
                sp.mark("compute")
        assert tracker.train.num_steps == 3
        assert tracker.train.total_samples == 48
        assert tracker.train.last is not None
        assert "compute" in tracker.train.last.marks
        assert tracker.train.samples_per_second > 0

    def test_multi_stage_integration(self):
        tracker = perf_tracker("cpu", warmup_steps=0)
        with tracker.train(batch_size=32):
            _ = torch.randn(10, 10)
        with tracker.eval(batch_size=64):
            _ = torch.randn(10, 10)
        assert tracker.train.num_steps == 1
        assert tracker.eval.num_steps == 1
        assert tracker.train.last.batch_size == 32
        assert tracker.eval.last.batch_size == 64
