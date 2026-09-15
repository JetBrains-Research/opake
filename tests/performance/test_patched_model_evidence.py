from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest
import torch
from tools.performance.patched_model_evidence import (
    CheckResult,
    EvidenceError,
    Stage,
    Workload,
    compare_runs,
    main,
    record_run,
    validate_run,
)

if TYPE_CHECKING:
    from pathlib import Path


def _factory(device: torch.device, config, seed: int) -> Workload:
    del device, seed
    durations = iter(config.get("durations", [0.01, 0.02]))

    def run(_: None) -> None:
        next(durations, 0.02)

    return Workload(
        workload_id="test.workload",
        config={"shape": [2, 3]},
        stages=(Stage("forward", "token", 6, "test", lambda: None, run),),
        checks=lambda: (
            CheckResult("output", "numerical", True, 0.0, 0.0, 1e-6, 1e-5),
            CheckResult("gradient", "gradient", True, 0.0, 0.0, 1e-6, 1e-5),
        ),
    )


def _run(monkeypatch, *, variant: str = "baseline"):
    from tools.performance import patched_model_evidence as evidence

    elapsed = iter([0.01, 0.02])

    class _Timer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            duration = next(elapsed)
            self.perf = type(
                "Perf",
                (),
                {
                    "step_time_sec": duration,
                    "samples_per_second": 6 / duration,
                    "memory_peak_gb": 2.0,
                    "memory_allocated_gb": 1.5,
                    "memory_reserved_gb": 2.5,
                },
            )()

    monkeypatch.setattr(evidence, "step_perf", lambda *args, **kwargs: _Timer())
    monkeypatch.setattr(
        evidence,
        "get_memory_stats",
        lambda device: type("Memory", (), {"allocated_gb": 1.0})(),
    )
    monkeypatch.setattr(
        evidence,
        "device_capabilities",
        lambda device: type("Capabilities", (), {"peak_memory_trackable": True})(),
    )
    monkeypatch.setattr(
        evidence,
        "_git_metadata",
        lambda: {"commit": "a" * 40, "dirty": False, "dirty_paths": []},
    )
    return record_run(
        _factory,
        variant=variant,
        device=torch.device("cpu"),
        config={},
        seed=956,
        warmup=0,
        repeats=2,
        command=["benchmark"],
    )


def _scale_metric(run, field: str, factor: float) -> None:
    stage = run["stages"][0]
    for sample in stage["samples"]:
        sample[field] *= factor
    for statistic in ("median", "p95", "min", "max"):
        stage["summary"][field][statistic] *= factor


def test_record_run_keeps_raw_stage_samples_and_reproducibility_metadata(monkeypatch):
    run = _run(monkeypatch)

    validate_run(run)
    assert run["metadata"]["seed"] == 956
    assert run["metadata"]["command"] == ["benchmark"]
    assert len(run["stages"][0]["samples"]) == 2
    assert run["stages"][0]["summary"]["duration_sec"]["median"] == 0.015
    assert run["stages"][0]["summary"]["duration_sec"]["p95"] == pytest.approx(0.0195)
    assert run["stages"][0]["selected_backend"] == "test"
    assert run["workload_counters"] == {}
    assert {check["kind"] for check in run["correctness"]} == {
        "numerical",
        "gradient",
    }


def test_comparison_reports_throughput_and_memory_independently(monkeypatch):
    baseline = _run(monkeypatch)
    candidate = copy.deepcopy(baseline)
    candidate["variant"] = "candidate"
    _scale_metric(candidate, "throughput", 0.8)
    _scale_metric(candidate, "peak_allocated_bytes", 0.5)

    report = compare_runs(baseline, candidate)

    stage = report["stages"][0]
    assert stage["throughput_status"] == "regression"
    assert stage["memory_status"] == "pass"
    assert report["performance_passed"] is False


def test_comparison_rejects_different_environment_fingerprints(monkeypatch):
    baseline = _run(monkeypatch)
    candidate = copy.deepcopy(baseline)
    candidate["comparison_fingerprint"] = "different"

    with pytest.raises(EvidenceError, match="fingerprint"):
        compare_runs(baseline, candidate)


def test_comparison_rejects_dirty_diagnostic_artifacts(monkeypatch):
    baseline = _run(monkeypatch)
    candidate = copy.deepcopy(baseline)
    candidate["metadata"]["git"]["dirty"] = True

    with pytest.raises(EvidenceError, match="dirty diagnostic"):
        compare_runs(baseline, candidate)


def test_validation_requires_both_numerical_and_gradient_checks(monkeypatch):
    run = _run(monkeypatch)
    run["correctness"] = [run["correctness"][0]]

    with pytest.raises(EvidenceError, match="numerical and gradient"):
        validate_run(run)


def test_validation_recomputes_summaries_from_raw_samples(monkeypatch):
    run = _run(monkeypatch)
    run["stages"][0]["samples"][0]["throughput"] = 1e99

    with pytest.raises(EvidenceError, match="does not match raw samples"):
        validate_run(run)


def test_inexact_memory_is_not_an_enforceable_comparison(monkeypatch):
    baseline = _run(monkeypatch)
    candidate = copy.deepcopy(baseline)
    for run in (baseline, candidate):
        for sample in run["stages"][0]["samples"]:
            sample["peak_exact"] = False
    _scale_metric(candidate, "peak_allocated_bytes", 2)

    report = compare_runs(baseline, candidate)

    assert report["stages"][0]["memory_peak_exact"] is False
    assert report["stages"][0]["memory_status"] == "unavailable"


@pytest.mark.parametrize("budget", [float("nan"), float("inf")])
def test_comparison_rejects_nonfinite_budgets(monkeypatch, budget):
    baseline = _run(monkeypatch)

    with pytest.raises(ValueError, match="max_memory_regression"):
        compare_runs(baseline, copy.deepcopy(baseline), max_memory_regression=budget)


def test_record_run_rejects_dirty_worktree_by_default(monkeypatch):
    from tools.performance import patched_model_evidence as evidence

    monkeypatch.setattr(
        evidence,
        "_git_metadata",
        lambda: {"commit": "a" * 40, "dirty": True, "dirty_paths": ["file.py"]},
    )

    with pytest.raises(EvidenceError, match="worktree is dirty"):
        record_run(
            _factory,
            variant="candidate",
            device=torch.device("cpu"),
            config={},
            seed=956,
            warmup=0,
            repeats=1,
        )


def test_compare_cli_only_enforces_wall_clock_when_requested(
    monkeypatch, tmp_path: Path
):
    baseline = _run(monkeypatch)
    candidate = copy.deepcopy(baseline)
    candidate["variant"] = "candidate"
    _scale_metric(candidate, "throughput", 0.5)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    assert main(["compare", str(baseline_path), str(candidate_path)]) == 0
    assert (
        main(
            [
                "compare",
                str(baseline_path),
                str(candidate_path),
                "--enforce-performance",
            ]
        )
        == 3
    )
