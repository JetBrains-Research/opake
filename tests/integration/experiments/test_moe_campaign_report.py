"""Synthetic metadata tests for reporting, not evidence of experiment outcomes."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from examples.moe_privacy.campaign_report import build_report, write_report

ARMS = ("reference", "reference_aux", "dp", "dp_aux")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _code(plus=0.5):
    return {
        name: {
            "plus_pass1": plus,
            "base_pass1": plus + 0.05,
            "task_count": count,
            "smoke": False,
            "protocol_sha256": f"synthetic-{name}-protocol",
        }
        for name, count in (("humaneval", 164), ("mbpp", 378))
    }


def _campaign(root, seeds=(0, 1, 2), *, completed=True, pooled=False):
    config = {
        "model_id": "synthetic/report-test",
        "model_revision": "fixed-test-revision",
        "dataset_languages": ("python",),
        "train_sequences": 8192,
        "validation_sequences": 256,
        "test_sequences": 1024,
        "expected_batch_size": 32,
        "steps": 256,
        "target_epsilon": 8.0,
        "delta": 1e-5,
    }
    _write(
        root / "campaign.json",
        {
            "config": config,
            "seeds": seeds,
            "arms": ARMS,
            "protocol": "matched-clipped-lagged-v2",
        },
    )
    if not completed:
        return root
    _write(root / "base/code_metrics.json", _code(0.4))
    for seed in seeds:
        for arm in ARMS:
            private, auxiliary = arm.startswith("dp"), arm.endswith("_aux")
            folder = root / "main" / f"seed-{seed}" / arm
            summary = {
                "status": "completed",
                "arm": arm,
                "model_seed": seed,
                "config": copy.deepcopy(config),
                "privacy": {
                    "private": private,
                    "epsilon": 7.9995 if private else None,
                    "delta": config["delta"] if private else None,
                    "noise_multiplier": 1.0 if private else 0.0,
                    "steps": config["steps"],
                    "sample_rate": 32 / 8192,
                },
                "data": {
                    f"{part}_sha256": character * 64
                    for part, character in (
                        ("train", "a"),
                        ("validation", "b"),
                        ("test", "c"),
                    )
                },
                "trainability": {"trainable_parameters": 100},
                "execution": {"peak_cuda_memory_allocated_bytes": 8_000_000_000},
                "training_seconds": 3600,
                "measurements": {
                    "test_nll": (2.2 if private else 2.0) * (1.001 if auxiliary else 1),
                    "test_token_accuracy": 0.6,
                    "test_load_cv": (1.1 if private else 1)
                    * (0.8 + seed * 0.01 if auxiliary else 1),
                    "test_max_mean_load": 1.8,
                    "test_effective_experts": 6.5,
                },
            }
            if pooled:
                summary["measurements"].update(
                    test_pooled_load_cv=0.1 if auxiliary else 0.4,
                    test_pooled_max_mean_load=1.1 if auxiliary else 1.6,
                    test_pooled_effective_experts=7.5 if auxiliary else 6.0,
                )
            summary["config"]["dataset_languages"] = ["python"]
            _write(folder / "summary.json", summary)
            _write(
                folder / "code_metrics.json",
                _code((0.45 if private else 0.5) + (0.005 if auxiliary else 0)),
            )
    return root


def _change(root, relative, section, key, value):
    path = root / relative
    content = json.loads(path.read_text())
    content[section][key] = value
    _write(path, content)


def test_three_seed_paired_aggregation_and_privacy_utility(tmp_path):
    report = build_report(_campaign(tmp_path))
    assert report["counts"]["complete"] == 12
    assert report["baseline"]["metrics"]["humaneval_plus_pass1"] == 0.4
    aggregate = report["aggregates"]["reference_aux"]["test_load_cv"]
    assert aggregate["mean"] == pytest.approx(0.81)
    assert aggregate["sample_sd"] == pytest.approx(0.01)
    for name in ("reference_aux_minus_reference", "dp_aux_minus_dp"):
        comparison = report["comparisons"][name]
        assert comparison["conclusion"] == "supported"
        assert comparison["complete_paired_seeds"] == [0, 1, 2]
        effect = comparison["effects"]["test_load_cv_relative_pct"]
        assert effect["n"] == 3
        assert effect["mean"] == pytest.approx(-19)
        assert effect["sample_sd"] == pytest.approx(1)
        assert effect["ci95"] == pytest.approx(
            [-19 - 4.303 / math.sqrt(3), -19 + 4.303 / math.sqrt(3)]
        )
    utility = report["comparisons"]["dp_minus_reference"]["effects"]
    assert utility["test_nll_relative_pct"]["mean"] == pytest.approx(10)
    assert utility["humaneval_plus_pass1_pp"]["mean"] == pytest.approx(-5)
    assert report["rows"][0]["metrics"]["training_hours"] == 1
    assert report["rows"][0]["metrics"]["peak_memory_gb"] == 8


def test_legacy_summaries_leave_optional_pooled_metrics_pending(tmp_path):
    report = build_report(_campaign(tmp_path))
    assert report["counts"]["complete"] == 12
    assert report["counts"]["pooled_metrics_pending"] == 12
    for row in report["rows"]:
        assert row["pooled_status"] == "pending"
        assert len(row["missing_pooled_metrics"]) == 3
        assert row["metrics"]["test_pooled_load_cv"] is None
    for name in ("reference_aux_minus_reference", "dp_aux_minus_dp"):
        comparison = report["comparisons"][name]
        response = comparison["pooled_objective_response"]
        assert comparison["conclusion"] == "supported"
        assert response["status"] == "pending"
        assert response["pending_pairs"] == 3
        assert response["evidence"] == {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "ci95": None,
        }


def test_pooled_improvement_does_not_establish_per_layer_practical_benefit(tmp_path):
    root = _campaign(tmp_path, pooled=True)
    for seed in (0, 1, 2):
        for arm in ARMS:
            _change(
                root,
                f"main/seed-{seed}/{arm}/summary.json",
                "measurements",
                "test_load_cv",
                1.0,
            )
    report = write_report(root)
    assert report["counts"]["pooled_metrics_pending"] == 0
    assert report["criterion"]["margins"] == {
        "test_load_cv_relative_pct": -10,
        "test_nll_relative_pct": 2,
        "humaneval_plus_pass1_pp": -2,
    }
    for name in ("reference_aux_minus_reference", "dp_aux_minus_dp"):
        comparison = report["comparisons"][name]
        assert comparison["complete_paired_seeds"] == [0, 1, 2]
        assert comparison["conclusion"] == "inconclusive"
        assert comparison["criterion_evidence"]["test_load_cv_relative_pct"][
            "mean"
        ] == pytest.approx(0)
        response = comparison["pooled_objective_response"]
        assert response["status"] == "improved"
        assert response["pending_pairs"] == 0
        assert response["evidence"]["n"] == 3
        assert response["evidence"]["mean"] == pytest.approx(-75)
        assert response["evidence"]["ci95"] == pytest.approx([-75, -75])
        assert (
            comparison["effects"]["test_pooled_load_cv_relative_pct"]
            == response["evidence"]
        )
    for arm in ARMS:
        metrics = report["aggregates"][arm]
        assert metrics["test_load_cv"]["mean"] == 1
        assert metrics["test_pooled_load_cv"]["mean"] == pytest.approx(
            0.1 if arm.endswith("_aux") else 0.4
        )
    header, _, *rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in (root / "report.md").read_text().splitlines()
        if line.startswith("|")
    ]
    row = next(row for row in rows if row[:2] == ["0", "reference_aux"])
    cells = dict(zip(header, row, strict=True))
    assert float(cells["Mean per-layer load CV"]) == 1
    assert float(cells["Pooled-index load CV"]) == 0.1
    assert float(cells["Pooled-index max/mean load"]) == 1.1
    assert float(cells["Pooled-index effective experts"]) == 7.5


@pytest.mark.parametrize("control_cv", [None, 0.0])
def test_pooled_pairs_exclude_missing_values_and_zero_denominators(
    tmp_path, control_cv
):
    root = _campaign(tmp_path, pooled=True)
    _change(
        root,
        "main/seed-2/dp/summary.json",
        "measurements",
        "test_pooled_load_cv",
        control_cv,
    )
    report = build_report(root)
    assert report["counts"]["complete"] == 12
    assert report["counts"]["pooled_metrics_pending"] == int(control_cv is None)
    comparison = report["comparisons"]["dp_aux_minus_dp"]
    assert comparison["conclusion"] == "supported"
    response = comparison["pooled_objective_response"]
    assert response["status"] == "inconclusive"
    assert response["pending_pairs"] == 1
    assert response["evidence"]["n"] == 2
    assert response["evidence"]["mean"] == pytest.approx(-75)
    assert comparison["paired_values"][2]["test_pooled_load_cv_relative_pct"] is None


@pytest.mark.parametrize(
    ("values", "status"),
    [
        ((0.8, 0.8, 0.8), "worsened"),
        ((0.2, 0.3, 0.5), "inconclusive"),
        ((0.4, 0.4, 0.4), "inconclusive"),
    ],
)
def test_pooled_response_uses_paired_intervals_not_only_means(tmp_path, values, status):
    root = _campaign(tmp_path, pooled=True)
    for seed, value in enumerate(values):
        _change(
            root,
            f"main/seed-{seed}/dp_aux/summary.json",
            "measurements",
            "test_pooled_load_cv",
            value,
        )
    comparison = build_report(root)["comparisons"]["dp_aux_minus_dp"]
    response = comparison["pooled_objective_response"]
    effect = response["evidence"]
    assert response["status"] == status
    assert effect["n"] == 3
    width = 4.303 * effect["sample_sd"] / math.sqrt(3)
    assert effect["ci95"] == pytest.approx(
        [effect["mean"] - width, effect["mean"] + width]
    )
    assert comparison["conclusion"] == "supported"


def test_pooled_metrics_do_not_replace_required_per_layer_metrics(tmp_path):
    root = _campaign(tmp_path, pooled=True)
    _change(
        root, "main/seed-2/dp_aux/summary.json", "measurements", "test_load_cv", None
    )
    report = build_report(root)
    row = report["rows"][-1]
    assert row["status"] == "metrics_pending"
    assert row["pooled_status"] == "completed"
    assert row["missing_metrics"] == ["test_load_cv"]
    comparison = report["comparisons"]["dp_aux_minus_dp"]
    assert comparison["conclusion"] == "inconclusive"
    assert comparison["pooled_objective_response"]["status"] == "improved"


@pytest.mark.parametrize(
    "metric",
    [
        "test_pooled_load_cv",
        "test_pooled_max_mean_load",
        "test_pooled_effective_experts",
    ],
)
@pytest.mark.parametrize("value", [-1, math.nan])
def test_invalid_optional_pooled_metrics_raise(tmp_path, metric, value):
    root = _campaign(tmp_path)
    _change(root, "main/seed-0/dp_aux/summary.json", "measurements", metric, value)
    with pytest.raises(ValueError, match=metric):
        build_report(root)


def test_pending_rows_missing_metrics_and_state_cannot_complete_training(tmp_path):
    root = _campaign(tmp_path)
    (root / "main/seed-1/reference/summary.json").unlink()
    (root / "main/seed-2/dp_aux/code_metrics.json").unlink()
    _change(
        root, "main/seed-2/reference_aux/summary.json", "measurements", "test_nll", None
    )
    _write(root / "campaign_state.json", {"status": "completed"})
    report = build_report(root)
    rows = {(row["seed"], row["arm"]): row for row in report["rows"]}
    assert len(rows) == 12
    assert rows[1, "reference"]["status"] == "pending"
    assert rows[2, "dp_aux"]["status"] == "code_pending"
    assert rows[2, "reference_aux"]["status"] == "metrics_pending"
    assert rows[1, "reference"]["metrics"]["test_nll"] is None
    assert rows[2, "dp_aux"]["metrics"]["humaneval_plus_pass1"] is None
    assert report["aggregates"]["reference"]["test_nll"]["n"] == 2
    assert report["counts"]["training_pending"] == 1
    assert report["counts"]["code_pending"] == 1
    assert report["counts"]["complete"] == 9
    assert report["comparisons"]["dp_aux_minus_dp"]["pending_pairs"] == 1
    assert report["comparisons"]["dp_aux_minus_dp"]["conclusion"] == "inconclusive"


def test_empty_campaign_ignores_pilots_and_retains_base_pending(tmp_path):
    root = _campaign(tmp_path, completed=False)
    _write(root / "pilot/summary.json", {"status": "completed", "invalid": True})
    report = build_report(root)
    assert report["counts"]["training_pending"] == 12
    assert report["counts"]["code_pending"] == 12
    assert report["baseline"]["code_status"] == "pending"
    assert report["aggregates"]["dp"]["test_nll"] == {
        "n": 0,
        "mean": None,
        "sample_sd": None,
        "ci95": None,
    }


def test_one_seed_cannot_establish_support(tmp_path):
    report = build_report(_campaign(tmp_path, seeds=(0,), pooled=True))
    comparison = report["comparisons"]["dp_aux_minus_dp"]
    assert comparison["conclusion"] == "inconclusive"
    assert comparison["effects"]["test_nll_relative_pct"]["ci95"] is None
    assert comparison["effects"]["test_nll_relative_pct"]["sample_sd"] is None
    response = comparison["pooled_objective_response"]
    assert response["status"] == "inconclusive"
    assert response["evidence"]["n"] == 1
    assert response["evidence"]["ci95"] is None


def test_favorable_means_do_not_override_uncertain_intervals(tmp_path):
    root = _campaign(tmp_path)
    for seed, cv in enumerate((0.7, 0.8, 1.0)):
        _change(
            root,
            f"main/seed-{seed}/reference_aux/summary.json",
            "measurements",
            "test_load_cv",
            cv,
        )
    comparison = build_report(root)["comparisons"]["reference_aux_minus_reference"]
    effect = comparison["effects"]["test_load_cv_relative_pct"]
    assert effect["mean"] < -10
    assert effect["ci95"][1] > -10
    assert comparison["conclusion"] == "inconclusive"


@pytest.mark.parametrize(("size", "critical"), [(2, 12.706), (10, 2.262)])
def test_student_intervals_use_number_of_training_seeds(tmp_path, size, critical):
    report = build_report(_campaign(tmp_path, seeds=tuple(range(size))))
    effect = report["comparisons"]["reference_aux_minus_reference"]["effects"][
        "test_load_cv_relative_pct"
    ]
    width = critical * effect["sample_sd"] / math.sqrt(size)
    assert effect["ci95"] == pytest.approx(
        [effect["mean"] - width, effect["mean"] + width]
    )


def test_stage_failure_is_operational_not_completion_evidence(tmp_path):
    root = _campaign(tmp_path, completed=False)
    _write(
        root / "campaign_state.json",
        {
            "stages": {
                "main/seed-0/dp": {"status": "failed"},
                "main/seed-0/dp_aux": {"status": "completed"},
            }
        },
    )
    report = build_report(root)
    assert report["counts"]["training_completed"] == 0
    assert report["counts"]["training_failed"] == 1
    assert report["counts"]["training_pending"] == 11


@pytest.mark.parametrize("metric", ["test_load_cv", "test_nll", "humaneval_plus_pass1"])
def test_negative_balance_or_utility_effect_is_not_supported(tmp_path, metric):
    root = _campaign(tmp_path)
    for seed in (0, 1, 2):
        folder = f"main/seed-{seed}/dp_aux"
        if metric == "humaneval_plus_pass1":
            _change(root, f"{folder}/code_metrics.json", "humaneval", "plus_pass1", 0.4)
        else:
            _change(root, f"{folder}/summary.json", "measurements", metric, 3.0)
    comparison = build_report(root)["comparisons"]["dp_aux_minus_dp"]
    assert comparison["conclusion"] == "observed trade-off"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("config", "model_id", "different", "configuration"),
        ("config", "model_revision", "different", "configuration"),
        ("config", "test_sequences", 16, "configuration"),
        ("data", "train_sha256", "different", "data"),
        ("data", "validation_sha256", "different", "data"),
        ("data", "test_sha256", "different", "data"),
        ("privacy", "epsilon", 8.1, "epsilon"),
        ("privacy", "epsilon", 7.9, "epsilon"),
        ("privacy", "delta", 1e-6, "delta"),
        ("privacy", "sample_rate", 0.1, "sampling"),
        ("privacy", "steps", 255, "sampling"),
        ("privacy", "private", False, "private"),
        ("trainability", "trainable_parameters", 101, "trainability"),
    ],
)
def test_incomparable_runs_raise(tmp_path, section, key, value, message):
    root = _campaign(tmp_path)
    _change(root, "main/seed-2/dp_aux/summary.json", section, key, value)
    with pytest.raises(ValueError, match=message):
        build_report(root)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("smoke", True, "smoke"),
        ("task_count", 163, "task_count"),
        ("protocol_sha256", "different", "protocol"),
        ("plus_pass1", 1.1, "pass1"),
    ],
)
@pytest.mark.parametrize("location", ["base", "main/seed-2/dp_aux"])
def test_invalid_benchmark_results_raise(tmp_path, key, value, message, location):
    root = _campaign(tmp_path)
    _change(root, f"{location}/code_metrics.json", "humaneval", key, value)
    with pytest.raises(ValueError, match=message):
        build_report(root)


def test_failed_summary_is_not_used_and_zero_denominator_is_unknown(tmp_path):
    root = _campaign(tmp_path)
    _write(root / "main/seed-1/reference/summary.json", {"status": "failed"})
    _change(root, "main/seed-0/dp/summary.json", "measurements", "test_load_cv", 0.0)
    report = build_report(root)
    assert report["counts"]["training_failed"] == 1
    comparison = report["comparisons"]["dp_aux_minus_dp"]
    assert comparison["effects"]["test_load_cv_relative_pct"]["n"] == 2
    assert comparison["conclusion"] == "inconclusive"


@pytest.mark.parametrize("seeds", [[], [0, 0], list(range(11))])
def test_invalid_seed_plan_raises(tmp_path, seeds):
    with pytest.raises(ValueError, match="seeds"):
        build_report(_campaign(tmp_path, seeds=seeds, completed=False))


def test_atomic_outputs_and_cli(tmp_path):
    root = _campaign(tmp_path)
    _write(
        root / "main/seed-0/dp/wandb_run.json", {"url": "https://example.invalid/run"}
    )
    original = set(root.iterdir())
    report = write_report(root)
    assert json.loads((root / "report.json").read_text()) == report
    assert set(root.iterdir()) - original == {root / "report.json", root / "report.md"}
    assert (root / "report.md").stat().st_size > 0
    row = next(row for row in report["rows"] if row["seed"] == 0 and row["arm"] == "dp")
    assert row["wandb_url"] == "https://example.invalid/run"
    script = Path(__file__).parents[3] / "examples/moe_privacy/campaign_report.py"
    subprocess.run(
        [sys.executable, "-B", str(script), "--root", str(root)],
        check=True,
        capture_output=True,
    )
    assert json.loads((root / "report.json").read_text()) == report


def test_interrupted_replacement_preserves_report_and_removes_temporary(
    tmp_path, monkeypatch
):
    root = _campaign(tmp_path)
    write_report(root)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}

    def fail_replace(self, target):
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        write_report(root)
    assert {
        path.name: path.read_bytes() for path in root.iterdir() if path.is_file()
    } == before
