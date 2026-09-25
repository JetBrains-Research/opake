"""Controller boundaries are tested without pretending to train or score a model."""

import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from examples.moe_privacy import campaign
from examples.moe_privacy.sft_run import SFTExperimentConfig


def test_protocol_hash_tracks_conditions_not_model_score():
    result = {
        "benchmark": {"sha256": "public-tasks"},
        "generation": {"max_new_tokens": 512},
        "evalplus_version": "0.3.1",
        "image_id": "immutable-image",
        "limits": {"cpus": 2, "memory_gb": 8, "timeout_seconds": 3600},
        "base": {"pass@1": 0.6},
        "plus": {"pass@1": 0.5, "wilson_95": [0.4, 0.6]},
        "task_count": 164,
        "mode": "full",
        "syntax_parse_success_rate": 0.8,
        "timeout_rate": 0.1,
        "truncation_rate": 0.2,
        "generation_seconds": 10.0,
        "generated_tokens": 1000,
    }
    first = campaign.normalize_code_result(result)
    other = deepcopy(result)
    other["plus"]["pass@1"] = 0.7
    assert (
        campaign.normalize_code_result(other)["protocol_sha256"]
        == first["protocol_sha256"]
    )
    other["generation"]["max_new_tokens"] = 256
    assert (
        campaign.normalize_code_result(other)["protocol_sha256"]
        != first["protocol_sha256"]
    )
    assert not first["smoke"]
    result["mode"] = "smoke"
    assert campaign.normalize_code_result(result)["smoke"]


def test_stage_runs_actual_process_and_records_failures(tmp_path):
    (tmp_path / "logs").mkdir()
    campaign.run_stage(
        [sys.executable, "-c", "print('controller process check')"],
        root=tmp_path,
        name="ok",
        deadline=time.time() + 90,
    )
    assert json.loads((tmp_path / "campaign_state.json").read_text())["stage_completed"]
    with pytest.raises(subprocess.CalledProcessError):
        campaign.run_stage(
            [sys.executable, "-c", "raise SystemExit(3)"],
            root=tmp_path,
            name="failed",
            deadline=time.time() + 90,
        )
    assert (tmp_path / "logs" / "failed.log").is_file()


def test_expired_stage_never_starts_process(tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        campaign.run_stage(
            ["nonexistent-command"], root=tmp_path, name="never", deadline=time.time()
        )
    assert list(tmp_path.iterdir()) == []


def test_stage_timeout_interrupts_owned_child_for_cleanup(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(campaign, "STAGE_GRACE_SECONDS", 0)
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_stage(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            root=tmp_path,
            name="timeout",
            deadline=time.time() + 0.5,
        )
    assert "KeyboardInterrupt" in (tmp_path / "logs" / "timeout.log").read_text()


def test_append_rejects_repeating_a_completed_benchmark_before_model_load(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(SFTExperimentConfig())))
    output = tmp_path / "result"
    output.mkdir()
    (output / "code_metrics.json").write_text(
        json.dumps({"humaneval": {"plus_pass1": 0.5}})
    )
    args = SimpleNamespace(
        config=config_path,
        output_dir=output,
        arm="base",
        append_benchmark=True,
        benchmarks=["humaneval"],
    )
    with pytest.raises(ValueError, match="only new benchmarks"):
        campaign.evaluate_checkpoint(args)
    assert list(output.iterdir()) == [output / "code_metrics.json"]


def test_appended_benchmark_cannot_change_checkpoint(tmp_path):
    config = SFTExperimentConfig()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "result"
    (output / "humaneval-execution").mkdir(parents=True)
    (output / "code_metrics.json").write_text(
        json.dumps({"humaneval": {"plus_pass1": 0.5}})
    )
    (output / "humaneval-execution" / "summary.json").write_text(
        json.dumps({"model": {"id": config.model_id, "revision": "0" * 40}})
    )
    args = SimpleNamespace(
        config=config_path,
        output_dir=output,
        arm="base",
        append_benchmark=True,
        benchmarks=["mbpp"],
    )
    with pytest.raises(ValueError, match="same base checkpoint"):
        campaign.evaluate_checkpoint(args)


@pytest.mark.parametrize(
    "deadline", ["2000-01-01T00:00:00Z", "2099-01-01T00:00:00Z", "2099-01-01T00:00:00"]
)
def test_bad_deadline_rejected_before_creating_campaign(tmp_path, deadline):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(SFTExperimentConfig())))
    args = SimpleNamespace(
        config=path, deadline=deadline, output_dir=tmp_path / "output"
    )
    with pytest.raises(ValueError, match="deadline"):
        campaign.run_campaign(args)
    assert not args.output_dir.exists()


def test_controller_stops_on_first_stage_error_and_preserves_report(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(SFTExperimentConfig(test_sequences=1024))))
    root = tmp_path / "campaign"
    args = SimpleNamespace(
        config=config_path,
        output_dir=root,
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        seeds=[0, 1, 2],
        group="test",
        image="sha256:test",
        benchmark_dir=tmp_path,
        benchmarks=["humaneval"],
        wandb_mode="disabled",
        max_new_tokens=512,
        docker_sudo=False,
    )
    calls = []

    def stage_error(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(campaign, "run_stage", stage_error)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.run_campaign(args)
    assert len(calls) == 1
    assert calls[0][2:4] == ["-m", "examples.moe_privacy.sft_run"]
    assert json.loads((root / "campaign_state.json").read_text())["status"] == "failed"
    assert (root / "report.md").is_file()
    assert not (root / "main").exists()
