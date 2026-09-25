"""Native-control configuration, evidence checks, and non-interrupting queue policy."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.moe_privacy import trl_campaign
from examples.moe_privacy.campaign import save
from examples.moe_privacy.sft_run import SFTExperimentConfig


@pytest.fixture
def reference():
    return {
        "status": "completed",
        "arm": "dp",
        "model_seed": 0,
        "config": {"steps": 256, "router_aux_loss_coef": 0.2},
        "data": {key: {"sha256": f"public-{key}"} for key in trl_campaign.PARTITIONS},
        "trainability": {
            "trainable_parameters": 128,
            "target_parameters": ["expert"],
            "geometry": {"num_experts": 4},
            "rank": 4,
            "alpha": 8,
        },
        "privacy": {
            "private": True,
            "epsilon": 8,
            "noise_multiplier": 0.4,
            "load_release": False,
            "steps": 256,
            "balancing": "off",
        },
        "measurements": {"test_nll": 0.5, "test_token_accuracy": 0.9},
        "training_seconds": 3600,
    }


@pytest.fixture
def native(reference):
    result = copy.deepcopy(reference)
    result["arm"] = "trl_reference"
    result["privacy"].update(private=False, epsilon=None, noise_multiplier=0)
    return result


@pytest.mark.parametrize(
    ("filename", "name", "coefficient"),
    [
        ("sft_trl_baselines.json", "mellum2-magicoder-native-trl", 0.2),
        (
            "sft_trl_aux0001.json",
            "mellum2-magicoder-native-trl-aux-coef0.001-ratio-na",
            0.001,
        ),
    ],
)
def test_native_config_preserves_public_task_and_optimizer_budget(
    filename, name, coefficient
):
    directory = Path(trl_campaign.__file__).parent / "configs"
    private = json.loads((directory / "sft_balancing_aux02.json").read_text())
    template = json.loads((directory / "sft_trl_baselines.json").read_text())
    native = json.loads((directory / filename).read_text())
    assert native == {**template, "name": name, "router_aux_loss_coef": coefficient}
    config = SFTExperimentConfig(**native)
    trl_campaign.validate_shared_config(native, private)
    assert config.router_aux_loss_coef == coefficient
    assert native["steps"] == private["steps"]
    assert native["expected_batch_size"] == private["expected_batch_size"]
    assert native["microbatch_size"] == 4
    assert native["expected_batch_size"] // native["microbatch_size"] == 8
    assert native["expected_batch_size"] % native["microbatch_size"] == 0
    assert native["clipping_norm"] == 1.0
    assert native["gradient_checkpointing"] is True


@pytest.mark.parametrize(
    "field",
    [
        "learning_rate",
        "steps",
        "model_id",
        "model_revision",
        "dataset_revision",
        "dataset_languages",
        "excluded_train_prompt_hashes",
        "data_seed",
        "validation_seed",
        "sequence_length",
        "train_sequences",
        "test_sequences",
        "expected_batch_size",
        "lora_rank",
        "lora_alpha",
    ],
)
def test_reject_changed_comparison_settings(field):
    config = {field: 1}
    with pytest.raises(ValueError, match="unmatched comparison"):
        trl_campaign.validate_shared_config(config, {field: 2})


def test_validate_real_completion_metadata(reference, native):
    trl_campaign.validate_result(native, reference, native["config"], "trl_reference")


@pytest.mark.parametrize(
    ("status", "arm"),
    [
        ("queued", "trl_reference"),
        ("running", "trl_reference"),
        ("failed", "trl_reference"),
        ("completed", "trl_reference_aux"),
        ("completed", "dp"),
    ],
)
def test_reject_incomplete_or_wrong_native_arm(reference, native, status, arm):
    native.update(status=status, arm=arm)
    with pytest.raises(ValueError, match="did not complete the requested arm"):
        trl_campaign.validate_result(
            native, reference, native["config"], "trl_reference"
        )


@pytest.mark.parametrize("coefficient", [0.2, 0.001])
def test_reject_reusing_another_native_coefficient(reference, native, coefficient):
    native["arm"] = "trl_reference_aux"
    native["config"]["router_aux_loss_coef"] = coefficient
    requested = dict(
        native["config"], router_aux_loss_coef=0.001 if coefficient == 0.2 else 0.2
    )
    with pytest.raises(ValueError, match="training configuration"):
        trl_campaign.validate_result(native, reference, requested, "trl_reference_aux")


@pytest.mark.parametrize(
    "field", ["trainable_parameters", "target_parameters", "geometry", "rank", "alpha"]
)
def test_reject_changed_trainable_scope(reference, native, field):
    native["trainability"][field] = None
    with pytest.raises(ValueError, match="unmatched trainability"):
        trl_campaign.validate_result(
            native, reference, native["config"], "trl_reference"
        )


def test_reject_changed_initialization(reference, native):
    native["model_seed"] = 1
    with pytest.raises(ValueError, match="initialization does not match"):
        trl_campaign.validate_result(
            native, reference, native["config"], "trl_reference"
        )


@pytest.mark.parametrize("partition", trl_campaign.PARTITIONS)
def test_reject_different_data_partition(reference, native, partition):
    native["data"][partition]["sha256"] = "other-records"
    with pytest.raises(ValueError, match=f"unmatched {partition}"):
        trl_campaign.validate_result(
            native, reference, native["config"], "trl_reference"
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("private", True),
        ("epsilon", 0),
        ("noise_multiplier", 0.1),
        ("load_release", True),
        ("steps", 2),
    ],
)
def test_reject_private_or_incomplete_native_result(reference, native, key, value):
    native["privacy"][key] = value
    with pytest.raises(ValueError, match="non-private"):
        trl_campaign.validate_result(
            native, reference, native["config"], "trl_reference"
        )


def test_wait_never_stops_or_restarts_private_training(tmp_path, monkeypatch):
    states = iter(("active", "inactive"))
    calls = []
    save(tmp_path / "campaign_state.json", {"status": "completed"})

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"ActiveState={next(states)}\nResult=success\n")

    monkeypatch.setattr(trl_campaign.subprocess, "run", inspect)
    monkeypatch.setattr(trl_campaign.time, "time", lambda: 0)
    monkeypatch.setattr(trl_campaign.time, "sleep", lambda _: None)
    trl_campaign.wait_for_previous("private.service", tmp_path, deadline=60)
    assert len(calls) == 2
    assert all(
        command[:3] == ["systemctl", "show", "private.service"] for command in calls
    )


@pytest.mark.parametrize("state", ["failed", "inactive"])
def test_predecessor_failure_or_missing_receipt_blocks_training(
    tmp_path, monkeypatch, state
):
    monkeypatch.setattr(
        trl_campaign.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"ActiveState={state}\n"),
    )
    monkeypatch.setattr(trl_campaign.time, "time", lambda: 0)
    with pytest.raises(RuntimeError):
        trl_campaign.wait_for_previous("private.service", tmp_path, deadline=60)


def test_deadline_prevents_any_next_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(trl_campaign.time, "time", lambda: 100)
    with pytest.raises(TimeoutError, match="no native training"):
        trl_campaign.wait_for_previous("private.service", tmp_path, deadline=60)


def test_report_retains_pending_runs_and_checks_code_protocol(tmp_path, reference):
    existing = tmp_path / "confirmation"
    baseline = existing / "main/seed-0/dp"
    baseline.mkdir(parents=True)
    save(baseline / "summary.json", reference)
    code = {
        "smoke": False,
        "task_count": 164,
        "protocol_sha256": "fixed",
        "plus_pass1": 0.5,
    }
    save(baseline / "code_metrics.json", {"humaneval": code})
    root = tmp_path / "native"
    root.mkdir()
    report = trl_campaign.write_comparison(root, existing, tmp_path / "followup")
    pending = [row for row in report["rows"] if row["variant"] in trl_campaign.ARMS]
    assert all(row["status"] == "pending" for row in pending)
    assert all("humaneval_plus_pass1" not in row for row in pending)
    assert "not an isolation of privacy noise" in report["comparison_limits"]
    scored = root / "results/trl_reference"
    scored.mkdir(parents=True)
    save(scored / "summary.json", reference)
    save(scored / "code_metrics.json", {"humaneval": dict(code, smoke=True)})
    with pytest.raises(ValueError, match="executable-code protocol"):
        trl_campaign.write_comparison(root, existing, tmp_path / "followup")
