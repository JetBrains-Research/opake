"""Behavioral checks for the synthetic Mellum experiment, using the real trainer."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from examples.moe_privacy.run import (
    ExperimentConfig,
    GrammarDataset,
    epsilon_for_run,
    routing_metrics,
    run_experiment,
    training_arguments,
)


@pytest.fixture
def config():
    return ExperimentConfig(
        name="test",
        train_sequences=32,
        validation_sequences=4,
        sequence_length=8,
        expected_batch_size=4,
        microbatch_size=1,
        steps=2,
        eval_every=1,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": 1.5},
        {"expected_batch_size": 33},
        {"microbatch_size": 5},
        {"delta": 0},
        {"delta": float("nan")},
        {"target_epsilon": float("inf")},
        {"load_noise_ratio": 0},
        {"filter_beta": 1},
        {"num_experts": 2},
        {"sequence_length": 3},
        {"vocab_size": 100},
        {"validation_seed": 20260917},
    ],
)
def test_config_rejects_invalid_experiment(config, changes):
    with pytest.raises(ValueError, match=r"must|require"):
        replace(config, **changes)


def test_public_data_are_reproducible_and_bounded(config):
    train = GrammarDataset(config, 32, config.data_seed)
    again = GrammarDataset(config, 32, config.data_seed)
    validation = GrammarDataset(config, 32, config.validation_seed)
    assert train.fingerprint() == again.fingerprint()
    assert train.fingerprint() != validation.fingerprint()
    assert train.ids.shape == (32, config.sequence_length)
    assert train.ids.min() > 0
    assert train.ids.max() < config.vocab_size
    row = train[0]
    torch.testing.assert_close(row["input_ids"], row["labels"])
    assert row["attention_mask"].sum() == config.sequence_length


@pytest.mark.parametrize("arm", ["reference", "dp", "dp_aux"])
def test_arm_wiring_and_secret_randomness(config, arm, tmp_path, monkeypatch):
    seeds = iter([100, 200])
    monkeypatch.setattr(
        "examples.moe_privacy.run.secrets.randbits", lambda _: next(seeds)
    )
    args = training_arguments(config, arm, tmp_path)
    assert args.seed == 100
    assert args.data_seed == 200
    assert args.sampling_mode == "poisson"
    assert args.optim == "sgd"
    assert args.clipping_norm == config.clipping_norm
    assert args.microbatch_size == 1
    assert args.router_aux_loss_coef == (
        config.router_aux_loss_coef if arm == "dp_aux" else 0
    )
    assert args.privacy_accounting == (arm != "reference")
    assert args.privacy_noise_multiplier == (0 if arm == "reference" else None)
    if arm == "dp_aux":
        assert args.router_aux_kwargs["max_tokens"] == config.sequence_length
        assert args.router_aux_kwargs["mean_tokens"] == config.sequence_length
        assert args.router_aux_kwargs["ratio"] == config.load_noise_ratio


def test_joint_accounting_prices_load_release(config):
    plain = epsilon_for_run(config, "dp", 1.0, config.steps)
    matched = epsilon_for_run(
        config, "dp_aux", math.sqrt(1 + config.load_noise_ratio), config.steps
    )
    under_noised = epsilon_for_run(config, "dp_aux", 1.0, config.steps)
    assert matched == pytest.approx(plain, rel=1e-9)
    assert under_noised > plain
    assert epsilon_for_run(config, "reference", 0, config.steps) is None


def test_public_routing_metrics_distinguish_balance_from_collapse():
    balanced = routing_metrics(np.ones((3, 2, 8)))
    assert balanced["load_cv"] == pytest.approx(0)
    assert balanced["routing_entropy"] == pytest.approx(1)
    assert balanced["max_expert_share"] == pytest.approx(1 / 8)
    counts = np.zeros((3, 2, 8))
    counts[:, :, :2] = 1
    collapsed = routing_metrics(counts)
    assert collapsed["load_cv"] > 0
    assert collapsed["routing_entropy"] == pytest.approx(1 / 3)
    assert collapsed["max_expert_share"] == pytest.approx(0.5)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("arm", "device"),
    [
        ("reference", "cpu"),
        ("dp", "cpu"),
        ("dp_aux", "cpu"),
        pytest.param("dp_aux", "cuda", marks=pytest.mark.cuda),
    ],
)
def test_real_mellum_smoke_writes_safe_artifacts(config, arm, device, tmp_path, capsys):
    output = tmp_path / arm
    result = run_experiment(
        config, arm, 0, output, checks=arm == "dp_aux", device=device
    )
    assert result["status"] == "completed"
    execution = result["execution"]
    assert torch.device(execution["device"]).type == device
    assert execution["device_name"]
    assert execution["cuda_version"] == torch.version.cuda
    if device == "cuda":
        assert execution["peak_cuda_memory_allocated_bytes"] > 0
    else:
        assert execution["peak_cuda_memory_allocated_bytes"] is None
    assert "device" not in result["config"]
    assert result["privacy"]["steps"] == 2
    assert result["parameters"] > 0
    assert math.isfinite(result["final"]["eval_loss"])
    assert result["final"]["step"] == 2
    assert result["final"]["eval_max_expert_share"] > 0
    assert result["privacy"]["load_release"] == (arm == "dp_aux")
    if arm != "reference":
        assert 0 < result["privacy"]["epsilon"] <= config.target_epsilon
    assert (output / "model" / "model.safetensors").is_file()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["privacy"] == result["privacy"]
    assert saved["execution"] == execution
    if arm == "dp_aux":
        checks = json.loads((output / "checks.json").read_text())
        assert checks["passed"]
        assert checks["execution"]["device"] == execution["device"]
        assert checks["execution"]["device_name"] == execution["device_name"]
        assert checks["execution"]["cuda_version"] == execution["cuda_version"]
        if device == "cuda":
            assert checks["execution"]["peak_cuda_memory_allocated_bytes"] > 0
    assert not (output / "training_args.bin").exists()
    assert not (output / "trainer_state.json").exists()
    for row in (output / "metrics.jsonl").read_text().splitlines():
        assert "loss" not in json.loads(row)
        assert "aux_loss" not in row
    assert "train_loss" not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        run_experiment(config, arm, 0, output)
