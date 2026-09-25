"""Public-data boundaries and privacy wiring for pretrained MoE SFT."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from examples.moe_privacy import sft_run
from examples.moe_privacy.run import epsilon_for_run
from examples.moe_privacy.sft_data import (
    CompletionCollator,
    fingerprint,
    prepare_split,
    prompt_key,
)
from examples.moe_privacy.sft_run import SFTExperimentConfig, training_arguments
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal

CONFIGS = Path(sft_run.__file__).parent / "configs"
PRIVATE_COEFFICIENT_CONFIGS = (
    ("sft_balancing_aux0001_ratio1.json", 0.001, 1.0),
    ("sft_balancing_aux0001_ratio01.json", 0.001, 0.1),
    ("sft_balancing_aux02_ratio01.json", 0.2, 0.1),
)


@pytest.fixture
def tokenizer():
    words = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "###",
        "Instruction",
        ":",
        "Response",
        "write",
        "code",
        "one",
        "two",
        "three",
        "four",
        "return",
        "1",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


@pytest.fixture
def records():
    return Dataset.from_list(
        [
            {"prompt": f"write code {word}", "completion": "return 1"}
            for word in ("one", "two", "three", "four")
        ]
    )


def test_selection_is_reproducible_and_excludes_held_out_prompts(tokenizer, records):
    excluded = {prompt_key("write   code one")}
    first, info = prepare_split(
        records, tokenizer, count=3, max_length=32, seed=1, excluded=excluded
    )
    second, again = prepare_split(
        records, tokenizer, count=3, max_length=32, seed=1, excluded=excluded
    )
    assert info == again
    assert fingerprint(first) == fingerprint(second) == info["sha256"]
    assert set(info["prompt_hashes"]).isdisjoint(excluded)
    assert len(set(info["prompt_hashes"])) == 3


def test_collator_keeps_prompts_in_router_mask_not_task_labels(tokenizer, records):
    data, _ = prepare_split(records, tokenizer, count=2, max_length=32, seed=1)
    collator = CompletionCollator(32, tokenizer.pad_token_id)
    batch = collator(list(data))
    assert batch["input_ids"].shape == (2, 32)
    assert (batch["attention_mask"].sum(1) > batch["labels"].ne(-100).sum(1)).all()
    for index, row in enumerate(data):
        length = len(row["input_ids"])
        torch.testing.assert_close(
            batch["attention_mask"][index, :length],
            torch.ones(length, dtype=torch.long),
        )
        assert batch["labels"][index, length:].eq(-100).all()
        assert batch["attention_mask"][index, length:].eq(0).all()
        expected = torch.tensor(row["completion_mask"]).bool()
        torch.testing.assert_close(batch["labels"][index, :length].ne(-100), expected)
    assert collator([])["labels"].shape == (0, 32)


def test_truncation_never_crosses_public_token_bound(tokenizer, records):
    data, info = prepare_split(records, tokenizer, count=2, max_length=11, seed=2)
    assert info["truncated_answers"] == 2
    assert all(len(row["input_ids"]) == 11 for row in data)
    assert all(any(row["completion_mask"][1:]) for row in data)
    with pytest.raises(ValueError, match="usable public records"):
        prepare_split(records, tokenizer, count=1, max_length=4, seed=2)


@pytest.mark.parametrize(
    "row",
    [
        {"input_ids": [1, 2], "completion_mask": [0, 0]},
        {"input_ids": [1, 2], "completion_mask": [0, 2]},
        {"input_ids": [1, 2], "completion_mask": [1]},
        {"input_ids": [1] * 5, "completion_mask": [0, 1, 1, 1, 1]},
    ],
)
def test_collator_rejects_invalid_records(row):
    with pytest.raises(ValueError, match=r"record|supervised"):
        CompletionCollator(4, 0)([row])


@pytest.mark.parametrize(
    "changes",
    [
        {"model_revision": "main"},
        {"dataset_revision": "bad"},
        {"steps": 0},
        {"clipping_norm": float("nan")},
        {"delta": 0},
        {"filter_beta": 1},
        {"router_aux_loss_coef": 0},
        {"load_noise_ratio": 0},
        {"load_noise_ratio": float("nan")},
        {"expected_batch_size": 4097},
        {"microbatch_size": 33},
    ],
)
def test_configuration_rejects_invalid_parameters(changes):
    with pytest.raises(ValueError, match=r"must|require"):
        replace(SFTExperimentConfig(), **changes)


@pytest.mark.parametrize(
    ("filename", "coefficient", "ratio"), PRIVATE_COEFFICIENT_CONFIGS
)
def test_private_coefficient_configs_change_only_declared_settings(
    filename, coefficient, ratio
):
    reference = json.loads((CONFIGS / "sft_balancing_aux02.json").read_text())
    actual = json.loads((CONFIGS / filename).read_text())
    assert actual == {
        **reference,
        "name": f"mellum2-magicoder-dp-aux-coef{coefficient:g}-ratio{ratio:g}",
        "router_aux_loss_coef": coefficient,
        "load_noise_ratio": ratio,
    }
    config = SFTExperimentConfig(**actual)
    assert config.steps == 256
    assert config.sequence_length == 512
    assert config.filter_beta == 0.95
    assert (config.lora_rank, config.lora_alpha) == (4, 8)
    assert config.clipping_norm == 0.1
    assert config.train_sequences == 8192
    assert config.expected_batch_size == 32
    assert config.target_epsilon == 8.0
    assert config.delta == 1e-5


@pytest.mark.parametrize(
    "filename", [None, *(case[0] for case in PRIVATE_COEFFICIENT_CONFIGS)]
)
@pytest.mark.parametrize("arm", ["reference", "reference_aux", "dp", "dp_aux"])
def test_completion_sft_uses_joint_privacy_wiring(filename, arm, tmp_path):
    config = (
        SFTExperimentConfig(**json.loads((CONFIGS / filename).read_text()))
        if filename
        else SFTExperimentConfig()
    )
    args = training_arguments(config, arm, tmp_path, device="cpu")
    private = arm in ("dp", "dp_aux")
    balanced = arm in ("reference_aux", "dp_aux")
    assert not args.remove_unused_columns
    assert args.microbatch_size == 1
    assert args.per_device_train_batch_size == config.expected_batch_size
    assert args.max_steps == config.steps
    assert args.optim == "sgd"
    assert args.learning_rate == config.learning_rate
    assert args.lr_scheduler == "constant"
    assert args.weight_decay == 0
    assert args.clipping_mode == "fixed"
    assert args.clipping_norm == config.clipping_norm
    assert args.sampling_mode == "poisson"
    assert args.privacy_noise_mechanism == "gaussian"
    assert args.privacy_accounting == private
    assert args.privacy_target_epsilon == (8.0 if private else None)
    assert args.privacy_target_delta == config.delta
    assert args.privacy_noise_multiplier == (None if private else 0.0)
    assert args.router_aux_loss_coef == (config.router_aux_loss_coef if balanced else 0)
    assert args.performance_kernels_config["router_fp32"]
    assert not args.use_performance_kernels
    if balanced:
        assert args.router_aux_kwargs == {
            "max_tokens": config.sequence_length,
            "mean_tokens": config.sequence_length,
            "ratio": config.load_noise_ratio,
            "filter_beta": config.filter_beta,
        }
    else:
        assert args.router_aux_kwargs == {}


def _calibrate_sft_noise(config, args):
    def process(noise):
        mechanism = dpsgd_acc.gaussian(noise)
        if args.router_aux_loss_coef:
            mechanism = dpsgd_acc.moe_aux(
                mechanism, ratio=args.router_aux_kwargs["ratio"]
            )
        return (
            dpsgd_acc.poisson(
                mechanism, args.per_device_train_batch_size / config.train_sequences
            )
            * args.max_steps
        )

    settings = args.noise_calibration_kwargs
    return cal.calibrate(
        cal.epsilon_budget(
            args.privacy_target_epsilon, delta=args.privacy_target_delta
        ),
        process,
        settings["min"],
        settings["max"],
        tolerance=settings["tolerance"],
    )


@pytest.mark.slow
@pytest.mark.parametrize("filename", [case[0] for case in PRIVATE_COEFFICIENT_CONFIGS])
def test_private_coefficient_configs_calibrate_joint_budget(filename, tmp_path):
    config = SFTExperimentConfig(**json.loads((CONFIGS / filename).read_text()))
    results = {}
    for arm in ("dp", "dp_aux"):
        args = training_arguments(config, arm, tmp_path / arm, device="cpu")
        result = _calibrate_sft_noise(config, args)
        assert result.converged
        epsilon = epsilon_for_run(config, arm, result.param, config.steps)
        assert 0 < epsilon <= config.target_epsilon
        assert epsilon == pytest.approx(result.achieved, abs=1e-9)
        results[arm] = result

    plain, joint = results["dp"], results["dp_aux"]
    assert joint.param == pytest.approx(
        plain.param * math.sqrt(1 + config.load_noise_ratio), rel=2e-3
    )
    assert (
        epsilon_for_run(config, "dp_aux", plain.param, config.steps)
        > config.target_epsilon
    )
    assert epsilon_for_run(
        replace(config, router_aux_loss_coef=2.0), "dp_aux", joint.param, config.steps
    ) == pytest.approx(joint.achieved, abs=1e-9)
