import sys
from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))
_config = import_module("opaque_examples.sft.config")
SFTJobConfig = _config.SFTJobConfig
parse_sft_config = _config.parse_sft_config
resolve_sft_config = _config.resolve_sft_config


_MELLUM2_MAGICODER_SMOKE: dict[str, object] = {
    "model_name_or_path": "JetBrains/Mellum2-12B-A2.5B-Base",
    "model_revision": "main",
    "dataset_name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "dataset_revision": "main",
    "prompt_field": "problem",
    "completion_field": "solution",
    "completion_only_loss": True,
    "packing": False,
    "loss_type": "chunked_nll",
    "max_length": 2048,
    "max_train_samples": 512,
    "max_eval_samples": 64,
    "eval_fraction": 0.01,
    "split_seed": 42,
    "num_train_epochs": 1.0,
    "max_steps": 2,
    "logical_batch_size": 128,
    "physical_microbatch_size": 2,
    "auto_microbatch_backoff": True,
    "target_epsilon": 8.0,
    "target_delta": None,
    "noise_multiplier": None,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.0,
    "bf16": True,
    "attn_implementation": "sdpa",
    "use_performance_kernels": False,
    "gradient_checkpointing": True,
    "chunked_nll": True,
    "save_steps": 1,
    "eval_steps": 1,
    "logging_steps": 1,
}

_MELLUM2_MAGICODER_PRODUCTION: dict[str, object] = {
    "model_name_or_path": "JetBrains/Mellum2-12B-A2.5B-Base",
    "model_revision": "main",
    "dataset_name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "dataset_revision": "main",
    "dataset_config": None,
    "dataset_split": "train",
    "dataset_text_field": "text",
    "prompt_field": "problem",
    "completion_field": "solution",
    "completion_only_loss": True,
    "packing": False,
    "loss_type": "chunked_nll",
    "compute_loss_func": False,
    "assistant_only_loss": False,
    "eos_token": None,
    "log_completion_metrics": False,
    "max_length": 2048,
    "max_train_samples": 75_000,
    "max_eval_samples": 750,
    "eval_fraction": 0.01,
    "split_seed": 42,
    "num_train_epochs": 1.0,
    "max_steps": 500,
    "learning_rate": 1e-5,
    "logical_batch_size": 128,
    "physical_microbatch_size": 2,
    "auto_microbatch_backoff": True,
    "target_epsilon": 8.0,
    "target_delta": None,
    "noise_multiplier": None,
    "max_grad_norm": 1.0,
    "lora_rank": 4,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "chat_template_path": None,
    "bf16": True,
    "attn_implementation": "sdpa",
    "use_performance_kernels": False,
    "gradient_checkpointing": True,
    "activation_offloading": False,
    "chunked_nll": True,
    "save_steps": 100,
    "eval_steps": 100,
    "eval_on_start": True,
    "per_device_eval_batch_size": None,
    "logging_steps": 5,
    "noise_bias_correction": False,
    "no_wandb": False,
    "seed": 42,
    "output_dir": "trainer_output/mellum2-magicoder",
}


def test_explicit_mellum_smoke_mapping_has_the_training_contract() -> None:
    config = resolve_sft_config(_MELLUM2_MAGICODER_SMOKE)

    assert config.model_name_or_path == "JetBrains/Mellum2-12B-A2.5B-Base"
    assert config.dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K"
    assert config.prompt_field == "problem"
    assert config.completion_field == "solution"
    assert config.completion_only_loss is True
    assert config.max_length == 2048
    assert config.target_epsilon == 8.0
    assert config.target_delta is None
    assert config.noise_multiplier is None
    assert config.bf16 is True
    assert config.attn_implementation == "sdpa"
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.gradient_checkpointing is True
    assert config.chunked_nll is True
    assert config.logical_batch_size == 128
    assert config.physical_microbatch_size < config.logical_batch_size
    assert config.auto_microbatch_backoff is True
    assert config.num_train_epochs == 1.0
    assert config.max_steps == 2


def test_parser_uses_only_the_provided_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sft", "--preset", "smoke"])

    config = parse_sft_config(["--model-name", "model", "--dataset", "owner/dataset"])

    assert config.model == "model"
    assert config.dataset == "owner/dataset"


def test_production_full_mapping_and_cli_resolution_are_equivalent() -> None:
    expected = resolve_sft_config(_MELLUM2_MAGICODER_PRODUCTION)

    actual = parse_sft_config(
        [
            "--model-name",
            "JetBrains/Mellum2-12B-A2.5B-Base",
            "--dataset",
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "--prompt-field",
            "problem",
            "--completion-field",
            "solution",
            "--completion-only-loss",
            "--loss-type",
            "chunked_nll",
            "--max-length",
            "2048",
            "--max-train-samples",
            "75000",
            "--max-eval-samples",
            "750",
            "--epochs",
            "1",
            "--max-steps",
            "500",
            "--batch-size",
            "128",
            "--microbatch-size",
            "2",
            "--auto-microbatch-backoff",
            "--lora-target-modules",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "--no-performance-kernels",
            "--gradient-checkpointing",
            "--save-steps",
            "100",
            "--eval-steps",
            "100",
            "--logging-steps",
            "5",
            "--output-dir",
            "trainer_output/mellum2-magicoder",
        ]
    )

    assert actual == expected


def test_output_directory_is_explicitly_configurable() -> None:
    config = parse_sft_config(
        [
            "--model",
            "model",
            "--dataset",
            "owner/dataset",
            "--output-dir",
            "/tmp/sft-output",
        ]
    )

    assert config.output_dir == "/tmp/sft-output"


def test_config_is_frozen() -> None:
    config = resolve_sft_config(_MELLUM2_MAGICODER_SMOKE)

    with pytest.raises(FrozenInstanceError):
        config.max_steps = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_fraction": 0.0}, "eval_fraction"),
        ({"eval_fraction": 1.0}, "eval_fraction"),
        ({"logical_batch_size": 0}, "logical_batch_size"),
        ({"save_steps": 0}, "save_steps"),
        ({"target_epsilon": 0.0}, "target_epsilon"),
        ({"target_delta": 1.0}, "target_delta"),
        ({"lora_rank": 0}, "lora_rank"),
        ({"completion_only_loss": False}, "completion_only_loss"),
        (
            {"noise_multiplier": 1.0},
            "target_epsilon",
        ),
    ],
)
def test_programmatic_validation_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_sft_config({**_MELLUM2_MAGICODER_SMOKE, **overrides})


def test_fixed_noise_is_valid_when_target_epsilon_is_cleared() -> None:
    config = resolve_sft_config(
        {
            **_MELLUM2_MAGICODER_SMOKE,
            "noise_multiplier": 1.0,
            "target_epsilon": None,
        }
    )

    assert config.noise_multiplier == 1.0
    assert config.target_epsilon is None


def test_unknown_fields_and_duplicate_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown SFT configuration field"):
        resolve_sft_config({**_MELLUM2_MAGICODER_SMOKE, "typo_field": True})

    with pytest.raises(ValueError, match=r"model_name_or_path.*supplied twice"):
        resolve_sft_config(
            {
                **_MELLUM2_MAGICODER_SMOKE,
                "model": "duplicate/model",
            }
        )


def test_programmatic_aliases_are_normalized() -> None:
    config = resolve_sft_config(
        {
            "model": "model",
            "dataset": "owner/dataset",
            "lora_target_modules": ["q_proj", "v_proj"],
        }
    )

    assert config.model_name_or_path == "model"
    assert config.dataset_name == "owner/dataset"
    assert config.lora_target_modules == ("q_proj", "v_proj")


def test_existing_cli_aliases_are_preserved() -> None:
    config = parse_sft_config(
        [
            "--model-name",
            "model",
            "--dataset",
            "owner/dataset",
            "--dataset-config",
            "subset",
            "--dataset-split",
            "validation",
            "--dataset-text-field",
            "content",
            "--num-train-samples",
            "12",
            "--num-eval-samples",
            "4",
            "--stop-at-step",
            "3",
            "--microbatch-size",
            "2",
            "--clipping-norm",
            "0.5",
            "--log-steps",
            "2",
            "--lora-modules",
            "q_proj",
            "v_proj",
            "--activation-offloading",
            "--auto-find-microbatch-size",
            "--no-performance-kernels",
            "--no-wandb",
        ]
    )

    assert config.dataset_config == "subset"
    assert config.dataset_split == "validation"
    assert config.dataset_text_field == "content"
    assert config.max_train_samples == 12
    assert config.max_eval_samples == 4
    assert config.max_steps == 3
    assert config.microbatch_size == 2
    assert config.clipping_norm == 0.5
    assert config.log_steps == 2
    assert config.lora_modules == ("q_proj", "v_proj")
    assert config.activation_offloading is True
    assert config.auto_find_microbatch_size is True
    assert config.use_performance_kernels is False
    assert config.no_wandb is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--preset", "smoke"],
        ["--model", "model", "--dataset", "owner/dataset", "--eval-fraction", "1"],
        ["--model", "model", "--dataset", "owner/dataset", "--unknown-option"],
    ],
)
def test_invalid_cli_input_uses_argparse_exit_behavior(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_sft_config(argv)

    assert exc_info.value.code == 2


def test_model_and_dataset_are_required() -> None:
    with pytest.raises(ValueError, match="model_name_or_path"):
        resolve_sft_config({})

    with pytest.raises(SystemExit) as exc_info:
        parse_sft_config([])

    assert exc_info.value.code == 2
