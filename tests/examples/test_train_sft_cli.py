"""Contract tests for the reusable SFT command-line entry point."""

import sys
from pathlib import Path
from types import ModuleType

import pytest
import train_sft_trainer
from opaque_examples.sft import resolve_sft_config

_MELLUM2_MAGICODER_SMOKE: dict[str, object] = {
    "model_name_or_path": "JetBrains/Mellum2-12B-A2.5B-Base",
    "dataset_name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "prompt_field": "problem",
    "completion_field": "solution",
    "completion_only_loss": True,
    "loss_type": "chunked_nll",
    "max_length": 2048,
    "max_train_samples": 512,
    "max_eval_samples": 64,
    "num_train_epochs": 1.0,
    "max_steps": 2,
    "logical_batch_size": 128,
    "physical_microbatch_size": 2,
    "auto_microbatch_backoff": True,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "use_performance_kernels": False,
    "gradient_checkpointing": True,
    "save_steps": 1,
    "eval_steps": 1,
}

_MELLUM2_MAGICODER_SMOKE_ARGV = [
    "--model",
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
    "512",
    "--max-eval-samples",
    "64",
    "--epochs",
    "1",
    "--max-steps",
    "2",
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
    "1",
    "--eval-steps",
    "1",
]


def _install_runner(monkeypatch: pytest.MonkeyPatch, calls: list[object]) -> None:
    runner = ModuleType("opaque_examples.sft.runner")

    def run_sft(config, output_dir):
        calls.append((config, output_dir))

    runner.run_sft = run_sft
    monkeypatch.setitem(sys.modules, "opaque_examples.sft.runner", runner)


def test_cli_delegates_resolved_config_and_explicit_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []
    _install_runner(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_sft_trainer.py", *_MELLUM2_MAGICODER_SMOKE_ARGV],
    )

    assert train_sft_trainer.main(tmp_path) == 0
    assert calls == [(resolve_sft_config(_MELLUM2_MAGICODER_SMOKE), tmp_path)]


def test_cli_uses_configured_output_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    _install_runner(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft_trainer.py",
            *_MELLUM2_MAGICODER_SMOKE_ARGV,
            "--output-dir",
            "custom-output",
        ],
    )

    assert train_sft_trainer.main() == 0
    assert calls[0][1] == Path("custom-output")


def test_cli_preserves_argparse_failure_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_sft_trainer.py", "--unknown-option"])

    with pytest.raises(SystemExit) as error:
        train_sft_trainer.main()

    assert error.value.code == 2
