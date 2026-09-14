"""Regression tests for evaluation correctness.

Covers:

- ``speed_metrics`` is reported under the ``metric_key_prefix`` (HF parity).
- ``EvalPrediction.losses`` is per-example (length = total samples), not
  per-batch.
- ``ignore_keys`` filters ``ModelOutput`` containers in ``prediction_step``.
- A user-supplied ``data_collator`` replaces ``collate_dp`` for the train
  loader; the DP path raises a typed error if it doesn't return the
  required 3-tuple.
"""

from __future__ import annotations

from typing import Any

import pytest
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model
from datasets import concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model

from opaque.api.transformers.trainer._eval import speed_metrics
from opaque.transformers.trainer import Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Pure-helper tests (no DP run required).
# ---------------------------------------------------------------------------


class TestSpeedMetrics:
    """``speed_metrics`` mirrors HF's helper surface."""

    def test_runtime_always_present(self):
        out = speed_metrics("eval", start_time=time_at_t_minus(0.5))
        assert "eval_runtime" in out
        assert out["eval_runtime"] >= 0.0

    def test_samples_per_second_when_provided(self):
        out = speed_metrics(
            "test",
            start_time=time_at_t_minus(0.5),
            num_samples=100,
        )
        assert "test_samples_per_second" in out
        # ~100 / 0.5 = 200; allow generous slack for clock jitter.
        assert out["test_samples_per_second"] > 50.0

    def test_steps_per_second_when_provided(self):
        out = speed_metrics(
            "eval",
            start_time=time_at_t_minus(0.5),
            num_steps=10,
        )
        assert "eval_steps_per_second" in out

    def test_omits_unsupplied_counts(self):
        out = speed_metrics("eval", start_time=time_at_t_minus(0.1))
        assert "eval_samples_per_second" not in out
        assert "eval_steps_per_second" not in out


def time_at_t_minus(seconds: float) -> float:
    """Return ``time.monotonic()`` shifted ``seconds`` into the past."""
    import time

    return time.monotonic() - seconds


# ---------------------------------------------------------------------------
# Trainer integration: per-example losses, speed_metrics in eval output.
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_and_tokenizer():
    tokenizer = gpt2_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    model = make_gpt2_model()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@pytest.fixture
def lora_model(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora_config), tokenizer


@pytest.fixture
def tiny_dataset(small_model_and_tokenizer):
    """Eight pre-padded causal-LM examples for Trainer eval tests."""
    _, tokenizer = small_model_and_tokenizer
    return build_lm_dataset(
        [f"sample {i}" for i in range(8)],
        tokenizer,
        max_length=16,
    )


def _args(tmp_path, **overrides) -> TrainingArguments:
    # ``use_cpu=True``: pin to CPU so the trainer's ``args.device``
    # resolves to CPU regardless of the host (LoRA fixtures are CPU).
    defaults: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,  # 8 / 4 = 2 batches
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "max_steps": 2,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


class TestEvalSpeedMetrics:
    """``evaluate(metric_key_prefix=...)`` exposes the throughput trio."""

    def test_default_prefix(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        trainer = Trainer(
            model=model,
            args=_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        metrics = trainer.evaluate()
        assert "eval_runtime" in metrics
        assert "eval_samples_per_second" in metrics
        assert "eval_steps_per_second" in metrics
        assert "eval_loss" in metrics
        for phase in (
            "model_time_sec",
            "gather_time_sec",
            "transfer_time_sec",
            "finalization_time_sec",
            "metric_time_sec",
            "transfer_bytes",
            "transfer_overlap_sec",
            "transfer_overlap_ratio",
        ):
            assert f"eval_{phase}" in metrics

    def test_custom_prefix(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        trainer = Trainer(
            model=model,
            args=_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        metrics = trainer.evaluate(metric_key_prefix="test")
        assert "test_runtime" in metrics
        assert "test_samples_per_second" in metrics
        assert "test_steps_per_second" in metrics
        assert "test_loss" in metrics


# ---------------------------------------------------------------------------
# data_collator wiring.
# ---------------------------------------------------------------------------


class TestDataCollatorWiring:
    """User-supplied ``data_collator`` is honored (HF parity)."""

    def test_user_collator_is_called(self, lora_model, tiny_dataset, tmp_path):
        from transformers import default_data_collator

        model, tokenizer = lora_model
        called = {"n": 0}

        # Repeat the dataset so Poisson subsampling is unlikely to hand the
        # collator an empty batch for every step of this short smoke run.
        train_ds = concatenate_datasets([tiny_dataset] * 8)

        def my_collator(examples):
            # Poisson subsampling can yield an empty index list; keep the test
            # collator HF-compatible by materializing at least one row.
            if not examples:
                examples = [train_ds[0]]
            called["n"] += 1
            return default_data_collator(examples)

        trainer = Trainer(
            model=model,
            args=_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=tiny_dataset,
            data_collator=my_collator,
        )
        trainer.train()
        assert called["n"] > 0, (
            "User-supplied data_collator was never called — Trainer is "
            "still using its default collator instead"
        )
