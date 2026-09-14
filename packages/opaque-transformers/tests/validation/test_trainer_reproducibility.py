"""Reproducibility & contract foundations (Stage 2 of the de-SFT plan).

Verifies the three foundations Trainer now wires:

- ``set_seed(args.seed)`` runs at construction so non-DP randomness
  (model init, ``compute_metrics`` ``torch.randn`` calls, …) is
  reproducible run-to-run.
- ``self._device = self.args.device`` (forwards to
  :meth:`TrainingArguments._setup_devices`) so ``use_cpu`` /
  ``use_mps_device`` / ``no_cuda`` actually take effect.
- ``TrainingArguments.__post_init__`` is idempotent so a single args
  instance can be passed to multiple ``Trainer`` constructions
  (sweep parity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model

from opaque.transformers.trainer import Trainer, TrainingArguments

# ``_hf_shared`` is in the parent of ``validation/``; mirror the import
# convention used by the sibling tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_model_and_tokenizer():
    tokenizer = gpt2_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    model = make_gpt2_model()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _build_lora_model(base_model, seed: int = 42):
    """LoRA-wrap a base GPT-2 with a fixed adapter shape.

    PEFT initialises LoRA-A / LoRA-B from the global ``torch`` RNG at
    the time of the ``get_peft_model`` call.  ``Trainer.__init__``
    only seeds *after* the user has built the model, so for
    bit-identical-run assertions the test must seed PEFT
    construction itself.  ``set_seed`` here mirrors what HF's
    ``Trainer.__init__`` does inside the trainer; we just hoist it to
    the model-construction site.
    """
    from transformers import set_seed

    set_seed(seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(base_model, lora_config)


def _args(tmp_path, **overrides) -> TrainingArguments:
    """Reproducibility-test args (CPU-pinned, deterministic).

    ``privacy_noise_multiplier=0.0`` is the canonical knob for asserting
    bit-identical runs — DP-SGD reduces to vanilla per-example SGD
    when σ=0, so two runs with the same ``args.seed`` should produce
    bit-identical parameters.  Enabling DP noise (σ>0) would require
    tolerance-based comparison.
    """
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "max_steps": 3,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        # NM=0.0 = non-private; target_eps is meaningless on this path.
        "privacy_noise_multiplier": 0.0,
        "clipping_norm": 1.0,
        "learning_rate": 1e-3,
        "seed": 42,
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _train_two_steps(model, tokenizer, dataset, args) -> dict[str, torch.Tensor]:
    """Run a tiny training pass and return a snapshot of trainable params."""
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()
    return {
        n: p.data.clone().cpu() for n, p in model.named_parameters() if p.requires_grad
    }


# ---------------------------------------------------------------------------
# Bit-identical runs at σ=0 — set_seed + worker_init_fn deliver
# ---------------------------------------------------------------------------


class TestBitIdenticalRuns:
    """Two runs with the same seed produce identical parameter tensors at σ=0.

    The DP RNG chain is independent of ``args.seed`` only when noise
    is on (σ>0); at σ=0 the entire DP-SGD step is deterministic given
    the data and the optimizer, and ``args.seed`` controls the
    sampler key + any non-DP randomness in the data pipeline.
    """

    def test_bit_identical_with_zero_noise(
        self,
        gpt2_model_and_tokenizer,
        tmp_path,
    ):
        base_model_a, tokenizer = gpt2_model_and_tokenizer
        lora_a = _build_lora_model(base_model_a)
        ds = build_lm_dataset(
            ["foo bar baz", "qux quux corge", "alpha beta gamma", "delta epsilon"],
            tokenizer,
            max_length=12,
        )

        run_a = _train_two_steps(
            lora_a,
            tokenizer,
            ds,
            _args(tmp_path / "run_a"),
        )

        # Fresh model / fresh trainer with the same seed.
        base_model_b = make_gpt2_model()
        base_model_b.config.pad_token_id = tokenizer.pad_token_id
        lora_b = _build_lora_model(base_model_b)
        run_b = _train_two_steps(
            lora_b,
            tokenizer,
            ds,
            _args(tmp_path / "run_b"),
        )

        assert run_a.keys() == run_b.keys()
        for name in run_a:
            assert torch.equal(run_a[name], run_b[name]), (
                f"param {name!r} diverged between two seeded runs"
            )

    def test_different_seeds_diverge(
        self,
        gpt2_model_and_tokenizer,
        tmp_path,
    ):
        """Same data + different ``args.seed`` ⇒ different params (at σ=0).

        Catches the case where ``set_seed`` is unwired and the seed is
        ignored, which would make this test silently pass with
        identical params.
        """
        base_a, tokenizer = gpt2_model_and_tokenizer
        lora_a = _build_lora_model(base_a)
        ds = build_lm_dataset(
            ["foo", "bar", "baz", "qux"],
            tokenizer,
            max_length=8,
        )

        run_a = _train_two_steps(
            lora_a,
            tokenizer,
            ds,
            _args(tmp_path / "a", seed=42),
        )

        base_b = make_gpt2_model()
        base_b.config.pad_token_id = tokenizer.pad_token_id
        lora_b = _build_lora_model(base_b)
        run_b = _train_two_steps(
            lora_b,
            tokenizer,
            ds,
            _args(tmp_path / "b", seed=123),
        )

        diverged = any(not torch.equal(run_a[n], run_b[n]) for n in run_a)
        assert diverged, (
            "different seeds must yield different params at σ=0; "
            "got bit-identical tensors which suggests args.seed is "
            "being ignored or set_seed isn't wired"
        )

    def test_data_seed_changes_sampling_with_same_model_seed(
        self,
        gpt2_model_and_tokenizer,
        tmp_path,
    ):
        """Same ``seed`` but different ``data_seed`` should diverge at σ=0.

        This isolates data-path randomness from model/global randomness:
        LoRA init is kept fixed and only the sampler seed differs.
        """
        base_a, tokenizer = gpt2_model_and_tokenizer
        lora_a = _build_lora_model(base_a, seed=7)
        ds = build_lm_dataset(
            [f"sample {i}" for i in range(64)],
            tokenizer,
            max_length=10,
        )

        run_a = _train_two_steps(
            lora_a,
            tokenizer,
            ds,
            _args(
                tmp_path / "data_seed_a",
                seed=42,
                data_seed=11,
                max_steps=6,
                per_device_train_batch_size=16,
            ),
        )

        base_b = make_gpt2_model()
        base_b.config.pad_token_id = tokenizer.pad_token_id
        lora_b = _build_lora_model(base_b, seed=7)
        run_b = _train_two_steps(
            lora_b,
            tokenizer,
            ds,
            _args(
                tmp_path / "data_seed_b",
                seed=42,
                data_seed=73,
                max_steps=6,
                per_device_train_batch_size=16,
            ),
        )

        diverged = any(not torch.equal(run_a[n], run_b[n]) for n in run_a)
        assert diverged, (
            "changing data_seed with fixed seed should change sampling trajectory; "
            "got identical params"
        )


# ---------------------------------------------------------------------------
# args.device honored — use_cpu pins to CPU
# ---------------------------------------------------------------------------


class TestDeviceFlagEffect:
    """``args.use_cpu`` / ``args.device`` actually relocates the model."""

    def test_use_cpu_pins_to_cpu(self, gpt2_model_and_tokenizer, tmp_path):
        """``use_cpu=True`` resolves the trainer device to ``cpu``."""
        model, tokenizer = gpt2_model_and_tokenizer
        lora = _build_lora_model(model)
        ds = build_lm_dataset(["a b", "c d"], tokenizer, max_length=8)

        trainer = Trainer(
            model=lora,
            args=_args(tmp_path, use_cpu=True),
            processing_class=tokenizer,
            train_dataset=ds,
            eval_dataset=ds,
        )

        assert trainer._device.type == "cpu"
        # Model should also be moved (HF parity).
        for p in lora.parameters():
            assert p.device.type == "cpu"

    def test_args_device_is_source_of_truth(
        self,
        gpt2_model_and_tokenizer,
        tmp_path,
    ):
        """``trainer._device`` follows ``args.device``, not the model's
        original placement.

        Pre-Stage-2 the trainer resolved ``self._device`` via
        ``next(model.parameters()).device``, which silently ignored
        ``args.use_cpu`` / ``args.use_mps_device`` / ``args.no_cuda``.
        """
        model, tokenizer = gpt2_model_and_tokenizer
        lora = _build_lora_model(model)
        ds = build_lm_dataset(["x", "y"], tokenizer, max_length=8)

        args = _args(tmp_path, use_cpu=True)
        trainer = Trainer(
            model=lora,
            args=args,
            processing_class=tokenizer,
            train_dataset=ds,
            eval_dataset=ds,
        )

        # ``args.device`` and ``trainer._device`` agree.
        assert trainer._device == args.device


# ---------------------------------------------------------------------------
# __post_init__ idempotency — single args instance survives reuse
# ---------------------------------------------------------------------------


class TestPostInitIdempotency:
    """``TrainingArguments.__post_init__`` is safe to re-enter."""

    def test_manual_reentry_is_a_noop(self, tmp_path):
        """Calling ``__post_init__`` a second time leaves fields stable."""
        a = TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            logging_steps=2,
            privacy_noise_multiplier=0.0,
        )
        # Snapshot post-first-init state.
        before = {
            k: getattr(a, k)
            for k in (
                "output_dir",
                "logging_dir",
                "eval_strategy",
                "logging_strategy",
                "save_strategy",
                "lr_scheduler",
                "logging_steps",
                "use_cpu",
                "include_for_metrics",
                "local_rank",
            )
        }

        # Second call: must not re-trigger mutations / FutureWarnings.
        a.__post_init__()

        after = {k: getattr(a, k) for k in before}
        assert after == before, (
            "__post_init__ re-entry mutated previously-derived fields; "
            f"diff: {[(k, before[k], after[k]) for k in before if before[k] != after[k]]}"
        )

    def test_reuse_args_across_two_trainers(
        self,
        gpt2_model_and_tokenizer,
        tmp_path,
    ):
        """The same ``TrainingArguments`` instance constructs two trainers."""
        model_a, tokenizer = gpt2_model_and_tokenizer
        lora_a = _build_lora_model(model_a)
        ds = build_lm_dataset(["a", "b"], tokenizer, max_length=8)

        args = _args(tmp_path)

        Trainer(
            model=lora_a,
            args=args,
            processing_class=tokenizer,
            train_dataset=ds,
            eval_dataset=ds,
        )

        # A fresh model with the *same* args object — would have
        # raised on FutureWarning re-emission or on ``output_dir``
        # being already-prefixed if ``__post_init__`` re-fired.
        model_b = make_gpt2_model()
        model_b.config.pad_token_id = tokenizer.pad_token_id
        lora_b = _build_lora_model(model_b)
        Trainer(
            model=lora_b,
            args=args,
            processing_class=tokenizer,
            train_dataset=ds,
            eval_dataset=ds,
        )

        # Sentinel must remain set.
        assert getattr(args, "_dp_post_init_done", False) is True


# ---------------------------------------------------------------------------
# report_to — Phase 5b: no longer raises, integration callbacks wired
# ---------------------------------------------------------------------------


class TestReportToRaises:
    """Phase 5b: ``report_to`` no longer raises; integration callbacks are wired."""

    @pytest.mark.parametrize("value", ["wandb", "tensorboard", "all", ["wandb"]])
    def test_non_default_report_to_no_longer_raises(self, value, tmp_path):
        # Phase 5b removed the ValueError; these values must construct cleanly.
        TrainingArguments(
            output_dir=str(tmp_path),
            report_to=value,
            privacy_noise_multiplier=0.0,
        )

    @pytest.mark.parametrize("value", [None, "none", [], ["none"]])
    def test_default_report_to_is_silent(self, value, tmp_path):
        # All four sentinels must construct without error.
        args = TrainingArguments(
            output_dir=str(tmp_path),
            report_to=value,
            privacy_noise_multiplier=0.0,
        )
        assert args.report_to == []
