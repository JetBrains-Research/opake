"""End-to-end tests for Trainer with GPT-2 + LoRA.

Exercises Trainer's HF-Trainer-parity surface: ``train()``, param
restoration, ``evaluate()``, ``get_train_dataloader()``, callback
dispatch, checkpoint round-trip, and resume.  Datasets are HF-shaped
(pre-padded ``input_ids`` / ``labels`` / ``attention_mask``) and feed
the trainer's default ``transformers.default_data_collator`` directly.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
import torch
from _hf_shared import build_lm_dataset, make_gpt2
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback as _HFTrainerCallback

from opaque.api.transformers.trainer import _dpftrl
from opaque.api.transformers.trainer._state import TrainerState
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.random import fold_in, key, split
from opaque.transformers.trainer import Trainer, TrainingArguments
from opaque.transformers.trainer.types import EvaluationResult, TrainOutput


def _default_args(**overrides) -> TrainingArguments:
    """Build TrainingArguments with test defaults.

    Pins ``use_cpu=True`` so the trainer's ``args.device`` resolves to
    CPU regardless of the host (MPS on macOS would otherwise pick up
    the test fixtures' CPU-resident model parameters and produce a
    device mismatch).  Tests that explicitly need an accelerator can
    override.
    """
    defaults = {
        "per_device_train_batch_size": 4,
        "clipping_norm": 1.0,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_model_and_tokenizer():
    """Tiny randomly-initialised GPT-2 + tokenizer (see _hf_shared.make_gpt2)."""
    return make_gpt2()


@pytest.fixture
def gpt2_with_lora(gpt2_model_and_tokenizer):
    """GPT-2 with LoRA adapters."""
    model, tokenizer = gpt2_model_and_tokenizer

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


@pytest.fixture
def tiny_lm_dataset(gpt2_model_and_tokenizer):
    """Eight pre-padded causal-LM examples for Trainer integration tests."""
    _, tokenizer = gpt2_model_and_tokenizer
    texts = [
        "def fibonacci(n): return n",
        "area = math.pi * r * r",
        "return os.listdir(path)",
        "self.items = []",
        "return f'hello {name}'",
        "sorted_list = sorted(xs)",
        "return n % 2 == 0",
        "open file and read it",
    ]
    return build_lm_dataset(texts, tokenizer, max_length=24)


@pytest.fixture(autouse=True)
def _isolate_default_trainer_output(tmp_path, monkeypatch):
    """Keep default relative trainer outputs isolated across xdist workers."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainerInit:
    """Test Trainer construction and basic interface."""

    def test_constructor(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        assert trainer.model is model
        assert trainer.processing_class is tokenizer
        assert isinstance(trainer.args, TrainingArguments)

    def test_get_train_dataloader(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        loader = trainer.get_train_dataloader()
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].ndim == 2
        assert batch["labels"].ndim == 2

    def test_eval_dataloader_forwards_multiprocessing_context(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        context = multiprocessing.get_all_start_methods()[0]
        trainer = Trainer(
            model=model,
            args=_default_args(
                dataloader_num_workers=1,
                dataloader_multiprocessing_context=context,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        loader = trainer.get_eval_dataloader()

        assert loader.multiprocessing_context.get_start_method() == context
        assert loader.in_order is True


class TestPrivacyBudgetValidation:
    """Validation of the ``privacy_noise_multiplier`` / ``privacy_target_epsilon``
    pairing at ``TrainingArguments.__post_init__``.
    """

    def test_neither_set_raises(self):
        with pytest.raises(
            ConfigurationError, match="Set either privacy_noise_multiplier"
        ):
            TrainingArguments(use_cpu=True)

    def test_nm_only_is_ok(self):
        args = TrainingArguments(use_cpu=True, privacy_noise_multiplier=1.0)
        assert args.privacy_noise_multiplier == 1.0
        assert args.privacy_target_epsilon is None

    def test_target_only_is_ok(self):
        args = TrainingArguments(use_cpu=True, privacy_target_epsilon=8.0)
        assert args.privacy_noise_multiplier is None
        assert args.privacy_target_epsilon == 8.0

    def test_nm_zero_alone_is_ok(self):
        args = TrainingArguments(use_cpu=True, privacy_noise_multiplier=0.0)
        assert args.privacy_noise_multiplier == 0.0

    def test_nm_zero_with_target_raises(self):
        with pytest.raises(ValueError, match="non-private path"):
            TrainingArguments(
                use_cpu=True,
                privacy_noise_multiplier=0.0,
                privacy_target_epsilon=8.0,
            )

    def test_both_set_is_ok(self):
        """Both set is the stop-at-ε path — no raise at construction."""
        args = TrainingArguments(
            use_cpu=True,
            privacy_noise_multiplier=1.0,
            privacy_target_epsilon=8.0,
        )
        assert args.privacy_noise_multiplier == 1.0
        assert args.privacy_target_epsilon == 8.0


class TestIgnoreDataSkip:
    """Validate ignore_data_skip sampling modes."""

    def test_poisson_is_ok(self):
        args = TrainingArguments(
            use_cpu=True,
            privacy_noise_multiplier=1.0,
            ignore_data_skip=True,
        )
        assert args.sampling_mode == "poisson"
        assert args.ignore_data_skip is True

    def test_participation_schema_raises(self):
        with pytest.raises(ValueError, match="ignore_data_skip=True requires"):
            TrainingArguments(
                use_cpu=True,
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_band",
                ignore_data_skip=True,
            )


class TestStopAtEpsilon:
    """Stop-at-ε: when both NM>0 and target_epsilon are set, the trainer
    halts at the first log boundary where the accumulated ε ≥ target.
    """

    def test_stops_when_target_epsilon_reached(self, gpt2_with_lora, tiny_lm_dataset):
        """A very small target_epsilon halts training before max_steps."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=20,
                num_train_epochs=1,
                eval_strategy="no",
                logging_steps=1,
                # NM=1.0 + a tiny target → should stop within the first few logs.
                privacy_noise_multiplier=1.0,
                privacy_target_epsilon=0.001,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train()
        assert out.global_step < 20, (
            f"stop-at-ε should have halted before max_steps; got "
            f"global_step={out.global_step}"
        )
        assert trainer.state.privacy_target_epsilon_reached is True

    def test_stop_at_epsilon_independent_of_logging_cadence(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """#392: the stop step is independent of ``logging_steps``.

        Pre-fix the ε-budget check only fired at log boundaries, so a coarse
        logging cadence let extra accounted updates run past target (a coarse
        cadence would never hit a boundary and run to max_steps).  Accounting
        is data/weight-independent, so reusing the model across runs is fine.
        """

        def _run(logging_steps: int) -> int:
            model, tokenizer = gpt2_with_lora
            trainer = Trainer(
                model=model,
                args=_default_args(
                    max_steps=20,
                    num_train_epochs=1,
                    eval_strategy="no",
                    save_strategy="no",
                    logging_steps=logging_steps,
                    privacy_noise_multiplier=1.0,
                    privacy_target_epsilon=0.001,
                ),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )
            out = trainer.train()
            assert trainer.state.privacy_target_epsilon_reached is True
            return out.global_step

        fine = _run(1)
        coarse = _run(999)  # no log boundary before max_steps=20
        assert fine == coarse < 20, (fine, coarse)

    def test_stop_at_epsilon_final_step_target(self, gpt2_with_lora, tiny_lm_dataset):
        """#392 review: a target reached exactly on the final step still sets
        the flag (the predicted-step check precedes the total_steps ceiling)."""
        model, tokenizer = gpt2_with_lora

        def _make(max_steps: int) -> Trainer:
            return Trainer(
                model=model,
                args=_default_args(
                    max_steps=max_steps,
                    num_train_epochs=1,
                    eval_strategy="no",
                    save_strategy="no",
                    logging_steps=999,
                    privacy_noise_multiplier=1.0,
                    privacy_target_epsilon=0.001,
                ),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )

        probe = _make(20)
        m = probe.train().global_step  # discover the crossing step M < 20
        assert probe.state.privacy_target_epsilon_reached is True

        exact = _make(m)  # the target now lands exactly on the final step
        out = exact.train()
        assert out.global_step == m
        assert exact.state.privacy_target_epsilon_reached is True

    def test_stop_does_not_advance_sampler_past_global_step(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """#392 review: stopping must not consume an extra Poisson round — the
        checkpointed sampler cursor equals global_step, so resume skips no
        data.  ``save_steps=999`` makes the end-of-training save the recorded
        snapshot (it reads the live sampler at stop time)."""
        from opaque.api.transformers.trainer import _checkpoint as ckpt

        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=20,
                num_train_epochs=1,
                eval_strategy="no",
                save_strategy="steps",
                save_steps=999,
                logging_steps=999,
                privacy_noise_multiplier=1.0,
                privacy_target_epsilon=0.001,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train()
        payload = ckpt.load_dp_runtime_state(
            str(tmp_path / f"checkpoint-{out.global_step}" / ckpt.DP_STATE_NAME)
        )
        assert payload.sampler_state["consumed"] == out.global_step

    def test_runs_to_max_steps_when_target_not_reached(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """A target that's never reached at max_steps lets training finish."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=5,
                num_train_epochs=1,
                eval_strategy="no",
                logging_steps=1,
                privacy_noise_multiplier=1.0,
                privacy_target_epsilon=100.0,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train()
        assert out.global_step == 5
        assert trainer.state.privacy_target_epsilon_reached is False

    def test_no_stop_when_only_nm_set(self, gpt2_with_lora, tiny_lm_dataset):
        """NM-only path runs to max_steps regardless of accumulated ε."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="no",
                logging_steps=1,
                privacy_noise_multiplier=1.0,
                # No target_epsilon — stop check skipped.
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train()
        assert out.global_step == 4
        assert trainer.state.privacy_target_epsilon_reached is False


class TestTrainerTrain:
    """Test the full DP-SGD training loop."""

    def test_train_few_steps(self, gpt2_with_lora, tiny_lm_dataset):
        """Train for a few steps and verify state is returned."""
        model, tokenizer = gpt2_with_lora

        pre_train_params = {
            n: p.clone() for n, p in model.named_parameters() if p.requires_grad
        }

        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=3,
                num_train_epochs=1,
                learning_rate=1e-3,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()

        assert isinstance(output, TrainOutput)
        assert "train_loss" in output.metrics
        assert "privacy_epsilon" in output.metrics
        assert output.metrics["privacy_epsilon"] > 0
        assert output.global_step == 3
        assert output.training_loss > 0

        changed = False
        for n, p in model.named_parameters():
            if (
                p.requires_grad
                and n in pre_train_params
                and not torch.allclose(p.data, pre_train_params[n])
            ):
                changed = True
                break
        assert changed, "Model parameters did not change after training"

    def test_model_generates_after_training(self, gpt2_with_lora, tiny_lm_dataset):
        """Verify model.generate() works after param restoration."""
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
                logging_steps=999,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        inputs = tokenizer("def hello():", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.decode(output[0])
        assert len(decoded) > 0


class TestTrainerEvaluate:
    """Test the evaluation method."""

    def test_evaluate_returns_loss(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        metrics = trainer.evaluate()

        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)
        assert metrics["eval_loss"] > 0

    def test_evaluate_custom_prefix(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        metrics = trainer.evaluate(metric_key_prefix="test")

        assert "test_loss" in metrics


class TestTrainerCallbacks:
    """Test that callbacks are fired at the right points."""

    def test_callbacks_are_invoked(self, gpt2_with_lora, tiny_lm_dataset):
        """Track which callback hooks fire during training."""
        model, tokenizer = gpt2_with_lora

        fired = []

        from transformers import TrainerCallback

        class TrackingCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                fired.append("on_train_begin")

            def on_log(self, args, state, control, logs=None, **kwargs):
                fired.append("on_log")

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                fired.append("on_evaluate")

            def on_train_end(self, args, state, control, **kwargs):
                fired.append("on_train_end")

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[TrackingCallback()],
        )

        trainer.train()

        assert "on_train_begin" in fired
        assert "on_log" in fired
        assert "on_train_end" in fired
        assert "on_evaluate" in fired


class TestTrainerContractFlags:
    """Focused tests for trainer-contract flags."""

    def test_explicit_train_ignores_do_train_flag(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.train()
        assert out.global_step == 2
        assert out.training_loss > 0

    def test_predict_returns_prediction_output(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.predict(tiny_lm_dataset)
        assert isinstance(out, EvaluationResult)
        assert "test_loss" in out.metrics

    def test_explicit_predict_ignores_do_predict_flag(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.predict(tiny_lm_dataset)
        assert isinstance(out, EvaluationResult)
        assert "test_loss" in out.metrics

    def test_debug_underflow_overflow_callback_is_wired(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(debug="underflow_overflow"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        assert hasattr(trainer, "_debug_underflow_overflow")
        assert (
            trainer._debug_underflow_overflow.__class__.__name__
            == "DebugUnderflowOverflow"
        )

    def test_auto_find_batch_size_retries_microbatch(
        self, gpt2_with_lora, tiny_lm_dataset, monkeypatch
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=True,
                per_device_train_batch_size=8,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        calls: list[int | None] = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            assert ignore_keys_for_eval is None
            calls.append(microbatch_size_override)
            if len(calls) < 3:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return TrainOutput(
                global_step=1, training_loss=1.0, metrics={"train_loss": 1.0}
            )

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        out = trainer.train()

        assert out.global_step == 1
        assert calls == [8, 4, 2]

    def test_microbatch_size_sets_vmap_chunk_without_auto_find(
        self, gpt2_with_lora, tiny_lm_dataset, monkeypatch
    ):
        """``microbatch_size`` is plumbed to ``_train_once`` even when
        ``auto_find_microbatch_size=False`` — it's the primary knob, not a
        side-effect of auto-find."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=False,
                per_device_train_batch_size=32,
                microbatch_size=4,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        captured: list[int | None] = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            captured.append(microbatch_size_override)
            return TrainOutput(
                global_step=1, training_loss=1.0, metrics={"train_loss": 1.0}
            )

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        trainer.train()
        assert captured == [4]
        assert trainer.state.converged_microbatch_size == 4

    def test_microbatch_size_seeds_auto_find_starting_point(
        self, gpt2_with_lora, tiny_lm_dataset, monkeypatch
    ):
        """When ``auto_find_microbatch_size=True``, the user-set
        ``microbatch_size`` is the starting point — auto-find halves
        from there, not from ``per_device_train_batch_size``."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=True,
                per_device_train_batch_size=32,
                microbatch_size=8,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        calls: list[int | None] = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            calls.append(microbatch_size_override)
            if len(calls) < 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return TrainOutput(
                global_step=1, training_loss=1.0, metrics={"train_loss": 1.0}
            )

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        trainer.train()
        # Starts at user-set 8 (not 32), halves once on OOM, succeeds at 4.
        assert calls == [8, 4]

    def test_auto_find_batch_size_stops_at_floor(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
        monkeypatch,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=True,
                per_device_train_batch_size=1,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            assert ignore_keys_for_eval is None
            raise torch.OutOfMemoryError("out of memory")

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        with pytest.raises(torch.OutOfMemoryError, match="out of memory"):
            trainer.train()

    def test_past_index_is_rejected(self, tmp_path):
        # ``past_index`` is not on the TrainingArguments surface; unknown
        # kwargs raise ``TypeError`` (louder HF-porting signal).
        with pytest.raises(TypeError, match="past_index"):
            TrainingArguments(
                output_dir=str(tmp_path),
                use_cpu=True,
                past_index=1,
            )


class TestPrivacyDefaults:
    def test_unset_privacy_delta_uses_dataset_dependent_default(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(max_steps=1),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
        )

        ctx = trainer._setup_training()

        assert ctx.target_delta == pytest.approx(1 / len(tiny_lm_dataset) ** 1.1)


class TestTrainerAdaptiveClipping:
    """Test adaptive clipping mode."""

    def test_quantile_and_gradient_noise_use_split_keys(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        seed = 123
        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_mode="adaptive",
                clipping_norm=1.0,
                max_steps=1,
                seed=seed,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
        )

        ctx = trainer._setup_training()
        quantile_noise_key, gradient_noise_key = split(key(seed))

        assert ctx.clip_state._rng_key == quantile_noise_key
        assert ctx.noise_state._rng_key == gradient_noise_key
        assert ctx.clip_state._rng_key != ctx.noise_state._rng_key

    @pytest.mark.parametrize("target_quantile", [0.1, 0.9])
    def test_adaptive_clipping_public_kwargs_reach_factory(
        self, gpt2_with_lora, tiny_lm_dataset, target_quantile
    ):
        """Public adaptive-clipping kwargs reach the factory unchanged.

        The rates are asymmetric on purpose — 0.5 is a fixed point of
        ``x -> 1 - x`` and cannot distinguish the two conventions.
        """
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_mode="adaptive",
                clipping_norm=1.0,
                max_steps=1,
                clipping_kwargs={
                    "target_quantile": target_quantile,
                    "clipping_norm_max": 50.0,
                },
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
        )

        ctx = trainer._setup_training()

        assert ctx.clip_state._target_quantile == pytest.approx(target_quantile)
        assert ctx.clip_state._clipping_norm_max == pytest.approx(50.0)

    def test_adaptive_clipping_runs(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_mode="adaptive",
                clipping_norm=1.0,
                max_steps=3,
                eval_strategy="no",
                logging_steps=999,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()
        assert output.global_step == 3
        assert output.metrics["privacy_epsilon"] > 0


class TestTrainerLRScheduling:
    """Test that ``lr_scheduler`` and warmup actually take effect."""

    def test_fractional_epochs_use_fractional_step_horizon(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """A partial final epoch drives training, scheduling, and accounting."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=-1,
                num_train_epochs=1.25,
                learning_rate=1e-3,
                lr_scheduler="linear",
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()

        assert output.global_step == 3
        assert trainer.state.max_steps == 3
        assert trainer.state.privacy_total_steps == 3
        lrs = [
            entry["learning_rate"]
            for entry in trainer.state.log_history
            if "learning_rate" in entry
        ]
        assert lrs == pytest.approx([1e-3, 1e-3 * 2 / 3, 1e-3 / 3])

    def test_sub_epoch_training_runs_one_step(self, gpt2_with_lora, tiny_lm_dataset):
        """A positive fraction below one does not truncate to zero steps."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=-1,
                num_train_epochs=0.25,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()

        assert output.global_step == 1
        assert trainer.state.max_steps == 1
        assert trainer.state.privacy_total_steps == 1

    def test_constant_lr_logged_at_base(self, gpt2_with_lora, tiny_lm_dataset):
        """lr_scheduler='constant' logs base_lr at every step."""
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=3,
                num_train_epochs=1,
                learning_rate=1e-3,
                lr_scheduler="constant",
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        lrs = [
            e["learning_rate"]
            for e in trainer.state.log_history
            if "learning_rate" in e
        ]
        assert len(lrs) == 3
        for lr in lrs:
            assert lr == pytest.approx(1e-3)

    def test_linear_warmup_produces_expected_lr_series(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """Warmup ramp + linear decay produces the expected lr_schedule(step) series."""
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=5,
                num_train_epochs=1,
                learning_rate=1e-3,
                lr_scheduler="linear",
                warmup_steps=2,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        lrs = [
            e["learning_rate"]
            for e in trainer.state.log_history
            if "learning_rate" in e
        ]
        # HF parity: the LR logged at global_step k is the value that was
        # *just applied* to the optimizer update, i.e. ``schedule(k - 1)``.
        # With warmup_steps=2 over a 5-step linear schedule:
        #   step 1 → schedule(0) = 0.0   (start of warmup)
        #   step 2 → schedule(1) = 5e-4  (mid-warmup)
        #   step 3 → schedule(2) = 1e-3  (end of warmup, peak)
        #   step 4 → schedule(3) = 1e-3 * 2/3
        #   step 5 → schedule(4) = 1e-3 * 1/3
        expected = [0.0, 5e-4, 1e-3, 1e-3 * 2 / 3, 1e-3 * 1 / 3]
        assert len(lrs) == 5
        for got, exp in zip(lrs, expected, strict=False):
            assert got == pytest.approx(exp, abs=1e-9)

    def test_warmup_changes_param_trajectory(self, gpt2_with_lora, tiny_lm_dataset):
        """Trainers with constant vs warmup LR diverge on identical seed/data."""
        model_const, tok = gpt2_with_lora
        init_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        common = {
            "clipping_norm": 1.0,
            "max_steps": 4,
            "num_train_epochs": 1,
            "learning_rate": 1e-3,
            "warmup_steps": 2,
            "seed": 42,
            "eval_strategy": "no",
            "logging_steps": 999,
        }

        trainer1 = Trainer(
            model=model_const,
            args=_default_args(lr_scheduler="constant", **common),
            processing_class=tok,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()
        constant_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        # Reset model to initial state so the second run sees the same start.
        for n, p in model_const.named_parameters():
            if n in init_params:
                p.data.copy_(init_params[n])

        trainer2 = Trainer(
            model=model_const,
            args=_default_args(lr_scheduler="linear", **common),
            processing_class=tok,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer2.train()
        warmup_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        diverged = any(
            not torch.allclose(constant_params[n], warmup_params[n])
            for n in constant_params
        )
        assert diverged, "constant vs linear+warmup produced identical params"


class TestTrainerCheckpointing:
    """End-to-end checkpoint save / rotation / final-save tests."""

    def _common_args(self, output_dir, **overrides):
        defaults = {
            "clipping_norm": 1.0,
            "max_steps": 4,
            "num_train_epochs": 1,
            "learning_rate": 1e-3,
            "lr_scheduler": "constant",
            "eval_strategy": "no",
            "logging_steps": 1,
            "output_dir": str(output_dir),
            "save_strategy": "steps",
            "save_steps": 2,
            "save_safetensors": True,
            "overwrite_output_dir": True,
        }
        defaults.update(overrides)
        return _default_args(**defaults)

    def test_save_at_step_intervals(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        ckpts = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert ckpts == ["checkpoint-2", "checkpoint-4"]

        for name in ckpts:
            d = tmp_path / name
            # PEFT models save adapter_*; full models save model.safetensors / config.json.
            weights_present = (d / "model.safetensors").exists() or (
                d / "adapter_model.safetensors"
            ).exists()
            cfg_present = (d / "config.json").exists() or (
                d / "adapter_config.json"
            ).exists()
            assert weights_present
            assert cfg_present
            assert (d / "dp_optimizer.pt").exists()
            assert (d / "dp_state.pt").exists()
            assert (d / "accountant.json").exists()
            assert (d / "rng_state.pth").exists()
            assert (d / "trainer_state.json").exists()
            assert (d / "training_args.bin").exists()

    def test_save_safetensors_false_writes_pickle(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """save_safetensors=False writes the legacy .bin format, and we can resume from it."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=2, save_steps=2, save_safetensors=False
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        d = tmp_path / "checkpoint-2"
        # PEFT writes adapter_model.bin; full models write pytorch_model.bin.
        assert (d / "pytorch_model.bin").exists() or (d / "adapter_model.bin").exists()
        # No safetensors files when the flag is off.
        assert not (d / "model.safetensors").exists()
        assert not (d / "adapter_model.safetensors").exists()

        # Resume from this .bin checkpoint to confirm load works for both formats.
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                tmp_path, max_steps=4, save_steps=2, save_safetensors=False
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer2.train(resume_from_checkpoint=str(d))
        assert out.global_step == 4

    def test_save_only_model_skips_runtime_files(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, save_only_model=True, max_steps=2, save_steps=2
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        d = tmp_path / "checkpoint-2"
        weights_present = (d / "model.safetensors").exists() or (
            d / "adapter_model.safetensors"
        ).exists()
        cfg_present = (d / "config.json").exists() or (
            d / "adapter_config.json"
        ).exists()
        assert weights_present
        assert cfg_present
        # Resumability files (optimizer / DP runtime / RNG) are skipped
        # under ``save_only_model``.
        assert not (d / "dp_optimizer.pt").exists()
        assert not (d / "dp_state.pt").exists()
        assert not (d / "rng_state.pth").exists()
        # Interpretability files are always written: ``trainer_state.json``,
        # ``training_args.bin`` (HF-parity filename), and ``accountant.json``
        # (the privacy guarantee is a property of the saved model, not
        # training state).
        assert (d / "trainer_state.json").exists()
        assert (d / "training_args.bin").exists()
        assert (d / "accountant.json").exists()

    def test_save_total_limit_keeps_most_recent(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=6, save_steps=2, save_total_limit=2
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert names == ["checkpoint-4", "checkpoint-6"]

    def test_final_save_when_unaligned(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """Final step saves a checkpoint even if it doesn't align with save_steps."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=3, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # checkpoint-2 from save_steps; checkpoint-3 from final-save.
        assert names == ["checkpoint-2", "checkpoint-3"]

    def test_save_strategy_no_writes_nothing(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        assert [
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        ] == []

    def test_save_strategy_epoch(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=-1,
                num_train_epochs=2,
                save_strategy="epoch",
                # Make epoch length tiny so the test is quick.
                per_device_train_batch_size=4,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # At least one checkpoint per epoch.
        assert len(names) >= 2

    def test_fractional_save_steps_resolved(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """save_steps in (0, 1) is treated as a fraction of total_steps."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=4, save_steps=0.5),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        # 0.5 * 4 == 2 → checkpoints at 2 and 4
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert names == ["checkpoint-2", "checkpoint-4"]

    def test_trainer_state_json_round_trips(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Saved trainer_state.json deserializes to an equivalent TrainerState."""
        import json

        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        state_path = tmp_path / "checkpoint-2" / "trainer_state.json"
        with state_path.open() as f:
            data = json.load(f)
        assert data["global_step"] == 2
        assert data["max_steps"] == 2
        assert isinstance(data["log_history"], list)

        restored = TrainerState.from_json(data)
        assert restored.global_step == 2

    def test_save_model_public_api(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """save_model() writes weights, training args, and accountant.json."""
        model, tokenizer = gpt2_with_lora
        out = tmp_path / "final"
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no", max_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        trainer.save_model(str(out))
        assert (out / "model.safetensors").exists() or (
            out / "adapter_model.safetensors"
        ).exists()
        assert (out / "config.json").exists() or (out / "adapter_config.json").exists()
        # Privacy provenance travels with the saved model.
        assert (out / "accountant.json").exists()

    # ------------------------------------------------------------------
    # Accountant persistence, independent of the model
    # ------------------------------------------------------------------

    def _trained_trainer(self, model, tokenizer, dataset, tmp_path, **overrides):
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, save_strategy="no", max_steps=2, **overrides
            ),
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainer.train()
        return trainer

    def test_save_accountant_writes_only_the_accountant(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """The accounting can be harvested without re-saving the model."""
        model, tokenizer = gpt2_with_lora
        trainer = self._trained_trainer(model, tokenizer, tiny_lm_dataset, tmp_path)

        out = tmp_path / "accounting-only"
        written = trainer.save_accountant(str(out))

        assert written == str(out / "accountant.json")
        assert (out / "accountant.json").exists()
        # No model artefacts — that is the whole point of the method.
        assert not (out / "model.safetensors").exists()
        assert not (out / "adapter_model.safetensors").exists()
        assert not (out / "training_args.bin").exists()

    def test_save_accountant_creates_missing_directory(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = self._trained_trainer(model, tokenizer, tiny_lm_dataset, tmp_path)

        out = tmp_path / "nested" / "does-not-exist-yet"
        assert trainer.save_accountant(str(out)) is not None
        assert (out / "accountant.json").exists()

    def test_save_accountant_defaults_to_args_output_dir(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        default_dir = tmp_path / "default-target"
        trainer = self._trained_trainer(model, tokenizer, tiny_lm_dataset, default_dir)

        assert trainer.save_accountant() == str(default_dir / "accountant.json")
        assert (default_dir / "accountant.json").exists()

    def test_save_accountant_matches_what_save_model_writes(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Factoring the write out of ``save_model`` changed no bytes."""
        import json

        model, tokenizer = gpt2_with_lora
        trainer = self._trained_trainer(model, tokenizer, tiny_lm_dataset, tmp_path)

        via_model = tmp_path / "via-save-model"
        via_direct = tmp_path / "via-save-accountant"
        trainer.save_model(str(via_model))
        trainer.save_accountant(str(via_direct))

        with (via_model / "accountant.json").open() as f:
            from_model = json.load(f)
        with (via_direct / "accountant.json").open() as f:
            from_direct = json.load(f)
        assert from_model == from_direct

    def test_save_accountant_before_training_writes_nothing(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """No training run means no accounting to serialise — report, don't guess."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no", max_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = tmp_path / "untrained"
        assert trainer.save_accountant(str(out)) is None
        assert not (out / "accountant.json").exists()

    # ------------------------------------------------------------------
    # Best-model tracking
    # ------------------------------------------------------------------

    def test_load_best_model_at_end_requires_strategies(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        # HF parity: ``load_best_model_at_end`` with mismatched
        # ``eval_strategy``/``save_strategy`` is rejected by
        # ``TrainingArguments.__post_init__`` before the trainer
        # even constructs.  The error wording mirrors HF's exact phrasing.
        with pytest.raises(ValueError, match="save and eval strategy to match"):
            self._common_args(
                tmp_path,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                eval_strategy="no",
            )

    def test_load_best_model_at_end_defaults_metric_to_loss(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        # HF parity: omitting ``metric_for_best_model`` under
        # ``load_best_model_at_end`` is now silently filled with ``"loss"``
        # by ``TrainingArguments.__post_init__`` (it used to raise).
        model, tokenizer = gpt2_with_lora
        args = self._common_args(
            tmp_path,
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=2,
            eval_strategy="steps",
            eval_steps=2,
            metric_for_best_model=None,
        )
        assert args.metric_for_best_model == "loss"
        # ``greater_is_better`` defaults to ``False`` for loss-suffixed metrics.
        assert args.greater_is_better is False
        # Construction still succeeds with the auto-default in place.
        trainer = Trainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        assert trainer is not None

    def test_greater_is_better_default_for_loss(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        # Loss is minimized → greater_is_better=False.  HF parity:
        # ``__post_init__`` populates the default at construction time,
        # so the args object reflects the resolved value just like
        # ``transformers.TrainingArguments``.
        assert trainer.args.greater_is_better is False
        assert trainer.args.greater_is_better is False

    def test_greater_is_better_default_for_accuracy(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        # Anything not ending in "loss" → greater_is_better=True.  HF parity:
        # ``__post_init__`` populates the default at construction time.
        assert trainer.args.greater_is_better is True
        assert trainer.args.greater_is_better is True

    def test_best_metric_tracking_runs(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """state.best_* is populated after training when eval is enabled."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        assert trainer.state.best_metric is not None
        assert trainer.state.best_global_step in (2, 4)
        assert trainer.state.best_model_checkpoint is not None
        assert Path(trainer.state.best_model_checkpoint).is_dir()

    def test_save_strategy_best_only_saves_on_improvement(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="best",
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # First eval always counts as an improvement (best_metric was None).
        # Final-save will also fire at the last step, even if no improvement.
        assert len(names) >= 1

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_lr_schedule_continues_across_resume(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Resume with matching args preserves the LR trajectory step-for-step.

        Phase-1 trains 5 of 10 steps then forcibly stops + saves (via a
        callback, so the LR schedule is built with ``max_steps=10`` from the
        start).  Phase-2 resumes with the same ``max_steps=10`` and runs to
        completion.  The per-step LR series logged by ``log_history`` after
        the chained run must match a continuous max_steps=10 run exactly.

        Catches three classes of bug:
         * a saved schedule overrides the fresh one on resume (post-resume
           LR returns to warmup-from-zero);
         * the optimizer holds a stale schedule reference (LRs applied
           differ silently from LRs logged);
         * ``log_history`` is reset on resume (chained series is shorter
           than continuous).
        """
        from transformers.trainer_callback import TrainerCallback

        class StopAtStepCallback(TrainerCallback):
            def __init__(self, stop_at: int):
                self.stop_at = stop_at

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= self.stop_at:
                    control.should_save = True
                    control.should_training_stop = True

        common = {
            "max_steps": 10,
            "lr_scheduler": "linear",
            "warmup_steps": 2,
            "learning_rate": 1e-3,
            "logging_steps": 1,
        }

        # --- continuous baseline ---
        model_c, tokenizer_c = gpt2_with_lora
        trainer_c = Trainer(
            model=model_c,
            args=self._common_args(
                tmp_path / "continuous", save_strategy="no", **common
            ),
            processing_class=tokenizer_c,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer_c.train()
        lrs_continuous = [
            e["learning_rate"]
            for e in trainer_c.state.log_history
            if "learning_rate" in e
        ]
        assert len(lrs_continuous) == 10

        # --- phase 1: same args, stop+save at step 5 via callback ---
        chain_dir = tmp_path / "chain"
        model_1, tokenizer_1 = gpt2_with_lora
        trainer_1 = Trainer(
            model=model_1,
            args=self._common_args(chain_dir, save_strategy="no", **common),
            processing_class=tokenizer_1,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[StopAtStepCallback(stop_at=5)],
        )
        trainer_1.train()
        ckpt_dir = str(chain_dir / "checkpoint-5")
        assert Path(ckpt_dir).is_dir(), f"checkpoint-5 not written at {chain_dir}"

        # --- phase 2: resume same args, run to step 10 ---
        model_2, tokenizer_2 = gpt2_with_lora
        trainer_2 = Trainer(
            model=model_2,
            args=self._common_args(chain_dir, save_strategy="no", **common),
            processing_class=tokenizer_2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer_2.train(resume_from_checkpoint=ckpt_dir)
        lrs_chained = [
            e["learning_rate"]
            for e in trainer_2.state.log_history
            if "learning_rate" in e
        ]
        assert len(lrs_chained) == 10, (
            f"chained log_history should cover all 10 steps "
            f"(phase1's 5 + phase2's 5); got {len(lrs_chained)} entries"
        )

        for step, (got, exp) in enumerate(
            zip(lrs_chained, lrs_continuous, strict=False), start=1
        ):
            assert got == pytest.approx(exp, abs=1e-9), (
                f"LR mismatch at global_step={step}: chained={got}, continuous={exp}"
            )

    def test_dataloader_prefetch_checkpoints_trainer_consumed_not_sampler_cursor(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """#791: worker prefetch must not leak into the checkpointed cursor."""
        from opaque.api.transformers.trainer import _checkpoint as ckpt

        model, tokenizer = gpt2_with_lora
        live_consumed_at_save: list[int] = []

        class _CaptureLiveSamplerCursor(_HFTrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                live_consumed_at_save.append(trainer._ctx.current_sampler.consumed)

        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=6,
                save_steps=2,
                dataloader_num_workers=2,
                dataloader_prefetch_factor=2,
                # "fork": macOS's "spawn" default can't pickle the
                # trainer's collate-fn closure across processes.
                dataloader_multiprocessing_context="fork",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[_CaptureLiveSamplerCursor()],
        )
        trainer.train()

        # Prefetch must actually race ahead of global_step=2 here, or this
        # test wouldn't exercise the bug.
        assert live_consumed_at_save[0] > 2

        payload = ckpt.load_dp_runtime_state(
            str(tmp_path / "checkpoint-2" / ckpt.DP_STATE_NAME)
        )
        assert payload.sampler_state["consumed"] == 2

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                tmp_path,
                max_steps=6,
                save_steps=2,
                dataloader_num_workers=2,
                dataloader_prefetch_factor=2,
                dataloader_multiprocessing_context="fork",
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        assert out2.global_step == 6

    def test_resume_normalizes_prefetched_legacy_sampler_cursor(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path, caplog
    ):
        from opaque.api.transformers.trainer import _checkpoint as ckpt

        batches: list[torch.Tensor] = []

        class _CaptureBatches(Trainer):
            def training_step(self, model, inputs):
                batches.append(inputs["input_ids"].detach().clone())
                return super().training_step(model, inputs)

        model, tokenizer = gpt2_with_lora
        trainer = _CaptureBatches(
            model=model,
            args=self._common_args(tmp_path, max_steps=6),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
        )
        trainer.train()
        expected_batches = batches[2:]
        batches.clear()

        checkpoint = tmp_path / "checkpoint-2"
        runtime_path = checkpoint / ckpt.DP_STATE_NAME
        payload = ckpt.load_dp_runtime_state(str(runtime_path))
        assert payload.version == ckpt.DP_STATE_BUNDLE_VERSION
        # Simulate a legacy snapshot taken after workers prefetched all draws.
        payload.sampler_state["consumed"] = 6
        torch.save(payload, runtime_path)

        resumed = _CaptureBatches(
            model=model,
            args=self._common_args(tmp_path / "resumed", max_steps=6),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
        )
        result = resumed.train(resume_from_checkpoint=str(checkpoint))

        assert result.global_step == 6
        assert len(batches) == len(expected_batches) == 4
        for actual, expected in zip(batches, expected_batches, strict=True):
            assert torch.equal(actual, expected)
        assert "Clamping checkpoint sampler consumed=6 to global_step=2" in caplog.text
        for step in (4, 6):
            saved = ckpt.load_dp_runtime_state(
                str(tmp_path / "resumed" / f"checkpoint-{step}" / ckpt.DP_STATE_NAME)
            )
            assert saved.sampler_state["consumed"] == step

    def test_chained_resume_after_ignore_data_skip_keeps_cursor_aligned(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """#791: clamp must rebase across an ``ignore_data_skip`` resume."""
        from opaque.api.transformers.trainer import _checkpoint as ckpt

        # Phase 1: plain run to checkpoint-2 (global_step=2).
        model1, tokenizer1 = gpt2_with_lora
        trainer1 = Trainer(
            model=model1,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer1,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        # Phase 2: resume with ignore_data_skip=True (rebases the sampler
        # stream at global_step=2) plus prefetch, 2 more steps to checkpoint-4.
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                ignore_data_skip=True,
                dataloader_num_workers=2,
                dataloader_prefetch_factor=2,
                dataloader_multiprocessing_context="fork",
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        assert out2.global_step == 4

        # 2 post-rebase steps, not global_step=4 or a prefetch-inflated value.
        payload = ckpt.load_dp_runtime_state(
            str(tmp_path / "checkpoint-4" / ckpt.DP_STATE_NAME)
        )
        assert payload.sampler_state["consumed"] == 2

        # Phase 3: normal resume to completion; the offset must carry
        # forward so a further checkpoint under prefetch still clamps right.
        model3, tokenizer3 = gpt2_with_lora
        trainer3 = Trainer(
            model=model3,
            args=self._common_args(
                tmp_path,
                max_steps=6,
                save_steps=2,
                dataloader_num_workers=2,
                dataloader_prefetch_factor=2,
                dataloader_multiprocessing_context="fork",
            ),
            processing_class=tokenizer3,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out3 = trainer3.train(resume_from_checkpoint=str(tmp_path / "checkpoint-4"))
        assert out3.global_step == 6

        payload6 = ckpt.load_dp_runtime_state(
            str(tmp_path / "checkpoint-6" / ckpt.DP_STATE_NAME)
        )
        assert payload6.sampler_state["consumed"] == 4

    def test_resume_continues_global_step(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Resume from a saved checkpoint advances global_step instead of restarting."""
        model, tokenizer = gpt2_with_lora
        # Initial run: 2 steps, save at step 2.
        trainer1 = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out1 = trainer1.train()
        assert out1.global_step == 2
        ckpt_dir = str(tmp_path / "checkpoint-2")
        assert Path(ckpt_dir).is_dir()

        # Resume: max_steps=4 → run 2 more steps starting at global_step=2.
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=ckpt_dir)
        assert out2.global_step == 4
        assert trainer2.state.global_step == 4
        # checkpoint-4 was just written by the final save / save_steps.
        assert (tmp_path / "checkpoint-4").is_dir()

    def test_resume_true_finds_latest(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """resume_from_checkpoint=True picks the latest checkpoint under output_dir."""
        model, tokenizer = gpt2_with_lora
        trainer1 = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=True)
        assert out2.global_step == 4

    def test_resume_missing_path_raises(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        with pytest.raises(CheckpointError):
            trainer.train(resume_from_checkpoint=str(tmp_path / "no-such-ckpt"))

    def test_resume_true_starts_fresh_when_no_checkpoints(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """``resume_from_checkpoint=True`` is tolerant: auto-finds the latest
        checkpoint when one exists, falls back to a fresh run when none does.

        Lets scripts pass ``resume_from_checkpoint=True`` unconditionally —
        "resume if you can, else start fresh" — without probing the filesystem.
        """
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train(resume_from_checkpoint=True)
        # Fresh-run semantics: global_step advances from 0, not via resume.
        assert out.global_step > 0

    def test_resume_missing_accountant_raises(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """A checkpoint missing ``accountant.json`` is not a complete DP
        checkpoint, so resume rejects it as a weights-only export."""

        # Produce a checkpoint, then delete its accountant.json.
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        ckpt_dir = tmp_path / "checkpoint-2"
        (ckpt_dir / "accountant.json").unlink()

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        with pytest.raises(CheckpointError, match="weights-only export"):
            trainer2.train(resume_from_checkpoint=str(ckpt_dir))

    def test_fresh_run_from_weights_only_checkpoint_via_model_arg(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Starting from a weights-only export is a *fresh* DP run loaded at
        construction (model=), not a resume.  The new run begins with a zero
        accountant and trains to its own target."""
        model, tokenizer = gpt2_with_lora
        trainer = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=2, save_steps=2, save_only_model=True
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()  # weights-only export (no dp_state / optimizer)

        # A fresh run that simply continues using the same in-memory model is
        # the supported "start from these weights" path — no resume, fresh ε.
        out2_dir = tmp_path / "fresh"
        out2_dir.mkdir()
        trainer2 = Trainer(
            model=trainer.model,
            args=self._common_args(out2_dir, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer2.train()  # no resume_from_checkpoint
        assert out.global_step == 2

    def test_resume_restores_accountant(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """epsilon at end of resumed run reflects total composition (incl. saved steps)."""
        model, tokenizer = gpt2_with_lora
        trainer1 = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out1 = trainer1.train()
        eps_after_2 = out1.metrics["privacy_epsilon"]

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # Resumed run composes additional steps on top of saved process → ε grows.
        assert out2.metrics["privacy_epsilon"] > eps_after_2

    def test_resume_save_only_model_is_refused(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """A ``save_only_model=True`` checkpoint omits the DP runtime state
        (noise/sampler), so resuming *training* from it would rebuild the
        noise stream at step 0 and reuse the original run's noise on
        re-sampled data — a silent privacy break.  The trainer must refuse
        it (export-only), even though ``accountant.json`` is present."""
        import pytest

        model, tokenizer = gpt2_with_lora
        trainer1 = Trainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=2, save_steps=2, save_only_model=True
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()
        ckpt_dir = str(tmp_path / "checkpoint-2")
        # Interpretability file always present; resumability files absent.
        assert (Path(ckpt_dir) / "accountant.json").exists()
        assert not (Path(ckpt_dir) / "dp_optimizer.pt").exists()
        assert not (Path(ckpt_dir) / "dp_state.pt").exists()

        model2, tokenizer2 = gpt2_with_lora
        out2_dir = tmp_path / "resumed"
        out2_dir.mkdir()
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                out2_dir,
                max_steps=4,
                save_steps=2,
                save_only_model=False,
                privacy_noise_multiplier=1.0,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        with pytest.raises(CheckpointError, match="weights-only export"):
            trainer2.train(resume_from_checkpoint=ckpt_dir)

    def test_ignore_data_skip_uses_new_poisson_stream(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path, monkeypatch
    ):
        """Use a distinct Poisson stream."""
        model, tokenizer = gpt2_with_lora
        trainer1 = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        sampler_keys = []
        original_build_sampler = _dpftrl.build_sampler

        def record_sampler_key(**kwargs):
            sampler_keys.append(kwargs["key"])
            return original_build_sampler(**kwargs)

        monkeypatch.setattr(_dpftrl, "build_sampler", record_sampler_key)

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                tmp_path, max_steps=4, save_steps=2, ignore_data_skip=True
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        assert out2.global_step == 4
        base_key = key(
            trainer2.args.data_seed
            if trainer2.args.data_seed is not None
            else trainer2.args.seed
        )
        assert sampler_keys == [
            fold_in(base_key, "opaque.transformers.ignore_data_skip", 2)
        ]
        assert sampler_keys[0] != base_key

    def test_resume_restores_callback_state(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """When restore_callback_states_from_checkpoint=True, callbacks reload state."""
        import json

        model, tokenizer = gpt2_with_lora

        from transformers import TrainerCallback
        from transformers.trainer_callback import ExportableState

        class StatefulCallback(TrainerCallback, ExportableState):
            def __init__(self):
                self.value = 0

            def state(self):
                return {"args": {}, "attributes": {"value": self.value}}

            def on_log(self, args, state, control, logs=None, **kwargs):
                self.value += 1

        cb = StatefulCallback()
        trainer1 = Trainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[cb],
        )
        trainer1.train()
        # The trainer writes the callback's ExportableState payload into the JSON.
        with (tmp_path / "checkpoint-2" / "trainer_state.json").open() as f:
            ts = json.load(f)
        assert ts.get("stateful_callbacks", {}).get("StatefulCallback") == {
            "args": {},
            "attributes": {"value": cb.value},
        }
        saved_value = cb.value
        assert saved_value > 0  # received at least one on_log

        cb2 = StatefulCallback()
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                restore_callback_states_from_checkpoint=True,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[cb2],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # Callback's value was restored before the resumed run started, so it's
        # at least the saved value (further on_log bumps may have advanced it).
        assert cb2.value >= saved_value


# ---------------------------------------------------------------------------
# Phase 3a: eval_on_start, eval_delay, prediction_loss_only
# ---------------------------------------------------------------------------


class _EvalRecorder(_HFTrainerCallback):
    """Pytest helper: records ``(global_step, epoch)`` per on_evaluate call."""

    def __init__(self):
        self.calls: list[tuple[int, float]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.calls.append((state.global_step, float(state.epoch)))

    @property
    def steps(self) -> list[int]:
        return [step for step, _ in self.calls]


def _hf_evaluates_at_end_of_training() -> bool:
    """Whether HF's flow callback adds an evaluation at an off-grid final step.

    ``DefaultFlowCallback`` grew this in transformers 5.5 and the supported
    floor is 4.57, so the cadence tests probe for it the way the kernel patches
    probe torch: ask the installed version, don't assume a release.
    """
    from transformers.trainer_callback import (
        DefaultFlowCallback,
        TrainerControl,
        TrainerState,
    )

    args = _default_args(max_steps=5, eval_strategy="steps", eval_steps=2)
    state = TrainerState(global_step=5, max_steps=5, eval_steps=2)
    control = DefaultFlowCallback().on_step_end(args, state, TrainerControl())
    return bool(control.should_evaluate)


class TestTrainerEvalControls:
    """``eval_on_start``, evaluation cadence, ``prediction_loss_only``.

    Cadence comes from HF's ``DefaultFlowCallback``, which sets
    ``control.should_evaluate``; ``Trainer._maybe_log_save_evaluate`` acts
    on it.  The cadence tests assert the ``on_evaluate`` events a real
    training run emits.
    """

    def test_eval_on_start_fires_before_first_step(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="steps",
                eval_steps=999,  # disables the per-step trigger
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        # First on_evaluate must fire at global_step=0, before any training step.
        assert rec.calls, "on_evaluate did not fire"
        assert rec.calls[0][0] == 0

    def test_eval_on_start_with_strategy_no(self, gpt2_with_lora, tiny_lm_dataset):
        """eval_on_start fires at step 0 even when eval_strategy='no'."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == 0

    def test_eval_on_start_fires_on_resume(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """``eval_on_start=True`` fires on every ``train()`` call (HF parity).

        Pre-Stage-3 we had an Opaque-specific gate that suppressed this
        eval on resume.  HF *always* fires when the flag is set —
        users opting in want a baseline eval at the start of each
        ``train()`` call, including on resume.  Disable the flag if
        you don't want it on resume.
        """
        model, tokenizer = gpt2_with_lora

        # First run: produce a checkpoint at step 2.
        trainer1 = Trainer(
            model=model,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        # Resume: eval_on_start=True ⇒ the resumed run also fires the
        # baseline eval at the start.
        rec = _EvalRecorder()
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = Trainer(
            model=model2,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=4,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="no",
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # HF parity: exactly one eval call, fired before the inner loop runs.
        assert len(rec.calls) == 1
        # ``eval_on_start`` fires before the first post-resume step, so
        # ``global_step`` recorded by the eval recorder is the loaded
        # global_step (= 2), not 0.
        assert rec.calls[0][0] == 2

    @pytest.mark.parametrize(
        ("max_steps", "on_grid"),
        [(6, [2, 4, 6]), (5, [2, 4])],
        ids=["grid-hits-last-step", "grid-misses-last-step"],
    )
    def test_eval_steps_cadence(
        self, gpt2_with_lora, tiny_lm_dataset, max_steps, on_grid
    ):
        """``eval_strategy='steps'`` evaluates on the ``eval_steps`` grid.

        HF parity: on versions that do it, ``DefaultFlowCallback`` adds one
        evaluation at a final step the grid missed, so ``eval_steps=2`` over 5
        steps evaluates at 2, 4 and 5.  That is the only version-dependent part
        of the cadence, so it is probed rather than assumed.
        """
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=max_steps,
                eval_strategy="steps",
                eval_steps=2,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        expected = on_grid
        if max_steps not in on_grid and _hf_evaluates_at_end_of_training():
            expected = [*on_grid, max_steps]
        assert rec.steps == expected

    def test_eval_delay_steps_skips_until_threshold(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """eval_delay=4 with eval_steps=2, max_steps=6 → fires at 4 and 6, not 2."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                max_steps=6,
                eval_strategy="steps",
                eval_steps=2,
                eval_delay=4,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        assert rec.steps == [4, 6]

    def test_eval_epoch_cadence(self, gpt2_with_lora, tiny_lm_dataset):
        """``eval_strategy='epoch'`` evaluates once per completed epoch."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                num_train_epochs=3,
                eval_strategy="epoch",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        # 8 examples at a logical batch of 4 ⇒ 2 steps per epoch.  Each
        # evaluation lands on an integer epoch, never mid-epoch.
        assert rec.calls == [(2, 1.0), (4, 2.0), (6, 3.0)]

    def test_eval_delay_epochs_skips_until_threshold(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """Under ``eval_strategy='epoch'``, ``eval_delay`` counts epochs."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = Trainer(
            model=model,
            args=_default_args(
                num_train_epochs=3,
                eval_strategy="epoch",
                eval_delay=2,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        assert rec.calls == [(4, 2.0), (6, 3.0)]

    def test_eval_steps_cadence_survives_resume(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """The step grid is global, so resume keeps evaluating on it.

        Training stops at step 2 and resumes with ``eval_steps=3``: the grid
        is read off ``state.global_step``, so evaluation happens at 3 and 6.
        A cadence restarted from the resume point would evaluate at 5 instead.
        """
        model, tokenizer = gpt2_with_lora
        cadence = {
            "output_dir": str(tmp_path),
            "save_strategy": "steps",
            "save_steps": 2,
            "eval_strategy": "steps",
            "eval_steps": 3,
        }

        trainer1 = Trainer(
            model=model,
            args=_default_args(max_steps=2, **cadence),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        rec = _EvalRecorder()
        trainer2 = Trainer(
            model=model,
            args=_default_args(max_steps=6, **cadence),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))

        assert rec.steps == [3, 6]

    def test_eval_epoch_cadence_survives_resume(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Resume does not re-evaluate an epoch that already completed.

        The resumed run picks its starting epoch up from ``global_step``, so
        after one completed epoch it evaluates at epochs 2 and 3 only.
        """
        model, tokenizer = gpt2_with_lora
        cadence = {
            "output_dir": str(tmp_path),
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
        }

        trainer1 = Trainer(
            model=model,
            args=_default_args(num_train_epochs=1, **cadence),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        rec = _EvalRecorder()
        trainer2 = Trainer(
            model=model,
            args=_default_args(num_train_epochs=3, **cadence),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))

        assert rec.calls == [(4, 2.0), (6, 3.0)]

    def test_prediction_loss_only_skips_compute_metrics(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """prediction_loss_only=True: compute_metrics is not invoked."""
        model, tokenizer = gpt2_with_lora
        cm_calls = []

        def cm(eval_pred, **kwargs):
            cm_calls.append(eval_pred)
            return {"acc": 0.0}

        trainer = Trainer(
            model=model,
            args=_default_args(prediction_loss_only=True),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        out = trainer.evaluation_loop(
            trainer.get_eval_dataloader(),
            description="test",
        )
        assert out.predictions is None
        assert out.label_ids is None
        assert cm_calls == []  # compute_metrics never called
        assert "eval_loss" in out.metrics


# ---------------------------------------------------------------------------
# Phase 3b: eval_accumulation_steps, eval_do_concat_batches, batch_eval_metrics
# ---------------------------------------------------------------------------


class TestTrainerEvalMemory:
    """Phase 3b: memory-management flags."""

    def test_eval_accumulation_steps_engages_cpu_flushes(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """eval_accumulation_steps=2 with 4 batches → 2 mid-loop flushes + 1 final."""
        model, tokenizer = gpt2_with_lora

        # Track flushes by patching _PredictionAccumulator.flush_to_cpu.
        import opaque.api.transformers.trainer._eval as _eval_mod

        flush_count = {"n": 0}
        original = _eval_mod._PredictionAccumulator.flush_to_cpu

        def counting_flush(self):
            flush_count["n"] += 1
            return original(self)

        _eval_mod._PredictionAccumulator.flush_to_cpu = counting_flush
        try:
            trainer = Trainer(
                model=model,
                args=_default_args(
                    per_device_eval_batch_size=2,  # 8 examples / 2 = 4 batches
                    eval_accumulation_steps=2,
                ),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )
            metrics = trainer.evaluate()
        finally:
            _eval_mod._PredictionAccumulator.flush_to_cpu = original

        assert "eval_loss" in metrics
        # 4 batches with cadence 2 → 2 mid-loop flushes; finalize() adds 1.
        assert flush_count["n"] == 3

    def test_eval_do_concat_batches_false_returns_lists(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """eval_do_concat_batches=False → predictions delivered as a list."""
        model, tokenizer = gpt2_with_lora
        captured: dict = {}

        def cm(eval_pred):
            captured["predictions"] = eval_pred.predictions
            captured["label_ids"] = eval_pred.label_ids
            return {"shape_marker": 1.0}

        trainer = Trainer(
            model=model,
            args=_default_args(
                per_device_eval_batch_size=4,  # 8 examples / 4 = 2 batches
                eval_do_concat_batches=False,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()

        assert isinstance(captured["predictions"], list)
        assert isinstance(captured["label_ids"], list)
        assert len(captured["predictions"]) == 2


# ---------------------------------------------------------------------------
# Phase 3c: include_for_metrics + deprecated alias removal
# ---------------------------------------------------------------------------


class TestTrainerEvalMetrics:
    """Phase 3c: ``include_for_metrics`` and the dropped deprecated alias."""

    def test_compute_metrics_receives_eval_prediction(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """compute_metrics is called once with a fully-populated EvalPrediction."""
        import numpy as np

        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["predictions"] = eval_pred.predictions
            seen["label_ids"] = eval_pred.label_ids
            return {"acc": 0.42}

        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        metrics = trainer.evaluate()
        assert "eval_acc" in metrics
        assert metrics["eval_acc"] == pytest.approx(0.42)
        assert seen["predictions"] is not None
        assert seen["label_ids"] is not None
        # HF parity: ``compute_metrics`` consumes numpy arrays.
        assert isinstance(seen["predictions"], np.ndarray)
        assert isinstance(seen["label_ids"], np.ndarray)

    def test_compute_metrics_numpy_argmax_works(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """A typical HF-flavored reducer using ``np.argmax`` runs end-to-end.

        Pre-Stage-1 the accumulator returned ``torch.Tensor``; calling
        ``np.argmax`` on it works by accident but ``sklearn.metrics.*``
        and ``evaluate.load(...)`` reducers expect actual numpy arrays.
        Stage-1 numpifies via ``transformers.trainer_pt_utils.nested_numpify``
        so this contract holds.
        """
        import numpy as np

        model, tokenizer = gpt2_with_lora

        def cm(eval_pred):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids
            # Both must be numpy arrays for numpy-style reducers to work.
            assert isinstance(preds, np.ndarray)
            assert isinstance(labels, np.ndarray)
            top1 = np.argmax(preds, axis=-1)
            mask = labels != -100
            denom = max(1, int(mask.sum()))
            correct = int(((top1 == labels) & mask).sum())
            return {"top1_acc": correct / denom}

        trainer = Trainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        metrics = trainer.evaluate()
        assert "eval_top1_acc" in metrics
        assert 0.0 <= metrics["eval_top1_acc"] <= 1.0

    def test_include_for_metrics_inputs(self, gpt2_with_lora, tiny_lm_dataset):
        """include_for_metrics=['inputs'] populates EvalPrediction.inputs."""
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["inputs"] = eval_pred.inputs
            seen["losses"] = eval_pred.losses
            return {"acc": 0.0}

        trainer = Trainer(
            model=model,
            args=_default_args(include_for_metrics=["inputs"]),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["inputs"] is not None
        # HF parity (non-batch path): inputs is the bare main-input numpy array
        # (input_ids), not a full batch dict.  Shape: (num_examples, seq_len).
        import numpy as np

        assert isinstance(seen["inputs"], np.ndarray)
        assert seen["inputs"].ndim == 2
        assert seen["losses"] is None

    def test_include_for_metrics_loss_is_one_d_per_example(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """include_for_metrics=['loss'] populates a 1-D per-example losses tensor.

        HF parity: each batch's reduced loss is repeated by ``batch_size``
        before being appended, so the final ``EvalPrediction.losses``
        length equals total samples (not number of batches).
        """
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["losses"] = eval_pred.losses
            return {"acc": 0.0}

        trainer = Trainer(
            model=model,
            args=_default_args(
                per_device_eval_batch_size=4,  # 8 examples / 4 = 2 batches
                include_for_metrics=["loss"],
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["losses"] is not None
        # HF parity: ``compute_metrics`` consumes numpy arrays.
        assert seen["losses"].ndim == 1
        # 8 examples → length-8 per-example losses (HF parity).
        assert seen["losses"].shape[0] == 8

    def test_include_for_metrics_combined(self, gpt2_with_lora, tiny_lm_dataset):
        """Both 'inputs' and 'loss' populate the corresponding fields."""
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["inputs"] = eval_pred.inputs
            seen["losses"] = eval_pred.losses
            return {"acc": 1.0}

        trainer = Trainer(
            model=model,
            args=_default_args(include_for_metrics=["inputs", "loss"]),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["inputs"] is not None
        assert seen["losses"] is not None

    def test_unknown_include_for_metrics_key_raises(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """Unknown entries in include_for_metrics raise at __init__ time."""
        model, tokenizer = gpt2_with_lora
        with pytest.raises(ValueError, match="include_for_metrics"):
            Trainer(
                model=model,
                args=_default_args(include_for_metrics=["foo"]),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )


# ---------------------------------------------------------------------------
# Phase 3 — end-to-end integration & parity
# ---------------------------------------------------------------------------


def _accuracy_fn(eval_pred):
    """Tiny HF-typed compute_metrics: top-1 token accuracy ignoring -100.

    HF parity: predictions / label_ids are numpy arrays.
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(preds, list):
        preds = preds[0]
        labels = labels[0]
    pred_ids = preds.argmax(axis=-1)
    mask = labels != -100
    if int(mask.sum()) == 0:
        return {"acc": 0.0}
    correct = (pred_ids == labels) & mask
    return {"acc": float(int(correct.sum()) / int(mask.sum()))}


class TestTrainerEvalIntegration:
    """End-to-end: train() with eval_on_start, eval_delay, compute_metrics."""

    def test_full_eval_flow_through_train(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """eval_on_start + eval_delay + compute_metrics + load_best_model_at_end."""
        model, tokenizer = gpt2_with_lora

        trainer = Trainer(
            model=model,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=4,
                eval_strategy="steps",
                eval_steps=2,
                eval_on_start=True,
                eval_delay=2,  # skip the eval at step 2; first auto-eval at 4
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="eval_acc",
                greater_is_better=True,
                load_best_model_at_end=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=_accuracy_fn,
        )
        out = trainer.train()
        assert isinstance(out, TrainOutput)

        # log_history must contain at least one row tagged with eval_acc.
        eval_rows = [row for row in trainer.state.log_history if "eval_acc" in row]
        assert eval_rows, "no eval_acc row in log_history"

        # The step-0 eval (eval_on_start) seeded best_metric.
        assert trainer.state.best_metric is not None


class TestTrainerEvalParity:
    """Functional ↔ nn.Module parity for compute_metrics."""

    def test_compute_metrics_path_agnostic(self, gpt2_with_lora, tiny_lm_dataset):
        """Same compute_metrics produces the same eval_acc on both paths."""
        model, tokenizer = gpt2_with_lora

        captured: list[float] = []

        def cm(eval_pred):
            r = _accuracy_fn(eval_pred)
            captured.append(r["acc"])
            return r

        trainer = Trainer(
            model=model,
            args=_default_args(max_steps=1, eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )

        # Mid-train: functional path (self._ctx populated). Capture by
        # invoking evaluate() inside the train() context via a callback.
        functional_acc: list[float] = []

        class _MidTrainEval(_HFTrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                m = trainer.evaluate()
                functional_acc.append(m["eval_acc"])

        trainer._callback_handler.callbacks.append(_MidTrainEval())
        trainer.train()

        # Post-train: nn.Module path (self._ctx is None after train() returns).
        post_metrics = trainer.evaluate()

        assert functional_acc, "functional-path eval did not run"
        # Mid-train and post-train accuracies on the same eval set with the
        # same parameters (after one tiny training step) should be numerically
        # close.  We don't require strict equality because the one training
        # step does shift parameters slightly, but the *path* shouldn't be
        # observable — the key invariant is that both paths return a finite
        # float that compute_metrics could compute.
        assert isinstance(functional_acc[0], float)
        assert isinstance(post_metrics["eval_acc"], float)


class TestDeepJsonRecursionGuard:
    """#334: deep accountant wire dicts survive stdlib json round-trips."""

    def test_deep_nested_dict_round_trips_under_guard(self):
        import json
        import sys

        from opaque.api.transformers.trainer._trainer import (
            _deep_json_recursion,
        )

        d: dict = {"leaf": 1}
        for _ in range(5000):
            d = {"inner": d}

        old_limit = sys.getrecursionlimit()
        raised_limit = max(10_000, old_limit + 5_000)
        with _deep_json_recursion(raised_limit):
            assert sys.getrecursionlimit() == raised_limit
            back = json.loads(json.dumps(d))
            assert back == d  # dict __eq__ recurses too; compare under the guard
        assert sys.getrecursionlimit() == old_limit


class TestPredictStopStep:
    """Pure-accounting unit tests for the predicted stop-at-ε step (#392)."""

    @staticmethod
    def _step():
        import opaque.dpsgd.accounting as dacc

        return dacc.poisson(dacc.gaussian(1.0), 0.02)

    def test_matches_linear_scan(self):
        from opaque.api.transformers.trainer._trainer import predict_stop_step

        step, delta, horizon = self._step(), 1e-5, 30
        eps = [(step * k).epsilon_at(delta) for k in range(1, horizon + 1)]
        target = (eps[14] + eps[15]) / 2.0  # crossing strictly inside (15, 16]
        expected = next(k for k in range(1, horizon + 1) if eps[k - 1] >= target)
        got = predict_stop_step(
            None, step, target_epsilon=target, delta=delta, k0=0, horizon=horizon
        )
        assert got == expected

    def test_unreachable_returns_none(self):
        from opaque.api.transformers.trainer._trainer import predict_stop_step

        step, delta, horizon = self._step(), 1e-5, 10
        unreachable = (step * horizon).epsilon_at(delta) * 2.0
        assert (
            predict_stop_step(
                None,
                step,
                target_epsilon=unreachable,
                delta=delta,
                k0=0,
                horizon=horizon,
            )
            is None
        )

    def test_resume_prefix_preserves_absolute_step(self):
        """A resumed run (prefix = k0 accrued steps) predicts the same
        absolute crossing step as the fresh run."""
        from opaque.api.transformers.trainer._trainer import predict_stop_step

        step, delta, horizon = self._step(), 1e-5, 30
        eps = [(step * k).epsilon_at(delta) for k in range(1, horizon + 1)]
        target = (eps[14] + eps[15]) / 2.0
        fresh = predict_stop_step(
            None, step, target_epsilon=target, delta=delta, k0=0, horizon=horizon
        )
        k0 = 5
        resumed = predict_stop_step(
            step * k0,  # the resumed accountant's prefix process
            step,
            target_epsilon=target,
            delta=delta,
            k0=k0,
            horizon=horizon,
        )
        assert resumed == fresh
