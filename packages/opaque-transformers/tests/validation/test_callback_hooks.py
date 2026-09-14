"""Tests for the full HF ``TrainerCallback`` hook surface in Trainer.

Covers:

- All HF hooks fire in the documented order.
- ``TrainerControl.should_training_stop`` interrupts the loop.
- ``TrainerControl.should_save`` overrides force / suppress saves.
- ``on_pre_optimizer_step`` exposes clipped+noised grads via
  ``kwargs["grads"]`` so DP-aware callbacks can compute group norms
  without touching ``param.grad``.
- User-supplied ``TrainingArguments`` is never mutated (sweep parity).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback

from opaque.transformers.trainer import Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Local fixtures (kept here so the file is self-contained for callback work).
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
    """Tiny pre-padded causal-LM dataset for Trainer integration tests.

    The four texts are tokenised and pre-padded to a fixed length by
    :func:`build_lm_dataset` so HF's ``default_data_collator`` can
    consume them without further padding.
    """
    _, tokenizer = small_model_and_tokenizer
    return build_lm_dataset(
        ["hello a", "world b", "foo c", "bar d"],
        tokenizer,
        max_length=16,
    )


def _args(tmp_path, **overrides) -> TrainingArguments:
    # ``use_cpu=True``: pin to CPU so the trainer's ``args.device``
    # resolves to CPU regardless of the host (the LoRA fixture creates
    # a CPU-resident model).
    defaults: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "max_steps": 4,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# Hook recording — assert hook surface fires in HF-compatible order.
# ---------------------------------------------------------------------------


class _RecordingCallback(TrainerCallback):
    """HF-style callback that records every hook invocation.

    Each hook is explicitly defined so it shadows the no-op declared on
    :class:`transformers.TrainerCallback`.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def _record(self, name: str, state) -> None:
        self.events.append((name, state.global_step))

    def on_init_end(self, args, state, control, **kwargs):
        self._record("on_init_end", state)

    def on_train_begin(self, args, state, control, **kwargs):
        self._record("on_train_begin", state)

    def on_train_end(self, args, state, control, **kwargs):
        self._record("on_train_end", state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._record("on_epoch_begin", state)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._record("on_epoch_end", state)

    def on_step_begin(self, args, state, control, **kwargs):
        self._record("on_step_begin", state)

    def on_step_end(self, args, state, control, **kwargs):
        self._record("on_step_end", state)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._record("on_pre_optimizer_step", state)

    def on_optimizer_step(self, args, state, control, **kwargs):
        self._record("on_optimizer_step", state)

    def on_substep_end(self, args, state, control, **kwargs):
        self._record("on_substep_end", state)

    def on_save(self, args, state, control, **kwargs):
        self._record("on_save", state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._record("on_log", state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._record("on_evaluate", state)

    def on_prediction_step(self, args, state, control, **kwargs):
        self._record("on_prediction_step", state)


class TestHookSurface:
    """The full HF hook surface fires from Trainer."""

    def test_all_hooks_fire(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        cb = _RecordingCallback()

        trainer = Trainer(
            model=model,
            args=_args(tmp_path, max_steps=2, num_train_epochs=1),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[cb],
        )
        trainer.train()

        names = [n for n, _ in cb.events]

        # Required hooks (each must fire at least once).  ``on_substep_end``
        # is intentionally absent: DP-SGD has no substep — gradient
        # accumulation is folded into the Poisson round size, so each
        # iteration is one atomic optimizer step.
        for required in (
            "on_init_end",
            "on_train_begin",
            "on_epoch_begin",
            "on_step_begin",
            "on_pre_optimizer_step",
            "on_optimizer_step",
            "on_step_end",
            "on_epoch_end",
            "on_train_end",
        ):
            assert required in names, f"{required!r} did not fire; got {names}"

        # ``on_substep_end`` must NOT fire — DP-SGD has no substep concept.
        assert "on_substep_end" not in names, (
            "on_substep_end fired but Trainer should not emit it: "
            f"hook sequence was {names}"
        )

        # Ordering invariants.
        assert names.index("on_init_end") < names.index("on_train_begin")
        assert names.index("on_train_begin") < names.index("on_epoch_begin")
        assert names.index("on_epoch_begin") < names.index("on_step_begin")
        # Per step: pre_optimizer comes before optimizer comes before step_end.
        for i in range(len(names) - 2):
            triple = names[i : i + 3]
            if triple == ["on_pre_optimizer_step", "on_optimizer_step", "on_step_end"]:
                # Standard ordering observed.
                break
        else:
            pytest.fail(
                "Did not observe pre_optimizer/optimizer/step_end triple in order; "
                f"hook sequence was {names}"
            )
        assert names.index("on_train_end") == len(names) - 1


# ---------------------------------------------------------------------------
# TrainerControl flag honoring.
# ---------------------------------------------------------------------------


class _StopAtStepCallback(TrainerCallback):
    """Sets ``should_training_stop`` after the requested step."""

    def __init__(self, stop_at: int) -> None:
        self.stop_at = stop_at

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_at:
            control.should_training_stop = True


class _ForceSaveCallback(TrainerCallback):
    """Forces ``should_save`` once at the requested step."""

    def __init__(self, force_at: int) -> None:
        self.force_at = force_at
        self.fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.force_at and not self.fired:
            control.should_save = True
            self.fired = True


class TestControlFlags:
    """``TrainerControl.should_*`` flags drive flow control."""

    def test_should_training_stop_honored(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        stopper = _StopAtStepCallback(stop_at=2)

        trainer = Trainer(
            model=model,
            args=_args(tmp_path, max_steps=10, num_train_epochs=5),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[stopper],
        )
        result = trainer.train()
        assert result.global_step == 2, (
            "Callback set should_training_stop=True at step 2 but loop continued"
        )

    def test_should_save_force_outside_cadence(
        self, lora_model, tiny_dataset, tmp_path
    ):
        """Callback can force a save outside the trainer's internal cadence."""
        model, tokenizer = lora_model
        forcer = _ForceSaveCallback(force_at=2)

        trainer = Trainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                num_train_epochs=1,
                save_strategy="no",  # internal cadence wouldn't save
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[forcer],
        )
        trainer.train()

        # save_strategy='no' but the callback forced should_save=True at step 2.
        assert (tmp_path / "checkpoint-2").is_dir()


# ---------------------------------------------------------------------------
# DP-aware kwargs on optimizer hooks.
# ---------------------------------------------------------------------------


class _GradCaptureCallback(TrainerCallback):
    def __init__(self) -> None:
        self.last_grad_keys: set[str] | None = None
        self.last_param_keys: set[str] | None = None
        self.observed_grads: int = 0
        self.observed_params: int = 0

    def on_pre_optimizer_step(self, args, state, control, *, grads=None, **kwargs):
        assert grads is not None, "grads kwarg missing on on_pre_optimizer_step"
        grad_tree = grads.pytree
        self.last_grad_keys = set(grad_tree.keys())
        self.observed_grads += 1
        # Each value must be a tensor — clipped+noised gradient.
        for v in grad_tree.values():
            assert isinstance(v, torch.Tensor)

    def on_optimizer_step(
        self, args, state, control, *, trainable_params=None, **kwargs
    ):
        assert trainable_params is not None
        self.last_param_keys = set(trainable_params.keys())
        self.observed_params += 1


class TestOptimizerHookKwargs:
    """``on_pre_optimizer_step`` / ``on_optimizer_step`` expose DP kwargs."""

    def test_grads_and_params_kwargs(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        capture = _GradCaptureCallback()

        trainer = Trainer(
            model=model,
            args=_args(tmp_path, max_steps=2, num_train_epochs=1),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[capture],
        )
        trainer.train()

        assert capture.observed_grads >= 1
        assert capture.observed_params >= 1
        # Grads and params share the same keyspace (the trainable subset).
        assert capture.last_grad_keys == capture.last_param_keys


# ---------------------------------------------------------------------------
# Args mutation hygiene — same TrainingArguments reusable across instances.
# ---------------------------------------------------------------------------


class TestArgsMutationHygiene:
    """User-supplied ``TrainingArguments`` must not be mutated by Trainer."""

    def test_args_unchanged_by_construction(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        args = _args(
            tmp_path,
            metric_for_best_model="eval_loss",
            eval_strategy="steps",
            eval_steps=2,
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=2,
        )

        # ``_n_gpu`` is an HF-internal counter populated by
        # ``TrainingArguments._setup_devices`` the first time
        # ``args.device`` is read.  It transitions from sentinel ``-1``
        # to ``{0, 1}`` on first device resolution; this is HF parity,
        # not a leak of user-supplied configuration.  Strip it from
        # the snapshot so the test asserts only on the user-facing
        # field set.
        def _user_facing(a) -> dict:
            d = dataclasses.asdict(a)
            d.pop("_n_gpu", None)
            return d

        snapshot = _user_facing(args)

        # First Trainer: validates + would-mutate label_names / greater_is_better
        # / save_strategy in the legacy implementation.
        Trainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )

        assert _user_facing(args) == snapshot, (
            "Trainer mutated user-supplied args during __init__"
        )

        # Second construction with the *same* args object must succeed and
        # leave args unchanged again.  This also exercises
        # ``TrainingArguments.__post_init__`` idempotency: ``args``
        # has already been ``__post_init__``-d once at construction
        # time, so ``Trainer`` re-using it must not re-run the
        # mutating coercions.
        Trainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )

        assert _user_facing(args) == snapshot
