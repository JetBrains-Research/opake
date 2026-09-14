"""Regression: _restore_params must not re-derive trainability from the live
module's requires_grad flags.

The live module's requires_grad flags are not a reliable record of what was
trained under the functional training path. The old guard re-derived the
trainable set from the live module and could raise on a completing trainer run;
the guard now validates the provided keys against the model's state_dict instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.utils import SAFE_WEIGHTS_NAME

import opaque.api.transformers.trainer._checkpoint as ckpt
from opaque.accounting import Accountant
from opaque.api.engine.clipping.types import FixedClipState
from opaque.dpsgd.noise import gaussian_noise
from opaque.exceptions import ConfigurationError
from opaque.optimizers.types import ScheduleFreeState
from opaque.random import key
from opaque.serialization import state_dict as opaque_state_dict
from opaque.transformers.trainer import Trainer, TrainingArguments
from opaque.transformers.trainer.types import EvaluationResult


def _trainer(tmp_path, **overrides):
    model = nn.Linear(4, 2)
    kwargs = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    kwargs.update(overrides)
    args = TrainingArguments(**kwargs)
    return Trainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )


def _schedule_free_context(trainer):
    training = {
        name: torch.full_like(param, 2.0)
        for name, param in trainer._model.named_parameters()
    }
    published = {
        name: torch.full_like(param, 7.0)
        for name, param in trainer._model.named_parameters()
    }
    state = ScheduleFreeState(
        z={name: torch.full_like(param, 3.0) for name, param in training.items()},
        x=published,
        inner=(),
        step=2,
        beta=0.9,
    )
    return SimpleNamespace(
        trainable_params=training,
        opt_state=state,
        accounting=Accountant(),
    )


def test_restore_params_succeeds_when_module_requires_grad_neutralized(tmp_path):
    trainer = _trainer(tmp_path)
    # Simulate the post-functional state: every param requires_grad=False.
    for p in trainer._model.parameters():
        p.requires_grad_(False)

    trained = {
        name: torch.ones_like(p) for name, p in trainer._model.named_parameters()
    }
    # Must not raise (old guard raised here).
    trainer._restore_params(trained)

    for _, p in trainer._model.named_parameters():
        assert torch.equal(p, torch.ones_like(p))


def test_restore_params_rejects_foreign_keys(tmp_path):
    trainer = _trainer(tmp_path)
    bad = {"not_a_real_param": torch.zeros(1)}
    with pytest.raises(ConfigurationError, match="not present in the model"):
        trainer._restore_params(bad)


def test_evaluate_uses_schedule_free_published_params_and_restores_training_params(
    tmp_path, monkeypatch
):
    trainer = _trainer(tmp_path)
    trainer._eval_dataset = [{"x": torch.zeros(4)}]
    ctx = _schedule_free_context(trainer)
    trainer._ctx = ctx
    training = ctx.trainable_params

    def run_evaluation_loop(*args, **kwargs):
        del args, kwargs
        assert trainer._ctx.trainable_params is ctx.opt_state.x
        return EvaluationResult(
            predictions=None,
            label_ids=None,
            metrics={"eval_loss": 0.0},
            num_samples=1,
        )

    monkeypatch.setattr(trainer, "_run_evaluation_loop", run_evaluation_loop)

    assert trainer.evaluate() == {"eval_loss": 0.0}
    assert trainer._ctx.trainable_params is training


def test_predict_uses_schedule_free_published_params_and_restores_training_params(
    tmp_path, monkeypatch
):
    trainer = _trainer(tmp_path)
    ctx = _schedule_free_context(trainer)
    trainer._ctx = ctx
    training = ctx.trainable_params
    expected = EvaluationResult(
        predictions=torch.tensor([[1.0]]),
        label_ids=None,
        metrics={"test_loss": 0.0},
        num_samples=1,
    )

    def run_evaluation_loop(*args, **kwargs):
        del args, kwargs
        assert trainer._ctx.trainable_params is ctx.opt_state.x
        return expected

    monkeypatch.setattr(trainer, "_run_evaluation_loop", run_evaluation_loop)

    assert trainer.predict([{"x": torch.zeros(4)}]) is expected
    assert trainer._ctx.trainable_params is training


def test_schedule_free_checkpoint_saves_published_params_and_restores_module(tmp_path):
    trainer = _trainer(tmp_path, save_only_model=True)
    ctx = _schedule_free_context(trainer)
    trainer._ctx = ctx
    trainer.state.global_step = 1
    trainer._restore_params(ctx.trainable_params)

    checkpoint = trainer._save_checkpoint()

    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "checkpoint-1" / SAFE_WEIGHTS_NAME))
    for name, tensor in ctx.opt_state.x.items():
        torch.testing.assert_close(saved[name], tensor)
    for name, param in trainer._model.named_parameters():
        torch.testing.assert_close(param, ctx.trainable_params[name])
    assert checkpoint == str(tmp_path / "checkpoint-1")


def test_schedule_free_save_model_uses_published_params_and_restores_module(tmp_path):
    trainer = _trainer(tmp_path)
    ctx = _schedule_free_context(trainer)
    trainer._ctx = ctx
    trainer._restore_params(ctx.trainable_params)
    output_dir = tmp_path / "export"

    trainer.save_model(str(output_dir))

    from safetensors.torch import load_file

    saved = load_file(str(output_dir / SAFE_WEIGHTS_NAME))
    for name, tensor in ctx.opt_state.x.items():
        torch.testing.assert_close(saved[name], tensor)
    for name, param in trainer._model.named_parameters():
        torch.testing.assert_close(param, ctx.trainable_params[name])


def test_schedule_free_resume_reconstructs_training_iterate_from_optimizer_state(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    ctx = _schedule_free_context(trainer)
    clip_state = FixedClipState()
    _, noise_state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    ctx.clip_state = clip_state
    ctx.noise_state = noise_state

    resumed_state = ScheduleFreeState(
        z={
            name: torch.full_like(param, 5.0)
            for name, param in ctx.trainable_params.items()
        },
        x={
            name: torch.full_like(param, 1.0)
            for name, param in ctx.trainable_params.items()
        },
        inner=(),
        step=3,
        beta=0.25,
    )
    torch.save(
        opaque_state_dict(resumed_state),
        tmp_path / ckpt.DP_OPTIMIZER_NAME,
    )
    runtime = SimpleNamespace(
        clip_state=opaque_state_dict(clip_state),
        noise_state=opaque_state_dict(noise_state),
    )

    trainer._apply_runtime_state(ctx, runtime, accountant=None, ckpt_dir=str(tmp_path))

    assert isinstance(ctx.opt_state, ScheduleFreeState)
    for tensor in ctx.trainable_params.values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 4.0))
