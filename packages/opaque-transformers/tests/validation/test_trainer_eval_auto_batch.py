"""auto_find_microbatch_size also shrinks the eval batch on CUDA-OOM.

Eval has no gradient accumulation, so the eval batch is a pure throughput knob;
on OOM the search halves ``per_device_eval_batch_size`` and retries. OOM is
simulated by monkeypatching ``evaluation_loop`` / ``prediction_step`` so the
tests run on CPU.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from opaque.transformers.trainer import Trainer, TrainingArguments


class _TinyEvalModel(torch.nn.Module):
    main_input_name = "features"

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)

    def forward(self, features, labels=None):
        logits = self.proj(features.float())
        if labels is None:
            return {"logits": logits}
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def _dataset():
    return [
        {"features": torch.zeros(3), "labels": 0},
        {"features": torch.ones(3), "labels": 1},
        {"features": torch.full((3,), 0.5), "labels": 0},
        {"features": torch.full((3,), -0.5), "labels": 1},
    ]


def _trainer(tmp_path, **overrides):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        privacy_noise_multiplier=0.0,
        clipping_norm=1.0,
        save_strategy="no",
        seed=42,
        use_cpu=True,
        **overrides,
    )
    return Trainer(model=_TinyEvalModel(), args=args, eval_dataset=_dataset())


def test_eval_oom_halves_eval_batch_size(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, auto_find_microbatch_size=True)
    real_loop = trainer.evaluation_loop
    attempts: list[int] = []

    def flaky_loop(loader, **kwargs):
        size = trainer.args.per_device_eval_batch_size
        attempts.append(size)
        if size > 2:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_loop(loader, **kwargs)

    monkeypatch.setattr(trainer, "evaluation_loop", flaky_loop)

    metrics = trainer.evaluate()

    # OOM at batch 4, retried (and succeeds) at 2.
    assert attempts == [4, 2]
    assert trainer.args.per_device_eval_batch_size == 2
    assert "eval_loss" in metrics


def test_eval_oom_in_prediction_step_halves_batch(tmp_path, monkeypatch):
    """End-to-end: OOM from ``prediction_step`` still steps the eval batch down."""
    trainer = _trainer(tmp_path, auto_find_microbatch_size=True)
    real_step = trainer.prediction_step
    attempts: list[int] = []

    def flaky_step(*args, **kwargs):
        size = trainer.args.per_device_eval_batch_size
        attempts.append(size)
        if size > 2:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(trainer, "prediction_step", flaky_step)

    metrics = trainer.evaluate()

    assert trainer.args.per_device_eval_batch_size == 2
    assert attempts[0] == 4
    assert 2 in attempts
    assert "eval_loss" in metrics


def test_eval_oom_propagates_when_flag_off(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, auto_find_microbatch_size=False)

    def always_oom(loader, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(trainer, "evaluation_loop", always_oom)

    with pytest.raises(torch.OutOfMemoryError):
        trainer.evaluate()
    # No search → the batch size is left untouched.
    assert trainer.args.per_device_eval_batch_size == 4


def test_eval_oom_at_batch_size_one_raises(tmp_path, monkeypatch):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        privacy_noise_multiplier=0.0,
        clipping_norm=1.0,
        save_strategy="no",
        seed=42,
        use_cpu=True,
        auto_find_microbatch_size=True,
    )
    trainer = Trainer(model=_TinyEvalModel(), args=args, eval_dataset=_dataset())

    def always_oom(loader, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(trainer, "evaluation_loop", always_oom)

    with pytest.raises(torch.OutOfMemoryError, match="simulated"):
        trainer.evaluate()
    assert trainer.args.per_device_eval_batch_size == 1


def _fake_distributed(trainer) -> None:
    trainer._ddp = dataclasses.replace(trainer._ddp, is_distributed=True)


def test_eval_loop_collective_oom_when_sibling_steps_down(tmp_path, monkeypatch):
    """Surviving ranks must raise before end-of-loop DDP collectives.

    Mirrors the training-step guard: if any rank OOM'd, every rank raises a
    retryable ``OutOfMemoryError`` so nobody blocks forever in
    ``reduce_scalar`` / gather while the OOM rank has already exited.
    """
    trainer = _trainer(tmp_path, auto_find_microbatch_size=True)
    _fake_distributed(trainer)
    # Sibling OOM'd — local prediction_step succeeds.
    monkeypatch.setattr(trainer, "_cluster_needs_step_down", lambda local_oom: True)

    with pytest.raises(torch.OutOfMemoryError, match="collective eval batch retry"):
        trainer.evaluation_loop(
            trainer.get_eval_dataloader(),
            description="Evaluation",
            prediction_loss_only=True,
            ignore_keys=None,
            metric_key_prefix="eval",
        )


def test_eval_loop_local_oom_becomes_collective_under_ddp(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, auto_find_microbatch_size=True)
    _fake_distributed(trainer)
    seen_flags: list[bool] = []

    def record_and_step_down(local_oom: bool) -> bool:
        seen_flags.append(local_oom)
        return True

    monkeypatch.setattr(trainer, "_cluster_needs_step_down", record_and_step_down)

    def boom(*_args, **_kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(trainer, "prediction_step", boom)

    with pytest.raises(torch.OutOfMemoryError, match="collective eval batch retry"):
        trainer.evaluation_loop(
            trainer.get_eval_dataloader(),
            description="Evaluation",
            prediction_loss_only=True,
            ignore_keys=None,
            metric_key_prefix="eval",
        )
    assert seen_flags == [True]
