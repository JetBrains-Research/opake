"""Reporting integration compatibility tests for Trainer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers.trainer_callback import DefaultFlowCallback, TrainerCallback

import opaque.api.transformers.trainer._callback as callback_module
import opaque.api.transformers.trainer._checkpoint as ckpt
from opaque.transformers.trainer import Trainer, TrainingArguments


class _LossModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x, labels=None):
        logits = self.linear(x)
        loss = None
        if labels is not None:
            if logits.ndim == 1:
                logits_for_loss = logits.unsqueeze(0)
                labels_for_loss = labels.reshape(1)
            else:
                logits_for_loss = logits
                labels_for_loss = labels
            loss = torch.nn.functional.cross_entropy(logits_for_loss, labels_for_loss)
        return {"loss": loss, "logits": logits}


def _dataset() -> list[dict[str, torch.Tensor]]:
    return [
        {"x": torch.tensor([1.0, 0.0, 0.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([0.0, 0.0, 1.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([1.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 1.0, 1.0, 0.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([0.0, 0.0, 1.0, 1.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([1.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
    ]


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "use_cpu": True,
        "per_device_train_batch_size": 7,
        "per_device_eval_batch_size": 7,
        "max_steps": 1,
        "num_train_epochs": 1,
        "learning_rate": 1e-3,
        "eval_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 1,
        "clipping_norm": 1.0,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "remove_unused_columns": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _has_model_weights(checkpoint_dir: str) -> bool:
    return any(
        (Path(checkpoint_dir) / name).exists()
        for name in {ckpt.SAFE_WEIGHTS_NAME, ckpt.WEIGHTS_NAME}
    )


def test_train_sets_tokenizers_parallelism_default_before_callbacks(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    records = []

    class ReportingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            records.append(os.environ.get("TOKENIZERS_PARALLELISM"))

    trainer = Trainer(
        model=_LossModel(),
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        callbacks=[ReportingCallback()],
    )

    trainer.train()

    assert records == ["false"]


def test_train_preserves_explicit_tokenizers_parallelism(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")

    trainer = Trainer(
        model=_LossModel(),
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_reporting_callbacks_receive_privacy_logs_and_functional_slots(
    tmp_path, monkeypatch
):
    records = []

    class ReportingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            records.append(
                {
                    "event": "train_begin",
                    "optimizer": kwargs["optimizer"],
                    "lr_scheduler": kwargs["lr_scheduler"],
                    "model": kwargs["model"],
                    "privacy_resolved_delta": state.privacy_resolved_delta,
                    "privacy_resolved_noise_multiplier": (
                        state.privacy_resolved_noise_multiplier
                    ),
                    "privacy_calibration_source": (state.privacy_calibration_source),
                    "state_id": id(state),
                }
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            records.append(
                {
                    "event": "log",
                    "optimizer": kwargs["optimizer"],
                    "lr_scheduler": kwargs["lr_scheduler"],
                    "logs": dict(logs or {}),
                    "state_id": id(state),
                }
            )

        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = str(
                Path(args.output_dir) / f"checkpoint-{state.global_step}"
            )
            records.append(
                {
                    "event": "save",
                    "optimizer": kwargs["optimizer"],
                    "lr_scheduler": kwargs["lr_scheduler"],
                    "checkpoint_exists": Path(checkpoint_dir).is_dir(),
                    "state_id": id(state),
                }
            )

    class UserCallback(TrainerCallback):
        pass

    def fake_reporting_callbacks(report_to):
        assert report_to == ["tensorboard"]
        return [ReportingCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LossModel(),
        args=_args(tmp_path, report_to="tensorboard"),
        train_dataset=_dataset(),
        callbacks=[UserCallback()],
    )

    callbacks = trainer.callback_handler.callbacks
    default_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, DefaultFlowCallback)
    )
    reporting_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, ReportingCallback)
    )
    user_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, UserCallback)
    )
    assert default_index < reporting_index < user_index

    trainer.train()

    assert {record["event"] for record in records} >= {"train_begin", "log", "save"}
    assert all(record["optimizer"] is None for record in records)
    assert all(record["lr_scheduler"] is None for record in records)
    assert all(record["state_id"] == id(trainer.state) for record in records)
    assert any(record.get("checkpoint_exists") for record in records)
    train_begin = next(record for record in records if record["event"] == "train_begin")
    assert train_begin["privacy_resolved_delta"] is not None
    assert train_begin["privacy_resolved_noise_multiplier"] == 1.0
    assert train_begin["privacy_calibration_source"] == "fixed"

    step_logs = [
        record["logs"]
        for record in records
        if record["event"] == "log" and "privacy_clip_rate" in record["logs"]
    ]
    assert step_logs
    for key in {
        "loss",
        "learning_rate",
        "privacy_epsilon",
        "privacy_delta",
        "privacy_clip_rate",
        "privacy_clipping_norm",
        "privacy_noise_std",
        "privacy_noise_multiplier",
        "privacy_clipped_grad_norm_mean",
    }:
        assert key in step_logs[0]
    assert "dp_epsilon" not in step_logs[0]
    assert "clipped_grad_norm" not in step_logs[0]


def test_reporting_callbacks_receive_raw_per_group_privacy_logs(tmp_path, monkeypatch):
    records = []

    class ReportingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if "privacy_group_linear_grad_norm" in (logs or {}):
                records.append(dict(logs or {}))

    def fake_reporting_callbacks(report_to):
        assert report_to == ["wandb"]
        return [ReportingCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LossModel(),
        args=_args(
            tmp_path,
            report_to="wandb",
            save_strategy="no",
            clipping_norm={"fallback": 1.0, "linear": 1.0},
        ),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert records
    logs = records[0]
    for key in {
        "privacy_group_linear_grad_norm",
        "privacy_group_linear_clip_rate",
        "privacy_group_linear_clipping_norm",
        "privacy_group_linear_noise_std",
    }:
        assert key in logs
    assert logs["privacy_group_linear_grad_norm"] >= 0.0
    assert 0.0 <= logs["privacy_group_linear_clip_rate"] <= 1.0
    assert logs["privacy_group_linear_clipping_norm"] > 0.0


def test_privacy_reporting_rewrite_uses_visual_namespace():
    rewritten = callback_module.rewrite_logs_for_reporting(
        {
            "loss": 0.5,
            "eval_loss": 0.6,
            "privacy_epsilon": 1.2,
            "privacy_noise_multiplier": 0.8,
            "privacy_group_attention_q_proj_clipping_norm": 1.0,
            "privacy_group_attention_q_proj_grad_norm": 0.75,
        }
    )

    assert rewritten["train/loss"] == 0.5
    assert rewritten["eval/loss"] == 0.6
    assert rewritten["privacy/epsilon"] == 1.2
    assert rewritten["privacy/noise_multiplier"] == 0.8
    assert rewritten["privacy/group_attention_q_proj/clipping_norm"] == 1.0
    assert rewritten["privacy/group_attention_q_proj/grad_norm"] == 0.75
    assert "train/privacy_epsilon" not in rewritten


def test_wandb_reporting_callback_rewrites_privacy_logs(tmp_path, monkeypatch):
    instances = []

    class _Run:
        def __init__(self):
            self.summary = {}

    class _Wandb:
        def __init__(self):
            self.run = _Run()
            self.payloads = []

        def log(self, payload):
            self.payloads.append(dict(payload))

    class WandbCallback(TrainerCallback):
        def __init__(self):
            self._wandb = _Wandb()
            self._initialized = True
            instances.append(self)

        def setup(self, args, state, model=None):
            self._initialized = True

    def fake_reporting_callbacks(report_to):
        assert report_to == ["wandb"]
        return [WandbCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LossModel(),
        args=_args(
            tmp_path,
            report_to="wandb",
            save_strategy="no",
            clipping_norm={"fallback": 1.0, "linear": 1.0},
        ),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert instances
    payloads = instances[0]._wandb.payloads
    assert any("privacy/clip_rate" in payload for payload in payloads)
    assert any("privacy/group_linear/grad_norm" in payload for payload in payloads)
    assert all("train/privacy_epsilon" not in payload for payload in payloads)
    assert instances[0]._wandb.run.summary["privacy/epsilon"] > 0.0

    # Run-constants land in the summary but NOT in the per-step stream — only
    # ``epsilon`` (which accumulates) and ``delta`` + ``noise_multiplier``
    # (useful for multi-run comparison stacks) stay per-step. The calibration
    # keys, ``noise_std``, and ``converged_microbatch_size`` are summary-only.
    summary = instances[0]._wandb.run.summary
    assert "privacy/noise_std" in summary
    assert "privacy/calibration_source" in summary
    for key in (
        "privacy/noise_std",
        "privacy/calibration_source",
        "privacy/calibration_noise_multiplier",
        "privacy/calibration_achieved_epsilon",
        "privacy/calibration_converged",
    ):
        assert all(key not in payload for payload in payloads), (
            f"{key} should be summary-only, not per-step"
        )
    # ``delta`` and ``noise_multiplier`` are kept in the per-step stream by
    # design (single value per run → renders as a comparison stack across runs).
    assert any("privacy/delta" in payload for payload in payloads)
    assert any("privacy/noise_multiplier" in payload for payload in payloads)


def test_artifact_callback_sees_complete_dp_checkpoint_after_save(
    tmp_path, monkeypatch
):
    uploaded_checkpoints = []

    class ArtifactCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = str(
                Path(args.output_dir)
                / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            uploaded_checkpoints.append(
                {
                    "path": checkpoint_dir,
                    "files": {p.name for p in Path(checkpoint_dir).iterdir()},
                    "has_model_weights": _has_model_weights(checkpoint_dir),
                    "optimizer": kwargs["optimizer"],
                    "lr_scheduler": kwargs["lr_scheduler"],
                }
            )

    def fake_reporting_callbacks(report_to):
        assert report_to == ["wandb"]
        return [ArtifactCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LossModel(),
        args=_args(tmp_path, report_to="wandb"),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert len(uploaded_checkpoints) == 1
    checkpoint = uploaded_checkpoints[0]
    assert Path(checkpoint["path"]).is_dir()
    assert checkpoint["optimizer"] is None
    assert checkpoint["lr_scheduler"] is None
    assert checkpoint["has_model_weights"] is True
    assert {
        ckpt.DP_ACCOUNTANT_NAME,
        ckpt.TRAINER_STATE_NAME,
        ckpt.TRAINING_ARGS_NAME,
        ckpt.DP_OPTIMIZER_NAME,
        ckpt.DP_STATE_NAME,
        ckpt.RNG_STATE_NAME,
    }.issubset(checkpoint["files"])


def test_artifact_callback_save_only_model_still_sees_privacy_metadata(
    tmp_path, monkeypatch
):
    uploaded_checkpoints = []

    class ArtifactCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = (
                Path(args.output_dir)
                / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            uploaded_checkpoints.append({p.name for p in checkpoint_dir.iterdir()})

    def fake_reporting_callbacks(report_to):
        assert report_to == ["mlflow"]
        return [ArtifactCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LossModel(),
        args=_args(tmp_path, report_to="mlflow", save_only_model=True),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert len(uploaded_checkpoints) == 1
    files = uploaded_checkpoints[0]
    assert {
        ckpt.DP_ACCOUNTANT_NAME,
        ckpt.TRAINER_STATE_NAME,
        ckpt.TRAINING_ARGS_NAME,
    }.issubset(files)
    assert ckpt.DP_STATE_NAME not in files
    assert ckpt.DP_OPTIMIZER_NAME not in files
    assert ckpt.RNG_STATE_NAME not in files


def test_tensorboard_callback_smoke_logs_without_optimizer_or_scheduler(tmp_path):
    pytest.importorskip("tensorboard")

    trainer = Trainer(
        model=_LossModel(),
        args=_args(
            tmp_path, report_to="tensorboard", logging_dir=str(tmp_path / "runs")
        ),
        train_dataset=_dataset(),
    )

    trainer.train()

    assert trainer.callback_handler.optimizer is None
    assert trainer.callback_handler.lr_scheduler is None
    assert any((tmp_path / "runs").glob("events.out.tfevents.*"))
