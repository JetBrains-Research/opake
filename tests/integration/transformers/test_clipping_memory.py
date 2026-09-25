"""Streaming fixed clipping through real DPTrainer training and local export."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import (
    AutoModelForCausalLM,
    Qwen2Config,
    TrainerCallback,
    default_data_collator,
)

from opake.api.engine.clipping import _clipped_fun
from opake.api.transformers.trainer import _dp_trainer
from opake.transformers import DPTrainer, TrainingArguments

_STEPS = 2
_BATCH_SIZE = 3
_MICROBATCH_SIZE = 2
_CLIP_NORM = 0.1
_DIAGNOSTIC_KEYS = (
    "loss",
    "batch_size",
    "grad_norm",
    "privacy_clip_rate",
    "privacy_clipping_norm",
    "privacy_clipped_grad_norm_mean",
    "privacy_noise_std",
    "privacy_noise_multiplier",
)


def _snapshot(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    assert tensors, "expected a nonempty parameter/gradient collection"
    for name, tensor in tensors.items():
        assert tensor.numel() > 0, name
        assert torch.isfinite(tensor).all(), name
    return {name: tensor.detach().clone() for name, tensor in tensors.items()}


def _dataset(seed: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 32, (_BATCH_SIZE, 6), generator=generator)
    return [
        {
            "input_ids": row,
            "attention_mask": torch.ones_like(row),
            "labels": row.clone(),
        }
        for row in ids
    ]


class _StepTrace(TrainerCallback):
    def __init__(self, checkpointing: bool) -> None:
        self.checkpointing = checkpointing
        self.before: dict[str, torch.Tensor] = {}
        self.gradients: list[dict[str, torch.Tensor]] = []
        self.parameters: list[dict[str, torch.Tensor]] = []
        self.updates: list[dict[str, torch.Tensor]] = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        assert kwargs["model"].is_gradient_checkpointing == self.checkpointing
        self.before = _snapshot(kwargs["trainable_params"])
        self.gradients.append(_snapshot(kwargs["grads"].pytree))
        assert self.gradients[-1].keys() == self.before.keys()

    def on_optimizer_step(self, args, state, control, **kwargs):
        after = _snapshot(kwargs["trainable_params"])
        assert after.keys() == self.before.keys()
        updates = _snapshot({name: after[name] - self.before[name] for name in after})
        assert any(torch.count_nonzero(update).item() for update in updates.values())
        self.parameters.append(after)
        self.updates.append(updates)


@dataclass
class _Run:
    parameters: dict[str, torch.Tensor]
    trace: _StepTrace
    clipped: list[dict[str, torch.Tensor]]
    auxiliary: list[dict[str, torch.Tensor]]
    clip_states: list[Any]
    diagnostics: list[dict[str, float]]
    state: dict[str, Any]
    train_loss: float
    eval_loss: float


def _run_trainer(
    output_dir: Path, checkpointing: bool, monkeypatch: pytest.MonkeyPatch
) -> _Run:
    torch.manual_seed(123)
    model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
            tie_word_embeddings=False,
        )
    )
    initial = _snapshot(dict(model.named_parameters()))
    trace = _StepTrace(checkpointing)
    clipped = []
    auxiliary = []
    clip_states = []
    real_clipped_grad = _dp_trainer.clipped_grad

    def capture_clipped_grad(*args, **kwargs):
        assert kwargs["return_aux"] is True
        assert kwargs["clipping_norm"] == _CLIP_NORM
        assert kwargs["normalize_by"] == _BATCH_SIZE
        assert kwargs["microbatch_size"] == _MICROBATCH_SIZE
        grad_fn, clip_state = real_clipped_grad(*args, **kwargs)

        def capture_step(*args, **kwargs):
            result = grad_fn(*args, **kwargs)
            (grads, aux), next_state = result
            assert grads.max_norm == pytest.approx(_CLIP_NORM / _BATCH_SIZE)
            assert aux.batch_size == _BATCH_SIZE
            clipped.append(_snapshot(grads.pytree))
            auxiliary.append(
                _snapshot(
                    {
                        "loss_values": aux.loss_values,
                        "grad_norms": aux.grad_norms,
                        "clipped_grad_norms": aux.clipped_grad_norms,
                    }
                )
            )
            for values in auxiliary[-1].values():
                assert values.shape == (_BATCH_SIZE,)
            assert (aux.grad_norms > _CLIP_NORM).any()
            assert math.isfinite(aux.clipping_rate)
            assert 0.0 < aux.clipping_rate <= 1.0
            clip_states.append(copy.deepcopy(next_state))
            return result

        return capture_step, clip_state

    # q=1 guarantees a nonempty logical batch with a remainder microbatch.
    # All data are synthetic; the loss/norm diagnostics are not private releases.
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        per_device_train_batch_size=_BATCH_SIZE,
        per_device_eval_batch_size=2,
        microbatch_size=_MICROBATCH_SIZE,
        max_steps=_STEPS,
        learning_rate=1e-3,
        lr_scheduler="constant",
        clipping_mode="fixed",
        clipping_norm=_CLIP_NORM,
        privacy_noise_multiplier=0.5,
        privacy_accounting=False,
        sampling_mode="poisson",
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_performance_kernels=False,
        torch_compile=False,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        push_to_hub=False,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=456,
        data_seed=789,
    )
    with monkeypatch.context() as patch:
        patch.setattr(_dp_trainer, "clipped_grad", capture_clipped_grad)
        trainer = DPTrainer(
            model=model,
            args=args,
            train_dataset=_dataset(101),
            eval_dataset=_dataset(202),
            data_collator=default_data_collator,
            callbacks=[trace],
        )
        result = trainer.train()

    assert result.global_step == trainer.state.global_step == _STEPS
    assert math.isfinite(result.training_loss)
    assert result.training_loss > 0
    assert len(clipped) == len(auxiliary) == len(clip_states) == _STEPS
    assert len(trace.gradients) == len(trace.parameters) == len(trace.updates) == _STEPS
    parameters = _snapshot(dict(model.named_parameters()))
    assert parameters.keys() == initial.keys() == trace.parameters[-1].keys()
    torch.testing.assert_close(parameters, trace.parameters[-1], rtol=0, atol=0)
    assert any(not torch.equal(initial[name], parameters[name]) for name in initial)

    diagnostics = [
        {key: entry[key] for key in _DIAGNOSTIC_KEYS}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    assert len(diagnostics) == _STEPS
    for step, entry in enumerate(diagnostics):
        assert all(math.isfinite(value) for value in entry.values())
        assert entry["batch_size"] == _BATCH_SIZE
        assert entry["loss"] == pytest.approx(
            auxiliary[step]["loss_values"].mean().item()
        )
        assert entry["privacy_clipping_norm"] == pytest.approx(_CLIP_NORM / _BATCH_SIZE)
        assert entry["privacy_noise_std"] == pytest.approx(
            0.5 * _CLIP_NORM / _BATCH_SIZE
        )

    eval_metrics = trainer.evaluate()
    assert eval_metrics
    assert math.isfinite(eval_metrics["eval_loss"])
    assert eval_metrics["eval_loss"] > 0
    torch.testing.assert_close(
        dict(model.named_parameters()), parameters, rtol=0, atol=0
    )
    trainer.save_model()
    trainer.save_state()
    saved_state = json.loads((output_dir / "trainer_state.json").read_text())
    assert saved_state == trainer.state.to_json()
    reloaded = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True)
    restored = _snapshot(dict(reloaded.named_parameters()))
    assert restored.keys() == parameters.keys()
    torch.testing.assert_close(restored, parameters, rtol=0, atol=0)
    reloaded.eval()
    batch = default_data_collator(_dataset(202))
    with torch.no_grad():
        restored_loss = reloaded(**batch).loss.item()
    assert math.isfinite(restored_loss)
    assert restored_loss == pytest.approx(eval_metrics["eval_loss"], rel=1e-5, abs=1e-6)

    state = trainer.state.to_json()
    state.pop("log_history")  # Runtime/performance telemetry differs across runs.
    return _Run(
        parameters=parameters,
        trace=trace,
        clipped=clipped,
        auxiliary=auxiliary,
        clip_states=clip_states,
        diagnostics=diagnostics,
        state=state,
        train_loss=result.training_loss,
        eval_loss=eval_metrics["eval_loss"],
    )


@pytest.mark.parametrize("checkpointing", [False, True], ids=["eager", "checkpointed"])
def test_dptrainer_streaming_clipping_matches_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointing: bool
) -> None:
    calls = 0
    real_stream = _clipped_fun._stream_clip_and_sum

    def record_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(_clipped_fun, "_stream_clip_and_sum", record_stream)
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            _clipped_fun, "_streaming_supported", lambda *args, **kwargs: False
        )
        baseline = _run_trainer(tmp_path / "original", checkpointing, monkeypatch)
    assert calls == 0

    streaming = _run_trainer(tmp_path / "streaming", checkpointing, monkeypatch)
    assert calls >= _STEPS * math.ceil(_BATCH_SIZE / _MICROBATCH_SIZE)
    for actual, expected in (
        (streaming.parameters, baseline.parameters),
        (streaming.trace.parameters, baseline.trace.parameters),
        (streaming.trace.gradients, baseline.trace.gradients),
        (streaming.trace.updates, baseline.trace.updates),
        (streaming.clipped, baseline.clipped),
        (streaming.auxiliary, baseline.auxiliary),
        (streaming.diagnostics, baseline.diagnostics),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert streaming.clip_states == baseline.clip_states
    assert streaming.state == baseline.state
    assert streaming.train_loss == pytest.approx(
        baseline.train_loss, rel=1e-5, abs=1e-6
    )
    assert streaming.eval_loss == pytest.approx(baseline.eval_loss, rel=1e-5, abs=1e-6)
