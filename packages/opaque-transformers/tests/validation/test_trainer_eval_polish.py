"""Stage-4 eval polish: HF parity for prediction_step and EvalPrediction.inputs.

Trainer's ``prediction_step`` previously returned only
``output.logits`` — every auxiliary ``ModelOutput`` field
(``hidden_states``, ``attentions``, ``encoder_last_hidden_state``,
``image_embeds``, …) was silently dropped, breaking
``compute_metrics`` for seq2seq / encoder-decoder / vision models.
``EvalPrediction.inputs`` similarly forwarded the entire batch dict
instead of HF's ``main_input_name``-filtered single tensor.

Stage 4:

- ``prediction_step`` collapses every output field that survives
  the ``ignore_keys + ["loss"]`` filter into a tuple, then collapses
  to a bare tensor when length 1, ``None`` when length 0.
- ``ignore_keys`` defaults to ``[]``; the always-on kv_cache patch
  prevents ``past_key_values`` from being emitted in outputs.
- ``EvalPrediction.inputs`` carries only ``inputs[main_input_name]``
  (sniffed from ``model.main_input_name``; default ``"input_ids"``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import Tensor
from transformers.utils import ModelOutput

from opaque.transformers.trainer import Trainer, TrainingArguments
from opaque.transformers.trainer.types import TrainOutput

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model

if TYPE_CHECKING:
    from transformers.trainer_utils import EvalPrediction

# ---------------------------------------------------------------------------
# Test fixtures: tiny GPT-2 + tokenizer.  Reused across the integration
# tests; the prediction_step unit tests stub out the model.
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_lora_and_tokenizer():
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = gpt2_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    base = make_gpt2_model()
    base.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(base, lora_config), tokenizer


@pytest.fixture
def tiny_dataset(gpt2_lora_and_tokenizer):
    _, tokenizer = gpt2_lora_and_tokenizer
    return build_lm_dataset(
        [f"sample {i}" for i in range(8)],
        tokenizer,
        max_length=16,
    )


def _args(tmp_path, **overrides) -> TrainingArguments:
    """CPU-pinned, σ=0 args for deterministic eval runs."""
    defaults: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        # NM=0.0 = non-private; target_eps is meaningless on this path.
        "privacy_noise_multiplier": 0.0,
        "clipping_norm": 1.0,
        "max_steps": 1,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "seed": 42,
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


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


class _TupleEvalModel(torch.nn.Module):
    main_input_name = "features"

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)

    def forward(self, features, labels=None):
        logits = self.proj(features.float())
        aux = logits + 1.0
        if labels is None:
            return logits, aux
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss, logits, aux


def test_per_example_loss_only_eval_uses_scalar_closure(tmp_path):
    dataset = [
        {"features": torch.zeros(3), "labels": 0},
        {"features": torch.ones(3), "labels": 1},
    ]
    trainer = Trainer(
        model=_TinyEvalModel(),
        args=_args(tmp_path, include_for_metrics=["loss"]),
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    batch = {
        "features": torch.stack([row["features"] for row in dataset]),
        "labels": torch.tensor([row["labels"] for row in dataset]),
    }

    loss, predictions, labels = trainer.prediction_step(
        trainer.model, batch, prediction_loss_only=True
    )

    assert loss.shape == (len(dataset),)
    assert predictions is None
    assert labels is None
    assert trainer._eval_per_example_loss_only_fn is not None
    assert trainer._eval_per_example_loss_fn is None


class TestPredictionStepUnlabeledAndTupleOutputs:
    def test_auto_find_batch_size_retry_restores_model_state_and_rng(
        self,
        tmp_path,
        monkeypatch,
    ):
        dataset = [
            {"features": torch.zeros(3), "labels": 0},
            {"features": torch.ones(3), "labels": 1},
            {"features": torch.full((3,), 0.5), "labels": 0},
            {"features": torch.full((3,), -0.5), "labels": 1},
        ]
        model = _TinyEvalModel()
        trainer = Trainer(
            model=model,
            args=_args(
                tmp_path,
                auto_find_microbatch_size=True,
                per_device_train_batch_size=4,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        initial_state = {
            k: v.detach().clone() for k, v in trainer.model.state_dict().items()
        }
        torch.manual_seed(1234)
        initial_rng = torch.get_rng_state().clone()
        calls = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            assert ignore_keys_for_eval is None
            calls.append((resume_from_checkpoint, microbatch_size_override))
            if len(calls) == 1:
                with torch.no_grad():
                    for param in trainer.model.parameters():
                        param.add_(1.0)
                trainer.state.global_step = 7
                torch.manual_seed(9999)
                raise torch.OutOfMemoryError("CUDA out of memory")

            assert microbatch_size_override == 2
            assert trainer.state.global_step == 0
            for name, value in trainer.model.state_dict().items():
                assert torch.equal(value, initial_state[name])
            assert torch.equal(torch.get_rng_state(), initial_rng)
            return TrainOutput(global_step=0, training_loss=0.0, metrics={})

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)

        trainer.train()

        assert [call[1] for call in calls] == [4, 2]

    def test_auto_find_batch_size_retry_restores_requires_grad_partition(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Regression: an OOM raised mid-attempt that freezes the
        trainable params must not leak into the next attempt.  Before the
        fix the retry's ``load_state_dict(strict=False)`` restored values
        but not ``requires_grad``; the next attempt's ``make_functional``
        then captured a different trainable set and the post-training
        ``_restore_params`` guard crashed with "keys do not match the
        model's current requires_grad set" — which also aborted the
        step-N checkpoint write inside ``_save_checkpoint``.
        """
        dataset = [
            {"features": torch.zeros(3), "labels": 0},
            {"features": torch.ones(3), "labels": 1},
            {"features": torch.full((3,), 0.5), "labels": 0},
            {"features": torch.full((3,), -0.5), "labels": 1},
        ]
        model = _TinyEvalModel()
        trainer = Trainer(
            model=model,
            args=_args(
                tmp_path,
                auto_find_microbatch_size=True,
                per_device_train_batch_size=4,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainable_before = {
            name for name, p in trainer.model.named_parameters() if p.requires_grad
        }
        assert trainable_before, "fixture must have at least one trainable param"
        calls = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            calls.append(microbatch_size_override)
            if len(calls) == 1:
                # Simulate the real PEFT failure mode: an OOM raised after
                # the functional forward has frozen the trainable params.
                for p in trainer.model.parameters():
                    p.requires_grad_(False)
                raise torch.OutOfMemoryError("CUDA out of memory")
            # The retry must see the trainable partition fully restored.
            trainable_now = {
                name for name, p in trainer.model.named_parameters() if p.requires_grad
            }
            assert trainable_now == trainable_before
            return TrainOutput(global_step=0, training_loss=0.0, metrics={})

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        trainer.train()
        assert calls == [4, 2]

    def test_feature_only_training_does_not_require_input_ids(self, tmp_path):
        dataset = [
            {"features": torch.zeros(3), "labels": 0},
            {"features": torch.ones(3), "labels": 1},
            {"features": torch.full((3,), 0.5), "labels": 0},
            {"features": torch.full((3,), -0.5), "labels": 1},
        ]
        trainer = Trainer(
            model=_TinyEvalModel(),
            args=_args(
                tmp_path,
                max_steps=1,
                per_device_train_batch_size=2,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
        )

        output = trainer.train()

        assert output.global_step == 1

    def test_evaluate_without_metrics_uses_prediction_loss_only(
        self,
        tmp_path,
        monkeypatch,
    ):
        dataset = [
            {"features": torch.zeros(3), "labels": 0},
            {"features": torch.ones(3), "labels": 1},
        ]
        trainer = Trainer(
            model=_TinyEvalModel(),
            args=_args(tmp_path),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        seen = []
        original = trainer.prediction_step

        def wrapped_prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
        ):
            seen.append(prediction_loss_only)
            return original(model, inputs, prediction_loss_only, ignore_keys)

        monkeypatch.setattr(trainer, "prediction_step", wrapped_prediction_step)

        trainer.evaluate()

        assert seen == [True]

    def test_unlabeled_prediction_returns_logits_without_loss(self, tmp_path):
        dataset = [{"features": torch.zeros(3)} for _ in range(2)]
        trainer = Trainer(
            model=_TinyEvalModel(),
            args=_args(tmp_path),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        batch = next(iter(trainer.get_eval_dataloader(dataset)))
        assert batch["features"].device.type == "cpu"

        loss, logits, labels = trainer.prediction_step(
            trainer.model,
            batch,
            prediction_loss_only=False,
        )

        assert loss is None
        assert labels is None
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 2)

    def test_tuple_output_rejected_with_typeerror(self, tmp_path):
        """``forward`` returning a tuple is rejected (not silently coerced).

        Trainer's eval contract requires a dict-like ``ModelOutput``.
        Bare tuples are HF cruft from pre-``ModelOutput`` model APIs;
        wrap your forward to return a dict.
        """
        dataset = [
            {"features": torch.zeros(3), "labels": 0},
            {"features": torch.ones(3), "labels": 1},
        ]
        trainer = Trainer(
            model=_TupleEvalModel(),
            args=_args(tmp_path),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        batch = next(iter(trainer.get_eval_dataloader(dataset)))

        with pytest.raises(TypeError, match="dict-like ModelOutput"):
            trainer.prediction_step(
                trainer.model,
                batch,
                prediction_loss_only=False,
            )


# ---------------------------------------------------------------------------
# prediction_step parity: tuple-of-remaining-outputs after ignore_keys filter.
# ---------------------------------------------------------------------------


class TestPredictionStepLogitsCollapse:
    """``prediction_step`` mirrors HF's tuple/collapse semantics.

    HF builds ``logits = tuple(v for k, v in outputs.items() if k not in
    ignore_keys + ["loss"])`` then collapses to a bare tensor when
    ``len(logits) == 1``.  This must apply regardless of how many
    auxiliary fields the model returned.
    """

    def test_single_output_collapses_to_bare_tensor(
        self,
        gpt2_lora_and_tokenizer,
        tiny_dataset,
        tmp_path,
    ):
        """A model returning only ``loss`` + ``logits`` exposes a bare ``Tensor``.

        GPT-2's eval-time ``ModelOutput`` is ``{loss, logits}`` under the
        always-on kv_cache patch (which skips ``past_key_values``
        allocation).  After filtering ``["loss"]`` the tuple has length 1
        and must collapse to the raw ``logits`` tensor.
        """
        model, tokenizer = gpt2_lora_and_tokenizer
        trainer = Trainer(
            model=model,
            args=_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )

        loader = trainer.get_eval_dataloader(tiny_dataset)
        batch = next(iter(loader))
        # ``prediction_step`` is called from inside ``train()`` /
        # ``evaluate()`` normally; calling it directly post-construction
        # exercises the ``self._ctx is None`` branch (eval falls back to
        # the underlying ``nn.Module``).  Both paths share the same
        # tuple-collapse logic so this is sufficient.
        loss, logits, labels = trainer.prediction_step(
            trainer._model,
            batch,
            prediction_loss_only=False,
        )
        assert isinstance(logits, Tensor), (
            f"single-output model should expose a bare Tensor, got "
            f"{type(logits).__name__}"
        )
        # Shape: (bs, seq, vocab) for causal-LM logits.
        assert logits.ndim == 3
        assert isinstance(loss, Tensor)
        assert isinstance(labels, Tensor)

    def test_prediction_loss_only_short_circuits(
        self,
        gpt2_lora_and_tokenizer,
        tiny_dataset,
        tmp_path,
    ):
        """``prediction_loss_only=True`` skips logits/labels materialization."""
        model, tokenizer = gpt2_lora_and_tokenizer
        trainer = Trainer(
            model=model,
            args=_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )

        loader = trainer.get_eval_dataloader(tiny_dataset)
        batch = next(iter(loader))
        loss, logits, labels = trainer.prediction_step(
            trainer._model,
            batch,
            prediction_loss_only=True,
        )
        assert isinstance(loss, Tensor)
        assert logits is None
        assert labels is None


class _MultiOutputStubModel(torch.nn.Module):
    """Stub model whose ``forward`` returns a fixed ``ModelOutput``.

    Used to exercise ``prediction_step``'s logits-collection branch with
    a multi-field output (``loss + logits + hidden_states``) without
    needing a real HF model that emits one.
    """

    main_input_name = "input_ids"

    def __init__(self, output: Any) -> None:
        super().__init__()
        self._output = output
        # Trainer's optimizer setup needs at least one trainable param.
        self._noop = torch.nn.Linear(1, 1)

    def forward(self, **inputs):
        return self._output


class TestPredictionStepTupleCollapsePure:
    """Pure-Python tests for ``prediction_step``'s logits-collection branch.

    Use a stub model whose ``forward`` returns a fixed multi-output
    ``ModelOutput`` so we can hit the ``len(logits_tuple) > 1`` and
    length-1 collapse paths without finding an HF model that emits one.
    """

    def _make_fake_output(self):
        @dataclass
        class FakeOutput(ModelOutput):
            loss: Any = None
            logits: Any = None
            hidden_states: Any = None

        return FakeOutput(
            loss=torch.tensor(1.5),
            logits=torch.zeros(2, 3, 4),
            hidden_states=torch.zeros(2, 3, 8),
        )

    def test_multi_output_returns_tuple(self, tmp_path):
        """A model returning ``loss + logits + hidden_states`` exposes a tuple."""
        fake_out = self._make_fake_output()
        trainer = Trainer(
            model=_MultiOutputStubModel(fake_out),
            args=_args(tmp_path),
            train_dataset=[
                {"input_ids": torch.zeros(3, dtype=torch.long), "labels": 0}
            ],
            eval_dataset=[{"input_ids": torch.zeros(3, dtype=torch.long), "labels": 0}],
        )

        batch = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.zeros(2, 3, dtype=torch.long),
        }
        _, out_logits, _ = trainer.prediction_step(
            trainer._model,
            batch,
            prediction_loss_only=False,
            ignore_keys=[],
        )
        assert isinstance(out_logits, tuple), (
            f"multi-output model should expose a tuple, got {type(out_logits).__name__}"
        )
        assert len(out_logits) == 2
        assert torch.equal(out_logits[0], fake_out.logits)
        assert torch.equal(out_logits[1], fake_out.hidden_states)

    def test_ignore_keys_drops_auxiliary_outputs(self, tmp_path):
        """``ignore_keys=["hidden_states"]`` collapses to a bare tensor."""
        fake_out = self._make_fake_output()
        trainer = Trainer(
            model=_MultiOutputStubModel(fake_out),
            args=_args(tmp_path),
            train_dataset=[
                {"input_ids": torch.zeros(3, dtype=torch.long), "labels": 0}
            ],
            eval_dataset=[{"input_ids": torch.zeros(3, dtype=torch.long), "labels": 0}],
        )

        batch = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.zeros(2, 3, dtype=torch.long),
        }
        _, out_logits, _ = trainer.prediction_step(
            trainer._model,
            batch,
            prediction_loss_only=False,
            ignore_keys=["hidden_states"],
        )
        assert isinstance(out_logits, Tensor), (
            f"length-1 tuple should collapse to bare Tensor, got "
            f"{type(out_logits).__name__}"
        )
        assert torch.equal(out_logits, fake_out.logits)


# ---------------------------------------------------------------------------
# EvalPrediction.inputs ← model.main_input_name
# ---------------------------------------------------------------------------


class TestEvalPredictionInputsMainInputName:
    """``EvalPrediction.inputs`` carries the bare ``inputs[main_input_name]`` tensor.

    HF parity (``transformers.Trainer.evaluation_loop``,
    ``trainer.py:4687-4688``): the non-batched path collects
    ``inputs_decode = inputs[main_input_name]`` — a single tensor — and
    delivers it as ``EvalPrediction.inputs``.  No dict, no
    ``attention_mask``, no collator-side scratch columns leaked.
    """

    def test_default_main_input_name_is_input_ids(
        self,
        gpt2_lora_and_tokenizer,
        tiny_dataset,
        tmp_path,
    ):
        """For causal-LM models, ``EvalPrediction.inputs`` is the ``input_ids`` tensor.

        ``include_for_metrics=["inputs"]`` makes the trainer populate
        ``EvalPrediction.inputs`` with the bare main-input tensor (HF
        parity).  Capture it in a fake metrics fn and confirm it's a
        2-D ``input_ids`` array, not a dict with the full
        ``{input_ids, attention_mask}`` shape the collator produced.
        """
        captured: dict[str, Any] = {}

        def capture_metrics(ep: EvalPrediction):
            captured["inputs"] = ep.inputs
            return {"dummy": 0.0}

        model, tokenizer = gpt2_lora_and_tokenizer
        trainer = Trainer(
            model=model,
            args=_args(tmp_path, include_for_metrics=["inputs"]),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            compute_metrics=capture_metrics,
        )
        trainer.evaluate()

        inputs = captured.get("inputs")
        assert inputs is not None, "compute_metrics should have received inputs"
        # HF contract: bare numpy array (not a dict).
        import numpy as np

        assert isinstance(inputs, np.ndarray), (
            f"EvalPrediction.inputs must be a bare ndarray "
            f"(HF parity), got {type(inputs).__name__}"
        )
        assert inputs.ndim == 2, (
            f"EvalPrediction.inputs should be a 2-D (batch, seq_len) "
            f"input_ids tensor; got shape {inputs.shape}"
        )

    def test_main_input_name_override_propagates(
        self,
        gpt2_lora_and_tokenizer,
        tiny_dataset,
        tmp_path,
    ):
        """Vision-style models (``main_input_name='pixel_values'``) deliver pixel_values.

        We don't have a vision dataset on this CPU box, but the
        filtering logic only depends on ``model.main_input_name`` and
        ``batch.get(main_input_name)``.  Monkey-patch the attribute on
        the trainer's ``_model`` and add a synthetic ``pixel_values``
        column to the batch via a custom collator; confirm the
        captured ``EvalPrediction.inputs`` carries the
        ``pixel_values`` tensor (shape ``(N, 3, 4, 4)``).
        """
        from transformers import default_data_collator

        captured: dict[str, Any] = {}

        def capture_metrics(ep: EvalPrediction):
            captured["inputs"] = ep.inputs
            return {"dummy": 0.0}

        def vision_like_collator(examples):
            batch = default_data_collator(examples)
            # Synthesize a ``pixel_values`` column shaped (bs, 3, 4, 4).
            bs = batch["input_ids"].shape[0]
            batch["pixel_values"] = torch.zeros(bs, 3, 4, 4)
            return batch

        model, tokenizer = gpt2_lora_and_tokenizer
        trainer = Trainer(
            model=model,
            args=_args(tmp_path, include_for_metrics=["inputs"]),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            data_collator=vision_like_collator,
            compute_metrics=capture_metrics,
        )
        # Override the model's main_input_name post-construction (the
        # trainer reads it lazily inside ``evaluation_loop``, so this
        # takes effect).
        trainer._model.main_input_name = "pixel_values"

        trainer.evaluate()

        inputs = captured.get("inputs")
        assert inputs is not None
        # Bare numpy array of pixel_values (HF parity).
        import numpy as np

        assert isinstance(inputs, np.ndarray)
        assert inputs.shape[-3:] == (3, 4, 4), (
            f"vision-style main_input_name='pixel_values' should "
            f"deliver the (3, 4, 4) tensor; got shape {inputs.shape}"
        )

    def test_inputs_unset_when_include_for_metrics_omits_inputs(
        self,
        gpt2_lora_and_tokenizer,
        tiny_dataset,
        tmp_path,
    ):
        """Without ``include_for_metrics=["inputs"]``, ``EvalPrediction.inputs`` is None."""
        captured: dict[str, Any] = {}

        def capture_metrics(ep: EvalPrediction):
            captured["inputs"] = ep.inputs
            return {"dummy": 0.0}

        model, tokenizer = gpt2_lora_and_tokenizer
        trainer = Trainer(
            model=model,
            args=_args(tmp_path),  # no include_for_metrics override
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            compute_metrics=capture_metrics,
        )
        trainer.evaluate()

        # The accumulator path returns ``None`` for inputs when
        # ``include_inputs`` is False.
        assert captured.get("inputs") is None
