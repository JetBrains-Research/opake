"""HF Trainer contract regressions for Trainer."""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers.trainer_callback import DefaultFlowCallback, TrainerCallback

import opaque.api.transformers.trainer._callback as callback_module
import opaque.api.transformers.trainer._trainer as trainer_impl
from opaque.transformers.trainer import Trainer, TrainingArguments


class _LogitsOnlyModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return {"logits": self.linear(x)}


class _FusedAwareModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.fused_requests: list[bool] = []

    def forward(self, x, labels=None, loss_only=False, **kwargs):
        self.fused_requests.append(loss_only)
        logits = self.linear(x)
        loss = (
            torch.nn.functional.cross_entropy(
                logits,
                labels,
                label_smoothing=float(kwargs.get("label_smoothing", 0.0)),
            )
            if labels is not None
            else None
        )
        return {
            "loss": loss,
            "logits": None if loss_only else logits,
        }


class _TokenClassifierModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x, labels=None):
        logits = self.linear(x)
        loss = (
            torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits}


class _CausalLMModel(_TokenClassifierModel):
    def _get_name(self):
        return "GPT2LMHeadModel"

    def forward(self, x, labels=None):
        logits = self.linear(x)
        loss = (
            torch.nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, 3),
                labels[..., 1:].reshape(-1),
            )
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits}


class _StopOnInitCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        control.should_training_stop = True
        return control


class _ReportingCallback(TrainerCallback):
    pass


class _UserCallback(TrainerCallback):
    pass


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "save_strategy": "no",
        "use_cpu": True,
        # ``_LogitsOnlyModel`` is a synthetic non-HF fixture, so it does
        # not require the registered-family compatibility patches.
        "use_compat_patches": False,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def test_constructor_accepts_hf_positional_model_and_optional_datasets(tmp_path):
    model = _LogitsOnlyModel()
    args = _args(tmp_path)

    trainer = Trainer(model, args)

    assert trainer.model is model
    assert trainer.train_dataset is None
    assert trainer.eval_dataset is None


def test_label_less_predict_returns_logits_not_loss(tmp_path):
    args = _args(tmp_path, per_device_eval_batch_size=2)
    trainer = Trainer(model=_LogitsOnlyModel(), args=args)
    dataset = [{"x": torch.zeros(4)}, {"x": torch.ones(4)}]

    output = trainer.predict(dataset)

    assert output.predictions.shape == (2, 2)
    assert output.label_ids is None
    assert "test_loss" not in output.metrics


def test_callback_returned_control_is_preserved(tmp_path):
    trainer = Trainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path),
        callbacks=[_StopOnInitCallback()],
    )

    assert trainer.control.should_training_stop is True


def test_reporting_callbacks_precede_user_callbacks_and_see_functional_slots(
    tmp_path, monkeypatch
):
    def fake_reporting_callbacks(report_to):
        assert report_to == ["tensorboard"]
        return [_ReportingCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = Trainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, report_to="tensorboard"),
        callbacks=[_UserCallback()],
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
        if isinstance(callback, _ReportingCallback)
    )
    user_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, _UserCallback)
    )

    assert default_index < reporting_index < user_index
    assert trainer.callback_handler.optimizer is None
    assert trainer.callback_handler.lr_scheduler is None


def test_full_determinism_uses_hf_deterministic_seed_helper(tmp_path, monkeypatch):
    calls = []

    def fake_enable_full_determinism(seed):
        calls.append(("full", seed))

    def fake_set_seed(seed):
        calls.append(("seed", seed))

    monkeypatch.setattr(
        trainer_impl,
        "enable_full_determinism",
        fake_enable_full_determinism,
    )
    monkeypatch.setattr(trainer_impl, "set_seed", fake_set_seed)

    Trainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, full_determinism=True, seed=123),
    )

    assert calls == [("full", 123)]


def test_supported_causal_lm_installs_inert_fused_wrapper_automatically(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    trainer = Trainer(model=LlamaForCausalLM(config), args=_args(tmp_path))
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    eager = trainer.model(input_ids=input_ids, labels=input_ids, return_dict=True)
    fused = trainer.model(
        input_ids=input_ids,
        labels=input_ids,
        return_dict=True,
        loss_only=True,
    )

    assert trainer._fused_forward_uses_marker is True
    assert eager.logits is not None
    assert fused.logits is None
    assert torch.allclose(fused.loss, eager.loss, atol=1e-4, rtol=1e-4)


def test_unsupported_causal_lm_does_not_install_generic_fused_wrapper(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=32,
    )
    trainer = Trainer(model=GPT2LMHeadModel(config), args=_args(tmp_path))
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    output = trainer.model(input_ids=input_ids, labels=input_ids, return_dict=True)

    assert trainer._fused_forward_uses_marker is False
    assert output.logits is not None
    assert output.loss is not None


def test_model_native_per_example_loss_requests_fused_loss_only(tmp_path):
    trainer = Trainer(model=_FusedAwareModel(), args=_args(tmp_path))
    requests = []

    def fmodel(params, **inputs):
        del params
        requests.append(inputs.get("loss_only", False))
        logits = torch.tensor([2.0, -1.0])
        return {"loss": logits.sum(), "logits": logits}

    inputs = {"x": torch.zeros(4), "labels": torch.tensor(0)}
    loss = trainer.compute_per_example_loss(fmodel, {}, inputs)
    loss_with_logits, logits = trainer.compute_per_example_loss(
        fmodel, {}, inputs, return_logits=True
    )

    assert requests == [True, False]
    assert torch.equal(loss, loss_with_logits)
    assert logits is not None


def test_custom_per_example_loss_keeps_logits_available(tmp_path):
    trainer = Trainer(
        model=_FusedAwareModel(),
        args=_args(tmp_path),
        compute_loss_func=lambda output, labels: output["logits"].sum() + labels * 0,
    )
    requests = []

    def fmodel(params, **inputs):
        del params
        requests.append(inputs.get("loss_only", False))
        logits = torch.tensor([2.0, -1.0])
        return {"loss": logits.sum(), "logits": logits}

    trainer.compute_per_example_loss(
        fmodel, {}, {"x": torch.zeros(4), "labels": torch.tensor(0)}
    )

    assert requests == [False]


def test_prediction_step_requests_fused_loss_only_only_without_predictions(tmp_path):
    model = _FusedAwareModel()
    trainer = Trainer(model=model, args=_args(tmp_path))
    batch = {"x": torch.randn(2, 4), "labels": torch.tensor([0, 1])}

    loss, predictions, labels = trainer.prediction_step(
        model, dict(batch), prediction_loss_only=True
    )
    assert loss is not None
    assert predictions is None
    assert labels is None
    assert model.fused_requests == [True]

    loss, predictions, labels = trainer.prediction_step(
        model, dict(batch), prediction_loss_only=False
    )
    assert loss is not None
    assert predictions is not None
    assert labels is not None
    assert model.fused_requests == [True, False]


def test_default_eval_uses_same_custom_objective_as_training(tmp_path):
    model = _FusedAwareModel()
    batch = {"x": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0])}
    dataset = [
        {"x": x, "labels": label}
        for x, label in zip(batch["x"], batch["labels"], strict=True)
    ]

    def custom_loss(output, labels):
        return 3.0 * torch.nn.functional.cross_entropy(output["logits"], labels)

    trainer = Trainer(
        model=model,
        args=_args(tmp_path, per_device_eval_batch_size=2),
        eval_dataset=dataset,
        compute_loss_func=custom_loss,
    )

    eval_loss = trainer.evaluate()["eval_loss"]
    per_example_losses = []
    for x, label in zip(batch["x"], batch["labels"], strict=True):
        per_example_losses.append(
            trainer.compute_per_example_loss(
                lambda _params, **inputs: model(**inputs),
                {},
                {"x": x, "labels": label},
            )
        )

    assert eval_loss == pytest.approx(torch.stack(per_example_losses).mean().item())
    assert model.fused_requests == [False, False, False, False, False]


def test_default_eval_applies_label_smoothing(tmp_path):
    model = _FusedAwareModel()
    trainer = Trainer(
        model=model,
        args=_args(tmp_path, label_smoothing_factor=0.2),
    )
    batch = {"x": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0])}

    loss, predictions, labels = trainer.prediction_step(
        model,
        dict(batch),
        prediction_loss_only=False,
    )

    assert loss is not None
    assert predictions is not None
    assert labels is not None
    expected = torch.nn.functional.cross_entropy(
        predictions,
        labels,
        label_smoothing=0.2,
    )
    unsmoothed = torch.nn.functional.cross_entropy(predictions, labels)
    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, unsmoothed)


def test_default_eval_does_not_shift_token_classification_labels(tmp_path):
    model = _TokenClassifierModel()
    trainer = Trainer(
        model=model,
        args=_args(tmp_path, label_smoothing_factor=0.2),
    )
    batch = {
        "x": torch.randn(2, 4, 4),
        "labels": torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]]),
    }

    loss, predictions, labels = trainer.prediction_step(
        model,
        dict(batch),
        prediction_loss_only=False,
    )

    assert loss is not None
    assert predictions is not None
    assert labels is not None
    expected = torch.nn.functional.cross_entropy(
        predictions.reshape(-1, 3),
        labels.reshape(-1),
        label_smoothing=0.2,
    )
    assert torch.allclose(loss, expected)


def test_default_eval_shifts_causal_language_model_labels(tmp_path):
    model = _CausalLMModel()
    trainer = Trainer(
        model=model,
        args=_args(tmp_path, label_smoothing_factor=0.2),
    )
    batch = {
        "x": torch.randn(2, 4, 4),
        "labels": torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]]),
    }

    loss, predictions, labels = trainer.prediction_step(
        model,
        dict(batch),
        prediction_loss_only=False,
    )

    assert loss is not None
    assert predictions is not None
    assert labels is not None
    expected = torch.nn.functional.cross_entropy(
        predictions[..., :-1, :].reshape(-1, 3),
        labels[..., 1:].reshape(-1),
        label_smoothing=0.2,
    )
    assert torch.allclose(loss, expected)


def test_custom_loss_takes_precedence_over_label_smoothing(tmp_path):
    model = _FusedAwareModel()
    trainer = Trainer(
        model=model,
        args=_args(tmp_path, label_smoothing_factor=0.2),
        compute_loss_func=lambda output, labels: output["loss"],
    )
    batch = {"x": torch.randn(2, 4), "labels": torch.tensor([0, 1])}
    expected = torch.nn.functional.cross_entropy(
        model.linear(batch["x"]), batch["labels"]
    )

    loss, _, _ = trainer.prediction_step(
        model,
        dict(batch),
        prediction_loss_only=True,
    )
    train_loss = trainer.compute_per_example_loss(
        lambda _params, **inputs: model(**inputs),
        {},
        {"x": batch["x"][0], "labels": batch["labels"][0]},
    )

    assert loss is not None
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        train_loss,
        torch.nn.functional.cross_entropy(
            model.linear(batch["x"][0]),
            batch["labels"][0],
        ),
    )
    assert model.fused_requests == [False, False]


def test_label_smoothing_recomputes_loss_from_logits_for_vector_case(tmp_path):
    trainer = Trainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, label_smoothing_factor=0.2),
    )
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    labels = torch.tensor(0)
    unsmoothed_loss = torch.nn.functional.cross_entropy(
        logits.reshape(1, -1),
        labels.reshape(1),
    )

    def fmodel(params, **kwargs):
        # ``compute_per_example_loss`` expects dict-like output (the
        # ``ModelOutput`` contract); a plain dict satisfies it.
        return {"loss": unsmoothed_loss, "logits": logits}

    loss_fn, batch_argnums = trainer._build_per_example_loss(
        fmodel,
        frozen_params={},
        batch_keys=("x", "labels"),
    )

    actual = loss_fn({}, torch.zeros(4), labels)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(1, -1),
        labels.reshape(1),
        label_smoothing=0.2,
    )

    assert batch_argnums == (1, 2)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, unsmoothed_loss)


def test_public_save_model_writes_training_args(tmp_path):
    trainer = Trainer(model=_LogitsOnlyModel(), args=_args(tmp_path))

    trainer.save_model()

    assert (tmp_path / "training_args.bin").exists()


def test_train_signature_keeps_hf_subset():
    """``train()`` keeps the HF-compatible subset of parameter names.

    HPO is removed (``trial`` not accepted, ``hyperparameter_search`` gone)
    and the deprecated ``model_path`` kwarg is dropped along with ``**kwargs``.
    """
    from transformers import Trainer as HFTrainer

    trainer_train = inspect.signature(HFTrainer.train)
    opaque_train = inspect.signature(Trainer.train)
    for name in ["resume_from_checkpoint", "ignore_keys_for_eval"]:
        assert name in opaque_train.parameters
        assert name in trainer_train.parameters
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in opaque_train.parameters.values()
    )
    assert not hasattr(Trainer, "hyperparameter_search")


def test_process_helpers_are_single_process_true(tmp_path):
    trainer = Trainer(model=_LogitsOnlyModel(), args=_args(tmp_path))

    assert trainer.is_world_process_zero()
    assert trainer.is_local_process_zero()
