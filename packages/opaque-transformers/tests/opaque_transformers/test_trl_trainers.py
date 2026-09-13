# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke + contract tests for the TRL-style DP trainers (SFT / DPO).

Hermetic: builds a tiny Llama from an in-code config (no network) and runs a
couple of DP-SGD steps through ``SFTTrainer`` / ``DPOTrainer``. The autouse
patch fixture in ``conftest.py`` installs the vmap-safety runtime patches.
"""

from __future__ import annotations

import types
from operator import attrgetter

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from typing import ClassVar

import torch
from datasets import Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from opaque.alignment.dpo.loss import sequence_logp
from opaque.exceptions import ConfigurationError
from opaque.transformers.trl import (
    DPOConfig,
    DPOTrainer,
    SFTConfig,
    SFTTrainer,
)


def _tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def _tiny_qwen2() -> Qwen2ForCausalLM:
    # Qwen2 backbone is ``Qwen2Model`` (base_model_prefix ``"model"``), so the
    # PEFT-resolved fused-path prefix is ``base_model.model.model``.
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(cfg)


def _stub_tokenizer() -> types.SimpleNamespace:
    """Minimal processing_class for pre-tokenized data (no network)."""
    return types.SimpleNamespace(
        pad_token_id=0,
        pad_token="<pad>",
        eos_token="</s>",
        save_pretrained=lambda *a, **k: None,
    )


def _args(cls, tmp_path, **kw):
    return cls(
        output_dir=str(tmp_path),
        privacy_noise_multiplier=0.0,
        clipping_norm=1e9,
        per_device_train_batch_size=2,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        **kw,
    )


def _sft_dataset() -> Dataset:
    return Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5, 6]} for _ in range(8)])


def _chat_dataset() -> Dataset:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    return Dataset.from_list([{"messages": messages} for _ in range(4)])


class _ChatTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    special_tokens_map: ClassVar[dict[str, str]] = {}
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "user {{ message['content'] }} "
        "{% elif message['role'] == 'assistant' %}"
        "assistant {{ message['content'] }} "
        "{% endif %}"
        "{% endfor %}"
    )

    def save_pretrained(self, *args, **kwargs):
        return None

    def get_chat_template(self, chat_template=None, tools=None):
        return self.chat_template

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        return_dict,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        assert tokenize is True
        assert return_dict is True
        input_ids = [11, 12, 21, 22]
        result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_assistant_tokens_mask:
            result["assistant_masks"] = [0, 0, 1, 1]
        return result


def _pref_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "chosen_input_ids": [1, 2, 3, 7, 8],
                "rejected_input_ids": [1, 2, 3, 9, 10],
                "chosen_completion_mask": [0, 0, 0, 1, 1],
                "rejected_completion_mask": [0, 0, 0, 1, 1],
            }
            for _ in range(8)
        ]
    )


def _maybe_lora(use_peft: bool):
    """A tiny LoRA config (or ``None`` for full FT). Skips if PEFT is absent."""
    if not use_peft:
        return None
    lora = pytest.importorskip("peft")
    return lora.LoraConfig(target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")


# ----------------------------------------------------------------------
# Config behavior
# ----------------------------------------------------------------------
def test_sft_config_forces_remove_unused_columns_and_lr():
    cfg = SFTConfig(output_dir="x", privacy_noise_multiplier=0.0)
    assert cfg.remove_unused_columns is False
    assert cfg.learning_rate == 2e-5


def test_dpo_config_coerces_loss_type_and_defaults_weights():
    cfg = DPOConfig(output_dir="x", loss_type="sigmoid", privacy_noise_multiplier=0.0)
    assert cfg.loss_type == ["sigmoid"]
    assert cfg.loss_weights == [1.0]
    assert cfg.remove_unused_columns is False
    assert cfg.learning_rate == 1e-6

    mpo = DPOConfig(
        output_dir="x",
        loss_type=["sigmoid", "hinge"],
        privacy_noise_multiplier=0.0,
    )
    assert mpo.loss_weights == [1.0, 1.0]


def test_unsupported_param_is_absent_not_rejected():
    # ``padding_free`` is a TRL field with no DP meaning and is intentionally
    # absent, so passing it raises the standard dataclass TypeError.
    with pytest.raises(TypeError):
        DPOConfig(output_dir="x", padding_free=True, privacy_noise_multiplier=0.0)


def test_rpo_alpha_is_dropped():
    # rpo_alpha is not a field, so passing it is a standard unexpected-keyword
    # TypeError.
    with pytest.raises(TypeError):
        DPOConfig(output_dir="x", rpo_alpha=1.0, privacy_noise_multiplier=0.0)


def test_model_init_kwargs_field_present():
    # TRL-parity field for string-model loading; defaults to None on both configs.
    assert (
        DPOConfig(output_dir="x", privacy_noise_multiplier=0.0).model_init_kwargs
        is None
    )
    assert (
        SFTConfig(output_dir="x", privacy_noise_multiplier=0.0).model_init_kwargs
        is None
    )


def test_duplicate_loss_type_fails_fast():
    with pytest.raises(ValueError, match="duplicates"):
        DPOConfig(
            output_dir="x",
            loss_type=["sigmoid", "sigmoid"],
            privacy_noise_multiplier=0.0,
        )


def test_sync_ref_model_requires_reference_using_loss():
    # TR-DPO has nothing to sync toward when every head is reference-free.
    with pytest.raises(ValueError, match="sync_ref_model"):
        DPOConfig(
            output_dir="x",
            sync_ref_model=True,
            loss_type="simpo",
            privacy_noise_multiplier=0.0,
        )


def test_chosen_nll_is_a_reference_free_head():
    # ``chosen_nll`` (opaque's name for TRL's ``sft`` head) scores the policy's
    # own logp, so it is reference-free — TR-DPO has nothing to sync toward.
    with pytest.raises(ValueError, match="sync_ref_model"):
        DPOConfig(
            output_dir="x",
            sync_ref_model=True,
            loss_type="chosen_nll",
            privacy_noise_multiplier=0.0,
        )


def test_reference_free_flag_is_gone():
    # ``reference_free`` is dropped as a public flag — reference-need is derived
    # from ``loss_type``; passing it is a standard unexpected-keyword TypeError.
    with pytest.raises(TypeError):
        DPOConfig(output_dir="x", reference_free=True, privacy_noise_multiplier=0.0)


def test_robust_label_smoothing_validation():
    with pytest.raises(ValueError, match="label_smoothing"):
        DPOConfig(
            output_dir="x",
            loss_type=["robust"],
            label_smoothing=0.7,
            privacy_noise_multiplier=0.0,
        )


def test_unknown_loss_type_fails_at_dispatch(tmp_path):
    # ``aot`` has no per-example DP meaning and no head — a standard KeyError at
    # the dispatch table, not a curated rejection.
    with pytest.raises(KeyError):
        DPOTrainer(
            model=_tiny_model(),
            args=_args(DPOConfig, tmp_path, loss_type="aot", max_length=8),
            train_dataset=_pref_dataset(),
            processing_class=_stub_tokenizer(),
        )


# ----------------------------------------------------------------------
# SFT training
# ----------------------------------------------------------------------
def test_sft_chat_defaults_to_full_sequence_loss(tmp_path):
    tokenizer = _ChatTokenizer()
    original_chat_template = tokenizer.chat_template
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_chat_dataset(),
        processing_class=tokenizer,
    )

    assert tokenizer.chat_template == original_chat_template
    assert "completion_mask" not in trainer.train_dataset.column_names
    batch = trainer.data_collator([trainer.train_dataset[0]])
    assert torch.equal(batch["labels"], batch["input_ids"])


def test_sft_assistant_only_loss_masks_chat_prompt(tmp_path):
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            assistant_only_loss=True,
        ),
        train_dataset=_chat_dataset(),
        processing_class=_ChatTokenizer(),
    )

    batch = trainer.data_collator([trainer.train_dataset[0]])
    assert batch["labels"].tolist() == [[-100, -100, 21, 22]]


def test_sft_assistant_only_loss_requires_chat_dataset(tmp_path):
    with pytest.raises(ConfigurationError, match="requires a conversational dataset"):
        SFTTrainer(
            model=_tiny_model(),
            args=_args(
                SFTConfig,
                tmp_path,
                max_length=8,
                loss_type="nll",
                assistant_only_loss=True,
            ),
            train_dataset=_sft_dataset(),
            processing_class=_stub_tokenizer(),
        )


def test_sft_completion_only_loss_requires_prompt_completion_dataset(tmp_path):
    with pytest.raises(
        ConfigurationError, match="requires a prompt-completion dataset"
    ):
        SFTTrainer(
            model=_tiny_model(),
            args=_args(
                SFTConfig,
                tmp_path,
                max_length=8,
                loss_type="nll",
                completion_only_loss=True,
            ),
            train_dataset=_chat_dataset(),
            processing_class=_ChatTokenizer(),
        )


@pytest.mark.parametrize(
    ("completion_only_loss", "expected_labels"),
    [(None, [1, 2, 3, 4]), (True, [-100, -100, 3, 4])],
)
def test_sft_pretokenized_completion_mask_requires_explicit_option(
    tmp_path, completion_only_loss, expected_labels
):
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 3, 4],
                "completion_mask": [0, 0, 1, 1],
            }
            for _ in range(4)
        ]
    )
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            completion_only_loss=completion_only_loss,
        ),
        train_dataset=dataset,
        processing_class=_stub_tokenizer(),
    )

    batch = trainer.data_collator([trainer.train_dataset[0]])
    assert batch["labels"].tolist() == [expected_labels]


@pytest.mark.slow
def test_sft_chunked_nll_trains(tmp_path):
    # chunked_nll lets the model compute its own (fused, logits-free) loss.
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="chunked_nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


@pytest.mark.parametrize(
    "loss_type",
    [
        "nll",
        "dft",
    ],
)
def test_sft_trains_a_couple_steps(tmp_path, loss_type):
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type=loss_type),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_sft_logs_train_telemetry(tmp_path):
    # SFT rides the same (loss, aux) seam: entropy + mean_token_accuracy over the
    # supervised tokens are logged each step.
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.train()
    logged = set().union(*(row.keys() for row in trainer.state.log_history))
    assert "entropy" in logged
    assert "mean_token_accuracy" in logged


def test_sft_log_completion_metrics_false_skips_completion_keys(tmp_path):
    # log_completion_metrics=False trains fine and emits NO completion-metric
    # telemetry; the default (True) emits it (covered above).
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))
    logged = set().union(*(row.keys() for row in trainer.state.log_history))
    assert "entropy" not in logged
    assert "mean_token_accuracy" not in logged


def test_sft_activation_offloading_inherited_base_field(tmp_path):
    # ``activation_offloading`` is inherited from the base ``TrainingArguments``
    # — the config accepts it on ``SFTConfig`` and the base ``DPTrainer`` reader
    # sees the same flag; no SFT-side override.
    args = _args(
        SFTConfig, tmp_path, max_length=8, loss_type="nll", activation_offloading=True
    )
    assert args.activation_offloading is True
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=args,
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    # The reader in DPTrainer._setup_training reads ``args.activation_offloading``.
    assert trainer.args.activation_offloading is True
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_sft_eos_token_honored_when_set_else_tokenizer(tmp_path):
    # The ``eos_token`` field is meaningful for plain-text rows: when explicitly
    # set, that token's id is appended; when unset, the tokenizer's eos is used.
    text_rows = Dataset.from_list([{"text": "hello world"} for _ in range(4)])

    class _TextTokenizer:
        # Tiny tokenizer: split words on whitespace; the (no-whitespace) eos
        # string is appended directly to the text, so peel it off the tail.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"

        _vocab: ClassVar[dict[str, int]] = {
            "hello": 5,
            "world": 6,
            "</s>": 2,
            "<eos2>": 3,
        }
        _specials: ClassVar[tuple[str, ...]] = ("</s>", "<eos2>")

        def save_pretrained(self, *a, **k):
            return None

        def __call__(
            self, text, add_special_tokens=True, truncation=False, max_length=None
        ):
            trailing = []
            for special in self._specials:
                if text.endswith(special):
                    text = text[: -len(special)]
                    trailing = [self._vocab[special]]
                    break
            ids = [self._vocab[tok] for tok in text.split()] + trailing
            if max_length is not None and truncation:
                ids = ids[:max_length]
            return {"input_ids": ids}

    # Default: falls back to tokenizer.eos_token ("</s>" -> id 2).
    args_default = _args(SFTConfig, tmp_path, max_length=16, loss_type="nll")
    tok_default = _TextTokenizer()
    trainer_default = SFTTrainer(
        model=_tiny_model(),
        args=args_default,
        train_dataset=text_rows,
        processing_class=tok_default,
    )
    assert trainer_default.train_dataset[0]["input_ids"][-1] == 2

    # Explicit eos_token overrides the tokenizer's eos ("<eos2>" -> id 3).
    args_explicit = _args(
        SFTConfig, tmp_path, max_length=16, loss_type="nll", eos_token="<eos2>"
    )
    tok_explicit = _TextTokenizer()
    trainer_explicit = SFTTrainer(
        model=_tiny_model(),
        args=args_explicit,
        train_dataset=text_rows,
        processing_class=tok_explicit,
    )
    assert trainer_explicit.train_dataset[0]["input_ids"][-1] == 3


def test_sft_metrics_seam_failsafe_on_missing_logits(tmp_path):
    # Fail-safe: when the forward yields no logits (the CUDA fused logits-free
    # path), the telemetry dict is empty rather than crashing. On the CPU eager
    # fallback logits are present, so this is exercised with a stub forward.
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="chunked_nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )

    def fake_fmodel(_params, **_kw):
        return {"loss": torch.tensor(1.23), "logits": None}

    inputs = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 2, 3]),
    }
    loss, aux = trainer.compute_per_example_loss_and_metrics(fake_fmodel, {}, inputs)
    assert aux == {}
    assert float(loss) == pytest.approx(1.23)


def test_sft_honors_custom_compute_loss_func(tmp_path):
    # A custom per-example compute_loss_func(outputs, labels) -> scalar is routed
    # through the vmap path.
    torch.manual_seed(0)
    called = {"n": 0}

    def custom_loss(outputs, labels):
        called["n"] += 1
        logits = outputs.logits[..., :-1, :]
        tgt = labels[..., 1:]
        mask = tgt != -100
        tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.clamp(min=0).reshape(-1),
            reduction="none",
        ).reshape(tgt.shape)
        return (tok * mask).sum() / mask.sum().clamp(min=1)

    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
        compute_loss_func=custom_loss,
    )
    out = trainer.train()
    assert out.global_step == 2
    assert called["n"] > 0  # the custom loss actually ran
    assert torch.isfinite(torch.tensor(out.training_loss))


@pytest.mark.parametrize("loss_type", ["dft", "chunked_nll"])
def test_sft_self_reducing_loss_rejects_custom_loss_func(tmp_path, loss_type):
    # dft / chunked_nll compute their own loss; a custom loss has no logits to
    # work with, so it is rejected at construction (TRL parity).
    with pytest.raises(ValueError, match="custom compute_loss_func"):
        SFTTrainer(
            model=_tiny_model(),
            args=_args(SFTConfig, tmp_path, max_length=8, loss_type=loss_type),
            train_dataset=_sft_dataset(),
            processing_class=_stub_tokenizer(),
            compute_loss_func=lambda o, _labels: o.logits.sum(),
        )


# ----------------------------------------------------------------------
# DPO training
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("loss_type", "needs_ref"),
    [
        ("sigmoid", True),
        ("ipo", True),
        (["sigmoid", "hinge"], True),
        ("simpo", False),
        ("cpo", False),
        ("orpo", False),
    ],
)
def test_dpo_trains_a_couple_steps(tmp_path, loss_type, needs_ref):
    # Reference-using heads pass a ref_model; reference-free heads (simpo / cpo /
    # orpo) pass none and must train all the same.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model() if needs_ref else None,
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type=loss_type),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    # Reference-free runs attach no ref columns.
    cols = trainer.train_dataset.column_names
    assert ("ref_chosen_logps" in cols) is needs_ref
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_dpo_mixed_normalization_mpo_trains(tmp_path):
    # An MPO list mixing a summed (sigmoid) and a length-normalized (ipo) head
    # must work — normalization is per-head, not a single run-wide flag.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type=["sigmoid", "ipo"]),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_dpo_tr_dpo_syncs_reference(tmp_path, monkeypatch):
    # TR-DPO: full FT, reference recomputed per step from an EMA reference that
    # tracks the policy. With sync_steps=1 the reference must move during training.
    from opaque.api.alignment.dpo.reference import _precompute

    monkeypatch.setattr(_precompute.tempfile, "gettempdir", lambda: str(tmp_path))
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            sync_ref_model=True,
            ref_model_sync_steps=1,
            ref_model_mixup_alpha=0.5,
        ),
        train_dataset=_pref_dataset(),
        eval_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    name, param = next(iter(trainer._tr_ref.named_parameters()))
    before = param.detach().clone()
    out = trainer.train()
    after = dict(trainer._tr_ref.named_parameters())[name].detach()
    assert out.global_step == 2
    assert not torch.allclose(before, after)  # EMA moved the reference
    assert not (tmp_path / "opaque_ref_cache").exists()
    # Eval scores against the current EMA reference (exercises _inject_tr_ref_logps
    # in prediction_step), not the stale seed columns.
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    assert "eval_rewards/accuracies" in metrics


def test_dpo_logs_train_reward_metrics(tmp_path):
    # Reward telemetry rides the clipped-grad aux channel and is logged each step.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.train()
    logged = set().union(*(row.keys() for row in trainer.state.log_history))
    # Full TRL-parity logged set: rewards/* plus logps/*, logits/*, entropy and
    # mean_token_accuracy, all riding the same clipped-grad aux channel.
    for key in (
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
        "logits/chosen",
        "logits/rejected",
        "entropy",
        "mean_token_accuracy",
    ):
        assert key in logged, f"missing train telemetry: {key}"


def test_dpo_log_completion_metrics_off_skips_logits_telemetry(tmp_path):
    # With log_completion_metrics=False the logits-consuming telemetry
    # (entropy / mean_token_accuracy / logits/*) is not computed, but the free
    # reward + logp telemetry still rides the aux channel.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.train()
    logged = set().union(*(row.keys() for row in trainer.state.log_history))
    assert "rewards/chosen" in logged
    assert "logps/chosen" in logged
    for key in ("entropy", "mean_token_accuracy", "logits/chosen", "logits/rejected"):
        assert key not in logged, f"unexpected logits telemetry when off: {key}"


def test_dpo_evaluate_logs_reward_metrics(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        eval_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    # The same telemetry dict aggregates symmetrically at eval (eval_* prefix).
    for key in (
        "eval_rewards/accuracies",
        "eval_rewards/chosen",
        "eval_logps/chosen",
        "eval_logits/chosen",
        "eval_entropy",
        "eval_mean_token_accuracy",
    ):
        assert key in metrics, f"missing eval telemetry: {key}"


def test_dpo_reference_free_trains_without_precompute(tmp_path):
    # A reference-free loss_type (here ORPO) needs no ref_model and skips the
    # reference precompute entirely — no ref_* columns are attached.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="orpo"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    cols = trainer.train_dataset.column_names
    assert "ref_chosen_logps" not in cols
    assert "ref_rejected_logps" not in cols
    out = trainer.train()
    assert out.global_step == 2


def test_dpo_wpo_weighting_trains(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            use_weighting=True,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2


# ----------------------------------------------------------------------
# Fused logits-free policy-logp path (telemetry off → _use_fused_logp).
# Regression: under LoRA the backbone prefix must reach the backbone
# (Qwen2Model, returns last_hidden_state), not the inner causal-LM.
# ----------------------------------------------------------------------
def _fused_dpo_trainer(tmp_path, use_peft):
    return DPOTrainer(
        model=_tiny_qwen2(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="simpo",  # reference-free: no precompute, fused-eligible
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
        peft_config=_maybe_lora(use_peft),
    )


@pytest.mark.parametrize("use_peft", [False, True])
def test_dpo_fused_logp_resolves_backbone(tmp_path, use_peft):
    # The PEFT-wrap regression: handles resolve on the FINAL model so the prefix
    # reaches the backbone (Qwen2Model, returns last_hidden_state), not the inner
    # causal-LM. CPU-runnable — no Triton kernel involved.
    torch.manual_seed(0)
    trainer = _fused_dpo_trainer(tmp_path, use_peft)
    assert trainer._fused_logp_eligible
    assert trainer._use_fused_logp
    assert trainer._is_peft is use_peft

    backbone = attrgetter(trainer._backbone_prefix)(trainer.model)
    assert type(backbone).__name__ == "Qwen2Model"
    params = dict(trainer.model.named_parameters())
    assert trainer._lm_head_param_name in params

    device = next(trainer.model.parameters()).device
    ids = torch.tensor([1, 2, 3, 7, 8], device=device)
    attn = torch.ones_like(ids)
    # Backbone forward yields per-token hidden states (T, H) — the crash site.
    hidden = trainer._last_hidden_state(params, ids, attn)
    assert hidden.shape == (ids.shape[0], trainer.model.config.hidden_size)


@pytest.mark.cuda
@pytest.mark.parametrize("use_peft", [False, True])
def test_dpo_fused_logp_matches_eager(tmp_path, use_peft):
    # The fused summed logp routes through the CUDA-only Triton linear-CE kernel;
    # it must match the eager sequence_logp to the bit.
    torch.manual_seed(0)
    trainer = _fused_dpo_trainer(tmp_path, use_peft)
    params = dict(trainer.model.named_parameters())
    device = next(trainer.model.parameters()).device
    ids = torch.tensor([1, 2, 3, 7, 8], device=device)
    attn = torch.ones_like(ids)
    cmask = torch.tensor([0, 0, 0, 1, 1], device=device)

    fused = trainer._fused_logp(None, params, ids, attn, cmask)
    logits = torch.func.functional_call(
        trainer.model,
        params,
        (),
        {"input_ids": ids.unsqueeze(0), "attention_mask": attn.unsqueeze(0)},
    ).logits.squeeze(0)
    eager = sequence_logp(logits, ids, cmask)
    assert torch.allclose(fused, eager, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("use_peft", [False, True])
def test_dpo_fused_path_trains_end_to_end(tmp_path, use_peft):
    # The fused logp runs through the full DP vmap seam for a couple of steps.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_qwen2(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="simpo",
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
        peft_config=_maybe_lora(use_peft),
    )
    assert trainer._use_fused_logp
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


# ----------------------------------------------------------------------
# DP-purity: the per-example loss for example i depends only on example i.
# ----------------------------------------------------------------------
def _to_device(trainer, batch):
    device = next(trainer.model.parameters()).device
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _per_example_loss_inputs(trainer, batch):
    from opaque.functional import make_functional

    # The collator emits CPU tensors; move to the model device (the trainer's
    # _prepare_input does this in the real path — bypassed here).
    batch = _to_device(trainer, batch)
    fmodel, trainable, frozen = make_functional(trainer.model, partition_trainable=True)
    keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]

    def fn(tp, *batch_args):
        merged = {**frozen, **tp}
        return trainer.compute_per_example_loss(
            fmodel, merged, dict(zip(keys, batch_args, strict=False))
        )

    return fn, trainable, [batch[k] for k in keys], keys


def _per_example_losses_and_gradients(trainer, batch):
    """Return vmapped DPO losses and parameter gradients for every pair."""
    fn, trainable, batch_args, keys = _per_example_loss_inputs(trainer, batch)
    vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * len(keys))
    vmapped_grad = torch.vmap(torch.func.grad(fn), in_dims=(None,) + (0,) * len(keys))
    return (
        vmapped(trainable, *batch_args),
        vmapped_grad(trainable, *batch_args),
        keys,
    )


def _per_example_losses(trainer, batch):
    """vmap ``compute_per_example_loss`` over a collated batch (the DP path)."""
    fn, trainable, batch_args, keys = _per_example_loss_inputs(trainer, batch)
    vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * len(keys))
    return vmapped(trainable, *batch_args), keys


def test_sft_dp_purity_per_example_independence(tmp_path):
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = trainer.data_collator(rows)
    losses0, _ = _per_example_losses(trainer, batch)

    # Perturb only example 0's tokens; examples 1..3 must be untouched.
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["input_ids"][0, 1] = (batch2["input_ids"][0, 1] + 1) % 64
    losses1, _ = _per_example_losses(trainer, batch2)

    assert not torch.allclose(losses0[0], losses1[0])
    assert torch.allclose(losses0[1:], losses1[1:])


@pytest.mark.parametrize(
    ("loss_type", "loss_kwargs"),
    [
        ("sigmoid", {}),
        ("sigmoid", {"f_divergence_type": "forward_kl"}),
        ("sigmoid", {"ld_alpha": 0.5}),
        ("sigmoid", {"use_weighting": True}),
        ("ipo", {}),
        ("simpo", {}),
        ("bco_pair", {}),
    ],
)
def test_dpo_dp_purity_per_example_independence(tmp_path, loss_type, loss_kwargs):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig, tmp_path, max_length=8, loss_type=loss_type, **loss_kwargs
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = trainer.data_collator(rows)
    losses0, gradients0, _ = _per_example_losses_and_gradients(trainer, batch)

    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["chosen_input_ids"][0, 3] = (batch2["chosen_input_ids"][0, 3] + 1) % 64
    losses1, gradients1, _ = _per_example_losses_and_gradients(trainer, batch2)

    assert not torch.allclose(losses0[0], losses1[0])
    assert torch.allclose(losses0[1:], losses1[1:])
    for name in gradients0:
        assert torch.allclose(gradients0[name][1:], gradients1[name][1:]), name


# ----------------------------------------------------------------------
# Numeric parity: the trainer wires the primitives correctly.
# ----------------------------------------------------------------------
def test_sft_loss_matches_direct_nll(tmp_path):
    from opaque.alignment.sft.loss import nll_loss

    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        out = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        expected = nll_loss(out.logits, batch["labels"])
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_loss_matches_direct_sigmoid(tmp_path):
    from opaque.alignment.dpo.loss import sequence_logp, sigmoid_loss

    torch.manual_seed(0)
    beta = 0.1
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid", beta=beta),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        expected = sigmoid_loss(
            c_lp - batch["ref_chosen_logps"],
            r_lp - batch["ref_rejected_logps"],
            beta=beta,
        )
    assert torch.allclose(losses, expected, atol=1e-4)


def _completion_len(mask: torch.Tensor) -> torch.Tensor:
    """Mirror of DPOTrainer._completion_len (shifted completion-token count)."""
    return (mask[..., 1:] != 0).sum(-1).clamp(min=1)


def test_dpo_simpo_loss_matches_formula(tmp_path):
    # SimPO: -logσ(β·(c_avg − r_avg) − γ) with label smoothing, on the
    # length-normalized, reference-free policy logps.
    from opaque.alignment.dpo.loss import sequence_logp, simpo_loss

    torch.manual_seed(0)
    beta, gamma, eps = 2.0, 0.5, 0.1
    trainer = DPOTrainer(
        model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="simpo",
            beta=beta,
            simpo_gamma=gamma,
            label_smoothing=eps,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        c_avg = c_lp / _completion_len(batch["chosen_completion_mask"])
        r_avg = r_lp / _completion_len(batch["rejected_completion_mask"])
        expected = simpo_loss(c_avg, r_avg, beta=beta, gamma=gamma, label_smoothing=eps)
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_cpo_loss_matches_formula(tmp_path):
    # CPO: sigmoid_loss(c_sum, r_sum, β) + cpo_alpha · meanNLL(chosen), with
    # meanNLL = −c_sum / completion_len (per-token mean).
    from opaque.alignment.dpo.loss import sequence_logp, sigmoid_loss

    torch.manual_seed(0)
    beta, cpo_alpha = 0.1, 0.7
    trainer = DPOTrainer(
        model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="cpo",
            beta=beta,
            cpo_alpha=cpo_alpha,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        c_avg = c_lp / _completion_len(batch["chosen_completion_mask"])
        expected = sigmoid_loss(c_lp, r_lp, beta=beta) + cpo_alpha * (-c_avg)
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_orpo_loss_matches_formula(tmp_path):
    # ORPO: meanNLL(chosen) + orpo_lambda · odds_ratio_loss(c_norm, r_norm) on
    # length-normalized, reference-free policy logps.
    from opaque.alignment.dpo.loss import odds_ratio_loss, sequence_logp

    torch.manual_seed(0)
    orpo_lambda = 0.3
    trainer = DPOTrainer(
        model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="orpo",
            orpo_lambda=orpo_lambda,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        c_avg = c_lp / _completion_len(batch["chosen_completion_mask"])
        r_avg = r_lp / _completion_len(batch["rejected_completion_mask"])
        expected = (-c_avg) + orpo_lambda * odds_ratio_loss(c_avg, r_avg)
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_precompute_attaches_reference_columns(tmp_path):
    # The reference precompute should add the constant ref logp columns the
    # collator emits; verified indirectly by a successful non-reference-free run
    # and directly here on the prepared dataset.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    cols = trainer.train_dataset.column_names
    assert "ref_chosen_logps" in cols
    assert "ref_rejected_logps" in cols


def test_dpo_model_cache_identity_tracks_reference_state():
    from opaque.api.transformers.trl._dpo_trainer import (
        _model_cache_identity,
        _tensor_state_digest,
    )

    torch.manual_seed(0)
    reference = _tiny_model()
    equivalent = _tiny_model()
    equivalent.load_state_dict(reference.state_dict())

    identity = _model_cache_identity(reference, adapter_mode="explicit")
    assert identity == _model_cache_identity(equivalent, adapter_mode="explicit")

    with torch.no_grad():
        next(equivalent.parameters()).add_(1)
    assert identity != _model_cache_identity(equivalent, adapter_mode="explicit")

    scalar_state = torch.nn.Module()
    scalar_state.register_buffer("scalar", torch.tensor(1.0))
    assert _tensor_state_digest(scalar_state)

    bf16_state = torch.nn.Module()
    bf16_state.register_buffer("buffer", torch.tensor([1.0], dtype=torch.bfloat16))
    assert _tensor_state_digest(bf16_state)


def test_dpo_disabled_adapter_identity_ignores_adapter_weights():
    from opaque.api.transformers.trl._dpo_trainer import _model_cache_identity

    model = _tiny_peft_model()
    disabled_identity = _model_cache_identity(model, adapter_mode="disabled")
    explicit_identity = _model_cache_identity(model, adapter_mode="explicit")

    adapter_parameter = next(
        parameter for name, parameter in model.named_parameters() if ".lora_" in name
    )
    with torch.no_grad():
        adapter_parameter.add_(1)

    assert disabled_identity == _model_cache_identity(model, adapter_mode="disabled")
    assert explicit_identity != _model_cache_identity(model, adapter_mode="explicit")

    explicit_identity = _model_cache_identity(model, adapter_mode="explicit")
    model.disable_adapter_layers()
    assert explicit_identity != _model_cache_identity(model, adapter_mode="explicit")


# ----------------------------------------------------------------------
# Reference loading: string ref_model + model_init_kwargs threading
# ----------------------------------------------------------------------
def test_dpo_string_ref_model_loads_and_trains(tmp_path):
    # A string ref_model is loaded via AutoModelForCausalLM.from_pretrained,
    # attaches the ref columns, and trains.
    torch.manual_seed(0)
    ref_dir = tmp_path / "ref"
    _tiny_model().save_pretrained(str(ref_dir))

    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=str(ref_dir),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    cols = trainer.train_dataset.column_names
    assert "ref_chosen_logps" in cols
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_dpo_model_init_kwargs_reach_reference(tmp_path, monkeypatch):
    # model_init_kwargs (here a dtype) is threaded into the string-ref load — the
    # reference instantiates with the requested dtype. Spy on from_pretrained to
    # capture the model it returns and assert a param dtype.
    import transformers

    captured = {}
    orig = transformers.AutoModelForCausalLM.from_pretrained

    def spy(path, *a, **kw):
        model = orig(path, *a, **kw)
        captured["kwargs"] = kw
        captured["dtype"] = next(model.parameters()).dtype
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)

    torch.manual_seed(0)
    ref_dir = tmp_path / "ref"
    _tiny_model().save_pretrained(str(ref_dir))

    DPOTrainer(
        model=_tiny_model(),  # in-memory policy → no extra from_pretrained
        ref_model=str(ref_dir),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            model_init_kwargs={"torch_dtype": torch.float64},
            # Pin to CPU: float64 verifies the kwarg threading but MPS (CI's
            # Apple-silicon lane) can't hold a float64 tensor, so the reference's
            # ``.to(device)`` would raise there. The dtype path is device-agnostic.
            use_cpu=True,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert captured["kwargs"].get("torch_dtype") is torch.float64
    assert captured["dtype"] is torch.float64


def test_dpo_no_reference_available_raises_early(tmp_path):
    # An in-memory policy with no path, no ref_model, not PEFT, reference-using
    # loss → fail before tokenize/precompute, pointing at reference-free heads.
    with pytest.raises(ValueError, match="reference-free loss_type"):
        DPOTrainer(
            model=_tiny_model(),
            args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
            train_dataset=_pref_dataset(),
            processing_class=_stub_tokenizer(),
        )


# ----------------------------------------------------------------------
# Fused logits-free path: eligibility gating + CPU fallback-equivalence.
# On CPU the fused primitives' ``lce_available`` is False, so they fall back to
# the eager ``hidden @ Wᵀ`` projection — numerically identical to the eager
# logits path. These tests assert that equivalence and that an ineligible
# config keeps the eager path.
# ----------------------------------------------------------------------
def _tiny_model_tied() -> LlamaForCausalLM:
    """A tiny Llama with tied input/output embeddings (no ``lm_head.weight``)."""
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg)


# ---- SFT: eligibility gating -----------------------------------------
def test_sft_fused_eligible_when_telemetry_off(tmp_path):
    # nll + telemetry off → routes through the model-level fused forward.
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_loss_eligible is True
    assert trainer._fused_nll is True
    assert trainer._fused_dft is False
    # The kernel config is flipped on so the model computes its own NLL.
    assert trainer.args.performance_kernels_config["fused_linear_cross_entropy"] is True


def test_sft_dft_fused_eligible_resolves_lm_head(tmp_path):
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="dft",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_dft is True
    assert trainer._fused_nll is False
    assert trainer._backbone_prefix == "model"
    assert trainer._lm_head_param_name == "lm_head.weight"


@pytest.mark.parametrize("loss_type", ["nll", "dft"])
def test_sft_telemetry_on_keeps_eager(tmp_path, loss_type):
    # Telemetry on (the default) is logits-consuming → no fused path.
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type=loss_type),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_loss_eligible is False
    assert trainer._fused_nll is False
    assert trainer._fused_dft is False
    # No fused-CE forced for an eager nll/dft run.
    assert not (trainer.args.performance_kernels_config or {}).get(
        "fused_linear_cross_entropy", False
    )


def test_sft_custom_loss_func_keeps_eager(tmp_path):
    # A custom compute_loss_func needs the logits → ineligible even telemetry-off.
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
        compute_loss_func=lambda outputs, labels: outputs.logits.sum() * 0.0,
    )
    assert trainer._fused_loss_eligible is False
    assert trainer._fused_nll is False


# ---- SFT: CPU fallback-equivalence -----------------------------------
@pytest.mark.parametrize("model_factory", [_tiny_model, _tiny_model_tied])
def test_sft_fused_nll_matches_eager_on_cpu(tmp_path, model_factory):
    # The fused (model-level) NLL path == the eager nll_loss on logits, on CPU.
    from opaque.alignment.sft.loss import nll_loss

    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=model_factory(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="nll",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_nll is True
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    fused_losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        out = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        expected = nll_loss(out.logits, batch["labels"])
    assert torch.allclose(fused_losses, expected, atol=1e-4)


@pytest.mark.parametrize("model_factory", [_tiny_model, _tiny_model_tied])
def test_sft_fused_dft_matches_eager_on_cpu(tmp_path, model_factory):
    # The fused dft primitive (over the last hidden state) == eager dft_loss.
    from opaque.alignment.sft.loss import dft_loss

    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=model_factory(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="dft",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_dft is True
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    fused_losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        out = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        expected = dft_loss(out.logits, batch["labels"])
    assert torch.allclose(fused_losses, expected, atol=1e-4)


def test_sft_fused_dft_matches_eager_on_cpu_under_peft(tmp_path):
    """The fused-dft path under PEFT must walk the PEFT wrapper to the real
    backbone (``BaseModelOutputWithPast``), not the inner causal-LM. With
    ``peft_config=LoraConfig(...)`` the fused-dft loss must match eager
    ``dft_loss(model.logits, labels)`` on the PEFT-wrapped forward.
    """
    peft = pytest.importorskip("peft")
    from opaque.alignment.sft.loss import dft_loss

    torch.manual_seed(0)
    peft_cfg = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="none"
    )
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="dft",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
        peft_config=peft_cfg,
    )
    # The resolver must have detected PEFT and produced a dotted prefix; the
    # fused-dft seam stays eligible (PEFT-aware lm_head lookup resolves).
    assert trainer._fused_dft is True
    assert trainer._backbone_prefix == "base_model.model.model"
    assert trainer._lm_head_param_name is not None
    assert trainer._lm_head_param_name.startswith("base_model.model.")

    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    fused_losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        out = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        expected = dft_loss(out.logits, batch["labels"])
    assert torch.allclose(fused_losses, expected, atol=1e-4)


def test_sft_fused_last_hidden_state_is_last_layer_only(tmp_path):
    # The fused path must obtain ONLY the last hidden state (T, H) — not the full
    # (L+1, T, H) stack that output_hidden_states would return. Assert the helper
    # returns one (T, H) tensor that equals the model's final hidden state.
    from opaque.functional import make_functional

    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=8,
            loss_type="dft",
            log_completion_metrics=False,
        ),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    _fmodel, trainable, frozen = make_functional(
        trainer.model, partition_trainable=True
    )
    params = {**frozen, **trainable}
    hidden = trainer._last_hidden_state(
        params, batch["input_ids"], batch["attention_mask"]
    )
    with torch.no_grad():
        full = trainer.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
    assert hidden.shape == full.hidden_states[-1].shape  # (B, T, H) — NOT (L+1, ...)
    assert torch.allclose(hidden, full.hidden_states[-1], atol=1e-5)


# ---- DPO: eligibility gating -----------------------------------------
def test_dpo_fused_eligible_resolves_handles(tmp_path):
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._fused_logp_eligible is True
    assert trainer._use_fused_logp is True
    assert trainer._backbone_prefix == "model"
    assert trainer._lm_head_param_name == "lm_head.weight"


def test_dpo_ld_public_length_is_shorter_completion():
    chosen_mask = torch.tensor([[0, 0, 0, 1, 1, 1]])
    rejected_mask = torch.tensor([[0, 1, 1, 0, 0, 0]])
    chosen_prefix, rejected_prefix = DPOTrainer._ld_public_length(
        chosen_mask, rejected_mask
    )
    torch.testing.assert_close(chosen_prefix, torch.tensor([2]))
    torch.testing.assert_close(rejected_prefix, torch.tensor([2]))


def test_dpo_reference_logps_apply_ld_weighting():
    trainer = object.__new__(DPOTrainer)
    object.__setattr__(trainer, "_ld_alpha", 0.25)
    ref_model = _tiny_model().eval()
    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "chosen_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "chosen_completion_mask": torch.tensor([[0, 0, 1, 1, 1, 1]]),
        "rejected_input_ids": torch.tensor([[1, 2, 7, 8, 0, 0]]),
        "rejected_attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        "rejected_completion_mask": torch.tensor([[0, 0, 1, 1, 0, 0]]),
    }

    with torch.no_grad():
        result = trainer.compute_ref_log_probs(
            batch, ref_model, null_ref=False, to_cpu=False
        )
        chosen_logits = ref_model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits
        rejected_logits = ref_model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits

    expected_chosen = sequence_logp(
        chosen_logits,
        batch["chosen_input_ids"],
        batch["chosen_completion_mask"],
        ld_alpha=0.25,
        shared_prefix_len=2,
    )
    expected_rejected = sequence_logp(
        rejected_logits,
        batch["rejected_input_ids"],
        batch["rejected_completion_mask"],
        ld_alpha=0.25,
        shared_prefix_len=2,
    )
    torch.testing.assert_close(result["ref_chosen_logps"], expected_chosen)
    torch.testing.assert_close(result["ref_rejected_logps"], expected_rejected)


@pytest.mark.parametrize(
    "extra",
    [
        # Three intrinsic blockers: features that read per-token logps from logits
        # and thus can't run on the logits-free fused path.
        {"log_completion_metrics": False, "use_weighting": True},  # WPO reads logps
        {
            "log_completion_metrics": False,
            "f_divergence_type": "js_divergence",
        },  # non-reverse-KL remaps logps
        {"log_completion_metrics": False, "ld_alpha": 0.5},  # LD-DPO needs per-token
    ],
)
def test_dpo_ineligible_configs_keep_eager(tmp_path, extra):
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid", **extra),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._use_fused_logp is False


def test_dpo_log_completion_metrics_keeps_static_eligibility(tmp_path):
    # Telemetry no longer disables ``_use_fused_logp`` wholesale; the per-step
    # branch in ``compute_per_example_loss_and_metrics`` is what falls back to
    # the eager path when telemetry is on. Static eligibility stays True so
    # toggling --log-completion-metrics doesn't reshape the trainer.
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            log_completion_metrics=True,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._use_fused_logp is True
    assert trainer._log_completion_metrics is True


# ---- DPO: CPU fallback-equivalence -----------------------------------
def _dpo_fused_logps(trainer, batch):
    """vmap the fused ``_fused_logp`` over a collated batch (chosen + rejected)."""
    from opaque.functional import make_functional

    batch = _to_device(trainer, batch)
    fmodel, trainable, frozen = make_functional(trainer.model, partition_trainable=True)

    def fn(tp, c_ids, c_mask, c_cmask, r_ids, r_mask, r_cmask):
        params = {**frozen, **tp}
        c = trainer._fused_logp(fmodel, params, c_ids, c_mask, c_cmask)
        r = trainer._fused_logp(fmodel, params, r_ids, r_mask, r_cmask)
        return c, r

    vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * 6)
    return vmapped(
        trainable,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_completion_mask"],
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_completion_mask"],
    )


@pytest.mark.parametrize("model_factory", [_tiny_model, _tiny_model_tied])
def test_dpo_fused_logp_matches_eager_on_cpu(tmp_path, model_factory):
    # The fused chosen/rejected logps == the eager sequence_logp on logits, on CPU.
    from opaque.alignment.dpo.loss import sequence_logp

    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=model_factory(),
        ref_model=model_factory(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._use_fused_logp is True
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))

    fused_c, fused_r = _dpo_fused_logps(trainer, batch)
    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        eager_c = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        eager_r = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
    assert torch.allclose(fused_c, eager_c, atol=1e-4)
    assert torch.allclose(fused_r, eager_r, atol=1e-4)


@pytest.mark.parametrize("model_factory", [_tiny_model, _tiny_model_tied])
def test_dpo_fused_loss_matches_eager_on_cpu(tmp_path, model_factory):
    # End-to-end: the eligible (fused) per-example DPO loss == the eager logits
    # loss. We compare the fused trainer to an eager trainer built identically
    # (telemetry on forces eager) on the same weights and batch.
    from opaque.alignment.dpo.loss import sequence_logp, sigmoid_loss

    torch.manual_seed(0)
    beta = 0.1
    trainer = DPOTrainer(
        model=model_factory(),
        ref_model=model_factory(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            beta=beta,
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._use_fused_logp is True
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    fused_losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        expected = sigmoid_loss(
            c_lp - batch["ref_chosen_logps"],
            r_lp - batch["ref_rejected_logps"],
            beta=beta,
        )
    assert torch.allclose(fused_losses, expected, atol=1e-4)


def test_dpo_fused_trains_a_couple_steps(tmp_path):
    # The fused-eligible run trains end-to-end through the DP step machinery.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            log_completion_metrics=False,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    assert trainer._use_fused_logp is True
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


# ----------------------------------------------------------------------
# PEFT unwrap — fused-dft / fused-logp backbone resolution under PEFT.
# The resolver must walk a ``PeftModelForCausalLM`` to the real backbone
# (``BaseModelOutputWithPast``), not the inner causal-LM, so that
# ``_last_hidden_state`` finds ``last_hidden_state``.
# ----------------------------------------------------------------------
def _tiny_peft_model():
    peft = pytest.importorskip("peft")
    base = _tiny_model()
    cfg = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="none"
    )
    return peft.get_peft_model(base, cfg)


def test_sft_resolve_fused_handles_unwraps_peft():
    from opaque.api.transformers.trl._sft_trainer import _resolve_fused_handles

    peft_model = _tiny_peft_model()
    prefix, lm_head_name = _resolve_fused_handles(peft_model, eligible=True)
    # PEFT-aware dotted path that ``attrgetter`` walks all the way to the
    # real backbone (LlamaModel), not the wrapped causal-LM.
    assert prefix == "base_model.model.model"
    # lm_head param-name resolves on the OUTER model — under PEFT, every key
    # in named_parameters lives under ``base_model.model.``.
    assert lm_head_name is not None
    assert lm_head_name.startswith("base_model.model.")
    # Walking the returned prefix yields the unwrapped backbone — calling it
    # functionally returns a BaseModelOutputWithPast (with last_hidden_state),
    # which is what _last_hidden_state requires.
    from operator import attrgetter

    backbone = attrgetter(prefix)(peft_model)
    assert backbone.__class__.__name__ == "LlamaModel"


def test_dpo_resolve_fused_handles_unwraps_peft():
    from opaque.api.transformers.trl._dpo_trainer import _resolve_fused_handles

    peft_model = _tiny_peft_model()
    prefix, lm_head_name = _resolve_fused_handles(peft_model, eligible=True)
    assert prefix == "base_model.model.model"
    assert lm_head_name is not None
    assert lm_head_name.startswith("base_model.model.")


def test_sft_resolve_fused_handles_bare_model_unchanged():
    """Non-PEFT path still returns the bare ``base_model_prefix`` (no regression)."""
    from opaque.api.transformers.trl._sft_trainer import _resolve_fused_handles

    bare = _tiny_model()
    prefix, lm_head_name = _resolve_fused_handles(bare, eligible=True)
    assert prefix == "model"  # LlamaForCausalLM.base_model_prefix == "model"
    assert lm_head_name is not None
    assert not lm_head_name.startswith("base_model.")


def test_sft_dft_eval_uses_training_objective(tmp_path):
    """#384: SFT eval with ``loss_type='dft'`` reports the DFT loss, not model CE.

    All eval examples are identical, so ``eval_loss`` equals the per-example DFT
    loss.  Pre-fix, SFT eval used the model's cross-entropy, so best-model
    selection ranked checkpoints on the wrong objective.
    """
    from opaque.alignment.sft.loss import dft_loss, nll_loss

    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="dft"),
        train_dataset=_sft_dataset(),
        eval_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    eval_loss = trainer.evaluate()["eval_loss"]

    batch = trainer._data_collator([{"input_ids": [1, 2, 3, 4, 5, 6]}])
    batch = {k: v.to(trainer._device) for k, v in batch.items()}
    trainer._model.eval()  # match eval mode (no dropout) for the reference
    with torch.no_grad():
        logits = trainer._model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
    dft_ref = dft_loss(logits, batch["labels"]).mean().item()
    ce_ref = nll_loss(logits, batch["labels"]).mean().item()

    # bf16 eval: match the DFT loss within tolerance, and be nowhere near CE
    # (~log V ≈ 4.15 here) — pre-fix eval reported CE.
    assert eval_loss == pytest.approx(dft_ref, rel=1e-2)
    assert abs(eval_loss - ce_ref) > 1.0


def _varied_len_dataset() -> Dataset:
    """Eval rows with contrasting real-token counts (2 vs 15 after shift)."""
    return Dataset.from_list(
        [{"input_ids": [1, 2, 3]}, {"input_ids": list(range(3, 19))}] * 2
    )


def _dft_trainer(tmp_path, *, compute_metrics=None, seed_models=0):
    torch.manual_seed(seed_models)  # identical weights across trainers
    return SFTTrainer(
        model=_tiny_model(),
        args=_args(
            SFTConfig,
            tmp_path,
            max_length=16,
            loss_type="dft",
            per_device_eval_batch_size=4,
        ),
        train_dataset=_sft_dataset(),
        eval_dataset=_varied_len_dataset(),
        processing_class=_stub_tokenizer(),
        compute_metrics=compute_metrics,
    )


def test_sft_dft_eval_loss_is_plain_per_example_mean(tmp_path):
    """#384 review: eval_loss is the plain per-example mean of the DFT scalars,
    not a mean reweighted by realized non-padding token counts."""
    from opaque.alignment.sft.loss import dft_loss

    seen = {}

    def cm(ep):
        seen["label_ids"] = ep.label_ids
        return {}

    trainer = _dft_trainer(tmp_path, compute_metrics=cm)
    eval_loss = trainer.evaluate()["eval_loss"]

    # labels still reach compute_metrics (the fix must not drop them).
    assert seen["label_ids"] is not None

    rows = [dict(r) for r in _varied_len_dataset()]
    batch = trainer._data_collator(rows)
    batch = {k: v.to(trainer._device) for k, v in batch.items()}
    trainer._model.eval()
    with torch.no_grad():
        logits = trainer._model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
    per_ex = dft_loss(logits.float(), batch["labels"])
    tok = (batch["labels"][..., 1:] != -100).sum(-1).float()
    plain = per_ex.mean().item()
    weighted = ((per_ex * tok).sum() / tok.sum()).item()

    # Self-check: the varied lengths make the two aggregations discriminable.
    assert abs(plain - weighted) > 2e-4
    # eval_loss must be the plain mean (bf16 tolerance << the plain/weighted gap).
    assert abs(eval_loss - plain) < 0.35 * abs(plain - weighted)


def test_sft_dft_eval_loss_consistent_across_metric_modes(tmp_path):
    """#384 review: the same checkpoint reports the same eval_loss whether or
    not compute_metrics is configured (pre-rework: plain vs token-weighted)."""
    loss_only = _dft_trainer(tmp_path / "a").evaluate()["eval_loss"]
    with_metrics = _dft_trainer(
        tmp_path / "b", compute_metrics=lambda ep: {}
    ).evaluate()["eval_loss"]
    assert loss_only == pytest.approx(with_metrics, rel=1e-6)


def test_sft_dft_loss_only_eval_uses_logits_free_closure(tmp_path, monkeypatch):
    """Loss-only DFT evaluation must not request a logits result just to discard it."""
    trainer = _dft_trainer(tmp_path)
    assert trainer._fused_dft is False  # telemetry normally keeps training eager
    rows = [dict(r) for r in _varied_len_dataset()]
    batch = trainer._data_collator(rows)
    called = 0
    original = trainer._last_hidden_state

    def logits_free_hidden(*args, **kwargs):
        nonlocal called
        called += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "_last_hidden_state", logits_free_hidden)
    loss, preds, labels = trainer.prediction_step(
        trainer._model, dict(batch), prediction_loss_only=True
    )

    assert loss.ndim == 1
    assert preds is None
    assert labels is None
    assert called == 1
    assert trainer._eval_per_example_loss_only_fn is not None
    assert trainer._eval_per_example_loss_fn is None


def test_sft_dft_prediction_step_respects_ignore_keys(tmp_path):
    """#384 review: ignore_keys=['logits'] filters predictions (base parity)."""
    trainer = _dft_trainer(tmp_path)
    rows = [dict(r) for r in _varied_len_dataset()]
    batch = trainer._data_collator(rows)

    loss, preds, labels = trainer.prediction_step(
        trainer._model, dict(batch), prediction_loss_only=False, ignore_keys=None
    )
    assert loss.ndim == 1
    assert preds is not None
    assert labels is not None

    loss, preds, labels = trainer.prediction_step(
        trainer._model,
        dict(batch),
        prediction_loss_only=False,
        ignore_keys=["logits"],
    )
    assert loss.ndim == 1
    assert preds is None
    assert labels is not None


@pytest.fixture
def _isolated_process_group():
    assert not torch.distributed.is_initialized(), (
        "a process group is already live before the test"
    )
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.slow
@pytest.mark.distributed
@pytest.mark.usefixtures("_isolated_process_group")
def test_dpo_brings_up_the_process_group_before_precomputing(tmp_path, monkeypatch):
    """The reference forward must find a live process group, or it cannot shard.

    The launcher environment is faked and ``init_process_group`` stubbed, which
    pins that the attempt happens and that it happens before ``ref`` runs; the
    real auto-init path needs a second process and lives in ``tests/distributed``.
    """
    from opaque.api.alignment.dpo.reference import _precompute

    # Keep the reference cache inside the test's tmp_path, out of reach of any
    # archive left in the shared system tempdir.
    monkeypatch.setattr(_precompute.tempfile, "gettempdir", lambda: str(tmp_path))

    import socket

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("gloo backend is not available")

    # What torchrun gives a rank: a multi-rank launcher env, no group yet.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    real_init = torch.distributed.init_process_group
    init_attempts: list[object] = []

    def _init_single_rank(*args, **kwargs):
        # Stand in for the peer rank: bring up a real 1-rank gloo group so the
        # rest of the trainer sees a consistent topology, rather than an env
        # that claims two ranks and a group that does not exist.
        init_attempts.append(kwargs.get("backend"))
        real_init(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )

    monkeypatch.setattr(torch.distributed, "init_process_group", _init_single_rank)

    group_live_at_forward: list[bool] = []
    real_forward = _precompute._reference_forward

    def _observing_forward(*args, **kwargs):
        group_live_at_forward.append(torch.distributed.is_initialized())
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(_precompute, "_reference_forward", _observing_forward)

    torch.manual_seed(0)
    DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )

    assert init_attempts, "trainer never brought up the process group"
    assert group_live_at_forward, "the reference forward never ran"
    assert all(group_live_at_forward), (
        "the reference forward ran before the process group was established"
    )
