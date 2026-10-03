# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""``trim_microbatch_padding``: length-ordered microbatches with padding trim.

EXPERIMENTAL, NOT YET DP-REVIEWED. These tests pin the mechanics (which
columns are dropped, which tensors are permuted) and the end-to-end claim that
training with the flag gives the same parameters as without it, up to
floating-point summation order, under fixed and adaptive clipping.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM

import opake.api.transformers.trainer._dp_trainer as dp_trainer_module
from opake.api.transformers.trainer._padding import (
    microbatch_padding_trimmer,
    sort_batch_by_length,
)
from opake.exceptions import ConfigurationError
from opake.transformers import TrainingArguments
from opake.transformers.trl import SFTConfig, SFTTrainer

KEYS = ("input_ids", "attention_mask", "labels")


def _batch(lengths, padded_len, label_tail=0):
    """Right-padded batch; ``label_tail`` valid labels continue past the mask."""
    n = len(lengths)
    ids = torch.zeros(n, padded_len, dtype=torch.long)
    mask = torch.zeros(n, padded_len, dtype=torch.long)
    labels = torch.full((n, padded_len), -100, dtype=torch.long)
    for r, length in enumerate(lengths):
        ids[r, :length] = torch.arange(1, length + 1)
        mask[r, :length] = 1
        labels[r, : min(length + label_tail, padded_len)] = 7
    return ids, mask, labels


def test_trimmer_drops_only_columns_without_tokens_or_labels():
    ids, mask, labels = _batch([3, 5], padded_len=9)
    weights = torch.ones(2)  # per-example, not a sequence tensor
    out = microbatch_padding_trimmer((*KEYS, "w"))(
        ("params", ids, mask, labels, weights)
    )
    assert out[0] == "params"
    assert [t.shape[1] for t in out[1:4]] == [5, 5, 5]
    assert torch.equal(out[1], ids[:, :5])
    assert torch.equal(out[3], labels[:, :5])
    assert out[4] is weights


def test_trimmer_keeps_columns_with_valid_labels_past_the_mask():
    ids, mask, labels = _batch([3, 4], padded_len=10, label_tail=2)
    out = microbatch_padding_trimmer(KEYS)(("p", ids, mask, labels))
    assert out[2].shape[1] == 6  # 4 attended + 2 labelled columns


def test_trimmer_is_a_no_op_without_trailing_padding():
    args = ("p", *_batch([6, 6], padded_len=6))
    assert microbatch_padding_trimmer(KEYS)(args) is args


def test_sort_orders_every_per_example_tensor_longest_first_stably():
    ids, mask, labels = _batch([2, 5, 2, 4], padded_len=6)
    tag = torch.tensor([10, 11, 12, 13])
    out = sort_batch_by_length((ids, mask, labels, tag), (*KEYS, "tag"))
    assert out[3].tolist() == [11, 13, 10, 12]  # lengths 5, 4, 2, 2 (stable ties)
    assert out[1].sum(dim=1).tolist() == [5, 4, 2, 2]


def test_trimming_requires_an_attention_mask():
    with pytest.raises(ConfigurationError, match="attention_mask"):
        microbatch_padding_trimmer(("input_ids", "labels"))


@pytest.mark.parametrize("other", [{"torch_compile": True}, {"cuda_graphs": True}])
def test_trimming_excludes_static_shape_features(tmp_path, other):
    with pytest.raises(ConfigurationError, match="trim_microbatch_padding"):
        TrainingArguments(
            output_dir=str(tmp_path),
            privacy_noise_multiplier=1.0,
            trim_microbatch_padding=True,
            **other,
        )


def test_trimming_flag_must_be_a_bool(tmp_path):
    with pytest.raises(ConfigurationError, match="must be a bool"):
        TrainingArguments(
            output_dir=str(tmp_path),
            privacy_noise_multiplier=1.0,
            trim_microbatch_padding=1,
        )


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
    )


def _train(tmp_path, *, trim, clipping_mode, monkeypatch=None, seen=None):
    if seen is not None:
        real = dp_trainer_module.microbatch_padding_trimmer

        def recording(keys):
            inner = real(keys)

            def trim_fn(args):
                out = inner(args)
                seen.append((args[1].shape[1], out[1].shape[1]))
                return out

            return trim_fn

        monkeypatch.setattr(dp_trainer_module, "microbatch_padding_trimmer", recording)
    lengths = [3 + (7 * i) % 18 for i in range(32)]  # 3..20, varied
    data = Dataset.from_list(
        [{"input_ids": [1 + (j % 60) for j in range(n)]} for n in lengths]
    )
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        pad_token="<pad>",
        eos_token="</s>",
        save_pretrained=lambda *a, **k: None,
    )
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=SFTConfig(
            output_dir=str(tmp_path),
            privacy_noise_multiplier=0.0,
            clipping_norm=0.05,  # small: clipping is active for every example
            clipping_mode=clipping_mode,
            per_device_train_batch_size=8,
            microbatch_size=2,
            max_steps=2,
            max_length=32,
            loss_type="nll",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=0,
            trim_microbatch_padding=trim,
        ),
        train_dataset=data,
        processing_class=tokenizer,
    )
    trainer.train()
    return {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}


@pytest.mark.parametrize("clipping_mode", ["fixed", "adaptive"])
def test_training_with_trimming_matches_training_without(
    tmp_path, monkeypatch, clipping_mode
):
    seen: list[tuple[int, int]] = []
    base = _train(tmp_path / "off", trim=False, clipping_mode=clipping_mode)
    trimmed = _train(
        tmp_path / "on",
        trim=True,
        clipping_mode=clipping_mode,
        monkeypatch=monkeypatch,
        seen=seen,
    )
    assert any(after < before for before, after in seen), "nothing was trimmed"
    initial = _tiny_model().state_dict()
    changed = [k for k in base if not torch.equal(base[k], initial[k])]
    assert changed, "training did not update any parameter"
    for k in base:
        torch.testing.assert_close(trimmed[k], base[k], atol=1e-6, rtol=1e-5, msg=k)
