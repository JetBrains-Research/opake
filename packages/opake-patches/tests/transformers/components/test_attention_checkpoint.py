# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-block checkpointing under the functional per-example path."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config

from opake.functional import make_functional
from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

_VOCAB = 64


def _tiny_lora_model(device):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa", dtype=torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        ),
    ).to(device)
    # Non-zero lora_B so every adapter factor receives an informative gradient.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    # Training mode (as under DPTrainer); dropout is zero so forwards are
    # deterministic.
    model.train()
    return model


def _attention_modules(model):
    return [
        module.self_attn for module in model.modules() if hasattr(module, "self_attn")
    ]


def _count_attention_calls(model):
    counts = []
    for attn in _attention_modules(model):
        forward = attn.forward
        calls = [0]

        def counted(*args, _forward=forward, _calls=calls, **kwargs):
            _calls[0] += 1
            return _forward(*args, **kwargs)

        attn.forward = counted
        counts.append(calls)
    return counts


def _per_example_grads(model, device, **forward_kwargs):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, _VOCAB, (3, 12), generator=generator).to(device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 8:] = 0

    def per_example_loss(params, ids, mask):
        out = fmodel(
            {**frozen, **params},
            input_ids=ids,
            attention_mask=mask,
            **forward_kwargs,
        )
        return out.logits.float().square().mean()

    return torch.vmap(torch.func.grad(per_example_loss), in_dims=(None, 0, 0))(
        trainable, input_ids, attention_mask
    )


def _checkpointed_model_with_counts(device):
    model = _tiny_lora_model(device)
    apply_model_patches(model, performance=True, compat=True, kernels=False)
    counts = _count_attention_calls(model)
    apply_model_patches(
        model,
        performance=True,
        compat=True,
        kernels=False,
        attention_checkpointing=True,
    )
    return model, counts


def test_attention_checkpointing_preserves_per_example_grads(device):
    reference_model = _tiny_lora_model(device)
    apply_model_patches(reference_model, performance=True, compat=True, kernels=False)
    reference = _per_example_grads(reference_model, device, use_cache=False)

    model = _tiny_lora_model(device)
    apply_model_patches(
        model,
        performance=True,
        compat=True,
        kernels=False,
        attention_checkpointing=True,
    )
    got = _per_example_grads(model, device, use_cache=False)

    assert reference.keys() == got.keys()
    for name in reference:
        torch.testing.assert_close(got[name], reference[name], msg=name)


def test_attention_checkpointing_recomputes_attention_in_backward(device):
    model, counts = _checkpointed_model_with_counts(device)

    _per_example_grads(model, device, use_cache=False)

    assert [calls[0] for calls in counts] == [2, 2]


def test_attention_checkpointing_with_default_use_cache_in_training(device):
    """``config.use_cache=True`` must not disable checkpointing in training.

    Trainer forwards leave ``use_cache`` unset; the KV-cache patch turns the
    cache off for training calls, so attention is still recomputed.
    """
    reference_model = _tiny_lora_model(device)
    apply_model_patches(reference_model, performance=True, compat=True, kernels=False)
    reference = _per_example_grads(reference_model, device, use_cache=False)

    model, counts = _checkpointed_model_with_counts(device)
    assert model.config.use_cache
    got = _per_example_grads(model, device)

    assert [calls[0] for calls in counts] == [2, 2]
    for name in reference:
        torch.testing.assert_close(got[name], reference[name], msg=name)


def test_attention_checkpointing_is_inactive_in_eval_mode(device):
    model, counts = _checkpointed_model_with_counts(device)
    model.eval()

    _per_example_grads(model, device, use_cache=False)

    assert [calls[0] for calls in counts] == [1, 1]


def test_attention_checkpointing_is_idempotent_and_inactive_without_grad(device):
    model = _tiny_lora_model(device)
    apply_model_patches(model, performance=True, compat=True, kernels=False)
    counts = _count_attention_calls(model)
    for _ in range(2):
        apply_model_patches(
            model,
            performance=True,
            compat=True,
            kernels=False,
            attention_checkpointing=True,
        )

    input_ids = torch.randint(0, _VOCAB, (2, 8)).to(device)
    with torch.no_grad():
        model(input_ids=input_ids)

    assert [calls[0] for calls in counts] == [1, 1]


def test_attention_checkpointing_keeps_kv_cache_outside_training(device):
    from transformers import DynamicCache

    model, counts = _checkpointed_model_with_counts(device)
    model.eval()

    input_ids = torch.randint(0, _VOCAB, (2, 8)).to(device)
    cache = DynamicCache(config=model.config)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    out.logits.float().sum().backward()

    assert [calls[0] for calls in counts] == [1, 1]
    assert cache.get_seq_length() == input_ids.shape[1]


def test_attention_checkpointing_is_off_by_default(device):
    model = _tiny_lora_model(device)
    apply_model_patches(model, performance=True, compat=True, kernels=False)
    counts = _count_attention_calls(model)

    _per_example_grads(model, device)

    assert [calls[0] for calls in counts] == [1, 1]
