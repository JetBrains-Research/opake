# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig

from opake.patches import apply_model_patches


def test_lora_patching_allowed_on_non_cuda_hosts(monkeypatch):
    import torch

    from opake.api.patches.peft._router import _lora_patching_allowed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _lora_patching_allowed()


def test_apply_model_patches_is_idempotent_for_lora_fusions():
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    model = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    layer = model.model.model.layers[0]

    apply_model_patches(model, performance=False, compat=True, lora=True)
    attn_forward = layer.self_attn.forward.__func__
    mlp_forward = layer.mlp.forward.__func__

    apply_model_patches(model, performance=False, compat=True, lora=True)

    assert layer.self_attn.forward.__func__ is attn_forward
    assert layer.mlp.forward.__func__ is mlp_forward
    assert getattr(layer.self_attn, "_opake_lora_qkv_patched", False)
    assert getattr(layer.mlp, "_opake_lora_mlp_patched", False)
