# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the qwen3_5_moe family. RoPE left to HF (special rotary); only the
standard ``Qwen3_5MoeRMSNorm`` is patched, not the gated linear-attention norms."""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"


apply_qwen3_5_moe_family_patches = make_apply_family_patches(
    family="qwen3_5_moe",
    module_path=_MODULE_PATH,
    rope_replacement=None,
)


apply_qwen3_5_moe_patches = make_apply_model_patches(
    family="qwen3_5_moe",
    family_apply=apply_qwen3_5_moe_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Qwen3_5MoeMLP",
        "experts": "Qwen3_5MoeExperts",
        "rms_norm": "Qwen3_5MoeRMSNorm",
        "decoder_layer": "Qwen3_5MoeDecoderLayer",
        "causal_lm": "Qwen3_5MoeForCausalLM",
    },
    activation_kind="swiglu",
    moe_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("qwen3_5_moe", apply_qwen3_5_moe_patches)


__all__ = ["apply_qwen3_5_moe_family_patches", "apply_qwen3_5_moe_patches"]
