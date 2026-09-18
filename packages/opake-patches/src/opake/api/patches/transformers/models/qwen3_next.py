# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the qwen3_next family (hybrid linear-attention MoE). RoPE left to
HF; only the standard ``Qwen3NextRMSNorm`` is patched, not the gated variants."""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.qwen3_next.modeling_qwen3_next"


apply_qwen3_next_family_patches = make_apply_family_patches(
    family="qwen3_next",
    module_path=_MODULE_PATH,
    rope_replacement=None,
)


apply_qwen3_next_patches = make_apply_model_patches(
    family="qwen3_next",
    family_apply=apply_qwen3_next_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Qwen3NextMLP",
        "experts": "Qwen3NextExperts",
        "rms_norm": "Qwen3NextRMSNorm",
        "decoder_layer": "Qwen3NextDecoderLayer",
        "causal_lm": "Qwen3NextForCausalLM",
    },
    activation_kind="swiglu",
    moe_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("qwen3_next", apply_qwen3_next_patches)


__all__ = ["apply_qwen3_next_family_patches", "apply_qwen3_next_patches"]
