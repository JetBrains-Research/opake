# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the qwen3_moe family. ``experts`` targets the v5 stacked
``Qwen3MoeExperts``; ``mlp`` covers dense / pre-v5 per-expert ``Qwen3MoeMLP``
(whichever exists). ``fused_add_rms_kind=None`` keeps the MoE decoder forward intact."""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.qwen3_moe.modeling_qwen3_moe"


apply_qwen3_moe_family_patches = make_apply_family_patches(
    family="qwen3_moe",
    module_path=_MODULE_PATH,
)


apply_qwen3_moe_patches = make_apply_model_patches(
    family="qwen3_moe",
    family_apply=apply_qwen3_moe_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Qwen3MoeMLP",
        "experts": "Qwen3MoeExperts",
        "rms_norm": "Qwen3MoeRMSNorm",
        "decoder_layer": "Qwen3MoeDecoderLayer",
        "causal_lm": "Qwen3MoeForCausalLM",
    },
    activation_kind="swiglu",
    moe_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("qwen3_moe", apply_qwen3_moe_patches)


__all__ = ["apply_qwen3_moe_family_patches", "apply_qwen3_moe_patches"]
