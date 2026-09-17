# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the hunyuan_v1_moe family (MoE)."""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe"


apply_hunyuan_v1_moe_family_patches = make_apply_family_patches(
    family="hunyuan_v1_moe",
    module_path=_MODULE_PATH,
)


apply_hunyuan_v1_moe_patches = make_apply_model_patches(
    family="hunyuan_v1_moe",
    family_apply=apply_hunyuan_v1_moe_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "HunYuanMoEV1MLP",
        "experts": "HunYuanMoEV1Experts",
        "rms_norm": "HunYuanMoEV1RMSNorm",
        "decoder_layer": "HunYuanMoEV1DecoderLayer",
        "causal_lm": "HunYuanMoEV1ForCausalLM",
    },
    activation_kind="swiglu",
    moe_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("hunyuan_v1_moe", apply_hunyuan_v1_moe_patches)


__all__ = ["apply_hunyuan_v1_moe_family_patches", "apply_hunyuan_v1_moe_patches"]
