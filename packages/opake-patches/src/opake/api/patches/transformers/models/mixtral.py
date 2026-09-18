# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the mixtral family (MoE)."""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.mixtral.modeling_mixtral"


apply_mixtral_family_patches = make_apply_family_patches(
    family="mixtral",
    module_path=_MODULE_PATH,
)


apply_mixtral_patches = make_apply_model_patches(
    family="mixtral",
    family_apply=apply_mixtral_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "experts": "MixtralExperts",
        "rms_norm": "MixtralRMSNorm",
        "decoder_layer": "MixtralDecoderLayer",
        "causal_lm": "MixtralForCausalLM",
    },
    moe_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("mixtral", apply_mixtral_patches)


__all__ = ["apply_mixtral_family_patches", "apply_mixtral_patches"]
