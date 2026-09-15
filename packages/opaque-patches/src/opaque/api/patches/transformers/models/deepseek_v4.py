# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the deepseek_v4 family. Experts and RoPE are not patched (scaled
experts + interleaved partial RoPE; mirrors Liger ``swiglu=False, rope=False``) —
only RMSNorm and cross-entropy."""

from __future__ import annotations

from opaque.api.patches.transformers._factory import make_apply_model_patches
from opaque.api.patches.transformers._family import make_apply_family_patches
from opaque.api.patches.transformers._registry import register_family
from opaque.api.patches.transformers.components.attention import (
    vmap_eager_attention_forward_with_sinks,
)

_MODULE_PATH = "transformers.models.deepseek_v4.modeling_deepseek_v4"


apply_deepseek_v4_family_patches = make_apply_family_patches(
    family="deepseek_v4",
    module_path=_MODULE_PATH,
    eager_attention_replacement=vmap_eager_attention_forward_with_sinks,
    rope_replacement=None,
)


apply_deepseek_v4_patches = make_apply_model_patches(
    family="deepseek_v4",
    family_apply=apply_deepseek_v4_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "rms_norm": "DeepseekV4RMSNorm",
        "causal_lm": "DeepseekV4ForCausalLM",
    },
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("deepseek_v4", apply_deepseek_v4_patches)


__all__ = ["apply_deepseek_v4_family_patches", "apply_deepseek_v4_patches"]
