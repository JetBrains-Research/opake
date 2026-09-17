# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the gpt_oss family. Experts are not patched (clamped SwiGLU + MXFP4
won't match the plain-SwiGLU kernel; mirrors Liger ``swiglu=False``) — only
RMSNorm, RoPE, and cross-entropy."""

from __future__ import annotations

from opaque.api.patches.transformers._factory import make_apply_model_patches
from opaque.api.patches.transformers._family import make_apply_family_patches
from opaque.api.patches.transformers._registry import register_family
from opaque.api.patches.transformers.components.attention import (
    vmap_eager_attention_forward_with_sinks,
)

_MODULE_PATH = "transformers.models.gpt_oss.modeling_gpt_oss"


apply_gpt_oss_family_patches = make_apply_family_patches(
    family="gpt_oss",
    module_path=_MODULE_PATH,
    eager_attention_replacement=vmap_eager_attention_forward_with_sinks,
)


apply_gpt_oss_patches = make_apply_model_patches(
    family="gpt_oss",
    family_apply=apply_gpt_oss_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "rms_norm": "GptOssRMSNorm",
        "causal_lm": "GptOssForCausalLM",
    },
    rms_norm_kind="llama",
    fused_add_rms_kind=None,
)


register_family("gpt_oss", apply_gpt_oss_patches)


__all__ = ["apply_gpt_oss_family_patches", "apply_gpt_oss_patches"]
