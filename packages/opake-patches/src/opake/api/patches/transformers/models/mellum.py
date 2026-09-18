# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the Mellum 2.0 MoE family (``JetBrains/Mellum2-12B-A2.5B``, tf v5.8+).

Stacked ``MellumExperts``, SwiGLU, and route-sensitive RMSNorm. Mellum keeps the
upstream Transformers vmap-safe RMSNorm because Triton reduction-order
differences in BF16 can change expert selection. ``fused_add_rms_kind=None``
keeps the MoE decoder forward (router logits / aux loss) intact. The original
dense Mellum (``model_type="llama"``) is served by the ``llama`` family.
"""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.mellum.modeling_mellum"


apply_mellum_family_patches = make_apply_family_patches(
    family="mellum",
    module_path=_MODULE_PATH,
    rope_replacement=None,
)


apply_mellum_patches = make_apply_model_patches(
    family="mellum",
    family_apply=apply_mellum_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "MellumMLP",
        "experts": "MellumExperts",
        "rms_norm": "MellumRMSNorm",
        "decoder_layer": "MellumDecoderLayer",
        "causal_lm": "MellumForCausalLM",
    },
    activation_kind="swiglu",
    moe_kind="swiglu",
    rms_norm_kind=None,
    fused_add_rms_kind=None,
    fused_linear_cross_entropy=False,
    chunked_linear_cross_entropy=2048,
)


register_family("mellum", apply_mellum_patches)


__all__ = ["apply_mellum_family_patches", "apply_mellum_patches"]
