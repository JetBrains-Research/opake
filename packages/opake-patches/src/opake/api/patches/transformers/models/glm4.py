# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the glm4 family — built via the patch factories.

The factories close over architecture-specific knobs (MLP kind, RMSNorm
casting, whether the family supports fused-add RMS) at construction
time, so the dispatch is bug-by-construction: e.g. Gemma's
``activation_kind="geglu_exact"`` cannot accidentally route to SwiGLU.

Registration: this module calls ``register_family`` at import time —
the same mechanism downstream users follow to add their own families.
"""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.glm4.modeling_glm4"


apply_glm4_family_patches = make_apply_family_patches(
    family="glm4",
    module_path=_MODULE_PATH,
    # GLM4's `apply_rotary_pos_emb` interleaves cos/sin over a partial rotary
    # dimension rather than the contiguous rotate_half split the default
    # kernel implements; leave RoPE unpatched to avoid incorrect math.
    rope_replacement=None,
)


apply_glm4_patches = make_apply_model_patches(
    family="glm4",
    family_apply=apply_glm4_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Glm4MLP",
        "rms_norm": "Glm4RMSNorm",
        "causal_lm": "Glm4ForCausalLM",
    },
    activation_kind="phi3_swiglu",
    rms_norm_kind="glm4",
    fused_add_rms_kind=None,
)


register_family("glm4", apply_glm4_patches)


__all__ = ["apply_glm4_family_patches", "apply_glm4_patches"]
