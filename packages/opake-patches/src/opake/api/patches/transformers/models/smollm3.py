# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the smollm3 family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.smollm3.modeling_smollm3"


apply_smollm3_family_patches = make_apply_family_patches(
    family="smollm3",
    module_path=_MODULE_PATH,
)


apply_smollm3_patches = make_apply_model_patches(
    family="smollm3",
    family_apply=apply_smollm3_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "SmolLM3MLP",
        "rms_norm": "SmolLM3RMSNorm",
        "decoder_layer": "SmolLM3DecoderLayer",
        "causal_lm": "SmolLM3ForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("smollm3", apply_smollm3_patches)


__all__ = ["apply_smollm3_family_patches", "apply_smollm3_patches"]
