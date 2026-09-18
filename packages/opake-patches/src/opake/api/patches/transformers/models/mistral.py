# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the mistral family — built via the patch factories.

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
from opake.api.patches.transformers.components.attention import (
    vmap_sdpa_attention_forward_sliding_window,
)
from opake.api.patches.transformers.components.masking import (
    apply_compact_sdpa_sliding_window_masking_patch,
)

_MODULE_PATH = "transformers.models.mistral.modeling_mistral"


apply_mistral_family_patches = make_apply_family_patches(
    family="mistral",
    module_path=_MODULE_PATH,
    sdpa_attention_replacement=vmap_sdpa_attention_forward_sliding_window,
    masking_module_patcher=apply_compact_sdpa_sliding_window_masking_patch,
)


apply_mistral_patches = make_apply_model_patches(
    family="mistral",
    family_apply=apply_mistral_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "MistralMLP",
        "rms_norm": "MistralRMSNorm",
        "decoder_layer": "MistralDecoderLayer",
        "causal_lm": "MistralForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("mistral", apply_mistral_patches)


__all__ = ["apply_mistral_family_patches", "apply_mistral_patches"]
