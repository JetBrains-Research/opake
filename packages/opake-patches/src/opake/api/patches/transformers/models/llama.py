# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the llama family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.llama.modeling_llama"


apply_llama_family_patches = make_apply_family_patches(
    family="llama",
    module_path=_MODULE_PATH,
)


apply_llama_patches = make_apply_model_patches(
    family="llama",
    family_apply=apply_llama_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "LlamaMLP",
        "rms_norm": "LlamaRMSNorm",
        "decoder_layer": "LlamaDecoderLayer",
        "causal_lm": "LlamaForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("llama", apply_llama_patches)


__all__ = ["apply_llama_family_patches", "apply_llama_patches"]
