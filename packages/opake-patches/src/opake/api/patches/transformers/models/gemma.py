# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the gemma family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.gemma.modeling_gemma"


apply_gemma_family_patches = make_apply_family_patches(
    family="gemma",
    module_path=_MODULE_PATH,
)


apply_gemma_patches = make_apply_model_patches(
    family="gemma",
    family_apply=apply_gemma_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "GemmaMLP",
        "rms_norm": "GemmaRMSNorm",
        "decoder_layer": "GemmaDecoderLayer",
        "causal_lm": "GemmaForCausalLM",
    },
    activation_kind="geglu_exact",
    rms_norm_kind="gemma",
    fused_add_rms_kind="gemma",
)


register_family("gemma", apply_gemma_patches)


__all__ = ["apply_gemma_family_patches", "apply_gemma_patches"]
