# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the gemma2 family — built via the patch factories.

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
    vmap_eager_attention_forward_gemma2,
    vmap_sdpa_attention_forward_gemma2,
)
from opake.api.patches.transformers.components.masking import (
    apply_compact_sdpa_sliding_window_masking_patch,
)

_MODULE_PATH = "transformers.models.gemma2.modeling_gemma2"


apply_gemma2_family_patches = make_apply_family_patches(
    family="gemma2",
    module_path=_MODULE_PATH,
    eager_attention_replacement=vmap_eager_attention_forward_gemma2,
    sdpa_attention_replacement=vmap_sdpa_attention_forward_gemma2,
    masking_module_patcher=apply_compact_sdpa_sliding_window_masking_patch,
)


apply_gemma2_patches = make_apply_model_patches(
    family="gemma2",
    family_apply=apply_gemma2_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Gemma2MLP",
        "rms_norm": "Gemma2RMSNorm",
        "causal_lm": "Gemma2ForCausalLM",
    },
    activation_kind="geglu_approx",
    rms_norm_kind="gemma2",
    fused_add_rms_kind=None,
)


register_family("gemma2", apply_gemma2_patches)


__all__ = ["apply_gemma2_family_patches", "apply_gemma2_patches"]
