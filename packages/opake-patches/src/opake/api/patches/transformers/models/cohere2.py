# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the cohere2 family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.cohere2.modeling_cohere2"


apply_cohere2_family_patches = make_apply_family_patches(
    family="cohere2",
    module_path=_MODULE_PATH,
)


apply_cohere2_patches = make_apply_model_patches(
    family="cohere2",
    family_apply=apply_cohere2_family_patches,
    module_path=_MODULE_PATH,
    classes={"mlp": "Cohere2MLP", "causal_lm": "Cohere2ForCausalLM"},
    activation_kind="swiglu",
    rms_norm_kind=None,
    fused_add_rms_kind=None,
)


register_family("cohere2", apply_cohere2_patches)


__all__ = ["apply_cohere2_family_patches", "apply_cohere2_patches"]
