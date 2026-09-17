# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the qwen3 family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.qwen3.modeling_qwen3"


apply_qwen3_family_patches = make_apply_family_patches(
    family="qwen3",
    module_path=_MODULE_PATH,
)


apply_qwen3_patches = make_apply_model_patches(
    family="qwen3",
    family_apply=apply_qwen3_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "Qwen3MLP",
        "rms_norm": "Qwen3RMSNorm",
        "decoder_layer": "Qwen3DecoderLayer",
        "causal_lm": "Qwen3ForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("qwen3", apply_qwen3_patches)


__all__ = ["apply_qwen3_family_patches", "apply_qwen3_patches"]
