# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the GPT-2 family — built via the patch factories.

GPT-2 uses absolute positional embeddings, standard GELU activations,
and standard LayerNorm (no SwiGLU/GeGLU, no RMSNorm, no RoPE).  The
family-level shims are disabled (``*_replacement=None``): GPT-2 attention
is not the llama-shaped GQA ``eager_attention_forward`` the default vmap
shim expects (it has no ``num_key_value_groups``), so applying it raises.
Only the per-model concerns apply:

- ``batchify`` (compat): transformers ≥ 4.47 added a ``logits_to_keep``
  slice (``hidden_states[:, slice_indices, :]``) that assumes a 3-D
  batch dim.  Under vmap the batch dim is absent, causing an IndexError.
- ``kv_cache`` (performance): skips per-forward DynamicCache allocation
  during training, which also avoids the cache's circular-reference
  memory leak under vmap.

Registration: this module calls ``register_family`` at import time —
the same mechanism downstream users follow to add their own families.
"""

from __future__ import annotations

from opake.api.patches.transformers._factory import make_apply_model_patches
from opake.api.patches.transformers._family import make_apply_family_patches
from opake.api.patches.transformers._registry import register_family

_MODULE_PATH = "transformers.models.gpt2.modeling_gpt2"


apply_gpt2_family_patches = make_apply_family_patches(
    family="gpt2",
    module_path=_MODULE_PATH,
    repeat_kv_replacement=None,
    eager_attention_replacement=None,
    rope_replacement=None,
)


apply_gpt2_patches = make_apply_model_patches(
    family="gpt2",
    family_apply=apply_gpt2_family_patches,
    module_path=_MODULE_PATH,
    classes={"causal_lm": "GPT2LMHeadModel"},
    # No activation_kind, rms_norm_kind, fused_add_rms_kind: GPT-2 uses
    # standard GELU and LayerNorm with no custom kernel paths.
)


register_family("gpt2", apply_gpt2_patches)


__all__ = ["apply_gpt2_family_patches", "apply_gpt2_patches"]
