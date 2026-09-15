"""Unified patching for Opaque.

User-facing entry points:

- :func:`apply_model_patches` — apply per-model HF + PEFT patches.
- :func:`apply_runtime_patches` — apply global runtime patches
  (vmap masking, empty-batch handling, checkpointing).
- :func:`apply_transformers_model_patches` — HF Transformers-only
  variant.
- :func:`apply_peft_model_patches` — PEFT-only variant.
- :func:`set_packed_sequences` / :func:`packed_sequences`: process-level
  policy telling the vmap-safe causal-mask builder whether rows are packed
  (all valid), so the attention kernel choice is a public property under DP
  training rather than a probe of the batch.

See :mod:`opaque.patches.kernels`, :mod:`opaque.patches.transformers`,
:mod:`opaque.patches.peft`, and :mod:`opaque.patches.torch` for the
power-user submodules.
"""

from opaque.api.patches import (
    apply_model_patches,
    apply_peft_model_patches,
    apply_runtime_patches,
    apply_transformers_model_patches,
    is_runtime_patched,
    packed_sequences,
    set_packed_sequences,
)

__all__ = [
    "apply_model_patches",
    "apply_peft_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "is_runtime_patched",
    "packed_sequences",
    "set_packed_sequences",
]
