"""Unified patching for Opaque.

User-facing entry points:

- :func:`apply_model_patches` — apply per-model HF + PEFT patches.
- :func:`apply_runtime_patches` — apply global runtime patches
  (vmap masking, empty-batch handling, checkpointing).
- :func:`apply_transformers_model_patches` — HF Transformers-only
  variant.
- :func:`apply_peft_model_patches` — PEFT-only variant.

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
)

__all__ = [
    "apply_model_patches",
    "apply_peft_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "is_runtime_patched",
]
