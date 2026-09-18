"""Patches transformers runtime façade."""

from opake.api.patches.transformers.runtime import (
    apply_collator_patches,
    apply_grouped_mm_patches,
    apply_masking_patches,
)

__all__ = [
    "apply_collator_patches",
    "apply_grouped_mm_patches",
    "apply_masking_patches",
]
