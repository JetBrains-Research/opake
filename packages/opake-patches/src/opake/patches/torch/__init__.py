"""PyTorch checkpoint patches — vmap-safe gradient checkpointing."""

from opake.api.patches.torch import apply_checkpoint_patch, is_checkpoint_patched

__all__ = [
    "apply_checkpoint_patch",
    "is_checkpoint_patched",
]
