"""PyTorch-version behavioral compatibility patches.

Currently exposes:

- :mod:`opake.api.patches.torch.checkpoint` — patches that make
  ``torch.utils.checkpoint`` compatible with ``vmap(grad(...))``, applied only
  where the running PyTorch lacks native support.

These are temporary shims until the corresponding fixes land in upstream
PyTorch.
"""

from opake.api.patches.torch.checkpoint import (
    apply_checkpoint_patch,
    is_checkpoint_patched,
)

__all__ = [
    "apply_checkpoint_patch",
    "is_checkpoint_patched",
]
