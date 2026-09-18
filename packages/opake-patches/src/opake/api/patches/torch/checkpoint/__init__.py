# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Enable non-reentrant gradient checkpointing under vmap(grad(...)).

``apply_checkpoint_patch`` installs only the patches the running PyTorch needs.
When torch supports the composition natively, only the HuggingFace glue is
applied; the torch-core patches are skipped (and several would be redundant, e.g.
re-scoping the already-scoped higher-order guard). Each patch lives in its own
module named for the surface it patches:

- ``saved_tensor_hooks_guard`` / ``save_on_cpu``  -- gated on the guard scoping
- ``noop_save_inputs``                            -- old-arch only (self-skips)
- ``create_graph`` / ``reparametrize_recompute``  -- gated on native support
- ``huggingface``                                 -- always (if transformers imports)

See :mod:`.native_support` for the capability probes.
"""

from __future__ import annotations

import logging

from opake.api.patches.torch.checkpoint import (
    create_graph,
    huggingface,
    native_support,
    noop_save_inputs,
    reparametrize_recompute,
    save_on_cpu,
    saved_tensor_hooks_guard,
)

logger = logging.getLogger(__name__)

_is_checkpoint_patched = False


def apply_checkpoint_patch(*, vmap_checkpointing: bool = True) -> None:
    """Patch PyTorch (as needed) to allow gradient checkpointing under
    vmap(grad(...)). Idempotent; a no-op when ``vmap_checkpointing`` is False.
    """
    global _is_checkpoint_patched
    if _is_checkpoint_patched or not vmap_checkpointing:
        return

    # The two capability probes are independent: a torch may have the
    # param-lifetime fix without the guard scoping, or vice versa. Gate each
    # backport on its own probe rather than short-circuiting on native support.
    if not native_support.saved_tensor_hooks_guard_scoped():
        saved_tensor_hooks_guard.apply()
        save_on_cpu.apply()

    if not native_support.native_checkpoint_support():
        noop_save_inputs.apply()  # self-skips on torch >= 2.12
        create_graph.apply()
        reparametrize_recompute.apply()

    huggingface.apply()

    _is_checkpoint_patched = True
    logger.debug("opake: applied checkpoint+functorch compatibility patches.")


def is_checkpoint_patched() -> bool:
    """True once :func:`apply_checkpoint_patch` has run."""
    return _is_checkpoint_patched


__all__ = ["apply_checkpoint_patch", "is_checkpoint_patched"]
