# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap-safe grouped-GEMM gating for MoE experts (transformers v5+).

Stacked-expert ``*Experts`` modules route through
``transformers.integrations.moe._grouped_mm``, whose ``_can_use_grouped_mm``
gate disqualifies the fast path on CPU under torch<=2.10 by reading
``weight.data_ptr() % 16`` (a 16-byte-alignment check for memmap'd
safetensors, see pytorch/pytorch#172440).

Under DP-SGD's ``vmap(grad(...))`` the CPU tensors are batched / grad-tracking
views with no storage, so ``data_ptr()`` raises
``RuntimeError: Cannot access data pointer of Tensor that doesn't have
storage`` and the experts forward crashes. The alignment concern only applies
to real memmap'd tensors; synthetic transform tensors are contiguous, so we
decide on grouped-GEMM availability alone. CUDA/MPS never hit the CPU branch,
so this is a CPU-under-vmap shim only.
"""

from __future__ import annotations

from opake.api.patches.torch.functorch_transform import under_functorch_transform


def apply_grouped_mm_patches(*, vmap_grouped_mm: bool = True) -> None:
    """Make ``transformers.integrations.moe._can_use_grouped_mm`` vmap-safe.

    No-op when transformers lacks the v5 grouped-MoE integration. Idempotent.
    """
    if vmap_grouped_mm is False:
        return

    try:
        import torch
        import transformers.integrations.moe as moe
    except ImportError:
        return

    orig = getattr(moe, "_can_use_grouped_mm", None)
    if orig is None or getattr(orig, "_opake_vmap_safe", False):
        return

    def _vmap_safe_can_use_grouped_mm(input, weight, offs):  # type: ignore[no-untyped-def]
        try:
            return orig(input, weight, offs)
        except RuntimeError:
            # No-storage tensor under functorch: the 16-byte-alignment guard
            # can't read data_ptr(). Alignment is irrelevant for synthetic
            # transform tensors, so fall back to availability-only.
            if under_functorch_transform():
                return hasattr(torch.nn.functional, "grouped_mm") or hasattr(
                    torch, "_grouped_mm"
                )
            raise

    _vmap_safe_can_use_grouped_mm._opake_vmap_safe = True  # type: ignore[attr-defined]
    _vmap_safe_can_use_grouped_mm._original = orig  # type: ignore[attr-defined]
    moe._can_use_grouped_mm = _vmap_safe_can_use_grouped_mm


__all__ = ["apply_grouped_mm_patches"]
