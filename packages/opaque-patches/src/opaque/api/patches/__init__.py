"""Unified patching for opaque.

This package provides:
- Model-level patches: `apply_model_patches()`
- Runtime-level patches: `apply_runtime_patches()`
"""

import logging

import torch.nn as nn

from opaque.api.patches.peft import apply_peft_model_patches
from opaque.api.patches.transformers._router import apply_transformers_model_patches

logger = logging.getLogger(__name__)

_runtime_patches_applied: bool = False


def apply_model_patches(
    model: nn.Module,
    *,
    performance: bool = True,
    compat: bool = True,
    peft: bool = True,
    fused_linear_cross_entropy: bool | None = None,
    **kwargs,
) -> None:
    """Apply global and instance-level patches for a specific model.

    This is a convenience orchestrator that handles both Transformers and
    PEFT natively.  Users with purely custom non-Transformers
    architectures can import and invoke ``apply_peft_model_patches``
    directly from the root namespace to apply LoRA kernels.

    Three umbrella flags drive the per-concern ``**kwargs``:

    - ``performance`` (default ``True``) — memory-efficiency patches
      that run on any host (``kv_cache`` and conditional fused linear CE).
    - ``compat`` (default ``True``) — vmap-safety wrappers
      (``eager_attention``, ``batchify``).
    - ``kernels`` (kwarg, defaults to ``performance``) — Triton kernel
      patches that require CUDA + Triton (``rope``, ``rms_norm``,
      ``activation``, ``cross_entropy``). Forced to ``False`` when
      CUDA / Triton can't be imported, so ``performance=True`` keeps
      ``kv_cache`` enabled on CPU / MPS hosts.

    ``cross_entropy`` installs the non-fused CE kernel via
    ``loss_function``; logits remain materialized, so callers that read
    ``outputs.logits`` (``compute_metrics``,
    ``preprocess_logits_for_metrics``, eval loops) continue to work.

    ``fused_linear_cross_entropy`` installs a conditional causal-LM forward.
    It inherits from ``performance`` when unset; pass ``False`` to disable it.
    The wrapper delegates to the original forward unless a labeled call sets
    ``loss_only=True``, which permits the optimized branch to return
    ``logits=None``.
    """
    global _runtime_patches_applied
    if not _runtime_patches_applied:
        apply_runtime_patches(performance=performance, compat=compat, **kwargs)

    try:
        from opaque.api.patches.transformers._router import (
            apply_transformers_model_patches,
        )

        apply_transformers_model_patches(
            model,
            performance=performance,
            compat=compat,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            **kwargs,
        )
    except ImportError:
        logger.debug("opaque: Hugging Face kernel patches not available.")

    if peft:
        try:
            from opaque.api.patches.peft import apply_peft_model_patches

            apply_peft_model_patches(
                model, performance=performance, compat=compat, **kwargs
            )
        except ImportError:
            pass


def apply_runtime_patches(
    *, performance: bool = True, compat: bool = True, **kwargs
) -> None:
    """Apply global runtime patches.

    Per-concern compat flags (default-on with ``compat=True``):
    ``vmap_masking`` (vmap-safe causal-mask builders),
    ``empty_batches`` (collator handling for Poisson-sampled empty
    batches), ``vmap_checkpointing`` (gradient-checkpointing shim),
    ``vmap_grouped_mm`` (vmap-safe grouped-GEMM gate for MoE experts).
    Per-model CE patches live on :func:`apply_model_patches`.
    """
    global _runtime_patches_applied
    _runtime_patches_applied = True

    vmap_masking = kwargs.get("vmap_masking", compat)
    empty_batches = kwargs.get("empty_batches", compat)
    vmap_checkpointing = kwargs.get("vmap_checkpointing", compat)
    vmap_grouped_mm = kwargs.get("vmap_grouped_mm", compat)

    if vmap_masking:
        try:
            from opaque.api.patches.transformers.runtime.masking import (
                apply_masking_patches,
            )

            apply_masking_patches(vmap_masking=vmap_masking)
        except ImportError:
            pass

    if vmap_grouped_mm:
        try:
            from opaque.api.patches.transformers.runtime.moe import (
                apply_grouped_mm_patches,
            )

            apply_grouped_mm_patches(vmap_grouped_mm=vmap_grouped_mm)
        except ImportError:
            pass

    if empty_batches:
        try:
            from opaque.api.patches.transformers.runtime.collator import (
                apply_collator_patches,
            )

            apply_collator_patches(empty_batches=empty_batches)
        except ImportError:
            pass

    if vmap_checkpointing:
        try:
            from opaque.api.patches.torch import apply_checkpoint_patch

            apply_checkpoint_patch(vmap_checkpointing=vmap_checkpointing)
        except ImportError:
            logger.warning(
                "opaque: checkpoint+functorch patches unavailable; "
                "gradient checkpointing under vmap(grad(...)) may break.",
                exc_info=True,
            )


def is_runtime_patched() -> bool:
    """``True`` once :func:`apply_runtime_patches` has run in this interpreter.

    ``DPTrainer`` applies the runtime patches during construction; this lets a
    script that drives HF primitives without ``DPTrainer`` check whether the
    global shims are installed.
    """
    return _runtime_patches_applied


__all__ = [
    "apply_model_patches",
    "apply_peft_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "is_runtime_patched",
]
