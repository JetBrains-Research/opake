# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Central patching orchestration for HuggingFace Transformers."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING

from opaque.api.patches.transformers._registry import detect_family, get_family_apply_fn
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

logger = logging.getLogger(__name__)


# Per-concern kwargs whose default bucket is ``kernels`` — they install
# Triton kernels that require CUDA + Triton at runtime, so the router
# auto-forces them off when those aren't available. Listed here
# (rather than computed by prefix) so the flag set is explicit and
# grep-able.
_KERNEL_KWARGS = (
    "rope",
    "rms_norm",
    "activation",
    "cross_entropy",
    "fused_linear_cross_entropy",
)


def _has_kernel_runtime() -> bool:
    """Return True iff CUDA + Triton are importable on this host — the runtime the
    Triton-only kernels (rope / rms_norm / activation / cross_entropy) need.

    Delegates to :func:`opaque.api.engine.device.fused_kernels_available` so the
    "what runs the fused kernels?" question has a single source of truth. Portable
    accelerations (grouped-GEMM MoE, chunked CE) do NOT gate on this — they run on
    MPS/CPU and default from ``performance``.
    """
    from opaque.api.engine.device import fused_kernels_available

    return fused_kernels_available()


def _is_hf_pretrained_model(model: nn.Module) -> bool:
    """Whether ``model`` is or wraps a Hugging Face pretrained model."""
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return False

    try:
        from peft import PeftMixedModel, PeftModel
    except ImportError:
        pass
    else:
        if isinstance(model, (PeftModel, PeftMixedModel)):
            get_base_model = getattr(model, "get_base_model", None)
            model = (
                get_base_model()
                if get_base_model is not None
                else model.base_model.model
            )

    return isinstance(model, PreTrainedModel)


def _patch_forward(
    target_cls: type[nn.Module] | None,
    factory: Callable | None,
    model: nn.Module | None,
) -> bool:
    """Helper to apply Liger-style global + instance forward replacement.

    Returns True if any patch was applied (either global or instance).
    """
    if target_cls is None or factory is None:
        return False

    patched = False

    # 1. Global class-level patching
    if hasattr(target_cls, "forward") and not hasattr(
        target_cls.forward, "__opaque_patched__"
    ):
        new_fwd = factory(target_cls.forward)
        new_fwd.__opaque_patched__ = True
        target_cls.forward = new_fwd
        patched = True

    # 2. Instance-level fallback patching
    if model is not None:
        for module in model.modules():
            fwd_fn = getattr(module.forward, "__func__", module.forward)
            if type(module) is target_cls and not hasattr(fwd_fn, "__opaque_patched__"):
                new_fwd = factory(type(module).forward)
                new_fwd.__opaque_patched__ = True
                module.forward = types.MethodType(new_fwd, module)
                patched = True

    return patched


def apply_transformers_model_patches(
    model: nn.Module,
    *,
    performance: bool = True,
    compat: bool = True,
    kernels: bool | None = None,
    fused_linear_cross_entropy: bool | None = None,
    **kwargs,
) -> None:
    """Apply Liger-style global + instance patching for kernels and compat wrappers.

    Three umbrella flags drive the per-concern kwargs:

    - ``performance`` — memory-efficiency patches that run on any host
      (``kv_cache`` and the conditional fused-linear-CE wrapper).
    - ``compat`` — vmap-safety wrappers (``eager_attention``, ``batchify``).
    - ``kernels`` — "use accelerated kernels" (``rope``, ``rms_norm``,
      ``activation``, ``cross_entropy``, ``grouped_moe``). Defaults to
      ``performance`` when ``None``. The flag is unconditional; the **per-kernel
      install** checks the environment (see the factory / family): the Triton-only
      kernels gate on :func:`_has_kernel_runtime` so they're never installed
      off-CUDA, while the portable ones (grouped-GEMM MoE on ``torch._grouped_mm``,
      chunked CE) run on any host.

    ``fused_linear_cross_entropy`` is promoted out of ``**kwargs`` and inherits
    from ``performance`` when unset. Its wrapper delegates to the original
    forward unless the caller explicitly requests a loss-only result.
    """
    if kernels is None:
        kernels = performance
    # Extract explicitly-passed dropout/batchify (not defaults)
    dropout_explicit = "dropout" in kwargs
    batchify_explicit = "batchify" in kwargs
    dropout = kwargs.get("dropout", compat)
    batchify = kwargs.get("batchify", compat)

    family = detect_family(model)
    apply_fn = get_family_apply_fn(family) if family is not None else None
    if family is None or apply_fn is None:
        unknown_hf_family = _is_hf_pretrained_model(model)
        if (unknown_hf_family and compat) or (
            (dropout_explicit and dropout) or (batchify_explicit and batchify)
        ):
            raise ConfigurationError(
                *(
                    "opaque: compatibility/dropout/batchify patches require a registered "
                    f"transformers family; got {family!r} ({type(model).__name__})",
                )
            )
        logger.warning(
            "opaque: no registered apply function for model family %s; "
            "no model-level patches applied.",
            family or type(model).__name__,
        )
        return

    apply_fn(
        model,
        performance=performance,
        compat=compat,
        kernels=kernels,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        **kwargs,
    )
    logger.debug("opaque: Applied model patches for %s", family)

    # ``dropout`` (compat): zero the model's dropout. DP-SGD trains without it,
    # and SDPA's fused dropout breaks vmap(grad). Model-wide traversal, so it
    # runs here rather than in the per-class factory. Opt out with
    # ``dropout=False`` (keeps the model's dropout).
    if dropout and model is not None:
        from opaque.api.patches.transformers.components.dropout import disable_dropout

        disable_dropout(model)

    # Apply batchify to PeftModel classes if needed
    if batchify and model is not None:
        try:
            import peft

            if isinstance(model, peft.PeftModel):
                cls = type(model)
                from opaque.api.patches.transformers.components.batchify import (
                    apply_batchify_patch,
                )

                apply_batchify_patch(cls, model)
        except ImportError:
            pass


__all__ = [
    "apply_transformers_model_patches",
]
