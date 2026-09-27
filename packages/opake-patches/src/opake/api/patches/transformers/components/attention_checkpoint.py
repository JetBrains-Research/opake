# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-block activation checkpointing for decoder layers.

Full-layer gradient checkpointing recomputes every projection of a decoder
layer. Checkpointing only the ``self_attn`` block keeps its input (the
normalized hidden states, position embeddings and mask) and recomputes the
Q/K/V projections, rotary embedding, attention kernel and output projection in
backward. The MLP and norms keep their regular saved activations.

The wrapper uses non-reentrant :func:`torch.utils.checkpoint.checkpoint`, which
composes with ``vmap(grad(...))`` through the checkpoint runtime patches
installed by :func:`opake.patches.apply_runtime_patches`.
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

logger = logging.getLogger(__name__)

_CHECKPOINTED_MARKER = "_opake_attention_checkpointed"


def _has_kv_cache(kwargs) -> bool:
    return (
        kwargs.get("past_key_values") is not None
        or kwargs.get("past_key_value") is not None
    )


def _make_checkpointed_attention_forward(module, forward):
    """Wrap a bound attention ``forward`` in non-reentrant checkpointing.

    As with Hugging Face gradient checkpointing, only training-mode calls are
    checkpointed. Calls also run without checkpointing when gradients are
    disabled or when a KV cache is supplied: recomputing a cache-updating
    forward would append the same keys and values twice.
    """

    @functools.wraps(forward)
    def checkpointed_forward(*args, **kwargs):
        if not module.training or not torch.is_grad_enabled() or _has_kv_cache(kwargs):
            return forward(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(
            forward, *args, use_reentrant=False, **kwargs
        )

    return checkpointed_forward


def apply_attention_checkpointing(model: nn.Module) -> int:
    """Checkpoint every decoder layer's ``self_attn`` block in place.

    Apply after all other forward replacements (for example fused LoRA QKV)
    so the checkpointed region covers the final attention forward. Idempotent.

    Args:
        model: Model whose decoder layers expose a ``self_attn`` submodule.

    Returns:
        Number of attention blocks newly wrapped.
    """
    count = 0
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if not isinstance(attn, nn.Module) or getattr(
            attn, _CHECKPOINTED_MARKER, False
        ):
            continue
        attn.forward = _make_checkpointed_attention_forward(attn, attn.forward)
        setattr(attn, _CHECKPOINTED_MARKER, True)
        count += 1
    if count:
        logger.debug(f"opake: Checkpointed {count} attention blocks")
    return count


__all__ = ["apply_attention_checkpointing"]
