# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Rotary-position-embedding replacements backed by Opake kernels."""

from __future__ import annotations

import torch


def _rotate_half(x):
    """Rotates half the hidden dims of the input (standard HF rotate_half)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _opake_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """RoPE using Opake Triton kernel.

    Replaces HF's apply_rotary_pos_emb at module level. Uses Opake_RoPE_QK
    which processes Q and K together with GQA support.

    Falls back to PyTorch when:
    - cos/sin cannot be reduced to 2D (e.g., variable position_ids)
    - tensors are not on CUDA (Triton requires CUDA)
    """
    # Triton kernels require CUDA
    if not q.is_cuda:
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
        k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
        return q_embed, k_embed

    from opake.api.patches.kernels.rope_embedding import Opake_RoPE_QK

    # HF provides cos/sin as (batch, seq_len, head_dim) or (seq_len, head_dim).
    # The kernel needs 2D (seq_len, head_dim) after squeeze.
    cos_2d = cos.unsqueeze(unsqueeze_dim).squeeze()
    sin_2d = sin.unsqueeze(unsqueeze_dim).squeeze()

    if cos_2d.dim() == 2:  # noqa: PLR2004 - kernel accepts a 2D RoPE table
        return Opake_RoPE_QK.apply(q, k, cos_2d, sin_2d, None)

    # Fallback to PyTorch for batched cos/sin (e.g., variable position_ids)
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed
