# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-sequence completion log-probability.

:func:`sequence_logp` is the standard DPO / causal-LM per-sequence logp: it
applies the next-token shift (predictions ``logits[..., :-1, :]`` align with
targets ``input_ids[..., 1:]``), gathers per-token log-probs via
:func:`selective_log_softmax`, masks to the completion span, and sums over the
sequence axis.

Negative-axis indexing is used throughout so it works both when called
per-example (``logits`` ``(T, V)``, ``input_ids`` ``(T,)``) and when called on a
batched input (``logits`` ``(B, T, V)``, ``input_ids`` ``(B, T)``).

When ``ld_alpha`` is set, the function implements the LD-DPO
length-desensitised split from equations (9)-(12) of
https://arxiv.org/abs/2409.06411. The caller supplies the paper's public length
(the shorter completion length) through the compatibility name
``shared_prefix_len``.

"""

from __future__ import annotations

import math

import torch

from opake.api.alignment._compute_dtype import _compute_dtype
from opake.api.alignment._fused_lce import lce_available, linear_nll_sum
from opake.exceptions import ConfigurationError

from ._gather import selective_log_softmax

__all__ = ["fused_sequence_logp", "sequence_logp"]


def sequence_logp(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    ld_alpha: float | None = None,
    shared_prefix_len: int | torch.Tensor | None = None,
    length_normalized: bool = False,
) -> torch.Tensor:
    """Sum (or per-token mean) of completion-token log-probabilities per sequence.

    Applies the causal-LM shift, computes per-token log-probs, multiplies by
    the (shifted) completion mask, and sums over the sequence (last remaining)
    axis.

    Args:
        logits: Float tensor of shape ``(..., T, V)``. The ``(...)`` leading
            dims may be empty (per-example under ``vmap``) or a batch axis.
        input_ids: Integer tensor of shape ``(..., T)`` (token ids).
        completion_mask: Tensor of shape ``(..., T)``; non-zero where a token
            belongs to the completion span and should contribute to the logp.
            It is cast to the logits dtype before multiplying.
        ld_alpha: LD-DPO (arXiv:2409.06411) length-desensitisation coefficient
            in ``[0, 1]``.
            When set, completion tokens beyond ``shared_prefix_len`` are weighted
            by ``ld_alpha`` instead of ``1.0``, damping
            the verbose tail's contribution; ``ld_alpha=1.0`` recovers the plain
            masked sum. The fused path does not support it — LD-DPO needs the
            per-token log-probs, so use this eager ``sequence_logp``.
        shared_prefix_len: LD-DPO public length (the shorter completion length),
            retained under this compatibility name. An ``int`` or a per-example
            tensor broadcasting against the shifted position axis. Required when
            ``ld_alpha`` is set; ignored otherwise. This single-sequence helper
            cannot derive it: callers claiming LD-DPO must pass the minimum of
            the paired completion lengths.
        length_normalized: Divide the per-sequence logp by its completion-token
            count, giving the per-token mean reward ``(1/|y|)·log π(y)`` used by
            the reference-free heads (SimPO / ORPO). Default ``False`` (plain sum).

    Returns:
        Float tensor of shape ``(...)`` (one logp per sequence; a per-token mean
        when ``length_normalized``).

    Raises:
        ValueError: If ``ld_alpha`` is outside ``[0, 1]`` or is set without
            ``shared_prefix_len``.
    """
    # Causal-LM shift: predict token t+1 from the logits at position t.
    shifted_logits = logits[..., :-1, :]
    target_ids = input_ids[..., 1:]
    target_mask = completion_mask[..., 1:]
    target_weight = target_mask.to(shifted_logits.dtype)
    per_token_logp = selective_log_softmax(shifted_logits, target_ids)

    weight = target_weight
    if ld_alpha is not None:
        if not math.isfinite(ld_alpha) or not 0.0 <= ld_alpha <= 1.0:
            raise ConfigurationError(
                *(f"ld_alpha (LD-DPO) must be finite and in [0, 1], got {ld_alpha}",)
            )
        if shared_prefix_len is None:
            raise ConfigurationError(*("ld_alpha (LD-DPO) requires shared_prefix_len",))
        completion_pos = target_mask.to(torch.bool).cumsum(dim=-1)
        prefix_len = shared_prefix_len
        if isinstance(prefix_len, torch.Tensor):
            while prefix_len.ndim < completion_pos.ndim:
                prefix_len = prefix_len.unsqueeze(-1)
        is_prefix = (completion_pos >= 1) & (completion_pos <= prefix_len)
        ld_weight = torch.where(
            is_prefix,
            torch.ones((), dtype=per_token_logp.dtype, device=per_token_logp.device),
            torch.full(
                (), ld_alpha, dtype=per_token_logp.dtype, device=per_token_logp.device
            ),
        )
        weight = target_weight * ld_weight

    acc_dtype = _compute_dtype(per_token_logp)
    logp = (per_token_logp * weight).to(acc_dtype).sum(dim=-1)
    if length_normalized:
        token_count = target_mask.to(torch.bool).sum(-1).clamp(min=1)
        logp = logp / token_count.to(acc_dtype)
    return logp


def fused_sequence_logp(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    length_normalized: bool = False,
) -> torch.Tensor:
    """Memory-efficient :func:`sequence_logp` over hidden states (fused, per-example).

    Mathematically ``sequence_logp(hidden_states @ lm_head_weight.T, input_ids,
    completion_mask)``, but on CUDA + half precision with
    ``opake-alignment[patches]`` installed it computes the completion log-prob
    via the patches fused linear-CE kernel — no ``(T, V)`` logits are
    materialised and the LSE is recomputed in the backward. Otherwise it falls
    back to the eager form.

    Since ``sequence_logp = Σ_completion log p = −Σ_completion CE``, the kernel is
    the preference-logp kernel: encode the completion span as the kept labels
    (everything else → ``-100``), take the kernel's CE sum, and negate.

    **Per-example.** Pass one sequence ``(T, H)`` and drive with
    ``vmap(grad(...))`` (how ``train_dpo`` already computes logp inside
    ``clipped_grad``); the kernel's merged vmap rules then make the chosen /
    rejected microbatch one forward + one backward kernel. Unlike
    :func:`sequence_logp`, it is *not* meant to be called directly on a batch
    axis. The LD-DPO ``ld_alpha`` split is not supported on the fused path; use
    the eager :func:`sequence_logp` for that.

    Args:
        hidden_states: Last-layer hidden states ``(T, H)`` for one sequence
            (batched to ``(B, T, H)`` only by an outer ``vmap``).
        lm_head_weight: LM-head weight ``(V, H)``; logits are
            ``hidden_states @ lm_head_weight.T``.
        input_ids: Token ids ``(T,)``.
        completion_mask: ``(T,)``; non-zero on completion-span tokens.
        length_normalized: Divide by the completion-token count for the per-token
            mean reward (SimPO / ORPO); matches ``sequence_logp``.

    Returns:
        Scalar completion log-prob (per-token mean when ``length_normalized``);
        matches :func:`sequence_logp`.
    """
    if lce_available(hidden_states):
        # Completion tokens keep their id; everything else → ignore_index, so the
        # kernel's CE sum is exactly −Σ_completion logp. The kernel applies the
        # next-token shift internally, matching sequence_logp's shifted mask.
        masked_labels = torch.where(
            completion_mask.to(torch.bool),
            input_ids,
            torch.full_like(input_ids, -100),
        )
        seq_logp = -linear_nll_sum(hidden_states, lm_head_weight, masked_labels)
        if length_normalized:
            count = completion_mask[..., 1:].sum(-1).clamp(min=1)
            seq_logp = seq_logp / count.to(seq_logp.dtype)
        return seq_logp
    logits = hidden_states @ lm_head_weight.transpose(-2, -1)
    return sequence_logp(
        logits, input_ids, completion_mask, length_normalized=length_normalized
    )
