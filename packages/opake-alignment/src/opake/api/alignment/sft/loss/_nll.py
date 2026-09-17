# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Standard causal-LM NLL (cross-entropy) loss.

Implements the per-example negative log-likelihood loss used in supervised
fine-tuning (SFT) for causal language models.  The shifted-labels
convention matches that of HuggingFace ``transformers`` and TRL:

    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

Tokens with ``shifted_labels == -100`` (the ignore index) are excluded from
the per-example mean, as in ``torch.nn.CrossEntropyLoss(ignore_index=-100)``.

The per-example divisor is this
example's own non-ignored token count — a quantity derived purely from this
example's data.  This guarantees that swapping one example's data changes
only that example's gradient, satisfying the per-record sensitivity bound
required by DP-SGD / DP-FTRL.  Unlike TRL's ``SFTTrainer``, which divides
by a *batch-level* ``num_items_in_batch``, this function performs **pre-clip
division** by the per-example token count.

``clamp(min=1)`` on the divisor guards the all-ignored-token edge case
(prevents division-by-zero and produces 0.0 for an all-masked row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opake.api.alignment._compute_dtype import _compute_dtype
from opake.api.alignment._fused_lce import lce_available, linear_nll_sum
from opake.api.alignment.logprob._gather import selective_log_softmax

_IGNORE_INDEX = -100

if TYPE_CHECKING:
    import torch

__all__ = ["fused_nll_loss", "nll_loss"]


def nll_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-example causal-LM NLL loss with per-example DP-safe divisor.

    Shifts ``logits[..., :-1, :]`` against ``labels[..., 1:]`` (standard
    causal-LM convention), masks positions where the shifted label is
    ``-100``, computes ``selective_log_softmax`` at the (clamped) shifted
    labels, and returns the **per-example mean** of the negative per-token
    log-probabilities over non-ignored tokens.

    The exact shifted-label token count is computed per example from this
    example's data only.

    Args:
        logits: Float tensor of shape ``(..., T, V)`` where ``T`` is the
            sequence length and ``V`` is the vocabulary size.  Leading dims
            ``(...)`` may be absent (per-example) or a batch axis ``(B,)``.
        labels: Integer tensor of shape ``(..., T)`` matching the leading
            dims of ``logits``.  Values of ``-100`` are ignored.

    Returns:
        Float tensor of shape ``(...)`` (0-dim for a single example, shape
        ``(B,)`` for a batch).  Each entry is the per-example mean NLL over
        non-ignored shifted tokens.
    """
    # Shift: logits[..., :-1, :] predicts labels[..., 1:]
    shifted_logits = logits[..., :-1, :]  # (..., T-1, V)
    shifted_labels = labels[..., 1:]  # (..., T-1)

    # Mask: 1 where the label is a real token (not -100)
    mask = (shifted_labels != _IGNORE_INDEX).to(shifted_logits.dtype)  # (..., T-1)

    # Clamp -100 → 0 so gather does not receive an out-of-range index.
    # The mask zeros out the contribution of these positions.
    clamped_labels = shifted_labels.clamp(min=0)  # (..., T-1)

    # Per-token log-probabilities at the (clamped) target indices
    logp = selective_log_softmax(shifted_logits, clamped_labels)  # (..., T-1)

    # Accumulate in >= fp32 with an exact token count so eager matches fused.
    acc_dtype = _compute_dtype(logp)
    n_valid = (shifted_labels != _IGNORE_INDEX).sum(-1).clamp(min=1)
    nll = (-logp * mask).to(acc_dtype).sum(-1) / n_valid.to(acc_dtype)  # (...)
    return nll


def fused_nll_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient :func:`nll_loss` over hidden states (fused, per-example).

    Mathematically ``nll_loss(hidden_states @ lm_head_weight.T, labels)``, but on
    CUDA + half precision with ``opake-alignment[patches]`` installed it computes
    the per-token CE against the projection via the patches fused linear-CE
    kernel — no ``(T, V)`` logits are materialised and the LSE is recomputed in
    the backward. Otherwise it falls back to the eager form.

    **Per-example.** Pass one example ``(T, H)`` and drive with
    ``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path); the kernel's merged
    vmap rules then make the whole microbatch one forward + one backward kernel.
    Unlike :func:`nll_loss`, this is *not* meant to be called directly on a batch
    axis (the fused path would collapse it to a single scalar).

    Args:
        hidden_states: Last-layer hidden states ``(T, H)`` for one example
            (batched to ``(B, T, H)`` only by an outer ``vmap``).
        lm_head_weight: LM-head weight ``(V, H)``; logits are
            ``hidden_states @ lm_head_weight.T``.
        labels: Token-id targets ``(T,)``; ``-100`` positions are ignored.

    Returns:
        Scalar per-example mean NLL (matches :func:`nll_loss`).
    """
    if lce_available(hidden_states):
        ce_sum = linear_nll_sum(hidden_states, lm_head_weight, labels)
        n_valid = (labels[..., 1:] != _IGNORE_INDEX).sum(-1).clamp(min=1)
        return ce_sum / n_valid
    return nll_loss(hidden_states @ lm_head_weight.transpose(-2, -1), labels)
