"""Token-level telemetry: prediction entropy and next-token accuracy.

These metrics are detached but un-noised; release is outside Opake's DP
accounting. Masked reductions handle all-masked batches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["entropy_from_logits", "mean_token_accuracy"]


def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-position Shannon entropy over masked (shifted) positions.

    Logits and mask are shifted so that position ``t`` scores the predictive
    distribution over the token at ``t + 1`` (standard causal-LM next-token
    alignment), matching :func:`mean_token_accuracy`. Callers therefore pass the
    full-length logits and mask; the shift happens here.

    Args:
        logits: Logits of shape ``(..., seq, vocab)``; the last axis is the
            categorical distribution per position.
        mask: Boolean/float mask of shape ``(..., seq)`` selecting which
            (shifted) target positions contribute to the mean.

    Returns:
        A detached scalar tensor: the entropy averaged over positions where the
        shifted ``mask`` is truthy (``0`` when no position is unmasked).
    """
    logits = logits[..., :-1, :]
    mask = mask[..., 1:]
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = mask.to(entropy.dtype)
    total = (entropy * mask).sum()
    return (total / mask.sum().clamp(min=1)).detach()


def mean_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean next-token prediction accuracy over masked (shifted) positions.

    Logits, labels, and mask are shifted so that position ``t`` predicts the
    token at ``t + 1`` (standard causal-LM next-token alignment).

    Args:
        logits: Logits of shape ``(..., seq, vocab)``.
        labels: Target token ids of shape ``(..., seq)``.
        mask: Boolean/float mask of shape ``(..., seq)`` selecting which
            (shifted) target positions count toward accuracy.

    Returns:
        A detached scalar tensor: the fraction of unmasked positions whose
        argmax prediction matches the label (``0`` when none are unmasked).
    """
    logits = logits[..., :-1, :]
    labels = labels[..., 1:]
    mask = mask[..., 1:]
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask.bool()
    mask = mask.to(logits.dtype)
    return (correct.sum() / mask.sum().clamp(min=1)).detach()
