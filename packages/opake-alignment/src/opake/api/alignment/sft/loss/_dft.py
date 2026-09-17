# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""DFT (Dynamic Fine-Tuning) per-example loss.

Implements the DFT loss from:

    "On the Generalization of SFT: A Reinforcement Learning Perspective with
    Reward Rectification", arXiv:2508.05629 (2025).

The per-token loss is the standard NLL weighted by the **stop-gradient**
(detached) softmax probability of the target token:

    p  = softmax(logits, dim=-1).gather(target).detach()   # stop-gradient
    logp = selective_log_softmax(logits, target)           # differentiable
    per_token_loss = -p * logp

The detach on ``p`` is essential: it means the gradient flows only through
``logp``, not through ``p``.  If ``p`` were not detached the gradient of the
weighted term ``-p·logp`` w.r.t. the logits would include a second term
from the derivative of ``p``, changing the loss landscape and breaking
numeric parity with the TRL reference implementation.

**DP-corrected divisor.**
TRL's ``SFTTrainer`` divides by a *batch-level* ``num_items_in_batch`` (the
total number of non-ignored tokens across the entire batch).  This is a
cross-example aggregate and is **not DP-safe** as a pre-clip divisor: its
value changes when any example in the batch changes, coupling per-example
gradients.  This implementation instead divides by this example's own
non-ignored shifted-label token count — computed
**before** gradient clipping.  This matches the per-example
pre-clip divisor rule.

``clamp(min=1)`` on the divisor guards the all-ignored-token edge case.

The output depends only on this example's data.
"""

from __future__ import annotations

import torch

from opake.api.alignment._compute_dtype import _compute_dtype
from opake.api.alignment._fused_lce import lce_available, linear_nll_sum
from opake.api.alignment.logprob._gather import selective_log_softmax

__all__ = ["dft_loss", "fused_dft_loss"]

_IGNORE_INDEX = -100


def dft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-example DFT loss with detached softmax weighting and DP-safe divisor.

    Shifts ``logits[..., :-1, :]`` against ``labels[..., 1:]``, computes
    the per-token DFT loss ``-detach(softmax_prob) * logp``, masks positions
    where the shifted label is ``-100``, and returns the **per-example mean**
    over non-ignored tokens using a per-example divisor (DP-safe).

    The weighting probability ``p`` is detached (stop-gradient), so the
    gradient of the loss w.r.t. ``logits`` equals that of ``-p.detach() *
    logp``.  This matches the TRL ``dft`` formula in spirit but uses the
    DP-corrected per-example divisor instead of TRL's batch-level
    ``num_items_in_batch``.

    Args:
        logits: Float tensor of shape ``(..., T, V)`` where ``T`` is the
            sequence length and ``V`` is the vocabulary size.
        labels: Integer tensor of shape ``(..., T)``.  Values of ``-100``
            are ignored (set by the collator for prompt / pad tokens).

    Returns:
        Float tensor of shape ``(...)`` (0-dim for a single example, ``(B,)``
        for a batch).  Each entry is
        the mean per-token loss over non-ignored shifted labels.
    """
    # Shift: logits[..., :-1, :] predicts labels[..., 1:]
    shifted_logits = logits[..., :-1, :]  # (..., T-1, V)
    shifted_labels = labels[..., 1:]  # (..., T-1)

    # Mask: 1 where the label is a real token (not -100)
    mask = (shifted_labels != _IGNORE_INDEX).to(shifted_logits.dtype)  # (..., T-1)

    # Clamp -100 → 0 so gather does not receive an out-of-range index.
    # The mask zeroes out the contribution of these positions afterwards.
    clamped_labels = shifted_labels.clamp(min=0)  # (..., T-1)

    # Per-token log-probabilities (differentiable through logits)
    logp = selective_log_softmax(shifted_logits, clamped_labels)  # (..., T-1)

    # Detached softmax probability at the target token (stop-gradient)
    # torch.softmax + gather, then detach — gradient does NOT flow through p.
    p = (
        torch.softmax(shifted_logits, dim=-1)
        .gather(-1, clamped_labels.unsqueeze(-1))
        .squeeze(-1)
        .detach()
    )  # (..., T-1), no gradient

    # Per-token DFT loss: -p * logp  (p is detached; logp is differentiable)
    per_token = -p * logp  # (..., T-1)

    # Accumulate in >= fp32 with an exact token count so eager matches fused.
    acc_dtype = _compute_dtype(per_token)
    n_valid = (shifted_labels != _IGNORE_INDEX).sum(-1).clamp(min=1)
    loss = (per_token * mask).to(acc_dtype).sum(-1) / n_valid.to(acc_dtype)  # (...)
    return loss


def fused_dft_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient :func:`dft_loss` over hidden states (fused, per-example).

    Mathematically ``dft_loss(hidden_states @ lm_head_weight.T, labels)``, but on
    CUDA + half precision with ``opake-alignment[patches]`` installed it computes
    the detached-confidence-weighted per-token CE against the projection via the
    patches fused linear-CE kernel in its ``use_token_scaling`` mode — no ``(T,
    V)`` logits are materialised and the LSE is recomputed in the backward.
    Otherwise it falls back to the eager form.

    **Per-example.** Pass one example ``(T, H)`` and drive with
    ``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path); see
    :func:`~opake.api.alignment.sft.loss.fused_nll_loss` for the per-example
    contract.

    Args:
        hidden_states: Last-layer hidden states ``(T, H)`` for one example
            (batched to ``(B, T, H)`` only by an outer ``vmap``).
        lm_head_weight: LM-head weight ``(V, H)``; logits are
            ``hidden_states @ lm_head_weight.T``.
        labels: Token-id targets ``(T,)``; ``-100`` positions are ignored.

    Returns:
        Scalar per-example DFT loss (matches :func:`dft_loss`).
    """
    if lce_available(hidden_states):
        ce_sum = linear_nll_sum(
            hidden_states, lm_head_weight, labels, use_token_scaling=True
        )
        n_valid = (labels[..., 1:] != _IGNORE_INDEX).sum(-1).clamp(min=1)
        return ce_sum / n_valid
    return dft_loss(hidden_states @ lm_head_weight.transpose(-2, -1), labels)
