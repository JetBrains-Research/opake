# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""WPO per-example weighting — Weighted Preference Optimization.

Implements the policy-probability reweighting from:

    Zhou, W., Agrawal, R., Zhang, S., Indurthi, S. R., Zhao, S., Song, K.,
    Xu, S., & Zhu, C. (2024). WPO: Enhancing RLHF with Weighted Preference
    Optimization. arXiv:2406.11827.

WPO simulates on-policy preference learning under an off-policy dataset by
weighting each preference pair by how likely the *current* policy is to have
produced the completion.  Equation 2 aligns each realised token probability by
the collision probability of the policy distribution before averaging::

    log_denom = logsumexp(2 * logits) - 2 * logsumexp(logits)
    aligned_logp = per_token_logps - log_denom
    weight = exp(sum(masked aligned_logp) / completion_token_count)

The computation runs under ``no_grad``, so the weight acts purely as a
per-example reweighting of the loss, not as an additional learnable path.
"""

from __future__ import annotations

import torch

from opake.api.alignment._compute_dtype import _compute_dtype

__all__ = ["wpo_weights"]


def wpo_weights(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-example WPO weight (arXiv:2406.11827).

    When ``logits`` are provided, computes the weight-aligned policy probability
    from Equation 2 of WPO and returns its detached geometric mean::

        log_denom = logsumexp(2 * logits, -1) - 2 * logsumexp(logits, -1)
        aligned_logp = per_token_logps - log_denom
        return exp(sum(masked aligned_logp) / completion_token_count)

    Omitting ``logits`` retains the unaligned geometric-mean weighting for
    backward compatibility. The result is **detached**: it contributes no
    gradient and serves only as a per-example multiplicative reweighting of the
    downstream loss.

    Args:
        per_token_logps: Per-token log-probabilities of the completion under
            the current policy. Shape ``(..., T)``; the leading dims may be
            empty (per-example under ``vmap``) or a batch axis.
        completion_mask: Tensor of shape ``(..., T)``; non-zero where a token
            belongs to the completion span. Cast to the logp dtype before
            multiplying.
        logits: Optional policy logits of shape ``(..., T, V)`` corresponding
            to ``per_token_logps``. When present, the WPO weight-alignment term
            is computed over the vocabulary dimension.

    Returns:
        Detached per-example weight tensor of shape ``(...)`` (one weight per
        sequence). All-zero mask rows are protected by ``clamp(min=1)`` so
        there is no division by zero.
    """
    acc_dtype = _compute_dtype(per_token_logps)
    if logits is not None:
        acc_dtype = torch.promote_types(acc_dtype, _compute_dtype(logits))

    mask = completion_mask.to(torch.bool)
    token_count = completion_mask.to(torch.bool).sum(dim=-1).clamp(min=1)
    with torch.no_grad():
        aligned_logps = per_token_logps.to(acc_dtype)
        if logits is not None:
            compute_logits = logits.to(acc_dtype)
            log_denom = torch.logsumexp(
                2.0 * compute_logits, dim=-1
            ) - 2.0 * torch.logsumexp(compute_logits, dim=-1)
            aligned_logps = aligned_logps - log_denom
        masked_logps = torch.where(mask, aligned_logps, 0.0)
        mean_logps = masked_logps.sum(dim=-1) / token_count.to(acc_dtype)
        weights = mean_logps.exp()
    return weights
