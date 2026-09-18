"""SimPO: simple, reference-free, length-normalized preference loss with a margin.

SimPO (Meng et al., "SimPO: Simple Preference Optimization with a
Reference-Free Reward", 2024) drops the reference model and scores each response
by its **length-normalized** average log-probability, with a target reward
margin ``gamma``::

    L = -log σ(β·(r_w - r_r) - γ),   r = (1/|y|) · log π(y)

It reduces to the length-normalized DPO sigmoid loss at ``gamma=0``. The
length-normalization and reference-free reward are the caller's responsibility:
pass log-ratios already divided by each completion's token count (use
``sequence_logp(..., length_normalized=True)``) and, being reference-free,
formed from the policy log-prob alone (no reference subtraction).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["simpo_loss"]


def simpo_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    gamma: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-example SimPO loss.

    Computes ``L = -log σ(m)·(1 - ε) - log σ(-m)·ε`` with
    ``m = β·(chosen_logratio - rejected_logratio) - γ``. At ``gamma=0`` and
    ``label_smoothing=0`` this is the length-normalized DPO sigmoid loss.

    Args:
        chosen_logratio: Per-example length-normalized reward for the chosen
            completion — the policy log-prob divided by the completion length.
            Reference-free, so no reference is subtracted. Shape ``(B,)`` or
            ``()``.
        rejected_logratio: Same for the rejected completion.
        beta: Reward scale (SimPO β).
        gamma: Target reward margin subtracted inside the sigmoid; ``0.0``
            (default) recovers the marginless length-normalized sigmoid.
        label_smoothing: Conservative label-smoothing coefficient ε ∈ [0, 0.5).

    Returns:
        Per-example scalar loss, same shape as the inputs.
    """
    margin = beta * (chosen_logratio - rejected_logratio) - gamma
    return (
        -F.logsigmoid(margin) * (1.0 - label_smoothing)
        - F.logsigmoid(-margin) * label_smoothing
    )
