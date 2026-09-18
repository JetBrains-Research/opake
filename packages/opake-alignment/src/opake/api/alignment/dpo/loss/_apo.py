# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Anchored Preference Optimisation (APO) loss variants.

Implements the two APO variants from:
    "Anchored Preference Optimization and Contrastive Revisions: Addressing
    Underspecification in Alignment" (arXiv:2408.06266).

Both variants operate on per-example ``(chosen_logratio, rejected_logratio)``
scalars. They are pure losses:

- :func:`apo_zero_loss` — anchored push: chosen up, rejected down independently.
- :func:`apo_down_loss` — asymmetric pull: pulls chosen *toward* the chosen side
  while simultaneously pulling chosen above rejected by at least the margin.

No cross-example aggregates; NaN-injection contract holds by construction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["apo_down_loss", "apo_zero_loss"]


def apo_zero_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """APO-zero loss (arXiv:2408.06266).

    Pushes the chosen log-ratio up and the rejected log-ratio down
    independently of each other.  Mathematically:

        loss = (1 − σ(β · chosen_lr)) + σ(β · rejected_lr)

    where σ is the sigmoid function.

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    losses_chosen = 1 - F.sigmoid(beta * chosen_logratio)
    losses_rejected = F.sigmoid(beta * rejected_logratio)
    return losses_chosen + losses_rejected


def apo_down_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """APO-down loss (arXiv:2408.06266).

    Pulls the chosen log-ratio down (σ(β·chosen)) while simultaneously
    enforcing that chosen exceeds rejected by a relative margin
    (1 − σ(β·Δ)):

        loss = σ(β · chosen_lr) + (1 − σ(β · (chosen_lr − rejected_lr)))

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    losses_chosen = F.sigmoid(beta * chosen_logratio)
    losses_rejected = 1 - F.sigmoid(beta * (chosen_logratio - rejected_logratio))
    return losses_chosen + losses_rejected
