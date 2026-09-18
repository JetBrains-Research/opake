# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""SPPO (Self-Play Preference Optimisation) hard-label loss.

Implements the SPPO-hard variant from:
    Wu et al., "Self-Play Preference Optimisation for Language Model
    Alignment" (see TRL ``loss_type="sppo_hard"``).

SPPO frames preference learning as a two-player zero-sum game.  The
"hard" variant uses the Nash-equilibrium target ±0.5 (as a probability,
i.e. 1/(2β) in log-ratio space) and minimises the squared deviation of
each log-ratio from its target:

    loss = (chosen_lr − 0.5 / β)² + (rejected_lr + 0.5 / β)²

The division ``0.5 / beta`` is a true floating-point division (not integer
floor division); ``beta`` is ``float`` so this is always correct.

This is a pure loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["sppo_loss"]


def sppo_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """SPPO hard-label loss.

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).  The Nash-equilibrium
            target in log-ratio space is ``0.5 / beta``; ``beta`` must be
            non-zero.

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    target = 0.5 / beta
    return (chosen_logratio - target).pow(2) + (rejected_logratio + target).pow(2)
