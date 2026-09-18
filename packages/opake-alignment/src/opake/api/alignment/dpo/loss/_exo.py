# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""EXO (Efficient Exact Optimisation) pairwise DPO loss.

Implements the simplified pairwise EXO objective from:
    Ji et al., "Towards Efficient Exact Optimization of Language Model
    Alignment" (arXiv:2402.00856, Eq. 16; see TRL ``loss_type="exo_pair"``).

The EXO loss is a KL-weighted cross-entropy between the soft preference
distribution and the label-smoothed target distribution.  With label
smoothing ``ε``, the target places probability ``(1 − ε)`` on the chosen
completion and ``ε`` on the rejected:

    loss = σ(βΔ) · (log σ(βΔ) − log(1 − ε))
         + σ(−βΔ) · (log σ(−βΔ) − log ε)

where Δ = chosen_logratio − rejected_logratio and all logarithms and
sigmoids are computed in the numerically stable log-domain via
:func:`torch.nn.functional.logsigmoid`.

When ``label_smoothing=0`` the target denominator ``log(ε)`` diverges, so
the implementation silently floors ``ε`` to ``1e-3`` — identical to TRL's
behaviour.

This is a pure loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["exo_loss"]


def exo_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    label_smoothing: float = 1e-3,
) -> torch.Tensor:
    """EXO pairwise loss (arXiv:2402.00856, simplified pairwise variant).

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).
        label_smoothing: Label-smoothing coefficient ε ∈ (0, 0.5).
            When 0 or negative, silently clamped to ``1e-3`` (same as TRL)
            to keep ``log(ε)`` finite.

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    ls = label_smoothing if label_smoothing > 0 else 1e-3
    logits = beta * (chosen_logratio - rejected_logratio)
    return F.sigmoid(logits) * (F.logsigmoid(logits) - math.log(1 - ls)) + F.sigmoid(
        -logits
    ) * (F.logsigmoid(-logits) - math.log(ls))
