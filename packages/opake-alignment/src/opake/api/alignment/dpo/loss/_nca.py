# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""NCA (Noise-Contrastive Alignment) pairwise DPO loss.

Implements the NCA-pair variant (see TRL ``loss_type="nca_pair"``):

    Derived from Noise-Contrastive Estimation (NCE); the NCA DPO variant
    treats the chosen completion as the "signal" and the rejected as noise,
    with an asymmetric 0.5-weight on the noise log-probabilities.

Formula:

    loss = −log σ(β · chosen_lr)
         − 0.5 · log σ(−β · chosen_lr)
         − 0.5 · log σ(−β · rejected_lr)

All three terms use numerically stable :func:`torch.nn.functional.logsigmoid`.

This is a pure loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["nca_loss"]


def nca_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """NCA pairwise loss.

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    cr = beta * chosen_logratio
    rr = beta * rejected_logratio
    return -F.logsigmoid(cr) - 0.5 * F.logsigmoid(-cr) - 0.5 * F.logsigmoid(-rr)
