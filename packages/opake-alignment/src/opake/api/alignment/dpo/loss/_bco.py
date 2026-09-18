# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""BCO (Binary Classifier Optimisation) pairwise DPO loss.

Implements the BCO-pair variant from:
    Jhunjhunwala et al., "BCO: Binary Classifier Optimisation for
    Preference Learning" (see TRL ``loss_type="bco_pair"``).

The BCO loss frames DPO as a binary classification problem with an optional
reward baseline ``delta``. TRL computes ``delta`` as a running cross-batch
mean reward and applies it externally (outside ``bco_pair``). In
``opake-alignment`` the baseline is exposed as an optional scalar keyword
argument (default ``0.0``) so the function stays strictly per-example and
matches the BCO paper's formula:

    loss = −log σ(β · chosen_lr − δ) − log σ(−(β · rejected_lr − δ))

where δ is the reward baseline. A nonzero ``delta`` must be public, separately
DP, or derived from prior DP outputs; detaching a raw cross-batch statistic is
not sufficient. When ``delta=0.0`` (the default) this is identical to TRL's
``bco_pair``. This is a pure loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["bco_loss"]


def bco_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    delta: float = 0.0,
) -> torch.Tensor:
    """BCO pairwise loss.

    The reward baseline ``delta`` corresponds to TRL's running-mean reward
    estimate. The default ``0.0`` gives the zero-baseline variant. A nonzero
    value must be public, separately DP, or derived from prior DP outputs;
    detaching a raw cross-batch statistic is not sufficient.

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w|x) − log π_ref(y_w|x)``.
        rejected_logratio: Per-example scalar ``log π(y_l|x) − log π_ref(y_l|x)``.
        beta: KL-regularisation temperature (DPO-style).
        delta: Public or DP-safe reward baseline. Default ``0.0``.

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    return -F.logsigmoid(beta * chosen_logratio - delta) - F.logsigmoid(
        -(beta * rejected_logratio - delta)
    )
