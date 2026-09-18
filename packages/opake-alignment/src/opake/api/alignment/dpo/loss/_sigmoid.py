"""DPO sigmoid (logistic) loss — Rafailov et al., 2023.

Implements the standard DPO loss from:

    Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., &
    Finn, C. (2023). Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model. NeurIPS 2023.

    Formula (with optional label smoothing):
        loss = -logsigmoid(β·Δ) · (1 − ε) − logsigmoid(−β·Δ) · ε
    where Δ = chosen_logratio − rejected_logratio.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["sigmoid_loss"]


def sigmoid_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-example DPO sigmoid (logistic) loss.

    Implements the original DPO objective from Rafailov et al. 2023 with
    optional label smoothing. The function is elementwise and works on both
    batched ``(B,)`` inputs and 0-dim scalars.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response,
            ``log π(y_w | x) − log π_ref(y_w | x)``. Shape ``(B,)`` or ``()``.
        rejected_logratio: Per-example log-ratio for the rejected response,
            ``log π(y_l | x) − log π_ref(y_l | x)``. Same shape as
            ``chosen_logratio``.
        beta: DPO temperature (reference-deviation strength). Positive float;
            typical values in [0.01, 0.5].
        label_smoothing: Label-smoothing coefficient ε ∈ [0, 1). At ε = 0 this
            reduces to the unsmoothed DPO loss. Defaults to ``0.0``.

    Returns:
        Per-example loss tensor of the same shape as the inputs.
    """
    logits = chosen_logratio - rejected_logratio
    return (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
