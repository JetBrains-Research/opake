"""DPO robust (label-smoothed) loss — label-smoothed Rafailov 2023.

Implements the robust variant of the DPO loss, which applies label smoothing
in a numerically stable manner. Derived from the label-smoothed extension of:

    Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., &
    Finn, C. (2023). Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model. NeurIPS 2023.

    Formula:
        loss = (−(1−ε)·logsigmoid(β·Δ) + ε·logsigmoid(−β·Δ)) / (1 − 2ε)
    where Δ = chosen_logratio − rejected_logratio.

**Domain restriction:** ``label_smoothing`` must satisfy ``0 ≤ ε < 0.5``.
At ε = 0.5 the denominator ``(1 − 2ε)`` is zero, causing division by zero.
The function does not guard against this: callers must enforce ε < 0.5.
At ε = 0 this reduces to the unsmoothed DPO sigmoid loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["robust_loss"]


def robust_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-example DPO robust (label-smoothed) loss.

    Implements the robust label-smoothed DPO objective. The function is
    elementwise and works on both batched ``(B,)`` inputs and 0-dim scalars.

    Unlike the sigmoid variant's label smoothing (which simply blends the
    reversed loss in), the robust variant normalises the blended loss by
    ``(1 − 2·label_smoothing)`` so that the gradient scale is preserved
    across smoothing levels.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response,
            ``log π(y_w | x) − log π_ref(y_w | x)``. Shape ``(B,)`` or ``()``.
        rejected_logratio: Per-example log-ratio for the rejected response,
            ``log π(y_l | x) − log π_ref(y_l | x)``. Same shape as
            ``chosen_logratio``.
        beta: DPO temperature (reference-deviation strength). Positive float;
            typical values in [0.01, 0.5].
        label_smoothing: Label-smoothing coefficient ε ∈ [0, 0.5). The
            denominator is ``(1 − 2·ε)``; passing ε ≥ 0.5 causes division by
            zero and is a caller error. Defaults to ``0.0``.

    Returns:
        Per-example loss tensor of the same shape as the inputs.
    """
    logits = chosen_logratio - rejected_logratio
    return (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        + F.logsigmoid(-beta * logits) * label_smoothing
    ) / (1 - 2 * label_smoothing)
