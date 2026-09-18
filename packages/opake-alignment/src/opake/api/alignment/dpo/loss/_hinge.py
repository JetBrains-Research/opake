"""DPO hinge loss — Liu et al., 2023.

Implements the hinge variant of the DPO objective from:

    Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P. J., &
    Liu, J. (2023). Statistical Rejection Sampling Improves Preference
    Optimization. arXiv:2309.06657.

    Formula:
        loss = relu(1 − β·Δ)
    where Δ = chosen_logratio − rejected_logratio.
"""

from __future__ import annotations

import torch

__all__ = ["hinge_loss"]


def hinge_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Per-example DPO hinge loss.

    Implements the hinge variant of the DPO objective from Liu et al. 2023.
    The function is elementwise and works on both batched ``(B,)`` inputs and
    0-dim scalars.

    The hinge loss is zero when the margin β·Δ ≥ 1 (chosen is clearly
    preferred over rejected by the temperature-scaled margin). It is linear
    in the deficit below 1 otherwise.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response,
            ``log π(y_w | x) − log π_ref(y_w | x)``. Shape ``(B,)`` or ``()``.
        rejected_logratio: Per-example log-ratio for the rejected response,
            ``log π(y_l | x) − log π_ref(y_l | x)``. Same shape as
            ``chosen_logratio``.
        beta: DPO temperature (reference-deviation strength). Positive float;
            typical values in [0.01, 0.5].

    Returns:
        Per-example loss tensor of the same shape as the inputs.
    """
    logits = chosen_logratio - rejected_logratio
    return torch.relu(1 - beta * logits)
