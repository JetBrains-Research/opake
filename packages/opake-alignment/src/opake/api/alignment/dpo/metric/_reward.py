"""Preference-reward telemetry for DPO-family training.

Computes the standard ``rewards/*`` diagnostics (chosen, rejected,
accuracies, margins) from per-example chosen/rejected log-ratios.

These metrics are detached but un-noised; release is outside Opake's DP
accounting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["reward_metrics"]


def reward_metrics(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Compute ``rewards/*`` telemetry from per-example preference log-ratios.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response,
            ``log pi(y_w) - log pi_ref(y_w)``.
        rejected_logratio: Per-example log-ratio for the rejected response,
            ``log pi(y_l) - log pi_ref(y_l)``.
        beta: DPO temperature; scales the log-ratios into implicit rewards.

    Returns:
        A dict of detached scalar tensors:

        - ``"rewards/chosen"``: mean implicit reward for chosen responses.
        - ``"rewards/rejected"``: mean implicit reward for rejected responses.
        - ``"rewards/accuracies"``: fraction of examples where the chosen
            log-ratio exceeds the rejected one.
        - ``"rewards/margins"``: mean implicit reward margin (chosen minus
            rejected).
    """
    chosen_reward = beta * chosen_logratio
    rejected_reward = beta * rejected_logratio
    accuracies = (chosen_logratio > rejected_logratio).float()
    margins = beta * (chosen_logratio - rejected_logratio)
    return {
        "rewards/chosen": chosen_reward.mean().detach(),
        "rewards/rejected": rejected_reward.mean().detach(),
        "rewards/accuracies": accuracies.mean().detach(),
        "rewards/margins": margins.mean().detach(),
    }
