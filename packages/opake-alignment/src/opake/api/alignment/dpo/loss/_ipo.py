"""IPO (Identity Preference Optimization) loss for DPO-family training.

Implements the squared-loss variant introduced in:

    Azar, M. G., Rowland, M., Piot, B., Guo, Z. D., Calandriello, D.,
    Valko, M., & Munos, R. (2024). A General Theoretical Paradigm to
    Understand Learning from Human Feedback. AISTATS 2024.
    https://arxiv.org/abs/2310.12036

Loss for example *i* depends only on example *i*'s
data. NaN-injection contract holds: replacing one example's inputs with NaN
affects only that example's gradient.

**Note on length normalisation.** The IPO paper derives the loss under the
assumption that log-ratios are normalised by completion length.  The caller
is responsible for passing length-normalised log-ratios when that is
desired; the loss function itself is purely algebraic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["ipo_loss"]


def ipo_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """IPO per-example loss (Azar 2024).

    Computes the squared deviation between the log-ratio difference and the
    inverse of twice the KL-regularisation temperature::

        L = (Δ - 1/(2β))²    where Δ = chosen_logratio - rejected_logratio

    When the caller passes length-normalised log-ratios (i.e. each log-ratio
    divided by the corresponding completion token count), this recovers the
    original IPO objective from the paper.

    Args:
        chosen_logratio: Per-example scalar ``log π(y_w) - log π_ref(y_w)``.
            May be 0-dim (single example) or 1-dim ``(B,)`` (batch).
        rejected_logratio: Per-example scalar ``log π(y_l) - log π_ref(y_l)``.
            Same shape as *chosen_logratio*.
        beta: KL-regularisation temperature (DPO β).  Controls the strength
            of the preference signal; larger β → the model must diverge more
            from the reference to achieve the same reward margin.

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    logits = chosen_logratio - rejected_logratio
    return (logits - 1.0 / (2.0 * beta)).pow(2)
