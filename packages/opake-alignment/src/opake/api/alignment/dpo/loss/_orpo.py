"""ORPO odds-ratio preference loss.

ORPO (Hong et al., "ORPO: Monolithic Preference Optimization without Reference
Model", 2024) is reference-free and contrasts the **odds** of the chosen vs
rejected completion::

    L_OR = -log σ( log odds(y_w) - log odds(y_r) ),
    odds(y) = p(y) / (1 - p(y)),   log p(y) = (1/|y|) · log π(y)

so ``log odds(y) = log p(y) - log(1 - p(y))``. The full ORPO objective adds an
NLL term on the chosen completion; combine this head with ``chosen_nll_loss``
via :func:`~opake.api.alignment.dpo.loss.mpo_combine` (weight ``lambda`` on the
odds-ratio term).

Unlike the log-ratio heads, this takes the per-completion **length-normalized
log-probabilities directly** (``log p(y)``, necessarily ``< 0``), because the
``log(1 - p)`` term is nonlinear in the log-prob and cannot be expressed from a
``chosen - rejected`` log-ratio. Use ``sequence_logp(..., length_normalized=True)``
to form the inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["odds_ratio_loss"]


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x < 0`` (Mächler, 2012)."""
    return torch.where(
        x > -math.log(2.0),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def odds_ratio_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
) -> torch.Tensor:
    """Per-example ORPO odds-ratio loss.

    Computes ``-log σ(log odds(y_w) - log odds(y_r))`` with
    ``log odds(y) = log p(y) - log(1 - p(y))``.

    Args:
        chosen_logp: Length-normalized log-probability ``log p(y_w)`` of the
            chosen completion (``< 0``). Shape ``(B,)`` or ``()``. Form it with
            ``sequence_logp(..., length_normalized=True)``.
        rejected_logp: Same for the rejected completion.

    Returns:
        Per-example scalar odds-ratio loss, same shape as the inputs.
    """
    log_odds = (chosen_logp - _log1mexp(chosen_logp)) - (
        rejected_logp - _log1mexp(rejected_logp)
    )
    return -F.logsigmoid(log_odds)
