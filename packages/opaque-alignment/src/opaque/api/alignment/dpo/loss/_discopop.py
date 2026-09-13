"""DiscoPOP (Discovering Preference Optimization Procedures) loss.

Implements the logistic/exponential blend discovered by offline meta-learning
of DPO-family objectives:

    Lu, C., et al. (2024). Discovering Preference Optimization Algorithms
    with and for Large Language Models. NeurIPS 2024.
    https://arxiv.org/abs/2406.08414

The modulation gate ``gate = σ(β·Δ / τ)`` blends a logistic component and an
exponential component via ``L = logistic·(1 − gate) + exp·gate``.  Because
``gate → 1`` for large positive ``β·Δ`` and ``gate → 0`` for negative ``β·Δ``,
the **exponential** component dominates when ``β·Δ`` is large positive and the
**logistic** component dominates when ``β·Δ`` is negative.

The exponential branch is clamped to the active compute dtype's safe range so
half precision cannot overflow while the representable region keeps the same
formula.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["discopop_loss"]


def _exp_clamp_for_dtype(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 11.0
    return 80.0


def discopop_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    discopop_tau: float = 0.05,
) -> torch.Tensor:
    """DiscoPOP per-example loss (Lu et al., 2024, equations 4-5).

    Blends a logistic component ``-log σ(β·Δ)`` and an exponential component
    ``exp(-β·Δ)`` using a soft gate parameterised by temperature *τ*::

        logits    = (chosen_logratio - rejected_logratio) * beta
        gate      = σ(logits / τ)          # → 1 for large positive logits
        logistic  = -log σ(logits)
        exp_comp  = exp(-logits)            # clamped to avoid overflow
        L         = logistic * (1 - gate) + exp_comp * gate

    Args:
        chosen_logratio: Per-example scalar log-ratio for the chosen
            completion.  May be 0-dim or ``(B,)``.
        rejected_logratio: Per-example scalar log-ratio for the rejected
            completion.  Same shape as *chosen_logratio*.
        beta: KL-regularisation temperature (DPO β).
        discopop_tau: Modulation temperature *τ* (default ``0.05``).
            Smaller values sharpen the transition between the two components.

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.

    Note:
        The exponential component overflows half precision when ``logits ≪ 0``.
        To prevent NaN/Inf while preserving gradient locality, the logits passed
        to ``torch.exp`` are clamped from below at a dtype-specific bound:
        ``11`` for ``float16`` and ``80`` otherwise.  This keeps the
        exponential finite in the active compute dtype while leaving the
        representable region unchanged.
    """
    logits = (chosen_logratio - rejected_logratio) * beta
    exp_clamp = _exp_clamp_for_dtype(logits.dtype)
    logits_clamped = logits.clamp(min=-exp_clamp)
    modulation = torch.sigmoid(logits / discopop_tau)
    logistic_component = -F.logsigmoid(logits)
    exp_component = torch.exp(-logits_clamped)
    return logistic_component * (1.0 - modulation) + exp_component * modulation
