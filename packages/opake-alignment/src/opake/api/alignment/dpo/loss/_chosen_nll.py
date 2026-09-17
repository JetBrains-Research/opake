"""SFT regulariser term for MPO (Multi-Policy Optimisation) blends.

Implements the supervised fine-tuning (SFT) regulariser that can be
combined with any DPO-family loss via the MPO blend (``_mpo.py``).  In
this context the SFT loss is the negative log-likelihood of the *chosen*
completion under the policy model — i.e. standard cross-entropy on the
chosen sequence.

**Signature flexibility.** To allow transparent composition inside the DPO
dispatch loop (``_mpo.py``), the function accepts and silently ignores any
extra positional or keyword arguments beyond the first.  This means it can
be called with the same ``(chosen_logp, rejected_logp, *, beta, ...)``
signature as the logratio-based variants without special-casing at the
call site.  The caller must ensure that the *first* positional argument is
``chosen_logp`` (the total sequence log-probability, not a log-ratio).

References:
    MPO blend: used in TRL's ``loss_type=["sigmoid", "sft"]`` composite
    loss.  The SFT component acts as a KL penalty towards the SFT
    checkpoint, stabilising DPO training when the policy drifts far from the
    supervised distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["chosen_nll_loss"]


def chosen_nll_loss(
    chosen_logp: torch.Tensor,
    /,
    *_args: object,
    **_kwargs: object,
) -> torch.Tensor:
    """SFT regulariser: negative log-likelihood of the chosen completion.

    Returns ``-chosen_logp``, i.e. the NLL of the chosen sequence under the
    current policy.  Extra positional and keyword arguments are silently
    ignored so this function can be dispatched alongside log-ratio–based
    variants in the MPO blend loop.

    Args:
        chosen_logp: Per-example scalar log-probability
            ``log π(y_w | x)`` (the full completion log-prob, **not** a
            log-ratio).  May be 0-dim (single example) or ``(B,)`` (batch).
        *_args: Ignored.  Present so the function can be called with the
            ``(chosen_logp, rejected_logratio, ...)`` signature of the other
            DPO variants without raising a ``TypeError``.
        **_kwargs: Ignored.  ``beta`` and other variant kwargs are accepted
            and discarded.

    Returns:
        Per-example NLL scalar (same shape as *chosen_logp*).
    """
    return -chosen_logp
