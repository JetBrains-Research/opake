"""Functional EMA reference-parameter update for TR-DPO.

TR-DPO (Trust-Region DPO) maintains a moving reference model that tracks the
policy via an exponential moving average (EMA):

    ref ← (1 − α) · ref + α · policy

This "soft" reference update is described in the TR-DPO paper and is the
primary mechanism for preventing the policy from straying too far from the
reference distribution during DP-DPO training (arXiv:2404.09656 §3).

This module exposes a single pure function, :func:`ema_update_reference`, that
applies the update leafwise over an arbitrary parameter pytree without
mutating either input.
"""

from __future__ import annotations

from typing import Any

import optree

__all__ = ["ema_update_reference"]

# Type alias: any nested structure of tensors understood by optree.
PyTree = Any


def ema_update_reference(
    ref_params: PyTree,
    policy_params: PyTree,
    alpha: float,
) -> PyTree:
    """Functional EMA update of the reference parameters toward the policy.

    Implements the TR-DPO core update (arXiv:2404.09656):

        ref ← (1 − alpha) · ref + alpha · policy

    Applied leafwise over a parameter pytree.  Returns a **new** pytree of
    the same structure and does **not** mutate either input.

    Args:
        ref_params: Current reference-model parameters as a pytree of
            :class:`torch.Tensor` leaves (e.g. the dict returned by
            ``make_functional(partition_trainable=True)``).
        policy_params: Current policy-model parameters as a pytree with the
            same structure as ``ref_params``.
        alpha: EMA interpolation coefficient in ``[0, 1]``.

            - ``alpha = 0`` — keep ``ref_params`` unchanged (output equals
              ``ref_params`` values).
            - ``alpha = 1`` — copy ``policy_params`` exactly (output equals
              ``policy_params`` values).

    Returns:
        A new pytree of tensors with the same structure as ``ref_params``,
        where each leaf ``r`` is replaced by
        ``(1 - alpha) * r + alpha * p`` for the corresponding policy leaf
        ``p``.

    Note:
        This function is **pure** (no in-place mutation) and is intended to
        be called **outside** the per-example gradient loop. It is safe to
        call under ``torch.no_grad()`` to avoid accumulating an autograd graph
        over the reference parameters.
    """
    return optree.tree_map(
        lambda r, p: (1 - alpha) * r + alpha * p,
        ref_params,
        policy_params,
    )
