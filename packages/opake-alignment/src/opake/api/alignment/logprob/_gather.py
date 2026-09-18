# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Selective log-softmax gather.

:func:`selective_log_softmax` computes ``log_softmax(logits, dim=-1)`` and
gathers the entry at ``indices`` along the last (vocabulary) axis. It works for
any leading shape ``(...)``, both per-example (``(T, V)`` / ``(T,)``) and over a
batched input (``(B, T, V)`` / ``(B, T)``).

Numerical stability comes from :func:`torch.log_softmax`, which uses the
log-sum-exp trick internally; we never materialise ``exp(logits)`` so large
logits do not overflow.

On CUDA the call dispatches to the patches chunked CE Triton kernel via
``opake_selective_log_softmax``, which never allocates the intermediate
``log_softmax(logits)`` tensor. Off CUDA (CPU CI, MPS) the eager
``log_softmax + gather`` path keeps the numeric contract intact.
"""

from __future__ import annotations

import torch

from opake.api.alignment._fused_lce import (
    import_ce_kernel,
    selective_log_softmax_available,
)

__all__ = ["selective_log_softmax"]


def selective_log_softmax(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-position log-probabilities at ``indices``.

    Equivalent to ``log_softmax(logits, dim=-1).gather(-1, indices[..., None])``
    followed by ``squeeze(-1)``, but written so it is numerically stable.

    Args:
        logits: Float tensor of shape ``(..., V)`` where ``V`` is the
            vocabulary size. The leading dims ``(...)`` are arbitrary.
        indices: Integer tensor of shape ``(...)`` matching the leading dims
            of ``logits``. Each value selects a vocabulary entry in ``[0, V)``.

    Returns:
        Float tensor of shape ``(...)`` holding ``log p`` for the selected
        token at each position.
    """
    if selective_log_softmax_available(logits):
        kernel_mod = import_ce_kernel()
        (logits,) = kernel_mod.follow_autocast(logits)
        return kernel_mod.opake_selective_log_softmax(logits, indices)
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=indices.unsqueeze(-1))
    return gathered.squeeze(-1)
