# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Give checkpoint's ``_NoopSaveInputs`` bookkeeping autograd Function a vmap rule.

Non-reentrant checkpoint on torch < 2.12 saves its inputs through a no-op
``autograd.Function`` that lacks a batching rule, so vmap over a checkpointed
region errors. The rule is a pure no-op data-wise.

torch >= 2.12 (PR #174327) replaced this Function with a C++ saved-tensor path
that already works under vmap, so ``_NoopSaveInputs`` is gone; :func:`apply`
self-skips in that case.
"""

from __future__ import annotations


def apply() -> None:
    """Install a vmap rule for legacy checkpoint input bookkeeping."""
    try:
        from torch.utils.checkpoint import _NoopSaveInputs
    except ImportError:
        return  # torch >= 2.12: symbol removed, no rule needed

    @staticmethod
    def _vmap(info, in_dims, *args):
        return _NoopSaveInputs.apply(*args), None

    _NoopSaveInputs.vmap = _vmap
