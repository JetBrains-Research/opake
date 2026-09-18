# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Inspect the active ``torch.func`` transform stack."""

from __future__ import annotations

import torch


def under_differentiating_transform(*, when_compiling: bool) -> bool:
    """Return whether an enclosing ``grad``, ``vjp``, or ``jvp`` is active."""
    if torch.compiler.is_compiling():
        return when_compiling
    try:
        from torch._C._functorch import TransformType, get_interpreter_stack

        differentiating = (TransformType.Grad, TransformType.Jvp)
        stack = get_interpreter_stack() or ()
        return any(interpreter.key() in differentiating for interpreter in stack)
    except Exception:  # pragma: no cover - API moved/unavailable
        return True
