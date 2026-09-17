# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Helpers for active functorch transforms."""

from __future__ import annotations

import torch


def under_functorch_transform() -> bool:
    """Return whether a functorch transform is active on this thread."""
    try:
        from torch._C._functorch import peek_interpreter_stack

        return peek_interpreter_stack() is not None
    except Exception:  # pragma: no cover - API moved/unavailable
        return False


def prev_grad_mode() -> bool:
    """Return the grad mode captured when the innermost ``grad`` was entered."""
    if torch.compiler.is_compiling():
        return torch.is_grad_enabled()
    try:
        from torch._C._functorch import (
            CGradInterpreterPtr,
            TransformType,
            peek_interpreter_stack,
        )

        interpreter = peek_interpreter_stack()
        if interpreter is None or interpreter.key() != TransformType.Grad:
            return torch.is_grad_enabled()
        return CGradInterpreterPtr(interpreter).prevGradMode()
    except Exception:  # pragma: no cover - API moved/unavailable
        return torch.is_grad_enabled()
