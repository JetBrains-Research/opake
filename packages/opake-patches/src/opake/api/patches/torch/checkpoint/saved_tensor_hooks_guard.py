# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Backport PyTorch's scoped saved-tensor-hooks guard."""

from __future__ import annotations

import functools

import torch

_HIGHER_ORDER_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks (e.g. torch.autograd.graph.save_on_cpu) under higher-order "
    "differentiation such as grad(grad) or hessian: the saved-tensor "
    "round-trip severs the higher-order graph and would silently produce "
    "incorrect gradients. First-order use is supported."
)
_COMPILE_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks (e.g. torch.autograd.graph.save_on_cpu) under torch.compile, "
    "where saved-tensor hooks are managed by AOTAutograd. Use eager execution "
    "for saved-tensor hooks with torch.func."
)


def apply() -> None:
    """Install the upstream-compatible scoped saved-tensor-hooks guard."""
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        unguarded = getattr(getattr(eager, name, None), "__wrapped__", None)
        if unguarded is not None:
            setattr(
                eager, name, _disable_saved_tensor_hooks_for_higher_order(unguarded)
            )


def _disable_saved_tensor_hooks_for_higher_order(fn):
    """Apply PyTorch's scoped hook guard to a pre-upstream transform."""

    @functools.wraps(fn)
    def dispatch(*args, **kwargs):
        if torch.compiler.is_compiling():
            with torch.autograd.graph.disable_saved_tensors_hooks(_COMPILE_MESSAGE):
                return fn(*args, **kwargs)
        try:
            from torch._C._functorch import TransformType, get_interpreter_stack

            stack = get_interpreter_stack()
            higher_order = bool(stack) and any(
                interpreter.key() in (TransformType.Grad, TransformType.Jvp)
                for interpreter in stack
            )
        except Exception:  # pragma: no cover - private API moved/unavailable
            higher_order = True
        if higher_order:
            with torch.autograd.graph.disable_saved_tensors_hooks(
                _HIGHER_ORDER_MESSAGE
            ):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return dispatch
