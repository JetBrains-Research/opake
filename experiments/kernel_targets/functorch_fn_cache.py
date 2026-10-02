"""Prototype: cache functorch's per-call generated autograd.Function classes.

Under torch.func.grad, every call of a custom autograd.Function goes through
torch._functorch.autograd_function.custom_function_call_grad, which calls
generate_single_level_function(interpreter, fn). That builds a new
_SingleLevelFunction subclass with type(...) on every call (and FunctionMeta
builds its backward class), torch 2.14 lines ~119-173.

Only the generated forward uses the interpreter (interpreter.lower() and its
level); setup_context, backward and jvp do not. So the class can be cached per
(fn, level, transform key) if forward reads the *current* interpreter, which
the patched generator stores in thread-local state right before the caller
applies it (custom_function_call_grad applies the returned class immediately).

install() / uninstall() swap the module global; nothing else changes.
Prototype only: not packaged, not reviewed upstream.
"""

from __future__ import annotations

import threading

import torch
import torch._functorch.autograd_function as _af
import torch.utils._pytree as pytree

_ORIG = _af.generate_single_level_function
_CACHE: dict = {}
_TLS = threading.local()


def _current():
    d = getattr(_TLS, "interp", None)
    if d is None:
        d = _TLS.interp = {}
    return d


def _make(autograd_function, level, key):
    def forward(*operands):
        interpreter = _current()[key]
        unwrapped = pytree.tree_map_only(torch.Tensor, lambda x: _af._unwrap_for_grad(x, level), operands)
        with torch.enable_grad(), _af._set_fwd_grad_enabled(True), interpreter.lower():
            out = _af.custom_function_call(autograd_function, *unwrapped)

        def wrap_fn(output):
            return _af._wrap_for_grad(output, level)

        return _af.wrap_outputs_maintaining_identity(out, unwrapped, operands, wrap_fn)

    def setup_context(ctx, inputs, output):
        return autograd_function.setup_context(ctx, inputs, output)

    def backward(ctx, *grads):
        return autograd_function.backward(ctx, *grads)

    def jvp(ctx, *tangents):
        return autograd_function.jvp(ctx, *tangents)

    return type(
        f"{autograd_function.__name__}Generated",
        (torch.autograd.function._SingleLevelFunction,),
        {"forward": staticmethod(forward), "backward": staticmethod(backward),
         "jvp": staticmethod(jvp), "setup_context": staticmethod(setup_context)},
    )


def cached_generate_single_level_function(interpreter, autograd_function):
    level = interpreter.level()
    key = (autograd_function, level, interpreter.key())
    cls = _CACHE.get(key)
    if cls is None:
        cls = _CACHE[key] = _make(autograd_function, level, key)
    _current()[key] = interpreter
    return cls


def install():
    _af.generate_single_level_function = cached_generate_single_level_function


def uninstall():
    _af.generate_single_level_function = _ORIG
