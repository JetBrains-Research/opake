# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Re-bind functional_call parameters during checkpoint recomputation.

``functional_call`` installs parameters only for the forward
(``_reparametrize_module`` restores the originals on exit), but non-reentrant
checkpoint recomputes the forward later, during backward. The recomputed forward
must read the same parameters, or it silently miscomputes / errors with a
functorch level mismatch under vmap (this is opake's main HF path:
``make_functional`` -> ``functional_call`` -> checkpoint inside the model).

Two cooperating wrappers:

- ``_reparametrize_module``: record each active reparametrization on a
  thread-local stack while it is installed.
- ``_CheckpointFrame.__init__``: snapshot that stack at frame-creation time (the
  forward) and re-establish those swaps around recomputation.

This mirrors the upstream native fix, injected through the stable
``_reparametrize_module`` / ``_CheckpointFrame`` symbols (present on both the old
and new checkpoint architectures) rather than torch's internal checkpoint
generator. Applied only when torch lacks native support.
"""

from __future__ import annotations

import contextlib
import threading

_active = threading.local()
_orig_reparametrize = None


def _stack() -> list:
    s = getattr(_active, "stack", None)
    if s is None:
        s = _active.stack = []
    return s


def apply() -> None:
    """Install functional-parameter rebinding for checkpoint recomputation."""
    _wrap_reparametrize_module()
    _wrap_checkpoint_frame()


def _wrap_reparametrize_module() -> None:
    global _orig_reparametrize
    import torch.nn.utils.stateless as stateless

    _orig_reparametrize = stateless._reparametrize_module

    @contextlib.contextmanager
    def _reparametrize_module(
        module,
        parameters_and_buffers,
        tie_weights=False,
        strict=False,
        stack_weights=False,
    ):
        params = (
            stateless._untie_named_tensors_map(module, parameters_and_buffers)
            if tie_weights
            else parameters_and_buffers
        )
        _stack().append((module, params))
        try:
            with _orig_reparametrize(
                module,
                parameters_and_buffers,
                tie_weights=tie_weights,
                strict=strict,
                stack_weights=stack_weights,
            ):
                yield
        finally:
            _stack().pop()

    stateless._reparametrize_module = _reparametrize_module


def _wrap_checkpoint_frame() -> None:
    from torch.utils.checkpoint import _CheckpointFrame

    orig_init = _CheckpointFrame.__init__

    def __init__(self, recompute_fn, *args, **kwargs):
        snapshot = list(_stack())
        if snapshot:

            def rebinding_recompute(*a, _fn=recompute_fn, _snapshot=snapshot):
                with contextlib.ExitStack() as stack:
                    for module, params in _snapshot:
                        stack.enter_context(
                            _orig_reparametrize(module, params, tie_weights=False)
                        )
                    return _fn(*a)

            recompute_fn = rebinding_recompute
        orig_init(self, recompute_fn, *args, **kwargs)

    _CheckpointFrame.__init__ = __init__
