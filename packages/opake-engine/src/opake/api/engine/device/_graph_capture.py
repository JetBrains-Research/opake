# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Flag for kernels to take their CUDA-graph-capturable code paths.

Some kernels have a faster eager path that cannot be captured in a CUDA graph,
for example one that compacts tokens with ``torch.nonzero`` (a host
synchronization). The CUDA-graph chunk compiler sets this flag for its warmup
runs and its capture, so those runs, and the graph replayed afterwards, use the
capturable path consistently.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_CAPTURE_SAFE = ContextVar("opake_capture_safe_kernels", default=False)


def capture_safe_kernels_active() -> bool:
    """Whether kernels should take their CUDA-graph-capturable code path."""
    return _CAPTURE_SAFE.get()


@contextmanager
def capture_safe_kernels() -> Iterator[None]:
    """Make kernels take their CUDA-graph-capturable code paths in this block."""
    token = _CAPTURE_SAFE.set(True)
    try:
        yield
    finally:
        _CAPTURE_SAFE.reset(token)
