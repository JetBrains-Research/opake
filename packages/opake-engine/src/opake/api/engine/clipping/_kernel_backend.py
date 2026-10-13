"""Registration seam for optional fused clipping implementations."""

from collections.abc import Callable
from typing import Any

_fused_clip_backend: Callable[..., Any] | None = None


def register_fused_clip_backend(backend: Callable[..., Any]) -> None:
    """Register the optional fused clip-and-sum implementation."""
    global _fused_clip_backend
    _fused_clip_backend = backend


def get_fused_clip_backend() -> Callable[..., Any] | None:
    """Return the registered fused clip-and-sum implementation, if any."""
    return _fused_clip_backend
