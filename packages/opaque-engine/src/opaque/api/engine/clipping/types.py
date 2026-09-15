"""Public type definitions for :mod:`opaque.api.engine.clipping`.

Re-exports the clipping-specific state and auxiliary dataclasses for
type annotations. The cross-cutting DP types (``ClipState`` base,
``ClippedPytree``, ``PerGroup``, ``MaxNorm``, ``clipped()`` factory)
live in :mod:`opaque.types`.
"""

from __future__ import annotations

from opaque.api.engine.clipping._auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)
from opaque.api.engine.clipping._clipped_fun import (
    ClippedFunAux,
    ClippingStats,
    FixedClipState,
)
from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
from opaque.api.engine.clipping._pytree import ClipPytreeAux
from opaque.api.engine.clipping._types import ClippedGradFn, ClippedGradResult

__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "ClipPytreeAux",
    "ClippedFunAux",
    "ClippingStats",
    "ClippedGradAux",
    "ClippedGradFn",
    "ClippedGradResult",
    "FixedClipState",
]
