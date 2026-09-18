"""Clipping state and auxiliary types for :mod:`opake.dpsgd.clipping`.

Includes adaptive (DP-SGD) and AUTO-S dataclasses used in type annotations.
"""

from __future__ import annotations

from opake.api.dpsgd.clipping._adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
)
from opake.api.engine.clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    ClippedGradFn,
    ClippedGradResult,
    FixedClipState,
)

__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "ClippedGradFn",
    "ClippedGradResult",
    "FixedClipState",
]
