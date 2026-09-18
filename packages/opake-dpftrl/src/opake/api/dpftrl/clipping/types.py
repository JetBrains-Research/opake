"""Clipping state and auxiliary types for :mod:`opake.dpftrl.clipping`."""

from __future__ import annotations

from opake.api.engine.clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    ClippedFunAux,
    ClippedGradAux,
    ClippedGradFn,
    ClippedGradResult,
    ClipPytreeAux,
    FixedClipState,
)

__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "ClipPytreeAux",
    "ClippedFunAux",
    "ClippedGradAux",
    "ClippedGradFn",
    "ClippedGradResult",
    "FixedClipState",
]
