"""AUTO-S clipping symbols (see :mod:`opake.dpsgd.clipping`)."""

from __future__ import annotations

from opake.api.engine.clipping import auto_clipped_grad
from opake.api.engine.clipping.fun import auto_clipped_fun
from opake.api.engine.clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)

__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "auto_clipped_fun",
    "auto_clipped_grad",
]
