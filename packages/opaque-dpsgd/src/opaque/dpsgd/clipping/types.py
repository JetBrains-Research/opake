"""DP-SGD clipping types façade.

Re-exports adaptive (DP-SGD) and AUTO-S clipping state / aux dataclasses
for type annotations from the impl trees.
"""

from opaque.api.dpsgd.clipping.types import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    MoeClipState,
)
from opaque.api.engine.clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    ClippedGradFn,
    ClippedGradResult,
    ClippingStats,
    FixedClipState,
)

__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "ClippingStats",
    "ClippedGradFn",
    "ClippedGradResult",
    "FixedClipState",
    "MoeClipState",
]
