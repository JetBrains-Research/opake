"""DP-SGD clipping impl — adaptive thresholding and the MoE load release.

Fixed and AUTO-S clipping live in :mod:`opaque.api.engine.clipping` and
are re-exported by the ``opaque.dpsgd.clipping`` façade alongside the
DP-SGD-only :func:`adaptive_clipped_grad` and :func:`moe_clipped_grad`.
"""

import opaque.api.dpsgd.clipping._distributed  # noqa: F401  (registers sync handlers)
from opaque.api.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opaque.api.dpsgd.clipping._moe import moe_clipped_grad

__all__ = ["adaptive_clipped_grad", "moe_clipped_grad"]
