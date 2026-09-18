"""DP-SGD clipping impl — adaptive thresholding.

Fixed and AUTO-S clipping live in :mod:`opake.api.engine.clipping` and
are re-exported by the ``opake.dpsgd.clipping`` façade alongside the
DP-SGD-only :func:`adaptive_clipped_grad`.
"""

import opake.api.dpsgd.clipping._distributed  # noqa: F401  (registers sync handlers)
from opake.api.dpsgd.clipping._adaptive import adaptive_clipped_grad

__all__ = ["adaptive_clipped_grad"]
