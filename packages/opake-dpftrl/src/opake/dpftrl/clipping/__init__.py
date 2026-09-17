"""DP-FTRL clipping façade — fixed threshold, AUTO-S, and per-group norms.

Adaptive thresholding is only available under :mod:`opake.dpsgd.clipping`
(DP-SGD). State and aux types live in :mod:`opake.dpftrl.clipping.types`.
"""

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad, per_group

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
