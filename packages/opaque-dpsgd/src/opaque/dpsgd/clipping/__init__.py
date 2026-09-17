"""DP-SGD clipping façade — fixed, AUTO-S, per-group, and adaptive thresholding.

Headline factories:

- :func:`clipped_grad` — fixed-threshold per-example clipping.
- :func:`auto_clipped_grad` — AUTO-S (Bu et al.,
  `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_, NeurIPS 2023).
- :func:`per_group` — build :class:`~opaque.types.PerGroup` groupings.
- :func:`adaptive_clipped_grad` — adaptive clipping (Andrew et al.,
  `Differentially Private Learning with Adaptive Clipping
  <https://arxiv.org/abs/1905.03871>`_, 2021).
- :func:`moe_clipped_grad` — MoE clipping with the batch router load
  released inside the clipper (accounted by :func:`opaque.dpsgd.accounting.moe_aux`).

Adaptive clipping is DP-SGD-only: its threshold drifts across steps,
which violates the constant-sensitivity assumption matrix-factorization
proofs require.

State and auxiliary dataclasses live in :mod:`opaque.dpsgd.clipping.types`.
AUTO-S function-level helpers live in :mod:`opaque.dpsgd.clipping.fun`.
"""

from opaque.api.dpsgd.clipping import adaptive_clipped_grad, moe_clipped_grad
from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad, per_group

__all__ = [
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    "clipped_grad",
    "moe_clipped_grad",
    "per_group",
]
