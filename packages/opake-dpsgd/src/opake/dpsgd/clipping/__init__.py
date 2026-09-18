"""DP-SGD clipping façade — fixed, AUTO-S, per-group, and adaptive thresholding.

Headline factories:

- :func:`clipped_grad` — fixed-threshold per-example clipping.
- :func:`auto_clipped_grad` — AUTO-S (Bu et al.,
  `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_, NeurIPS 2023).
- :func:`per_group` — build :class:`~opake.types.PerGroup` groupings.
- :func:`adaptive_clipped_grad` — adaptive clipping (Andrew et al.,
  `Differentially Private Learning with Adaptive Clipping
  <https://arxiv.org/abs/1905.03871>`_, 2021).

Adaptive clipping is DP-SGD-only: its threshold drifts across steps,
which violates the constant-sensitivity assumption matrix-factorization
proofs require.

State and auxiliary dataclasses live in :mod:`opake.dpsgd.clipping.types`.
AUTO-S function-level helpers live in :mod:`opake.dpsgd.clipping.fun`.
"""

from opake.api.dpsgd.clipping import adaptive_clipped_grad
from opake.api.engine.clipping import auto_clipped_grad, clipped_grad, per_group

__all__ = [
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
