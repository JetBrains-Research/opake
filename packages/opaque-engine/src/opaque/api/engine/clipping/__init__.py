"""Per-example gradient clipping (fixed threshold and AUTO-S).

Entry points:

- :func:`clipped_grad` — differentiate + fixed-threshold clip + sum
- :func:`auto_clipped_grad` — differentiate + AUTO-S smooth-scale + sum
  (Bu et al., `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_,
  NeurIPS 2023)
- :func:`per_group` — build :class:`opaque.types.PerGroup` groupings

Fixed clipping and AUTO-S give a constant, data-independent per-record
sensitivity bound, so they pair with mechanisms in :mod:`opaque.dpsgd.noise`
and :mod:`opaque.dpftrl.noise`. Adaptive thresholding
(:func:`opaque.dpsgd.clipping.adaptive_clipped_grad`) is DP-SGD-only: the
threshold moves across steps and does not meet the constant-sensitivity
assumption used in matrix-factorization analyses.

Power-user APIs live in :mod:`opaque.api.engine.clipping.fun`; state and aux types in
:mod:`opaque.api.engine.clipping.types`. Cross-cutting wrapper types
(:class:`opaque.types.ClippedPytree`, :class:`opaque.types.PerGroup`, …) live
in :mod:`opaque.types`.

Use :func:`opaque.distributed.sync` to synchronize clipping state or aux
objects across ranks.
"""

import opaque.api.engine.clipping._distributed  # noqa: F401  (registers sync handlers)
from opaque.api.engine.clipping._auto import auto_clipped_grad
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.clipping._per_group import per_group

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
