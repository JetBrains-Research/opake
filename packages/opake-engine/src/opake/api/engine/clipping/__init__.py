"""Per-example gradient clipping (fixed threshold and AUTO-S).

Entry points:

- :func:`clipped_grad` — differentiate + fixed-threshold clip + sum
- :func:`auto_clipped_grad` — differentiate + AUTO-S smooth-scale + sum
  (Bu et al., `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_,
  NeurIPS 2023)
- :func:`per_group` — build :class:`opake.types.PerGroup` groupings

Fixed clipping and AUTO-S give a constant, data-independent per-record
sensitivity bound, so they pair with mechanisms in :mod:`opake.dpsgd.noise`
and :mod:`opake.dpftrl.noise`. Adaptive thresholding
(:func:`opake.dpsgd.clipping.adaptive_clipped_grad`) is DP-SGD-only: the
threshold moves across steps and does not meet the constant-sensitivity
assumption used in matrix-factorization analyses.

Power-user APIs live in :mod:`opake.api.engine.clipping.fun`; state and aux types in
:mod:`opake.api.engine.clipping.types`. Cross-cutting wrapper types
(:class:`opake.types.ClippedPytree`, :class:`opake.types.PerGroup`, …) live
in :mod:`opake.types`.

Use :func:`opake.distributed.sync` to synchronize clipping state or aux
objects across ranks.
"""

import opake.api.engine.clipping._distributed  # noqa: F401  (registers sync handlers)
from opake.api.engine.clipping._auto import auto_clipped_grad
from opake.api.engine.clipping._clipped_grad import clipped_grad
from opake.api.engine.clipping._per_group import per_group

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
