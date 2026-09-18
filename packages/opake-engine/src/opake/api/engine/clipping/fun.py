"""Power-user clipping building blocks.

The headline entry points for DP training are
:func:`opake.api.engine.clipping.clipped_grad` and :func:`opake.api.engine.clipping.auto_clipped_grad`.
This submodule exposes the lower-level primitives they build on, for users
who need to clip arbitrary per-example outputs or compose custom pipelines:

- :func:`clipped_fun` — fixed-threshold clip + sum any per-example function
- :func:`auto_clipped_fun` — AUTO-S smooth scaling + sum any per-example
  function (function-level analogue of ``auto_clipped_grad``)
- :func:`clip_pytree` — clip an already-batched pytree to a fixed L2 norm
- :func:`auto_scale_pytree` — AUTO-S smooth-scale an already-batched pytree

These are stable, supported public API; they live one level deeper to keep
the headline surface concise.
"""

from __future__ import annotations

from opake.api.engine.clipping._auto import auto_clipped_fun
from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.api.engine.clipping._pytree import auto_scale_pytree, clip_pytree

__all__ = [
    "auto_clipped_fun",
    "auto_scale_pytree",
    "clip_pytree",
    "clipped_fun",
]
