"""Power-user clipping building blocks for DP-SGD.

Mirrors :mod:`opake.api.engine.clipping.fun`:

- :func:`clipped_fun` / :func:`auto_clipped_fun` — clip + sum any per-example
  function output (PyTree).
- :func:`clip_pytree` / :func:`auto_scale_pytree` — clip / AUTO-S an
  already-batched pytree.
"""

from __future__ import annotations

from opake.api.engine.clipping.fun import (
    auto_clipped_fun,
    auto_scale_pytree,
    clip_pytree,
    clipped_fun,
)

__all__ = [
    "auto_clipped_fun",
    "auto_scale_pytree",
    "clip_pytree",
    "clipped_fun",
]
