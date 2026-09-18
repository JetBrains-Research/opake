"""DP-FTRL clipping power-user helpers.

- :func:`clipped_fun` / :func:`auto_clipped_fun` — clip + sum any
  per-example function output (PyTree).
- :func:`clip_pytree` / :func:`auto_scale_pytree` — clip / AUTO-S an
  already-batched pytree.
"""

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
