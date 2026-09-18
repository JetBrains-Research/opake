"""Internal helpers for :mod:`opake.dpsgd.sampling`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def _maybe_truncate_indices(
    rng: Generator,
    indices: np.ndarray,
    truncated_batch_size: int | None,
) -> list[int]:
    """Uniform subsample when ``indices`` exceeds ``truncated_batch_size``."""
    if truncated_batch_size is not None and indices.size > truncated_batch_size:
        indices = rng.choice(indices, size=truncated_batch_size, replace=False)
    return indices.tolist()


def _plain_poisson_step_indices(
    rng: Generator,
    num_examples: int,
    sample_rate: float,
    truncated_batch_size: int | None,
) -> list[int]:
    """Bernoulli inclusion over ``range(num_examples)`` (plain Poisson draw)."""
    included = rng.random(num_examples) < sample_rate
    indices = np.where(included)[0]
    return _maybe_truncate_indices(rng, indices, truncated_batch_size)
