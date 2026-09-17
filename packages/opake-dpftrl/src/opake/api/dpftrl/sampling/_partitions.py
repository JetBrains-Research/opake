"""Internal utilities for sampling strategies."""

from __future__ import annotations

import enum

import numpy as np


class PartitionType(enum.Enum):
    """Specifies how examples should be assigned to groups."""

    INDEPENDENT = enum.auto()
    """Each example assigned to a group independently at random."""
    EQUAL_SPLIT = enum.auto()
    """Examples shuffled and split into groups of equal size."""


def _independent_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
    """Partition examples independently (multinomial assignment)."""
    sizes = rng.multinomial(num_examples, np.ones(num_groups) / num_groups)
    boundaries = np.cumsum(sizes)[:-1]
    indices = rng.permutation(num_examples).astype(dtype)
    return np.split(indices, boundaries)


def _equal_split_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
    """Partition examples by shuffling then splitting equally."""
    indices = rng.permutation(num_examples).astype(dtype)
    group_size = num_examples // num_groups
    groups = np.array_split(indices, num_groups)
    return [g[:group_size] for g in groups]
