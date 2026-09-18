"""Dataset sharding helpers for distributed DP training.

Pure functions that take explicit ``rank`` and ``world_size`` parameters. Use
:func:`local_shard` to slice a dataset for a given rank, then pass the
resulting ``Subset`` to the sampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Subset

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sized


def local_shard(dataset: Sized, *, rank: int = 0, world_size: int = 1) -> Subset:
    """Return the shard of ``dataset`` that belongs to ``rank``.

    Each rank gets a contiguous, non-overlapping slice; the last rank receives
    any remainder examples.

    Args:
        dataset: Any object with ``__len__`` (e.g. a PyTorch ``Dataset``).
        rank: Rank of the current worker (0-indexed). Defaults to 0.
        world_size: Total number of workers. Defaults to 1.

    Returns:
        A ``torch.utils.data.Subset`` view into ``dataset``.
    """
    start, end = _local_shard_bounds(len(dataset), rank=rank, world_size=world_size)
    return Subset(dataset, range(start, end))


def _local_shard_bounds(
    dataset_size: int, *, rank: int = 0, world_size: int = 1
) -> tuple[int, int]:
    """Return ``[start, end)`` index bounds for ``rank``."""
    if dataset_size < 0:
        raise ConfigurationError(*(f"dataset_size must be >= 0, got {dataset_size}",))
    if world_size < 1:
        raise ConfigurationError(*(f"world_size must be >= 1, got {world_size}",))
    if not 0 <= rank < world_size:
        raise ConfigurationError(
            *(
                f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}",
            )
        )

    if world_size == 1:
        return 0, dataset_size

    shard = dataset_size // world_size
    start = rank * shard
    end = dataset_size if rank == world_size - 1 else (start + shard)
    return start, end


__all__ = ["local_shard"]
