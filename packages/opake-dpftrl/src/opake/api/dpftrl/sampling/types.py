"""Public type definitions for :mod:`opake.dpftrl.sampling`.

Re-exports the partition-strategy enum used by
:class:`opake.dpftrl.sampling.CyclicPoissonSampler` for type annotations
and explicit construction.
"""

from __future__ import annotations

from opake.api.dpftrl.sampling._partitions import PartitionType

__all__ = ["PartitionType"]
