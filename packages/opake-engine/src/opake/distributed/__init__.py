"""Distributed primitives for DP training.

DDP detection (``is_distributed``, ``get_rank``, ``get_world_size``),
collectives (``all_reduce``, ``barrier``), gradient reductions
(``reduce_pytree``, ``sum_gradients``), wrapper-aware ``sync()``
dispatch, and ``local_shard`` for dataset partitioning.

The ``opake.distributed.collectives`` and
``opake.distributed.gradients`` submodules expose the lower-level
power-user primitives directly.
"""

from opake.api.engine.distributed import (
    all_reduce,
    barrier,
    gather_for_metrics,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    local_shard,
    num_processes,
    process_index,
    sum_gradients,
    sum_gradients_,
    sync,
    wait_for_everyone,
)
from opake.api.engine.distributed.gradients import reduce_pytree, reduce_pytree_

__all__ = [
    "all_reduce",
    "barrier",
    "gather_for_metrics",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "local_shard",
    "num_processes",
    "process_index",
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
    "sync",
    "wait_for_everyone",
]
