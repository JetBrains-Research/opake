"""Distributed training helpers for differential privacy.

Headline DP-DDP flow:

- **detection**: :func:`is_distributed`, :func:`get_rank`, :func:`get_world_size`
- **collectives**: :func:`all_reduce`
- **gradient aggregation**: :func:`sum_gradients` (out-of-place) and
  :func:`sum_gradients_` (in-place; mutates the clipped pytree leaves).
- **state sync**: :func:`sync` (type-dispatched; handles clipping + noise
  states, registered DP runtime objects).
- **dataset sharding**: :func:`local_shard`.

The two documented power-user submodules collect lower-level primitives:

- :mod:`opake.distributed.collectives` — ``all_reduce`` (+ in-place),
  ``barrier``, and the detection helpers.
- :mod:`opake.distributed.gradients` — ``reduce_pytree`` (+ in-place) and
  ``sum_gradients`` (+ in-place).
"""

from opake.api.engine.distributed._shard import local_shard
from opake.api.engine.distributed._state import register_sync_type, sync
from opake.api.engine.distributed.collectives import (
    all_reduce,
    barrier,
    gather_for_metrics,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    num_processes,
    process_index,
    wait_for_everyone,
)
from opake.api.engine.distributed.gradients import sum_gradients, sum_gradients_

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
    "register_sync_type",
    "sum_gradients",
    "sum_gradients_",
    "sync",
    "wait_for_everyone",
]
