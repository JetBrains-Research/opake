# opake.distributed

Distributed training utilities for differential privacy with DDP.

## Overview

The `opake.distributed` module provides composable primitives for multi-GPU
DP training:

- **Core**: `is_distributed()`, `get_rank()`, `get_world_size()`
- **Collectives**: `all_reduce()` (+ in-place variant in the `collectives` submodule)
- **Gradient aggregation**: `sum_gradients()` (copy-returning) and `sum_gradients_()` (in-place)
- **State sync**: `sync()` (type-dispatched; handles clipping + noise
  states and registered DP runtime objects)
- **Sharding**: `local_shard()`

DDP is the only supported parallelism strategy.
See [User Guide: Distributed Training](../user-guide/distributed.md) for usage.

## Core Utilities

::: opake.distributed.is_distributed
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.get_rank
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.get_world_size
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.collectives.all_reduce
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.collectives.all_reduce_
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.collectives.barrier
    options:
        show_source: true
        heading_level: 3

## Gradient Aggregation

::: opake.distributed.sum_gradients
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.gradients.sum_gradients_
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.gradients.reduce_pytree
    options:
        show_source: true
        heading_level: 3

::: opake.distributed.gradients.reduce_pytree_
    options:
        show_source: true
        heading_level: 3

## State Synchronization

::: opake.distributed.sync
    options:
        show_source: true
        heading_level: 3

The `sync()` machinery is type-dispatched: clipping and noise states
register themselves with `opake.distributed._state.register_sync_type`
and provide the right per-state aggregation rule. Lower-level scalar
reductions, tensor gathers, and object syncs live in `_state.py`;
they're internal plumbing for the registered DP runtime objects rather
than headline API.

## Privacy Ordering

The order of operations matters for DP guarantees:

```python
grads, clip_state = grad_fn(params, x, y, state=clip_state)  # 1. Clip
grads = dist_utils.sum_gradients(grads)                       # 2. Aggregate (copy)
noisy_grads, noise_state = noise_fn(grads, noise_state)       # 3. Noise
```

`sync` accepts one or more objects and returns synchronized values in the
same order:

```python
clip_state, aux = dist_utils.sync(clip_state, aux)
noise_state = dist_utils.sync(noise_state)
```

See [examples/distributed_dp_training.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/distributed_dp_training.py)
for a complete working script.
