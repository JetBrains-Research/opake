"""Thin wrappers around ``torch.distributed`` collectives.

Provides ``is_distributed``, ``get_rank``, ``get_world_size``, ``all_reduce``
and its in-place variant, and ``barrier``. All are safe to call outside a
process group; non-distributed contexts fall through to sensible defaults
(rank 0, world_size 1, no-op barrier).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from opake.exceptions import ConfigurationError, OperationError


def is_distributed() -> bool:
    """Return True if ``torch.distributed`` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the world size (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


_OP_MAP: dict[str, object] = {}


def _resolve_op(op: str):
    """Map a string to a ``torch.distributed.ReduceOp``. Populated lazily."""
    if not _OP_MAP:
        _OP_MAP.update(
            {
                "sum": dist.ReduceOp.SUM,
                "mean": dist.ReduceOp.AVG,
                "max": dist.ReduceOp.MAX,
                "min": dist.ReduceOp.MIN,
                "product": dist.ReduceOp.PRODUCT,
            }
        )
    if op not in _OP_MAP:
        raise ConfigurationError(
            *(
                f"Invalid reduction operation: {op}. Must be one of: {list(_OP_MAP.keys())}",
            )
        )
    return _OP_MAP[op]


def all_reduce_(tensor: torch.Tensor, op: str = "sum") -> None:
    """All-reduce ``tensor`` across ranks in place.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.
        ValueError: If ``op`` is not a recognized reduction.
    """
    reduce_op = _resolve_op(op)
    if not is_distributed():
        raise OperationError(
            *(
                "torch.distributed is not initialized. "
                "Call torch.distributed.init_process_group() first.",
            )
        )
    dist.all_reduce(tensor, op=reduce_op)


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Return a reduced clone of ``tensor``; input is unchanged."""
    reduced = tensor.clone()
    all_reduce_(reduced, op=op)
    return reduced


def barrier() -> None:
    """Block until every rank reaches this call (no-op if not distributed)."""
    if is_distributed():
        dist.barrier()


def is_main_process() -> bool:
    """Return True on the rank-0 process (always True if not distributed)."""
    return get_rank() == 0


def num_processes() -> int:
    """Return the world size (1 if not distributed). Alias of ``get_world_size``."""
    return get_world_size()


def process_index() -> int:
    """Return the current rank (0 if not distributed). Alias of ``get_rank``."""
    return get_rank()


def wait_for_everyone() -> None:
    """Block until every rank reaches this call (no-op if not distributed).

    Module-level alias of :func:`barrier`, matching the
    ``accelerator.wait_for_everyone()`` idiom that callers port from.
    """
    barrier()


def gather_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather ``tensor`` across ranks and concatenate along dim 0.

    In a non-distributed context returns ``tensor`` unchanged. Intended for
    metric aggregation (e.g. detached KL means, reference-logprob shards),
    where duplicate samples from Poisson sampling are not a correctness
    concern. This is **not** a gradient primitive — do not use it inside the
    clipped/noised per-example gradient path.

    All ranks must pass tensors of identical shape and dtype.
    """
    if not is_distributed():
        return tensor
    world_size = get_world_size()
    # torch.cat cannot concatenate 0-dim tensors; promote a per-rank scalar
    # metric to 1-D so it gathers into a (world_size,) vector.
    local = tensor.unsqueeze(0) if tensor.dim() == 0 else tensor
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(gathered, dim=0)


__all__ = [
    "all_reduce",
    "all_reduce_",
    "barrier",
    "gather_for_metrics",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "num_processes",
    "process_index",
    "wait_for_everyone",
]
