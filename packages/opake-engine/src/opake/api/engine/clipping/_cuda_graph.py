# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay of the per-microbatch clipping kernel.

:class:`CudaGraphChunkCompiler` is a ``_chunk_compiler`` for
:func:`~opake.api.engine.clipping.clipped_grad` (and the factories that forward
to it). Each distinct input signature of the tensor-only chunk kernel
(``vmap(grad)`` + clipping + reduction for one microbatch) is captured once as a
CUDA graph and replayed afterwards. That removes the per-call Python and launch
overhead, which dominates DP-SGD steps whose microbatch kernels launch many
small operations.

Correctness rules:

* The capture key is the input pytree structure, every tensor's shape, stride,
  dtype, device and ``requires_grad``, and every non-tensor value. A changed
  Python value is captured anew instead of replayed; non-tensor values must be
  hashable.
* Before every replay all tensor inputs, including the clipping-norm tensor,
  are copied into the graph's static input buffers, so values that change per
  step (adaptive clipping thresholds, parameters) are always current.
* Outputs are cloned on return, because the next replay overwrites the graph's
  output buffers.
* Warmup runs (before the first capture only) and captures run under
  :func:`~opake.api.engine.device.capture_safe_kernels`, so kernels use their
  capturable code paths consistently. Later signatures, such as the partial last
  microbatch, are captured straight into the shared memory pool.

Host synchronizations and pageable host-to-device copies inside the kernel make
capture fail with a CUDA error; nothing falls back silently. Python state that
the kernel reads other than through its arguments (closures, globals) is frozen
at capture, which is why the chunk kernel is tensor-only. With more than
``max_graphs`` signatures, further signatures run eagerly.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from opake.api.engine.device import capture_safe_kernels
from opake.api.engine.pytree import tree_flatten, tree_map, tree_unflatten
from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


def _leaf_key(leaf: Any) -> tuple:
    if isinstance(leaf, torch.Tensor):
        return (
            "tensor",
            tuple(leaf.shape),
            tuple(leaf.stride()),
            leaf.dtype,
            str(leaf.device),
            leaf.requires_grad,
        )
    try:
        hash(leaf)
    except TypeError:
        raise ConfigurationError(
            *(
                "CUDA-graph replay keys captures on non-tensor arguments, which "
                f"must be hashable; got {type(leaf).__name__}.",
            )
        ) from None
    return ("value", type(leaf), leaf)


class CudaGraphChunkCompiler:
    """``_chunk_compiler`` that replays each microbatch kernel from a CUDA graph.

    One instance owns one CUDA-graph memory pool shared by all its captures, so
    use one instance per training run.

    Args:
        warmup_runs: Eager runs before the first capture (lazy initialization,
            autotuning). Later captures need none.
        max_graphs: Captured signatures before new ones run eagerly.
    """

    #: The graph replays the eager kernels it captured, so kernel choices that
    #: are only valid eagerly (the fused clip backend) stay valid.
    replays_eager_kernels = True

    def __init__(self, *, warmup_runs: int = 2, max_graphs: int = 32) -> None:
        if warmup_runs < 1:
            raise ConfigurationError(*("warmup_runs must be >= 1",))
        self.warmup_runs = warmup_runs
        self.max_graphs = max_graphs
        self.captures = 0
        self.replays = 0
        self.capture_seconds = 0.0
        self._pool = None
        self._warmed_up = False
        self._limit_logged = False

    def __call__(self, fn: Callable) -> Callable:
        return _GraphedChunk(fn, self)


class _GraphedChunk:
    def __init__(self, fn: Callable, owner: CudaGraphChunkCompiler) -> None:
        self._fn = fn
        self._owner = owner
        self._graphs: dict = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        leaves, treedef = tree_flatten((args, kwargs))
        device = next(
            (x.device for x in leaves if isinstance(x, torch.Tensor) and x.is_cuda),
            None,
        )
        if device is None:
            return self._fn(*args, **kwargs)
        key = (treedef, tuple(_leaf_key(x) for x in leaves))
        entry = self._graphs.get(key)
        if entry is None:
            owner = self._owner
            if owner.captures >= owner.max_graphs:
                if not owner._limit_logged:
                    log.warning(
                        "CUDA-graph replay reached max_graphs=%d; new microbatch "
                        "signatures run eagerly.",
                        owner.max_graphs,
                    )
                    owner._limit_logged = True
                return self._fn(*args, **kwargs)
            entry = self._graphs[key] = self._capture(leaves, treedef, device)
        graph, static_leaves, static_out = entry
        tensor_idx = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
        torch._foreach_copy_(
            [static_leaves[i] for i in tensor_idx], [leaves[i] for i in tensor_idx]
        )
        graph.replay()
        self._owner.replays += 1
        return tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x, static_out
        )

    def _capture(self, leaves: list, treedef: Any, device: torch.device) -> tuple:
        owner = self._owner
        start = time.perf_counter()
        static_leaves = [
            x.clone() if isinstance(x, torch.Tensor) else x for x in leaves
        ]
        static_args, static_kwargs = tree_unflatten(treedef, static_leaves)
        with torch.cuda.device(device), capture_safe_kernels():
            if not owner._warmed_up:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(owner.warmup_runs):
                        self._fn(*static_args, **static_kwargs)
                torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            # Release blocks the warmup cached for its side stream before the
            # capture allocates its own pool.
            torch.cuda.empty_cache()
            if owner._pool is None:
                owner._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=owner._pool):
                static_out = self._fn(*static_args, **static_kwargs)
            torch.cuda.synchronize()
        owner._warmed_up = True
        owner.captures += 1
        elapsed = time.perf_counter() - start
        owner.capture_seconds += elapsed
        log.info("Captured microbatch CUDA graph %d in %.1fs", owner.captures, elapsed)
        return graph, static_leaves, static_out
