# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Make ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` vmap-safe.

The stock pinned path allocates the host buffer with ``torch.empty(tensor.size(),
...)``. Under vmap ``tensor.size()`` is the unbatched logical shape and
``empty`` has no batch rule, so the buffer is unbatched and ``copy_`` cannot fill
it from the batched source. ``empty_like`` carries the batch dim through.

Applied only when torch has not already fixed this (it ships with the guard
scoping; see :mod:`.native_support`).
"""

from __future__ import annotations

import torch


def apply() -> None:
    """Install a vmap-safe pinned-memory ``save_on_cpu`` implementation."""
    import torch.autograd.graph as autograd_graph

    orig = autograd_graph.save_on_cpu

    class _VmapSaveOnCpu(orig):
        def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
            device_module = getattr(torch, device_type, torch.cuda)

            def pack_to_cpu(tensor):
                if not pin_memory:
                    return (tensor.device, tensor.cpu())
                is_pinnable = device_module.is_available() and not tensor.is_sparse
                packed = torch.empty_like(tensor, device="cpu", pin_memory=is_pinnable)
                packed.copy_(tensor, non_blocking=is_pinnable)
                return (tensor.device, packed)

            def unpack_from_cpu(packed):
                device, tensor = packed
                return tensor.to(device, non_blocking=pin_memory)

            # Skip orig.__init__ (it installs the unbatched-buffer hooks); go
            # straight to the grandparent saved_tensors_hooks.
            torch.autograd.graph.saved_tensors_hooks.__init__(
                self, pack_to_cpu, unpack_from_cpu
            )

    autograd_graph.save_on_cpu = _VmapSaveOnCpu
