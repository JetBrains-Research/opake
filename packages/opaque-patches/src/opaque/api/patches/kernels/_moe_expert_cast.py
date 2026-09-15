# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded low-precision shadows for frozen MoE expert banks."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ._moe_memory import _workspace_budget_bytes

_MAX_SHADOW_CACHE_BYTES = 4 * 1024**3
_SHADOW_POLICY_VERSION = 1
_EXCLUSIVE_STORAGE_USE_COUNT = 2
_CACHE: OrderedDict[tuple[object, ...], _Shadow] = OrderedDict()
_CACHE_BYTES = 0
_LOCK = threading.Lock()


@dataclass
class _Shadow:
    source: weakref.ReferenceType[torch.Tensor]
    value: torch.Tensor
    nbytes: int
    ready: torch.cuda.Event | None


def _cache_budget(device: torch.device) -> int:
    return min(_MAX_SHADOW_CACHE_BYTES, _workspace_budget_bytes(device))


def _source_key(tensor: torch.Tensor, dtype: torch.dtype) -> tuple[object, ...] | None:
    # Views are deliberately recast. Their storage may be mutated through an
    # independently versioned alias, which cannot be validated from this tensor.
    try:
        if tensor._is_view():
            return None
        storage = tensor.untyped_storage()
        if torch._C._storage_Use_Count(storage._cdata) != _EXCLUSIVE_STORAGE_USE_COUNT:
            return None
        return (
            _SHADOW_POLICY_VERSION,
            id(tensor),
            storage._cdata,
            tensor._version,
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
            dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )
    except (AttributeError, RuntimeError, TypeError):
        # Functional/BatchedTensor wrappers may intentionally hide storage.
        return None


def _drop(key: tuple[object, ...]) -> None:
    global _CACHE_BYTES
    entry = _CACHE.pop(key, None)
    if entry is not None:
        _CACHE_BYTES -= entry.nbytes


def _trim(budget: int) -> None:
    global _CACHE_BYTES
    while _CACHE and budget < _CACHE_BYTES:
        _, entry = _CACHE.popitem(last=False)
        _CACHE_BYTES -= entry.nbytes


def _evict_device(device: torch.device) -> None:
    for key, entry in tuple(_CACHE.items()):
        if entry.value.device == device:
            _drop(key)


def _release_source(
    key: tuple[object, ...], source: weakref.ReferenceType[torch.Tensor]
) -> None:
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None and entry.source is source:
            _drop(key)


def clear_expert_shadow_cache() -> None:
    """Clear retained frozen-expert shadows (primarily for tests)."""
    global _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def expert_shadow_cache_info() -> tuple[int, int]:
    """Return ``(entries, bytes)`` for diagnostics and regression tests."""
    with _LOCK:
        return len(_CACHE), _CACHE_BYTES


def cast_expert_banks(dtype: torch.dtype, *banks: torch.Tensor):
    """Cast expert banks under the MoE lifetime policy.

    Frozen, owning tensors use versioned shadows within a bounded per-process
    cache. Trainable tensors and storage views always recast, keeping the cast
    out of saved autograd state and bounding its lifetime to one kernel call.
    """
    global _CACHE_BYTES
    itemsize = torch.empty((), dtype=dtype).element_size()
    candidate_bytes = sum(
        bank.numel() * itemsize
        for bank in banks
        if bank.dtype != dtype
        and not bank.requires_grad
        and _source_key(bank, dtype) is not None
    )
    cast = []
    for bank in banks:
        if bank.dtype == dtype:
            cast.append(bank)
            continue
        if bank.requires_grad:
            cast.append(bank.to(dtype))
            continue

        key = _source_key(bank, dtype)
        budget = _cache_budget(bank.device)
        nbytes = bank.numel() * itemsize
        if key is None:
            cast.append(bank.to(dtype))
            continue

        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None and entry.source() is bank:
                _CACHE.move_to_end(key)
            else:
                if entry is not None:
                    _drop(key)
                entry = None
        if entry is not None:
            if entry.ready is not None:
                torch.cuda.current_stream(bank.device).wait_event(entry.ready)
            cast.append(entry.value)
            continue

        if candidate_bytes > budget:
            with _LOCK:
                _evict_device(bank.device)
            cast.append(bank.to(dtype))
            continue

        with _LOCK:
            _trim(budget)

        shadow = bank.to(dtype)
        ready = None
        if bank.is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(bank.device))
        source = weakref.ref(
            bank, lambda ref, cache_key=key: _release_source(cache_key, ref)
        )
        with _LOCK:
            # Remove older versions for this exact tensor without affecting
            # shadows owned by other expert banks sharing the same storage.
            for old_key in tuple(_CACHE):
                if old_key[1] == id(bank):
                    _drop(old_key)
            _CACHE[key] = _Shadow(source, shadow, nbytes, ready)
            _CACHE_BYTES += nbytes
            _trim(budget)
        cast.append(shadow)
    return tuple(cast)
