"""Shared cache decorators for PLD-producing process methods."""

from __future__ import annotations

import functools
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import TYPE_CHECKING

from .discretization import (
    DiscretizationConfig,
    _use_discretization,
    get_discretization,
)

if TYPE_CHECKING:
    from ._base import Pld

_CacheKey = tuple[DiscretizationConfig, Hashable, int | None]
_IdentityKey = tuple[DiscretizationConfig, int | None]
_MISSING = object()


class _WeakIdentityPldCache:
    """Keep bounded PLD entries without retaining process objects."""

    def __init__(
        self, maxsize: int | None, *, retain_per_instance: bool = False
    ) -> None:
        self._maxsize = maxsize
        self._retain_per_instance = retain_per_instance
        self._shared_entries: OrderedDict[_CacheKey, Pld] = OrderedDict()
        self._identity_entries: dict[
            int, tuple[weakref.ReferenceType[object], OrderedDict[_IdentityKey, Pld]]
        ] = {}
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def _get_shared(self, key: _CacheKey) -> Pld | object:
        cached = self._shared_entries.get(key, _MISSING)
        if cached is not _MISSING:
            self._shared_entries.move_to_end(key)
        return cached

    def _store(
        self, entries: OrderedDict[_CacheKey, Pld], key: _CacheKey, value: Pld
    ) -> None:
        if self._maxsize is None:
            entries[key] = value
        elif self._maxsize > 0:
            entries[key] = value
            if len(entries) > self._maxsize:
                entries.popitem(last=False)

    def _get_identity(self, process: object, key: _IdentityKey) -> Pld | object:
        entry = self._identity_entries.get(id(process))
        if entry is None or entry[0]() is not process:
            return _MISSING
        cached = entry[1].get(key, _MISSING)
        if cached is not _MISSING:
            entry[1].move_to_end(key)
        return cached

    def _remove_identity(
        self, process_id: int, process_ref: weakref.ReferenceType[object]
    ) -> None:
        with self._lock:
            entry = self._identity_entries.get(process_id)
            if entry is not None and entry[0] is process_ref:
                del self._identity_entries[process_id]

    def _store_identity(self, process: object, key: _IdentityKey, value: Pld) -> None:
        process_id = id(process)
        entry = self._identity_entries.get(process_id)
        if entry is None or entry[0]() is not process:
            process_ref = weakref.ref(
                process,
                lambda ref, process_id=process_id: self._remove_identity(
                    process_id, ref
                ),
            )
            entries: OrderedDict[_IdentityKey, Pld] = OrderedDict()
            self._identity_entries[process_id] = (process_ref, entries)
        else:
            entries = entry[1]
        self._store(entries, key, value)

    def get_or_compute(
        self,
        process: object,
        identity_key: _IdentityKey,
        shared_key: Callable[[], _CacheKey],
        compute: Callable[[], Pld],
    ) -> Pld:
        with self._lock:
            if self._retain_per_instance:
                cached = self._get_identity(process, identity_key)
                if cached is not _MISSING:
                    self._hits += 1
                    return cached
            key = shared_key()
            cached = self._get_shared(key)
            if cached is not _MISSING:
                if self._retain_per_instance:
                    self._store_identity(process, identity_key, cached)
                self._hits += 1
                return cached
            self._misses += 1

        result = compute()

        with self._lock:
            cached = self._get_shared(key)
            if cached is _MISSING:
                self._store(self._shared_entries, key, result)
                cached = result
            if self._retain_per_instance:
                self._store_identity(process, identity_key, cached)
            return cached

    def cache_clear(self) -> None:
        with self._lock:
            self._shared_entries.clear()
            self._identity_entries.clear()
            self._hits = 0
            self._misses = 0

    def cache_info(self) -> functools._CacheInfo:
        with self._lock:
            return functools._CacheInfo(
                self._hits,
                self._misses,
                self._maxsize,
                len(self._shared_entries),
            )


def _resolve_config(
    *,
    discretization: float | None,
    log_x_mass_truncation_bound: float | None,
    max_grid_size: int | None,
    max_conv_grid: int | None,
    seed: int | None,
    mc_resolution: float | None,
    mc_failure_probability: float | None,
) -> DiscretizationConfig:
    return get_discretization(
        discretization=discretization,
        log_x_mass_truncation_bound=log_x_mass_truncation_bound,
        max_grid_size=max_grid_size,
        max_conv_grid=max_conv_grid,
        seed=seed,
        mc_resolution=mc_resolution,
        mc_failure_probability=mc_failure_probability,
    )


def pld_cache(*, maxsize: int | None, retain_per_instance: bool = False):
    """Cache a ``DpProcess.pld`` method by resolved configuration and mechanism.

    ``retain_per_instance`` keeps each live process's entries available even
    after the structurally shared LRU evicts them. This lets an explicit cache
    boundary remain effective while it is part of a growing composition tree.
    """

    def decorator(method):
        cache = _WeakIdentityPldCache(maxsize, retain_per_instance=retain_per_instance)

        @functools.wraps(method)
        def wrapper(
            self,
            *,
            discretization: float | None = None,
            log_x_mass_truncation_bound: float | None = None,
            max_grid_size: int | None = None,
            max_conv_grid: int | None = None,
            seed: int | None = None,
            mc_resolution: float | None = None,
            mc_failure_probability: float | None = None,
        ) -> Pld:
            config = _resolve_config(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                seed=seed,
                mc_resolution=mc_resolution,
                mc_failure_probability=mc_failure_probability,
            )
            return cache.get_or_compute(
                self,
                (config, None),
                lambda: (config, self._pld_cache_key(), None),
                lambda: _compute_pld(method, self, config),
            )

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        return wrapper

    return decorator


def _compute_pld(
    method: Callable[..., Pld],
    process: object,
    config: DiscretizationConfig,
) -> Pld:
    with _use_discretization(config):
        return method(process)
