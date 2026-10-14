"""Shared cache decorators for PLD-producing process methods."""

from __future__ import annotations

import functools
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from threading import RLock
from typing import TYPE_CHECKING

from .discretization import (
    DiscretizationConfig,
    _use_discretization,
    get_discretization,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._base import Pld

_CacheKey = tuple[DiscretizationConfig, Hashable, int | None]
_IdentityKey = tuple[DiscretizationConfig, int | None]
_MISSING = object()


_active_captures: ContextVar[tuple[list[Pld], ...]] = ContextVar(
    "opake_accounting_pld_captures",
    default=(),
)
_pld_call_depth: ContextVar[int] = ContextVar(
    "opake_accounting_pld_call_depth",
    default=0,
)


@contextmanager
def _capture_pld_evaluations() -> Iterator[list[Pld]]:
    """Capture caller-visible PLDs without their nested composition work."""
    captured: list[Pld] = []
    token = _active_captures.set((*_active_captures.get(), captured))
    try:
        yield captured
    finally:
        _active_captures.reset(token)


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

        config, repeat_count = key
        non_mc_config = replace(config, mc_resolution=0.0, mc_failure_probability=0.0)
        for candidate_key, candidate in reversed(entry[1].items()):
            candidate_config, candidate_repeat_count = candidate_key
            if (
                candidate_repeat_count == repeat_count
                and candidate.mc_resolution == 0.0
                and candidate.mc_failure_probability == 0.0
                and replace(
                    candidate_config,
                    mc_resolution=0.0,
                    mc_failure_probability=0.0,
                )
                == non_mc_config
            ):
                entry[1].move_to_end(candidate_key)
                return candidate
        return _MISSING

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

    def get(
        self,
        process: object,
        identity_key: _IdentityKey,
        shared_key: Callable[[], _CacheKey],
    ) -> tuple[Pld | object, _CacheKey | None]:
        with self._lock:
            if self._retain_per_instance:
                cached = self._get_identity(process, identity_key)
                if cached is not _MISSING:
                    self._hits += 1
                    return cached, None
            key = shared_key()
            cached = self._get_shared(key)
            if cached is not _MISSING:
                if self._retain_per_instance:
                    self._store_identity(process, identity_key, cached)
                self._hits += 1
                return cached, key
            self._misses += 1
            return _MISSING, key

    def put(
        self,
        process: object,
        identity_key: _IdentityKey,
        shared_key: _CacheKey,
        result: Pld,
    ) -> Pld:
        with self._lock:
            cached = self._get_shared(shared_key)
            if cached is _MISSING:
                self._store(self._shared_entries, shared_key, result)
                cached = result
            if self._retain_per_instance:
                self._store_identity(process, identity_key, cached)
            return cached

    def get_or_compute(
        self,
        process: object,
        identity_key: _IdentityKey,
        shared_key: Callable[[], _CacheKey],
        compute: Callable[[], Pld],
    ) -> Pld:
        cached, key = self.get(process, identity_key, shared_key)
        if cached is not _MISSING:
            return cached
        assert key is not None
        return self.put(process, identity_key, key, compute())

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
            captures = _active_captures.get()
            if not captures:
                return cache.get_or_compute(
                    self,
                    (config, None),
                    lambda: (config, self._pld_cache_key(), None),
                    lambda: _compute_pld(method, self, config),
                )

            depth = _pld_call_depth.get()
            depth_token = _pld_call_depth.set(depth + 1)
            try:
                result = cache.get_or_compute(
                    self,
                    (config, None),
                    lambda: (config, self._pld_cache_key(), None),
                    lambda: _compute_pld(method, self, config),
                )
            finally:
                _pld_call_depth.reset(depth_token)

            if depth == 0:
                for captured in captures:
                    captured.append(result)
            return result

        def cache_get(self, **kwargs):
            config = _resolve_config(**kwargs)
            identity_key = (config, None)
            cached, shared_key = cache.get(
                self,
                identity_key,
                lambda: (config, self._pld_cache_key(), None),
            )
            return cached is not _MISSING, cached, (identity_key, shared_key)

        def cache_put(self, token, result):
            identity_key, shared_key = token
            assert shared_key is not None
            return cache.put(self, identity_key, shared_key, result)

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        wrapper.cache_get = cache_get
        wrapper.cache_put = cache_put
        return wrapper

    return decorator


def _compute_pld(
    method: Callable[..., Pld],
    process: object,
    config: DiscretizationConfig,
) -> Pld:
    with _use_discretization(config):
        return method(process)
