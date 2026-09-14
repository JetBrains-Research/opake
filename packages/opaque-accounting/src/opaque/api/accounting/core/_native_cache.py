"""Thread-safe LRU cache with a byte budget for native (Rust) handles.

Contributor-internal: not re-exported from the ``opaque.accounting`` façade.
"""

from __future__ import annotations

import os
import threading
import warnings
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")

#: Module-level registry holding *weak* references so test-constructed
#: caches (and any future dynamic caches) get garbage-collected without an
#: explicit unregister call. The b-min-sep cache that ships with
#: opaque-dpftrl lives at module scope so it stays alive for the lifetime
#: of the process anyway.
_registry: weakref.WeakSet[NativeCache] = weakref.WeakSet()


def _read_max_bytes(max_bytes_env: str | None, default_max_bytes: int) -> int:
    if max_bytes_env is None:
        return default_max_bytes
    raw = os.environ.get(max_bytes_env, "")
    if not raw.strip():
        return default_max_bytes
    try:
        return max(0, int(raw))
    except ValueError:
        return default_max_bytes


def _clear_all_native_caches() -> None:
    for cache in list(_registry):
        cache.clear()


class NativeCache:
    """LRU cache of native handles with a total-byte cap and entry cap."""

    def __init__(
        self,
        *,
        name: str,
        max_bytes_env: str | None,
        default_max_bytes: int,
        max_entries: int,
        nbytes_estimate: Callable[[tuple], int],
        destructor: Callable[[int], None],
    ) -> None:
        self._name = name
        self._max_bytes_env = max_bytes_env
        self._default_max_bytes = default_max_bytes
        self._max_entries = max_entries
        self._nbytes_estimate = nbytes_estimate
        self._destructor = destructor
        self._max_bytes = _read_max_bytes(max_bytes_env, default_max_bytes)
        self._cache: OrderedDict[tuple, int] = OrderedDict()
        self._lock = threading.Lock()
        _registry.add(self)

    @property
    def max_bytes(self) -> int:
        """Current byte cap (env-driven, refreshed on every cache lookup)."""
        return self._max_bytes

    def _refresh_max_bytes(self) -> None:
        """Re-read the env var so a runtime change to ``OPAQUE_*_MAX_BYTES``
        (e.g. shrink to 0 for an A/B benchmark) takes effect on the next
        lookup. The previous flat-file cache read the env var on every
        lookup; preserve that semantics here.
        """
        self._max_bytes = _read_max_bytes(self._max_bytes_env, self._default_max_bytes)

    def get_or_create(self, key: tuple, factory: Callable[[], int]) -> int | None:
        """Return a cached handle for ``key`` (creating it once if absent).

        .. warning::

            The returned handle is racy under concurrent
            :meth:`clear` / :func:`_clear_all_native_caches`: another
            thread may destruct the underlying native resource between
            the return of this method and the caller's use of the
            handle. Prefer :meth:`with_handle` (which holds the cache
            lock around the use) for any code path that runs inside or
            alongside :func:`opaque.accounting.calibration.calibrate`.
        """
        nbytes = self._admitted_nbytes(key)
        if nbytes is None:
            return None

        with self._lock:
            return self._get_or_create_locked(key, factory, nbytes)

    def with_handle(
        self,
        key: tuple,
        factory: Callable[[], int],
        use_handle: Callable[[int], _T],
    ) -> _T | None:
        """Atomically get-or-create the handle and run ``use_handle(handle)``.

        Holds the per-cache lock across both the lookup and the use, so a
        concurrent :func:`_clear_all_native_caches` (e.g. from
        :func:`opaque.accounting.calibration.calibrate`'s ``finally`` clause)
        cannot destruct the handle mid-use. Returns ``None`` when the
        cache is disabled (``max_bytes == 0`` or ``nbytes > max_bytes``);
        callers must fall back to a no-handle code path in that case.
        """
        nbytes = self._admitted_nbytes(key)
        if nbytes is None:
            return None

        with self._lock:
            handle = self._get_or_create_locked(key, factory, nbytes)
            return use_handle(handle)

    def _admitted_nbytes(self, key: tuple) -> int | None:
        self._refresh_max_bytes()
        if self._max_bytes == 0:
            return None
        nbytes = self._nbytes_estimate(key)
        if nbytes > self._max_bytes:
            setting = f" ({self._max_bytes_env})" if self._max_bytes_env else ""
            warnings.warn(
                f"{self._name} entry needs {nbytes:,} bytes, exceeding the "
                f"{self._max_bytes:,}-byte cache limit{setting}; "
                "this entry cannot be reused.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return nbytes

    def _get_or_create_locked(
        self, key: tuple, factory: Callable[[], int], nbytes: int
    ) -> int:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        current_bytes = sum(self._nbytes_estimate(k) for k in self._cache)
        while self._cache and (
            len(self._cache) >= self._max_entries
            or current_bytes + nbytes > self._max_bytes
        ):
            old_key, old_h = self._cache.popitem(last=False)
            current_bytes -= self._nbytes_estimate(old_key)
            self._safe_destruct(old_h)

        handle = factory()
        self._cache[key] = handle
        return handle

    def _safe_destruct(self, handle: int) -> None:
        try:
            self._destructor(handle)
        except Exception as exc:
            warnings.warn(
                f"NativeCache({self._name!r}): destructor failed for handle "
                f"{handle}: {exc}",
                stacklevel=2,
            )

    def clear(self) -> None:
        with self._lock:
            while self._cache:
                _k, h = self._cache.popitem(last=False)
                self._safe_destruct(h)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._cache


def native_cache(
    *,
    name: str,
    max_bytes_env: str | None,
    default_max_bytes: int,
    max_entries: int,
    nbytes_estimate: Callable[[tuple], int],
    destructor: Callable[[int], None],
) -> NativeCache:
    """Construct a :class:`NativeCache` (readable call sites for adapters)."""
    return NativeCache(
        name=name,
        max_bytes_env=max_bytes_env,
        default_max_bytes=default_max_bytes,
        max_entries=max_entries,
        nbytes_estimate=nbytes_estimate,
        destructor=destructor,
    )
