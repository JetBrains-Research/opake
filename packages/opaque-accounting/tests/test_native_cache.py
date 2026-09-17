"""Tests for :mod:`opaque.api.accounting.core._native_cache`."""

from __future__ import annotations

import os
import threading
import warnings

import pytest

from opaque.api.accounting.core._native_cache import (
    NativeCache,
    _clear_all_native_caches,
)


@pytest.fixture(autouse=True)
def _clear_global_native_caches() -> None:
    yield
    _clear_all_native_caches()


def test_basic_get_create_returns_handle() -> None:
    calls = [0]

    def factory() -> int:
        calls[0] += 1
        return 7

    c = NativeCache(
        name="basic",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=lambda _h: None,
    )
    k = ("a", 1)
    assert c.get_or_create(k, factory) == 7
    assert c.get_or_create(k, factory) == 7
    assert calls[0] == 1


def test_lru_eviction_by_entry_count() -> None:
    evicted: list[int] = []

    def destruct(h: int) -> None:
        evicted.append(h)

    c = NativeCache(
        name="e2",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=2,
        nbytes_estimate=lambda _k: 1,
        destructor=destruct,
    )
    c.get_or_create(("a",), lambda: 1)
    c.get_or_create(("b",), lambda: 2)
    c.get_or_create(("c",), lambda: 3)
    assert len(c) == 2
    assert 1 in evicted
    assert c.get_or_create(("b",), lambda: 99) == 2
    assert c.get_or_create(("c",), lambda: 99) == 3


def test_lru_eviction_by_bytes() -> None:
    evicted: list[int] = []

    c = NativeCache(
        name="bytes",
        max_bytes_env=None,
        default_max_bytes=100,
        max_entries=100,
        nbytes_estimate=lambda _k: 60,
        destructor=lambda h: evicted.append(h),
    )
    c.get_or_create(("a",), lambda: 10)
    c.get_or_create(("b",), lambda: 20)
    c.get_or_create(("c",), lambda: 30)
    assert 10 in evicted
    assert 20 in evicted
    assert len(c) == 1
    assert c.get_or_create(("c",), lambda: 99) == 30


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("cap", [0, 50, 100])
def test_admission_warns_only_for_oversized_entries(atomic, cap) -> None:
    calls = 0

    def factory() -> int:
        nonlocal calls
        calls += 1
        return 7

    cache = NativeCache(
        name="transcripts",
        max_bytes_env=None,
        default_max_bytes=cap,
        max_entries=10,
        nbytes_estimate=lambda _k: 100,
        destructor=lambda _h: None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            if atomic:
                result = cache.with_handle(("x",), factory, lambda handle: handle)
            else:
                result = cache.get_or_create(("x",), factory)
            assert result == (7 if cap == 100 else None)

    assert calls == (1 if cap == 100 else 0)
    assert len(caught) == (1 if cap == 50 else 0)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert "100 bytes" in str(caught[0].message)
        assert "50-byte" in str(caught[0].message)


def test_clear_releases_all_handles() -> None:
    evicted: list[int] = []

    c = NativeCache(
        name="clr",
        max_bytes_env=None,
        default_max_bytes=100,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=lambda h: evicted.append(h),
    )
    c.get_or_create(("a",), lambda: 1)
    c.get_or_create(("b",), lambda: 2)
    c.clear()
    assert len(c) == 0
    assert sorted(evicted) == [1, 2]


def test_clear_is_idempotent() -> None:
    evicted: list[int] = []

    c = NativeCache(
        name="idemp",
        max_bytes_env=None,
        default_max_bytes=100,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=lambda h: evicted.append(h),
    )
    c.get_or_create(("a",), lambda: 1)
    c.clear()
    c.clear()
    assert evicted == [1]


def test_env_override() -> None:
    env_name = "OPAQUE_NATIVE_CACHE_TEST_CAP_BYTES"
    old = os.environ.get(env_name)
    try:
        os.environ[env_name] = "999999"
        c = NativeCache(
            name="envcap",
            max_bytes_env=env_name,
            default_max_bytes=100,
            max_entries=10,
            nbytes_estimate=lambda _k: 1,
            destructor=lambda _h: None,
        )
        assert c.max_bytes == 999999
    finally:
        if old is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old


def test_env_disable_with_zero() -> None:
    env_name = "OPAQUE_NATIVE_CACHE_TEST_ZERO"
    old = os.environ.get(env_name)
    try:
        os.environ[env_name] = "0"
        c = NativeCache(
            name="off",
            max_bytes_env=env_name,
            default_max_bytes=10**9,
            max_entries=10,
            nbytes_estimate=lambda _k: 1,
            destructor=lambda _h: None,
        )
        calls = [0]

        def factory() -> int:
            calls[0] += 1
            return 1

        assert c.get_or_create(("k",), factory) is None
        assert calls[0] == 0
    finally:
        if old is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old


def test_thread_safety() -> None:
    n = 8
    barrier = threading.Barrier(n)
    factory_calls = [0]
    lock = threading.Lock()
    handles: list[int | None] = []

    c = NativeCache(
        name="par",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=lambda _h: None,
    )

    def factory() -> int:
        with lock:
            factory_calls[0] += 1
        return 42

    def worker() -> None:
        barrier.wait()
        h = c.get_or_create(("shared",), factory)
        barrier.wait()
        with lock:
            handles.append(h)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert factory_calls[0] == 1
    assert all(h == 42 for h in handles)


def test_destructor_swallow_warns() -> None:
    def boom(_h: int) -> None:
        raise RuntimeError("boom")

    c = NativeCache(
        name="warn",
        max_bytes_env=None,
        default_max_bytes=100,
        max_entries=1,
        nbytes_estimate=lambda _k: 1,
        destructor=boom,
    )
    c.get_or_create(("a",), lambda: 1)
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        c.get_or_create(("b",), lambda: 2)
        assert len(wrec) >= 1
