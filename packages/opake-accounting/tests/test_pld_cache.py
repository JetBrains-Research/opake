"""Regression tests for resolved PLD cache identity."""

from __future__ import annotations

import gc
import weakref

import pytest

import opake.accounting as acc
from opake.api.accounting.core._pld_cache import pld_cache


class _CachedProcess:
    def __init__(self) -> None:
        self.calls = 0

    def _pld_cache_key(self) -> str:
        return "cached-process"

    @pld_cache(maxsize=2)
    def pld(self) -> object:
        self.calls += 1
        return object()


class _UnboundedCachedProcess:
    def __init__(self) -> None:
        self.calls = 0

    def _pld_cache_key(self) -> str:
        return "unbounded-cached-process"

    @pld_cache(maxsize=None)
    def pld(self) -> object:
        self.calls += 1
        return object()


class _NamedCachedProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def _pld_cache_key(self) -> str:
        return self.name

    @pld_cache(maxsize=2)
    def pld(self) -> object:
        self.calls += 1
        return object()


@pytest.fixture(autouse=True)
def _restore_discretization() -> None:
    from opake.accounting import discretization

    original = discretization._default_config
    try:
        discretization._default_config = None
        yield
    finally:
        discretization._default_config = original


@pytest.mark.parametrize(
    "config",
    [
        {"discretization": 0.2},
        {"tail_mass_truncation": 1e-12},
        {"max_conv_grid": 16},
    ],
)
def test_existing_cached_process_tracks_global_discretization(
    config: dict[str, float | int],
) -> None:
    process = acc.cached(acc.eps_delta(0.11))

    default_pld = process.pld()
    acc.set_discretization(**config)
    changed_pld = process.pld()
    from opake.accounting import discretization

    discretization._default_config = None

    assert changed_pld is not default_pld
    assert process.pld() is default_pld


def test_query_override_shares_the_matching_resolved_cache_entry() -> None:
    process = acc.eps_delta(0.11)

    override_pld = process.pld(discretization=0.2)
    acc.set_discretization(discretization=0.2)

    assert process.pld() is override_pld


def test_distinct_processes_do_not_share_pld_cache_entries() -> None:
    first = acc.eps_delta(0.11)
    second = acc.eps_delta(0.12)
    first.pld.cache_clear()

    assert first.pld(discretization=0.1) is not second.pld(discretization=0.1)


def test_pld_cache_reuses_entries_evicts_lru_entry_and_clears() -> None:
    process = _CachedProcess()

    first = process.pld(discretization=0.1)
    assert process.pld(discretization=0.1) is first
    process.pld(discretization=0.2)
    process.pld(discretization=0.3)
    assert process.pld(discretization=0.1) is not first
    assert process.calls == 4

    process.pld.cache_clear()
    process.pld(discretization=0.1)
    assert process.calls == 5


def test_pld_cache_reuses_entries_for_equal_processes() -> None:
    _CachedProcess.pld.cache_clear()
    first = _CachedProcess()
    second = _CachedProcess()

    first_pld = first.pld(discretization=0.1)

    assert second.pld(discretization=0.1) is first_pld
    assert first.calls == 1
    assert second.calls == 0


def test_pld_cache_bound_applies_across_live_processes() -> None:
    _NamedCachedProcess.pld.cache_clear()
    processes = [_NamedCachedProcess(str(index)) for index in range(3)]

    first = processes[0].pld()
    processes[1].pld()
    processes[2].pld()

    assert processes[0].pld() is not first
    assert processes[0].calls == 2
    assert _NamedCachedProcess.pld.cache_info().currsize == 2


def test_pld_cache_supports_unbounded_entries() -> None:
    process = _UnboundedCachedProcess()

    first = process.pld(discretization=0.1)
    process.pld(discretization=0.2)

    assert process.pld(discretization=0.1) is first
    assert process.calls == 2


def test_pld_caches_do_not_retain_queried_processes() -> None:
    process = _CachedProcess()
    process.pld()
    process_ref = weakref.ref(process)

    del process
    gc.collect()

    assert process_ref() is None


def test_pld_cache_does_not_retain_composed_process_trees() -> None:
    left = acc.eps_delta(0.11)
    right = acc.eps_delta(0.12)
    process = left | right
    process.pld(discretization=0.2)
    process_refs = [weakref.ref(process), weakref.ref(left), weakref.ref(right)]

    del process, left, right
    gc.collect()

    assert all(process_ref() is None for process_ref in process_refs)
