"""Regression tests for cache keys of composed process trees."""

from __future__ import annotations

from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)


class _CacheKeyProbe(DpProcess):
    """Leaf that records ordinary versus count-sensitive key selection."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _pld_cache_key(self) -> tuple[object, ...]:
        return ("pld", self.name)

    def _repeated_pld_cache_key(self, count: int) -> tuple[object, ...]:
        return ("repeated_pld", self.name, count)

    def pld(self, **_kwargs: object) -> Pld:  # pragma: no cover - key probe only
        raise AssertionError("cache-key tests must not compute a PLD")


def test_repeated_key_preserves_cached_wrapper_structure() -> None:
    process = Repeated(CachedProcess(_CacheKeyProbe("leaf")), 7)

    assert process._pld_cache_key() == (
        "Repeated",
        7,
        "CachedProcess",
        ("repeated_pld", "leaf", 7),
    )


def test_repeated_composed_key_uses_ordinary_child_pld_keys() -> None:
    process = Repeated(
        Composed(_CacheKeyProbe("left"), _CacheKeyProbe("right")),
        7,
    )

    assert process._pld_cache_key() == (
        "Repeated",
        7,
        "Composed",
        ("pld", "left"),
        ("pld", "right"),
    )
