"""Hashing arbitrarily deep ``DpProcess`` composition trees is safe.

Builds the worst-case shapes — long left-skewed and right-skewed
``Composed`` chains, periodic ``cached(...)`` barriers, and
``Repeated`` wrappers around deep inner chains — and asserts
``hash(tree)`` returns a finite value (no ``RecursionError``). Also
asserts the hash / equality contract: structurally-equal trees
produce equal hashes, structurally-distinct trees produce distinct
hashes.
"""

from __future__ import annotations

import opake.accounting as acc
from opake.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)


def _leaf():
    """A cheap leaf mechanism for tree construction."""
    return acc.identity()


def _build_left_skewed(leaf, depth: int):
    """``Composed(Composed(...Composed(leaf, leaf)..., leaf), leaf)``."""
    chain = leaf
    for _ in range(depth):
        chain = Composed(chain, leaf)
    return chain


# ---------------------------------------------------------------------------
# Recursion-safety regressions
# ---------------------------------------------------------------------------


def test_deeply_nested_composed_hashable():
    """A 10 000-deep left-skewed Composed chain hashes without recursion."""
    chain = _build_left_skewed(_leaf(), 10_000)
    h = hash(chain)
    assert isinstance(h, int)


def test_deeply_nested_cached_inside_composed_hashable():
    """Long ``Composed`` chain with periodic ``cached(...)`` barriers is hashable.

    Mirrors the incremental-accounting pattern from ``cached``'s
    docstring — cache every N steps to compose deltas on top of a
    precomputed PLD.
    """
    chain = _leaf()
    for i in range(10_000):
        chain = Composed(chain, _leaf())
        if (i + 1) % 500 == 0:
            chain = acc.cached(chain)
    h = hash(chain)
    assert isinstance(h, int)


def test_deeply_nested_repeated_inside_composed_hashable():
    """A ``Repeated`` whose ``inner`` is itself a deep chain hashes safely."""
    deep_inner = _build_left_skewed(_leaf(), 5_000)
    repeated = Repeated(deep_inner, 100)
    outer_chain = _build_left_skewed(_leaf(), 5_000)
    full = Composed(outer_chain, repeated)
    h = hash(full)
    assert isinstance(h, int)


def test_right_skewed_composed_hashable():
    """A 10 000-deep right-skewed ``Composed`` chain hashes without recursion."""
    leaf = _leaf()
    chain = leaf
    for _ in range(10_000):
        chain = Composed(leaf, chain)
    h = hash(chain)
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# Hash / equality contract
# ---------------------------------------------------------------------------


def test_hash_consistent_with_eq_for_composed():
    leaf = _leaf()
    chain1 = _build_left_skewed(leaf, 100)
    chain2 = _build_left_skewed(leaf, 100)
    assert chain1 == chain2
    assert hash(chain1) == hash(chain2)


def test_hash_consistent_with_eq_for_repeated():
    leaf = _leaf()
    r1 = Repeated(leaf, 50)
    r2 = Repeated(leaf, 50)
    assert r1 == r2
    assert hash(r1) == hash(r2)


def test_hash_consistent_with_eq_for_cached():
    leaf = _leaf()
    c1 = CachedProcess(leaf)
    c2 = CachedProcess(leaf)
    assert c1 == c2
    assert hash(c1) == hash(c2)


def test_hash_distinguishes_different_depths():
    """The hash is structure-sensitive: different depths → different hashes.

    The hash is a deterministic digest of the tree, so for these fixed inputs
    the inequality is stable (not a probabilistic non-collision claim) — it
    guards against a degenerate hash that ignores depth.
    """
    leaf = _leaf()
    a = _build_left_skewed(leaf, 100)
    b = _build_left_skewed(leaf, 101)
    assert a != b
    assert hash(a) != hash(b)


def test_hash_distinguishes_composed_from_repeated():
    """``Composed(x, x)`` and ``Repeated(x, 2)`` hash differently.

    Deterministic for these fixed inputs (sentinel tags keep the two node
    kinds from colliding) — guards against the wrapper kind being dropped from
    the digest, not a universal non-collision guarantee.
    """
    leaf = _leaf()
    c = Composed(leaf, leaf)
    r = Repeated(leaf, 2)
    assert c != r
    assert hash(c) != hash(r)
