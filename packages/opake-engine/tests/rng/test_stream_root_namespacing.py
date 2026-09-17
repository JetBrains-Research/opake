"""Regression tests for engine-owned RNG stream roots."""

from opake.api.engine.noise_allocation import (
    PAIRED_FIRST_STREAM_FOLD,
    PAIRED_SECOND_STREAM_FOLD,
)
from opake.random import fold_in, key, split

_ENGINE_STREAM_ROOTS = (
    ("opake.paired.first", PAIRED_FIRST_STREAM_FOLD),
    ("opake.paired.second", PAIRED_SECOND_STREAM_FOLD),
)


def test_engine_stream_roots_match_registered_tags() -> None:
    for expected, actual in _ENGINE_STREAM_ROOTS:
        assert actual == expected, f"expected {expected!r}, got {actual!r}"


def test_engine_stream_roots_do_not_alias_sampled_integer_derivations() -> None:
    base = key(42)
    reachable = {child.seed for child in split(base, 16)}
    reachable |= {fold_in(base, index).seed for index in range(512)}
    reachable |= {fold_in(base, -index).seed for index in range(1, 64)}

    for _, tag in _ENGINE_STREAM_ROOTS:
        assert fold_in(base, tag).seed not in reachable, (
            f"stream root {tag!r} aliases a sampled integer derivation"
        )


def test_engine_stream_roots_are_mutually_distinct() -> None:
    seeds = [fold_in(key(7), tag).seed for _, tag in _ENGINE_STREAM_ROOTS]
    assert len(set(seeds)) == len(_ENGINE_STREAM_ROOTS), _ENGINE_STREAM_ROOTS
