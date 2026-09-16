"""Regression tests for Opaque's registered RNG stream roots."""

from opaque.api.auditing._coin_flip import (
    _CANARY_SELECTION_DOMAIN,
    _COIN_FLIP_DOMAIN,
)
from opaque.api.dpftrl.noise._engine import MF_GAUSSIAN_STREAM_FOLD
from opaque.api.dpftrl.sampling._b_min_sep import B_MIN_SEP_STREAM_FOLD
from opaque.api.dpftrl.sampling._balls_in_bins import BALLS_IN_BINS_STREAM_FOLD
from opaque.api.dpftrl.sampling._poisson import CYCLIC_POISSON_STREAM_FOLD
from opaque.api.dpsgd.clipping._adaptive import ADAPTIVE_CLIPPING_STREAM_FOLD
from opaque.api.dpsgd.noise._gaussian import GAUSSIAN_STREAM_FOLD
from opaque.api.dpsgd.noise._jme import (
    JME_FIRST_STREAM_FOLD,
    JME_SECOND_STREAM_FOLD,
)
from opaque.api.dpsgd.sampling._k_out_of_t import K_OUT_OF_T_STREAM_FOLD
from opaque.api.dpsgd.sampling._poisson import POISSON_STREAM_FOLD
from opaque.api.transformers._rng import IGNORE_DATA_SKIP_STREAM_FOLD
from opaque.random import fold_in, key, split

_STREAM_ROOTS = (
    ("opaque.dpftrl.mf_gaussian", MF_GAUSSIAN_STREAM_FOLD),
    ("opaque.dpftrl.b_min_sep", B_MIN_SEP_STREAM_FOLD),
    ("opaque.dpftrl.balls_in_bins", BALLS_IN_BINS_STREAM_FOLD),
    ("opaque.dpftrl.cyclic_poisson", CYCLIC_POISSON_STREAM_FOLD),
    ("opaque.dpsgd.gaussian", GAUSSIAN_STREAM_FOLD),
    ("opaque.dpsgd.jme.first", JME_FIRST_STREAM_FOLD),
    ("opaque.dpsgd.jme.second", JME_SECOND_STREAM_FOLD),
    ("opaque.dpsgd.adaptive_clipping", ADAPTIVE_CLIPPING_STREAM_FOLD),
    ("opaque.dpsgd.poisson", POISSON_STREAM_FOLD),
    ("opaque.dpsgd.k_out_of_t", K_OUT_OF_T_STREAM_FOLD),
    ("opaque.auditing.canary_selection", _CANARY_SELECTION_DOMAIN),
    ("opaque.auditing.coin_flip", _COIN_FLIP_DOMAIN),
    ("opaque.transformers.ignore_data_skip", IGNORE_DATA_SKIP_STREAM_FOLD),
)


def test_stream_roots_match_registered_tags() -> None:
    for expected, actual in _STREAM_ROOTS:
        assert actual == expected, f"expected {expected!r}, got {actual!r}"
        assert actual.startswith("opaque."), f"unnamespaced stream root: {actual!r}"


def test_stream_roots_do_not_alias_sampled_integer_derivations() -> None:
    base = key(42)
    reachable = {child.seed for child in split(base, 16)}
    reachable |= {fold_in(base, index).seed for index in range(512)}
    reachable |= {fold_in(base, -index).seed for index in range(1, 64)}

    for _, tag in _STREAM_ROOTS:
        assert fold_in(base, tag).seed not in reachable, (
            f"stream root {tag!r} aliases a sampled integer derivation"
        )


def test_stream_roots_are_mutually_distinct() -> None:
    tags = [actual for _, actual in _STREAM_ROOTS]
    seeds = [fold_in(key(7), tag).seed for tag in tags]
    assert len(set(tags)) == len(_STREAM_ROOTS), tags
    assert len(set(seeds)) == len(_STREAM_ROOTS), tags
