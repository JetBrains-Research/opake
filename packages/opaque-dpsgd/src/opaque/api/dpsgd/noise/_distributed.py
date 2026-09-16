"""Distributed-rank state validation for DP-SGD noise.

Registers a :class:`opaque.dpsgd.noise._gaussian.GaussianNoiseState`
and :class:`opaque.dpsgd.noise._jme.JmeNoiseState` sync handler with
:func:`opaque.distributed.sync` at import time.
Imported for its side effects from :mod:`opaque.dpsgd.noise`; not
re-exported.
"""

from __future__ import annotations

from opaque.api.dpsgd.noise._gaussian import GaussianNoiseState
from opaque.api.dpsgd.noise._jme import JmeNoiseState
from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    assert_string_equal,
    register_sync_type,
    sync_object,
)
from opaque.distributed import is_distributed

_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
    "_rng_key": "local",
}

_JME_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
    "_rng_key": "local",
    "_aggregate_norm": "assert_equal",
    "_allocation": "local",
    "_noise_multiplier": "assert_equal",
}


def _assert_rng_key_equal(
    state: GaussianNoiseState | JmeNoiseState,
    state_name: str,
) -> None:
    """Assert that the RNG key seed matches across ranks.

    Seeds are canonicalized to unsigned 64-bit, so roughly half of them fall
    outside the signed ``int64`` domain the scalar reductions use — every
    ``fold_in``-derived key has even odds of setting the top bit. They are
    compared as text instead: a seed is opaque identity material rather than a
    magnitude, and ``assert_string_equal`` costs one ``all_gather_object``
    against the two reductions ``assert_scalar_equal`` issues.
    """
    assert_string_equal(str(state._rng_key.seed), name=f"{state_name}.seed")


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate Gaussian noise state consistency across ranks.

    Asserts that all ranks share the same seed and step counter.  No-op
    outside ``torch.distributed``.
    """
    if not is_distributed():
        return state
    _assert_rng_key_equal(state, "GaussianNoiseState")
    return sync_object(state, field_ops=_NOISE_STATE_FIELD_OPS)


def sync_jme_noise_state(state: JmeNoiseState) -> JmeNoiseState:
    """Validate both JME streams and their public configuration."""
    if not is_distributed():
        return state
    assert_scalar_equal(state._aggregate_norm, name="JmeNoiseState.aggregate_norm")
    assert_scalar_equal(
        state._noise_multiplier,
        name="JmeNoiseState.noise_multiplier",
    )
    assert_string_equal(
        repr(state._allocation),
        name="JmeNoiseState.allocation",
    )
    _assert_rng_key_equal(state, "JmeNoiseState")
    return sync_object(state, field_ops=_JME_NOISE_STATE_FIELD_OPS)


register_sync_type(GaussianNoiseState, sync_gaussian_noise_state)
register_sync_type(JmeNoiseState, sync_jme_noise_state)


__all__ = ["sync_gaussian_noise_state", "sync_jme_noise_state"]
