"""Distributed-rank state validation for matrix-factorization noise.

Registers :class:`opaque.dpftrl.noise._engine.MFNoiseState` sync handling with
:func:`opaque.distributed.sync` at import time. Imported for its side effects
from :mod:`opaque.dpftrl.noise`; not re-exported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from opaque.api.dpftrl.noise._engine import MFNoiseState
from opaque.api.engine.distributed._state import (
    assert_string_equal,
    register_sync_type,
    sync_object,
)
from opaque.distributed import is_distributed
from opaque.types import PerGroup

_INT64_MAX = 2**63 - 1
_UINT64_MODULUS = 2**64


def _normalize_int64_fingerprint(value: int | None) -> int | None:
    """Convert legacy unsigned fingerprints to the signed int64 wire format."""
    if value is None or value <= _INT64_MAX:
        return value
    return value - _UINT64_MODULUS


def fingerprint_scalar_max_norm(c: float) -> int:
    """Deterministic signed 64-bit fingerprint for a scalar latched ``max_norm``."""
    payload = {"kind": "mf_noise_scalar_max_norm", "c": float(c)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def fingerprint_per_group_max_norm(pg: PerGroup) -> int:
    """Deterministic signed 64-bit fingerprint for cross-rank equality of
    ``PerGroup``.

    Called once when ``mf_gaussian_noise`` latches a ``PerGroup`` ``max_norm``; the
    result is stored on :class:`MFNoiseState` and equality is checked on each
    :func:`sync` (avoids re-hashing the full param→group map on the hot path).

    Scalar and per-group norms use disjoint ``kind`` payloads so a scalar latch
    on one rank cannot collide with a per-group latch fingerprint on another.

    Returns the leading signed 64 bits of the SHA-256 digest as an int (never
    coerced through float) so that cross-rank ``PerGroup`` mismatches can't
    slip past the equality check by colliding under float-precision loss. The
    signed representation is compatible with the ``torch.int64`` reductions
    used for distributed equality checks.
    """
    payload = {
        "kind": "mf_noise_per_group_max_norm",
        "groups": sorted(pg.groups.items()),
        "values": sorted(pg.values.items()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def mf_per_group_sync_fingerprint_for_latch(
    prior: MFNoiseState,
    max_norm: float | PerGroup,
) -> int | None:
    """Integer fingerprint for distributed sync; set on every latched ``max_norm``."""
    if prior._first_max_norm is None:
        if isinstance(max_norm, PerGroup):
            return fingerprint_per_group_max_norm(max_norm)
        return fingerprint_scalar_max_norm(float(max_norm))
    return _normalize_int64_fingerprint(prior._first_max_norm_sync_fingerprint)


_MF_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_inner_state": "local",
    "_step_counter": "assert_equal",
    "_rng_key": "local",
    "_first_max_norm_sync_fingerprint": "assert_optional_equal",
    "_first_max_norm": "local",
}


def _assert_rng_key_equal(state: MFNoiseState, state_name: str) -> None:
    """Assert that the RNG key seed matches across ranks.

    Seeds are canonicalized to unsigned 64-bit, so roughly half of them fall
    outside the signed ``int64`` domain the scalar reductions use — the same
    constraint that ``_normalize_int64_fingerprint`` handles for latched
    sensitivity fingerprints. A seed is opaque identity material rather than a
    magnitude, so it is compared as text.
    """
    assert_string_equal(str(state._rng_key.seed), name=f"{state_name}.seed")


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate MF noise state consistency across ranks.

    Asserts that all ranks share the same seed, step counter, and (once
    latched) first-call sensitivity bound.  No-op outside
    ``torch.distributed``.
    """
    if not is_distributed():
        return state
    fingerprint = _normalize_int64_fingerprint(state._first_max_norm_sync_fingerprint)
    if fingerprint != state._first_max_norm_sync_fingerprint:
        state = replace(state, _first_max_norm_sync_fingerprint=fingerprint)
    _assert_rng_key_equal(state, "MFNoiseState")
    return sync_object(state, field_ops=_MF_NOISE_STATE_FIELD_OPS)


register_sync_type(MFNoiseState, sync_mf_noise_state)


__all__ = [
    "fingerprint_per_group_max_norm",
    "fingerprint_scalar_max_norm",
    "mf_per_group_sync_fingerprint_for_latch",
    "sync_mf_noise_state",
]
