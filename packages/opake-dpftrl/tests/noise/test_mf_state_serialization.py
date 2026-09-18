"""Checkpoint layout tests for matrix-factorization noise state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from opake.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opake.exceptions import CheckpointError, ConfigurationError
from opake.random import key
from opake.serialization import from_state_dict, state_dict
from opake.types import PerGroup, clipped


def _template() -> dict[str, torch.Tensor]:
    return {"fallback": torch.zeros(3), "head": torch.zeros(2)}


def _bound(fallback: float, head: float) -> PerGroup:
    return PerGroup(
        groups={"fallback": "fallback", "head": "head"},
        values={"fallback": fallback, "head": head},
    )


def _make_noise(seed: int):
    return mf_gaussian_noise(
        _template(),
        band_mf_strategy(bands=3, momentum=0.9),
        n_steps=6,
        min_sep=6,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def _saved_latched_state():
    noise_fn, state = _make_noise(42)
    bound = _bound(1.0, 2.0)
    for _ in range(3):
        _, state = noise_fn(clipped(_template(), max_norm=bound), state)
    return noise_fn, state, state_dict(state)


def _latch(saved: dict[str, Any]) -> dict[str, Any]:
    return json.loads(saved["_first_max_norm"])


def _replace_latch(saved: dict[str, Any], latch: dict[str, Any]) -> None:
    saved["_first_max_norm"] = json.dumps(
        latch,
        sort_keys=True,
        separators=(",", ":"),
    )


def _inner_fields(saved: Mapping[str, Any]) -> list[str]:
    return sorted(
        field
        for field in saved
        if field == "_inner_state"
        or field.startswith(("_inner_state.", "_inner_state["))
    )


def _replace_inner_manifest(saved: dict[str, Any]) -> None:
    saved["_inner_state_fields"] = json.dumps(
        _inner_fields(saved),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_unlatched_state_round_trips() -> None:
    _, state = _make_noise(42)
    _, fresh = _make_noise(999)
    saved = state_dict(state)

    restored = from_state_dict(fresh, saved)

    assert _latch(saved)["kind"] == "none"
    assert restored._first_max_norm is None
    assert restored._first_max_norm_sync_fingerprint is None
    assert restored._rng_key == state._rng_key


def test_scalar_latch_round_trips() -> None:
    noise_fn, state = _make_noise(42)
    _, state = noise_fn(clipped(_template(), max_norm=1.25), state)
    _, fresh = _make_noise(999)
    saved = state_dict(state)

    restored = from_state_dict(fresh, saved)

    assert _latch(saved) == {"kind": "scalar", "value": 1.25}
    assert restored._first_max_norm == 1.25
    assert (
        restored._first_max_norm_sync_fingerprint
        == state._first_max_norm_sync_fingerprint
    )


def test_infinite_non_private_latch_round_trips() -> None:
    noise_fn, state = _make_noise(42)
    _, state = noise_fn(clipped(_template(), max_norm=math.inf), state)
    _, fresh = _make_noise(999)

    restored = from_state_dict(fresh, state_dict(state))

    assert restored._first_max_norm == math.inf
    assert (
        restored._first_max_norm_sync_fingerprint
        == state._first_max_norm_sync_fingerprint
    )


def test_per_group_latch_restores_into_fresh_state() -> None:
    noise_fn, expected, saved = _saved_latched_state()
    _, fresh = _make_noise(999)

    restored = from_state_dict(fresh, saved)

    assert saved["layout_version"] == 1
    assert _latch(saved)["kind"] == "per_group"
    assert not any(
        isinstance(value, (Mapping, list, tuple)) for value in saved.values()
    )
    assert restored._first_max_norm == expected._first_max_norm
    assert (
        restored._first_max_norm_sync_fingerprint
        == expected._first_max_norm_sync_fingerprint
    )
    with pytest.raises(ConfigurationError, match=r"varying ClippedPytree\.max_norm"):
        noise_fn(clipped(_template(), max_norm=_bound(1.5, 2.0)), restored)


def test_per_group_latch_preserves_numeric_representation() -> None:
    noise_fn, state = _make_noise(42)
    bound = PerGroup(
        groups={"fallback": "fallback", "head": "head"},
        values={"fallback": 1, "head": 2},
    )
    _, state = noise_fn(clipped(_template(), max_norm=bound), state)
    _, fresh = _make_noise(999)

    restored = from_state_dict(fresh, state_dict(state))

    assert restored._first_max_norm == bound
    assert type(restored._first_max_norm.values["fallback"]) is int


def test_rejects_inconsistent_latch_fingerprint() -> None:
    _, _, saved = _saved_latched_state()
    saved["_first_max_norm_sync_fingerprint"] += 1
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="fingerprint does not match"):
        from_state_dict(fresh, saved)


def test_rejects_fingerprint_for_empty_latch() -> None:
    _, state = _make_noise(42)
    saved = state_dict(state)
    saved["_first_max_norm_sync_fingerprint"] = 1
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="empty but its fingerprint is present"):
        from_state_dict(fresh, saved)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("_rng_key.impl", "fields do not match"),
        ("_first_max_norm", "fields do not match"),
        ("_first_max_norm_sync_fingerprint", "fields do not match"),
        ("_inner_state_fields", "fields do not match"),
    ],
)
def test_rejects_incomplete_layout(field: str, match: str) -> None:
    _, _, saved = _saved_latched_state()
    del saved[field]
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match=match):
        from_state_dict(fresh, saved)


def test_rejects_legacy_unversioned_layout() -> None:
    _, _, saved = _saved_latched_state()
    del saved["layout_version"]
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="legacy unversioned"):
        from_state_dict(fresh, saved)


def test_rejects_unknown_latch_kind() -> None:
    _, _, saved = _saved_latched_state()
    latch = _latch(saved)
    latch["kind"] = "unknown"
    _replace_latch(saved, latch)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="Unknown MF max-norm latch kind"):
        from_state_dict(fresh, saved)


def test_rejects_missing_latch_discriminator() -> None:
    _, _, saved = _saved_latched_state()
    latch = _latch(saved)
    del latch["kind"]
    _replace_latch(saved, latch)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="Unknown MF max-norm latch kind"):
        from_state_dict(fresh, saved)


def test_rejects_per_group_assignment_without_value() -> None:
    _, _, saved = _saved_latched_state()
    latch = _latch(saved)
    latch["values"].pop()
    _replace_latch(saved, latch)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="has no value for groups"):
        from_state_dict(fresh, saved)


def test_rejects_negative_per_group_value() -> None:
    _, _, saved = _saved_latched_state()
    latch = _latch(saved)
    latch["values"][0][1] = -1.0
    _replace_latch(saved, latch)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="must be a non-negative"):
        from_state_dict(fresh, saved)


def test_rejects_nan_per_group_value() -> None:
    _, _, saved = _saved_latched_state()
    latch = _latch(saved)
    latch["values"][0][1] = math.nan
    _replace_latch(saved, latch)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="non-negative, non-NaN"):
        from_state_dict(fresh, saved)


def test_rejects_missing_inner_state_field() -> None:
    _, _, saved = _saved_latched_state()
    inner_field = next(
        field
        for field in saved
        if field == "_inner_state"
        or field.startswith(("_inner_state.", "_inner_state["))
    )
    del saved[inner_field]
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="do not match their manifest"):
        from_state_dict(fresh, saved)


def test_rejects_unexpected_inner_state_field() -> None:
    _, _, saved = _saved_latched_state()
    saved["_inner_state.unexpected"] = 0
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="do not match their manifest"):
        from_state_dict(fresh, saved)


def test_rejects_missing_inner_field_with_rewritten_manifest() -> None:
    _, _, saved = _saved_latched_state()
    del saved[_inner_fields(saved)[0]]
    _replace_inner_manifest(saved)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="configured runtime"):
        from_state_dict(fresh, saved)


def test_rejects_unexpected_inner_field_with_rewritten_manifest() -> None:
    _, _, saved = _saved_latched_state()
    saved["_inner_state.unexpected"] = 0
    _replace_inner_manifest(saved)
    _, fresh = _make_noise(999)

    with pytest.raises(CheckpointError, match="configured runtime"):
        from_state_dict(fresh, saved)
