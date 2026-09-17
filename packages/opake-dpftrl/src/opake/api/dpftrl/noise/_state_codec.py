"""Serialization for matrix-factorization noise state."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

from opake.exceptions import CheckpointError
from opake.serialization import (
    from_state_dict,
    register_serializer,
    resolve_serializer,
    state_dict,
)
from opake.types import PerGroup

from ._engine import MFNoiseState

if TYPE_CHECKING:
    from opake.random.types import RngKey

_LAYOUT_VERSION = 1
_WIRE_ENTRY_SIZE = 2
_REQUIRED_FIELDS = frozenset(
    {
        "layout_version",
        "_step_counter",
        "_rng_key.seed",
        "_rng_key.impl",
        "_first_max_norm",
        "_first_max_norm_sync_fingerprint",
        "_inner_state_fields",
    }
)


@dataclasses.dataclass(frozen=True)
class _Payload:
    _inner_state: Any
    _step_counter: int
    _rng_key: RngKey
    _first_max_norm_sync_fingerprint: int | None


def _require_fields(
    saved: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(saved)
    if actual != expected:
        raise CheckpointError(
            *(
                f"{label} fields do not match the current layout: "
                f"missing={sorted(expected - actual, key=repr)}, "
                f"unexpected={sorted(actual - expected, key=repr)}.",
            )
        )


def _wire_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(key)
        value[key] = item
    return value


def _decode_json(value: Any, *, label: str) -> Any:
    if not isinstance(value, str):
        raise CheckpointError(*(f"{label} must be encoded as text.",))
    try:
        return json.loads(value, object_pairs_hook=_wire_object)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(*(f"{label} is not valid JSON.",)) from exc


def _encode_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_group_value(value: Any, *, label: str) -> int | float:
    if type(value) not in (int, float) or math.isnan(float(value)) or value < 0:
        raise CheckpointError(
            *(f"{label} must be a non-negative, non-NaN int or float.",)
        )
    return value


def _encode_max_norm(value: float | PerGroup | None) -> str:
    if value is None:
        return _encode_json({"kind": "none"})
    if not isinstance(value, PerGroup):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CheckpointError(
                *(f"MF max-norm latch has unsupported type {type(value).__name__}.",)
            )
        scalar = float(value)
        if scalar < 0:
            raise CheckpointError(*("MF scalar max-norm latch must be non-negative.",))
        return _encode_json({"kind": "scalar", "value": scalar})

    groups: list[list[Any]] = []
    used_groups: set[str] = set()
    for path, group in value.groups.items():
        if not isinstance(group, str):
            raise CheckpointError(*("MF per-group latch has an invalid group name.",))
        if any(
            isinstance(part, bool) or not isinstance(part, (str, int)) for part in path
        ):
            raise CheckpointError(*("MF per-group latch has an invalid path.",))
        groups.append([list(path), group])
        used_groups.add(group)

    values: list[list[Any]] = []
    for group, group_value in value.values.items():
        if not isinstance(group, str):
            raise CheckpointError(*("MF per-group latch has an invalid value name.",))
        values.append(
            [
                group,
                _require_group_value(
                    group_value,
                    label=f"MF per-group latch value for {group!r}",
                ),
            ]
        )
    missing_values = sorted(used_groups - set(value.values))
    if missing_values:
        raise CheckpointError(
            *(f"MF per-group latch has no value for groups {missing_values}.",)
        )
    return _encode_json({"kind": "per_group", "groups": groups, "values": values})


def _decode_per_group(saved: Mapping[str, Any]) -> PerGroup:
    _require_fields(
        saved,
        {"kind", "groups", "values"},
        label="MF per-group max-norm latch",
    )
    groups_wire = saved["groups"]
    values_wire = saved["values"]
    if not isinstance(groups_wire, list) or not isinstance(values_wire, list):
        raise CheckpointError(
            *("MF per-group max-norm latch has invalid groups or values.",)
        )

    groups: dict[tuple[str | int, ...], str] = {}
    used_groups: set[str] = set()
    for entry in groups_wire:
        if not isinstance(entry, list) or len(entry) != _WIRE_ENTRY_SIZE:
            raise CheckpointError(*("MF per-group latch has an invalid group entry.",))
        path, group = entry
        if (
            not isinstance(path, list)
            or not isinstance(group, str)
            or any(
                isinstance(part, bool) or not isinstance(part, (str, int))
                for part in path
            )
        ):
            raise CheckpointError(*("MF per-group latch has an invalid assignment.",))
        normalized_path = tuple(path)
        if normalized_path in groups:
            raise CheckpointError(
                *(f"MF per-group latch repeats path {normalized_path!r}.",)
            )
        groups[normalized_path] = group
        used_groups.add(group)

    values: dict[str, int | float] = {}
    for entry in values_wire:
        if not isinstance(entry, list) or len(entry) != _WIRE_ENTRY_SIZE:
            raise CheckpointError(*("MF per-group latch has an invalid value entry.",))
        group, value = entry
        if not isinstance(group, str):
            raise CheckpointError(*("MF per-group latch has an invalid value name.",))
        if group in values:
            raise CheckpointError(*(f"MF per-group latch repeats value {group!r}.",))
        values[group] = _require_group_value(
            value,
            label=f"MF per-group latch value for {group!r}",
        )
    missing_values = sorted(used_groups - set(values))
    if missing_values:
        raise CheckpointError(
            *(f"MF per-group latch has no value for groups {missing_values}.",)
        )
    return PerGroup(groups=groups, values=values)


def _decode_max_norm(value: Any) -> float | PerGroup | None:
    saved = _decode_json(value, label="MF max-norm latch")
    if not isinstance(saved, Mapping):
        raise CheckpointError(*("MF max-norm latch must encode an object.",))
    kind = saved.get("kind")
    if kind == "none":
        _require_fields(saved, {"kind"}, label="MF max-norm latch")
        return None
    if kind == "scalar":
        _require_fields(saved, {"kind", "value"}, label="MF max-norm latch")
        return float(
            _require_group_value(saved["value"], label="MF scalar max-norm latch")
        )
    if kind == "per_group":
        return _decode_per_group(saved)
    raise CheckpointError(*(f"Unknown MF max-norm latch kind {kind!r}.",))


def _validate_fingerprint(
    max_norm: float | PerGroup | None,
    fingerprint: int | None,
) -> None:
    if max_norm is None:
        if fingerprint is not None:
            raise CheckpointError(
                *("MF max-norm latch is empty but its fingerprint is present.",)
            )
        return
    if type(fingerprint) is not int:
        raise CheckpointError(*("MF max-norm latch requires an integer fingerprint.",))

    from ._distributed import (
        fingerprint_per_group_max_norm,
        fingerprint_scalar_max_norm,
    )

    expected = (
        fingerprint_per_group_max_norm(max_norm)
        if isinstance(max_norm, PerGroup)
        else fingerprint_scalar_max_norm(max_norm)
    )
    if fingerprint != expected:
        raise CheckpointError(
            *("MF max-norm latch fingerprint does not match its saved value.",)
        )


def _is_inner_field(field: object) -> bool:
    return isinstance(field, str) and (
        field == "_inner_state" or field.startswith(("_inner_state.", "_inner_state["))
    )


def _structure_only(value: Any) -> Any:
    """Preserve serializer paths without cloning model-sized tensor leaves."""
    import torch

    if isinstance(value, torch.Tensor):
        return 0
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _structure_only(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_structure_only(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_structure_only(item) for item in value)
    if isinstance(value, list):
        return [_structure_only(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _structure_only(item) for key, item in value.items()}
    return value


def _template_inner_fields(template: MFNoiseState) -> set[str]:
    payload = state_dict(
        _Payload(
            _inner_state=_structure_only(template._inner_state),
            _step_counter=0,
            _rng_key=template._rng_key,
            _first_max_norm_sync_fingerprint=None,
        )
    )
    return {field for field in payload if _is_inner_field(field)}


def _inner_state_dict(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Return MF inner-state fields relative to the inner-state root."""
    relative: dict[str, Any] = {}
    for field, value in saved.items():
        if field == "_inner_state":
            relative[""] = value
        elif field.startswith("_inner_state."):
            relative[field.removeprefix("_inner_state.")] = value
        elif field.startswith("_inner_state["):
            relative[field.removeprefix("_inner_state")] = value
    return relative


def _validate_inner_manifest(
    saved: Mapping[str, Any],
    template: MFNoiseState,
) -> None:
    declared = _decode_json(
        saved["_inner_state_fields"],
        label="MF inner-state field manifest",
    )
    if (
        not isinstance(declared, list)
        or any(not _is_inner_field(field) for field in declared)
        or declared != sorted(set(declared))
    ):
        raise CheckpointError(*("MF inner-state field manifest is invalid.",))

    actual = {field for field in saved if _is_inner_field(field)}
    declared_fields = set(declared)
    if actual != declared_fields:
        raise CheckpointError(
            *(
                "MF inner-state fields do not match their manifest: "
                f"missing={sorted(declared_fields - actual)}, "
                f"unexpected={sorted(actual - declared_fields)}.",
            )
        )

    configured = _template_inner_fields(template)
    if declared_fields != configured:
        # A registered inner state can provide a more actionable migration or
        # compatibility error than the outer structural mismatch.  In
        # particular, bounded BISR deliberately diagnoses legacy dense-history
        # checkpoints.  Ask that codec to validate its own slice, but retain
        # the fail-closed outer error if it accepts the slice.
        if resolve_serializer(type(template._inner_state)) is not None:
            from_state_dict(template._inner_state, _inner_state_dict(saved))
        raise CheckpointError(
            *(
                "MF inner-state fields do not match the configured runtime: "
                f"missing={sorted(configured - declared_fields)}, "
                f"unexpected={sorted(declared_fields - configured)}. Rebuild "
                "the mechanism with the checkpoint's strategy and parameter-tree "
                "structure.",
            )
        )


def _save(value: MFNoiseState) -> dict[str, Any]:
    max_norm = _encode_max_norm(value._first_max_norm)
    fingerprint = value._first_max_norm_sync_fingerprint
    _validate_fingerprint(value._first_max_norm, fingerprint)
    payload = state_dict(
        _Payload(
            _inner_state=value._inner_state,
            _step_counter=value._step_counter,
            _rng_key=value._rng_key,
            _first_max_norm_sync_fingerprint=fingerprint,
        )
    )
    inner_fields = sorted(field for field in payload if _is_inner_field(field))
    return {
        "layout_version": _LAYOUT_VERSION,
        **payload,
        "_first_max_norm": max_norm,
        "_inner_state_fields": _encode_json(inner_fields),
    }


def _load(template: MFNoiseState, saved: Mapping[str, Any]) -> MFNoiseState:
    version = saved.get("layout_version")
    if version is None:
        raise CheckpointError(
            *("Cannot restore a legacy unversioned MF noise-state checkpoint.",)
        )
    if type(version) is not int or version != _LAYOUT_VERSION:
        raise CheckpointError(
            *(
                f"Unsupported MF noise-state version {version!r}; "
                f"expected {_LAYOUT_VERSION}.",
            )
        )

    actual = set(saved)
    missing = sorted(_REQUIRED_FIELDS - actual)
    unexpected = sorted(
        (field for field in actual - _REQUIRED_FIELDS if not _is_inner_field(field)),
        key=repr,
    )
    if missing or unexpected:
        raise CheckpointError(
            *(
                "MF noise-state fields do not match the current layout: "
                f"missing={missing}, unexpected={unexpected}.",
            )
        )

    _validate_inner_manifest(saved, template)
    payload = from_state_dict(
        _Payload(
            _inner_state=template._inner_state,
            _step_counter=template._step_counter,
            _rng_key=template._rng_key,
            _first_max_norm_sync_fingerprint=(
                template._first_max_norm_sync_fingerprint
            ),
        ),
        {
            field: value
            for field, value in saved.items()
            if field not in {"layout_version", "_first_max_norm", "_inner_state_fields"}
        },
    )
    if type(payload._step_counter) is not int or payload._step_counter < 0:
        raise CheckpointError(
            *(
                "MF noise-state step must be a non-negative int, "
                f"got {payload._step_counter!r}.",
            )
        )
    if type(payload._rng_key.seed) is not int or not isinstance(
        payload._rng_key.impl, str
    ):
        raise CheckpointError(
            *(f"MF noise-state RNG key is invalid: {payload._rng_key!r}.",)
        )
    if (
        payload._first_max_norm_sync_fingerprint is not None
        and type(payload._first_max_norm_sync_fingerprint) is not int
    ):
        raise CheckpointError(
            *("MF max-norm latch fingerprint must be an int or None.",)
        )

    max_norm = _decode_max_norm(saved["_first_max_norm"])
    fingerprint = payload._first_max_norm_sync_fingerprint
    _validate_fingerprint(max_norm, fingerprint)
    return MFNoiseState(
        _inner_state=payload._inner_state,
        _step_counter=payload._step_counter,
        _rng_key=payload._rng_key,
        _first_max_norm=max_norm,
        _first_max_norm_sync_fingerprint=fingerprint,
    )


register_serializer(MFNoiseState, _save, _load)


__all__: list[str] = []
