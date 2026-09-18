"""Shared manifest engine for the HF / TRL → opake converters.

Every converter (HF in :mod:`._hf_convert`, TRL in
:mod:`opake.api.transformers.trl`) drives :func:`_apply_manifest` with its own
DIRECT/RENAME/TRANSFORM/REJECT/DROP buckets. This module holds that dispatcher
plus the DP-knob normalization and the dataclass-introspection helpers they all
share.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

from opake.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _normalize_dp_overrides(
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a dict of opake-side DP fields from kwargs.

    At least one of ``privacy_noise_multiplier`` or ``privacy_target_epsilon``
    must be set — opake's runtime would reject the config otherwise, and
    surfacing the error here gives a clearer message tied to the converter
    call site.
    """
    noise_mult = overrides.get("privacy_noise_multiplier")
    target_eps = overrides.get("privacy_target_epsilon")
    if noise_mult is None and target_eps is None:
        raise ConfigurationError(
            *(
                "Converting to an opake config requires a DP knob: pass either "
                "``privacy_noise_multiplier=<float>`` (fixed-noise mode) or "
                "``privacy_target_epsilon=<float>`` (calibrated-noise mode) as a "
                "keyword argument to the converter.",
            )
        )
    # Pass through every override verbatim — opake's TrainingArguments
    # __post_init__ does the cross-field validation.
    return dict(overrides)


def _is_default(value: Any, default: Any) -> bool:
    """``True`` if ``value`` matches ``default`` for a dataclass field.

    Handles the awkward cases the HF surface raises: dataclass instances
    (like ``AcceleratorConfig``) whose ``__eq__`` may not be defined, and
    pairs where ``default_factory`` constructs a fresh instance that
    isn't ``==`` to the value the user got.
    """
    if isinstance(default, dataclasses.Field):
        if default.default is not dataclasses.MISSING:
            default_value = default.default
        elif default.default_factory is not dataclasses.MISSING:
            default_value = default.default_factory()
        else:
            return False
    else:
        default_value = default

    # Fast path: identity match.
    if value is default_value:
        return True

    # Standard equality.
    try:
        if value == default_value:
            return True
    except Exception:  # pragma: no cover — pathological __eq__
        pass

    # Both dataclass instances? Compare via field dicts (handles
    # ``AcceleratorConfig`` and other HF nested dataclasses).
    if (
        not isinstance(value, type)
        and dataclasses.is_dataclass(value)
        and not isinstance(default_value, type)
        and dataclasses.is_dataclass(default_value)
    ):
        try:
            return _get_dataclass_field_values(value) == _get_dataclass_field_values(
                default_value
            )
        except Exception:  # pragma: no cover
            pass

    return False


def _get_dataclass_field_values(
    obj: Any,
) -> dict[str, Any]:
    """Return a name→value dict for every field of a dataclass instance."""
    if not dataclasses.is_dataclass(obj):
        raise InputTypeError(
            *(
                f"Expected a dataclass instance, got {type(obj).__name__}. "
                "The HF/TRL converters accept dataclass instances only.",
            )
        )
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _apply_manifest(
    *,
    source_values: Mapping[str, Any],
    source_defaults: Mapping[str, Any],
    direct: frozenset[str],
    rename: Mapping[str, str],
    transform: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
    reject: Mapping[str, Callable[[Any], str | None]],
    drop: Mapping[str, str | Callable[[Any], str | None]],
    source_label: str,
    strict: bool,
) -> dict[str, Any]:
    """Apply the bucketed manifest to ``source_values`` and return an opake dict.

    ``reject`` maps field name → callable that returns an error message
    string if the value is unsupported, or ``None`` if the value is
    benign (e.g. ``packing=False`` is fine, only ``packing=True`` is
    rejected). The callable receives the raw source value.

    ``drop`` maps field name → the reason surfaced in the ``RuntimeWarning``
    when the field is non-default: either a fixed string, or a callable
    receiving the raw value and returning ``None`` to stay silent (e.g. a
    coefficient set to 0 asks for nothing opake withholds).

    ``transform`` maps field name → callable that receives the full
    ``source_values`` dict and returns a partial opake-side dict to
    merge in. Used for multi-field transforms like the batch collapse.
    """
    opake: dict[str, Any] = {}
    errors: list[str] = []

    for name, value in source_values.items():
        # Layer 1: REJECT_IF_SET — non-default values run the rejector; a default
        # value is a silent drop, and a rejector returning None means the value
        # is benign for this rule (fall through to TRANSFORM / DIRECT).
        if name in reject:
            if _is_default(value, source_defaults.get(name)):
                continue  # default value → silent drop
            message = reject[name](value)
            if message is not None:
                errors.append(f"  - {source_label}.{name}={value!r}: {message}")
                continue
            # message is None → benign; fall through to other buckets.

        # Layer 2: DROP_WITH_WARN — drop and (optionally) warn.
        if name in drop:
            if not _is_default(value, source_defaults.get(name)):
                rule = drop[name]
                reason = rule(value) if callable(rule) else rule
                if reason is not None:
                    _warn_drop(source_label, name, value, reason, strict)
            continue

        # Layer 3: TRANSFORM — multi-field derivation, run once below.
        if name in transform:
            continue

        # Layer 4: RENAME — name swap, value preserved.
        if name in rename:
            opake[rename[name]] = value
            continue

        # Layer 5: DIRECT — copy as-is.
        if name in direct:
            opake[name] = value
            continue

        # Unclassified but untouched: no intent to honor, and nothing that
        # can move the accounting.
        if name in source_defaults and _is_default(value, source_defaults[name]):
            continue

        # Unclassified and user-set: never drop a knob someone configured.
        errors.append(
            f"  - {source_label}.{name}={value!r}: field is not classified by "
            f"the opake argument manifest (neither DIRECT, RENAME, TRANSFORM, "
            f"REJECT, nor DROP). This usually means upstream HF/TRL added a "
            f"new field on a version the opake manifest hasn't been updated "
            f"for. File a bug or pin a supported version."
        )

    if errors:
        raise ConfigurationError(
            *(f"Converting {source_label} to opake failed:\n" + "\n".join(errors),)
        )

    # Run all transforms — they see the full source dict.
    for name, transform_fn in transform.items():
        if name in source_values:
            opake.update(transform_fn(dict(source_values)))

    return opake


def _warn_drop(
    source_label: str, name: str, value: Any, reason: str, strict: bool
) -> None:
    """Emit a Python warning for a dropped non-default field."""
    msg = (
        f"opake: dropping {source_label}.{name}={value!r} — {reason}. "
        f"This field has no opake equivalent and is being discarded."
    )
    if strict:
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
    # When ``strict=False``, drop silently.


def _reject_if_truthy(message: str) -> Callable[[Any], str | None]:
    """Build a rejector that fires only when the user set a truthy value."""

    def inner(value: Any) -> str | None:
        if value:
            return message
        return None

    return inner
