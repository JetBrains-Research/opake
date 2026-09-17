"""Shared TRL → opaque conversion plumbing.

TRL ``SFTConfig`` / ``DPOConfig`` subclass HF ``TrainingArguments``, so each
converter delegates the HF-inherited subset of fields to the HF manifest in
:mod:`opaque.api.transformers.trainer._hf_convert` and only classifies
the TRL-specific fields with its own per-flavor manifest (defined alongside the
config class in ``_sft_config.py`` / ``_dpo_config.py``).

This module holds the bits both flavors share: the optional-dependency import
gate (``trl`` is the ``pip install opaque[trl]`` extra), the rejectors common
to SFT and DPO, and the two-layer dispatcher :func:`_convert_trl_config`.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

from ..trainer._convert import (  # noqa: F401  (_reject_if_truthy re-exported)
    _apply_manifest,
    _get_dataclass_field_values,
    _reject_if_truthy,
)
from ..trainer._hf_convert import (
    HF_DIRECT_FIELDS,
    HF_DROP_FIELDS,
    HF_REJECTED_FIELDS,
    HF_RENAME_MAP,
    HF_TRANSFORM_MAP,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _import_trl() -> Any:
    """Import the optional ``trl`` package or raise with a clear hint."""
    try:
        import trl
    except ImportError as e:  # pragma: no cover — gated by extras
        raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
            "Converting from TRL configs requires the optional ``trl`` "
            "dependency. Install with ``pip install opaque[trl]`` (or "
            "``pip install 'trl>=1.0,<2.0'`` if you manage deps yourself)."
        ) from e
    return trl


def _reject_truncation_mode(value: Any) -> str | None:
    if value == "keep_end":
        return (
            "TRL deprecated ``truncation_mode='keep_end'``; opaque only "
            "supports ``keep_start`` (the TRL 1.x default). Drop this field "
            "or set it to ``'keep_start'``."
        )
    return None


def _reject_pad_token(value: Any) -> str | None:
    if value is not None:
        return (
            "TRL deprecated ``pad_token`` in 1.x (removed in 2.0). Set "
            "``tokenizer.pad_token`` directly on the tokenizer before "
            "constructing the trainer instead of carrying it on the config."
        )
    return None


def _convert_trl_config(
    trl_cfg: Any,
    *,
    trl_direct: frozenset[str],
    trl_rename: dict[str, str],
    trl_transform: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    trl_reject: dict[str, Callable[[Any], str | None]],
    trl_drop: dict[str, str | Callable[[Any], str | None]],
    source_label: str,
    strict: bool,
    dp_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Shared TRL → opaque conversion dispatcher.

    Splits the TRL config's fields into the HF-inherited subset (handled
    by the HF manifest) and the TRL-specific subset (handled by the
    per-flavor manifest passed in). Then merges DP overrides on top.
    """
    source_values = _get_dataclass_field_values(trl_cfg)

    # Construct a baseline TRL instance to detect "user-set vs default".
    baseline_output_dir = source_values.get("output_dir") or tempfile.mkdtemp(
        prefix="opaque_trl_baseline_"
    )
    # Pass use_cpu=True if the source config has it, to avoid bf16 validation errors on CPU runners.
    baseline_kwargs = {"output_dir": baseline_output_dir}
    if source_values.get("use_cpu"):
        baseline_kwargs["use_cpu"] = True
    baseline = type(trl_cfg)(**baseline_kwargs)
    source_defaults = _get_dataclass_field_values(baseline)

    # The HF-base field names are the union of the HF manifest's buckets
    # — those are handled by the HF dispatcher with HF's own buckets.
    hf_field_names = (
        HF_DIRECT_FIELDS
        | HF_RENAME_MAP.keys()
        | HF_TRANSFORM_MAP.keys()
        | HF_REJECTED_FIELDS.keys()
        | HF_DROP_FIELDS.keys()
    )

    hf_values = {k: v for k, v in source_values.items() if k in hf_field_names}
    hf_defaults = {k: v for k, v in source_defaults.items() if k in hf_field_names}
    trl_values = {k: v for k, v in source_values.items() if k not in hf_field_names}
    trl_defaults = {k: v for k, v in source_defaults.items() if k not in hf_field_names}

    # Layer 1: HF base translation.
    hf_converted = _apply_manifest(
        source_values=hf_values,
        source_defaults=hf_defaults,
        direct=HF_DIRECT_FIELDS,
        rename=HF_RENAME_MAP,
        transform=HF_TRANSFORM_MAP,
        reject=HF_REJECTED_FIELDS,
        drop=HF_DROP_FIELDS,
        source_label=source_label,
        strict=strict,
    )

    # Layer 2: TRL-specific field translation.
    trl_converted = _apply_manifest(
        source_values=trl_values,
        source_defaults=trl_defaults,
        direct=trl_direct,
        rename=trl_rename,
        transform=trl_transform,
        reject=trl_reject,
        drop=trl_drop,
        source_label=source_label,
        strict=strict,
    )

    # Merge HF + TRL + DP overrides (DP wins at conflicts).
    converted: dict[str, Any] = {}
    converted.update(hf_converted)
    converted.update(trl_converted)
    # Performance kernels default ON in opaque but OFF in HF/TRL; default OFF on
    # conversion to match upstream (Liger in the HF manifest may set True).
    converted.setdefault("use_performance_kernels", False)
    converted.update(dp_overrides)
    return converted
