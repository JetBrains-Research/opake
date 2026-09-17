"""Canary test: every upstream HF / TRL config field is bucketed exactly once.

Drift catcher for the opake argument manifest. A new upstream
``TrainingArguments`` / ``SFTConfig`` / ``DPOConfig`` field that no bucket set
classifies fails the test; the fix is to add it to the right constant in
``opake.api.transformers.trainer._hf_convert`` (or the TRL manifests in
``opake.api.transformers.trl``).
"""

from __future__ import annotations

import dataclasses

import pytest

# Required upstream — HF transformers is a runtime dep of opake-transformers.
hf = pytest.importorskip("transformers")

from opake.api.transformers.trainer._hf_convert import (  # noqa: E402
    HF_DIRECT_FIELDS,
    HF_DROP_FIELDS,
    HF_LEGACY_FIELDS,
    HF_REJECTED_FIELDS,
    HF_RENAME_MAP,
    HF_TRANSFORM_MAP,
)
from opake.api.transformers.trl._dpo_convert import (  # noqa: E402
    TRL_DPO_DIRECT_FIELDS,
    TRL_DPO_DROP_FIELDS,
    TRL_DPO_REJECTED_FIELDS,
    TRL_DPO_RENAME_MAP,
    TRL_DPO_TRANSFORM_MAP,
)
from opake.api.transformers.trl._sft_convert import (  # noqa: E402
    TRL_SFT_DIRECT_FIELDS,
    TRL_SFT_DROP_FIELDS,
    TRL_SFT_REJECTED_FIELDS,
    TRL_SFT_RENAME_MAP,
    TRL_SFT_TRANSFORM_MAP,
)

# TRL is an optional ``opake[trl]`` extra; skip TRL-side checks if absent.
trl = pytest.importorskip("trl")


def _hf_inherited_field_names() -> set[str]:
    """Field names handled by the HF base manifest (inherited by TRL configs)."""
    return (
        set(HF_DIRECT_FIELDS)
        | set(HF_RENAME_MAP.keys())
        | set(HF_TRANSFORM_MAP.keys())
        | set(HF_REJECTED_FIELDS.keys())
        | set(HF_DROP_FIELDS.keys())
    )


def _trl_sft_only_field_names() -> set[str]:
    """TRL SFT field names NOT inherited from HF TrainingArguments."""
    all_trl_sft = {f.name for f in dataclasses.fields(trl.SFTConfig)}
    return all_trl_sft - _hf_inherited_field_names()


def _trl_dpo_only_field_names() -> set[str]:
    all_trl_dpo = {f.name for f in dataclasses.fields(trl.DPOConfig)}
    return all_trl_dpo - _hf_inherited_field_names()


def _opake_buckets_for_trl_sft() -> set[str]:
    return (
        set(TRL_SFT_DIRECT_FIELDS)
        | set(TRL_SFT_RENAME_MAP.keys())
        | set(TRL_SFT_TRANSFORM_MAP.keys())
        | set(TRL_SFT_REJECTED_FIELDS.keys())
        | set(TRL_SFT_DROP_FIELDS.keys())
    )


def _opake_buckets_for_trl_dpo() -> set[str]:
    return (
        set(TRL_DPO_DIRECT_FIELDS)
        | set(TRL_DPO_RENAME_MAP.keys())
        | set(TRL_DPO_TRANSFORM_MAP.keys())
        | set(TRL_DPO_REJECTED_FIELDS.keys())
        | set(TRL_DPO_DROP_FIELDS.keys())
    )


def _all_hf_field_names() -> set[str]:
    """Return every field name on HF's ``TrainingArguments`` dataclass."""
    return {f.name for f in dataclasses.fields(hf.TrainingArguments)}


def _opake_buckets_for_hf() -> set[str]:
    """Union of every HF field name covered by opake's manifest buckets."""
    return (
        set(HF_DIRECT_FIELDS)
        | set(HF_RENAME_MAP.keys())
        | set(HF_TRANSFORM_MAP.keys())
        | set(HF_REJECTED_FIELDS.keys())
        | set(HF_DROP_FIELDS.keys())
    )


def test_hf_manifest_buckets_are_disjoint():
    """No HF field appears in two different buckets — the manifest is a partition."""
    buckets = {
        "DIRECT": set(HF_DIRECT_FIELDS),
        "RENAME": set(HF_RENAME_MAP.keys()),
        "TRANSFORM": set(HF_TRANSFORM_MAP.keys()),
        "REJECT_IF_SET": set(HF_REJECTED_FIELDS.keys()),
        "DROP_WITH_WARN": set(HF_DROP_FIELDS.keys()),
    }
    for name_a, set_a in buckets.items():
        for name_b, set_b in buckets.items():
            if name_a >= name_b:
                continue
            overlap = set_a & set_b
            assert not overlap, (
                f"Manifest bucket overlap between {name_a} and {name_b}: "
                f"{sorted(overlap)}. Each HF field must be classified into "
                f"exactly one bucket."
            )


def test_hf_manifest_covers_every_upstream_field():
    """Every field on HF's ``TrainingArguments`` is bucketed.

    Drift case: HF adds a new field on a future release; this test
    surfaces it as the unclassified-field list. The fix is to add the
    new field to one of the constants in
    ``opake.api.transformers.trainer._hf_convert``.
    """
    upstream = _all_hf_field_names()
    covered = _opake_buckets_for_hf()
    missing = upstream - covered
    assert not missing, (
        f"HF ``TrainingArguments`` has fields not classified by the opake "
        f"argument manifest:\n  {sorted(missing)}\n\n"
        f"This usually means upstream transformers added new fields on a "
        f"release the opake manifest hasn't tracked yet. For each missing "
        f"field, decide its bucket and add it to the right constant in "
        f"``opake.api.transformers.trainer._hf_convert``:\n"
        f"  - HF_DIRECT_FIELDS: same name + semantics as opake\n"
        f"  - HF_RENAME_MAP: different name in opake\n"
        f"  - HF_TRANSFORM_MAP: needs a derivation callable\n"
        f"  - HF_REJECTED_FIELDS: unsupported on the DP-SGD path\n"
        f"  - HF_DROP_FIELDS: irrelevant on opake path (drop with warning)"
    )


def test_trl_sft_manifest_covers_every_upstream_field():
    """Every TRL ``SFTConfig`` field is bucketed by HF base OR TRL SFT manifest."""
    upstream_trl_only = _trl_sft_only_field_names()
    covered = _opake_buckets_for_trl_sft()
    missing = upstream_trl_only - covered
    assert not missing, (
        f"TRL ``SFTConfig`` has SFT-specific fields not classified by the "
        f"opake argument manifest:\n  {sorted(missing)}\n\n"
        f"For each missing field, add it to a constant in "
        f"``opake.api.transformers.trl._sft_config``: TRL_SFT_DIRECT_FIELDS, "
        f"TRL_SFT_RENAME_MAP, TRL_SFT_TRANSFORM_MAP, TRL_SFT_REJECTED_FIELDS, "
        f"or TRL_SFT_DROP_FIELDS."
    )


def test_trl_dpo_manifest_covers_every_upstream_field():
    """Every TRL ``DPOConfig`` field is bucketed by HF base OR TRL DPO manifest."""
    upstream_trl_only = _trl_dpo_only_field_names()
    covered = _opake_buckets_for_trl_dpo()
    missing = upstream_trl_only - covered
    assert not missing, (
        f"TRL ``DPOConfig`` has DPO-specific fields not classified by the "
        f"opake argument manifest:\n  {sorted(missing)}\n\n"
        f"For each missing field, add it to a constant in "
        f"``opake.api.transformers.trl._dpo_config``: TRL_DPO_DIRECT_FIELDS, "
        f"TRL_DPO_RENAME_MAP, TRL_DPO_TRANSFORM_MAP, TRL_DPO_REJECTED_FIELDS, "
        f"or TRL_DPO_DROP_FIELDS."
    )


def test_trl_sft_manifest_buckets_are_disjoint():
    buckets = {
        "DIRECT": set(TRL_SFT_DIRECT_FIELDS),
        "RENAME": set(TRL_SFT_RENAME_MAP.keys()),
        "TRANSFORM": set(TRL_SFT_TRANSFORM_MAP.keys()),
        "REJECT_IF_SET": set(TRL_SFT_REJECTED_FIELDS.keys()),
        "DROP_WITH_WARN": set(TRL_SFT_DROP_FIELDS.keys()),
    }
    for name_a, set_a in buckets.items():
        for name_b, set_b in buckets.items():
            if name_a >= name_b:
                continue
            overlap = set_a & set_b
            assert not overlap, (
                f"TRL SFT manifest bucket overlap between {name_a} and "
                f"{name_b}: {sorted(overlap)}."
            )


def test_trl_dpo_manifest_buckets_are_disjoint():
    buckets = {
        "DIRECT": set(TRL_DPO_DIRECT_FIELDS),
        "RENAME": set(TRL_DPO_RENAME_MAP.keys()),
        "TRANSFORM": set(TRL_DPO_TRANSFORM_MAP.keys()),
        "REJECT_IF_SET": set(TRL_DPO_REJECTED_FIELDS.keys()),
        "DROP_WITH_WARN": set(TRL_DPO_DROP_FIELDS.keys()),
    }
    for name_a, set_a in buckets.items():
        for name_b, set_b in buckets.items():
            if name_a >= name_b:
                continue
            overlap = set_a & set_b
            assert not overlap, (
                f"TRL DPO manifest bucket overlap between {name_a} and "
                f"{name_b}: {sorted(overlap)}."
            )


def test_hf_manifest_does_not_have_phantom_fields():
    """Every field in the opake manifest exists on HF's upstream class.

    Catches the reverse drift: HF removes a field, and opake's manifest
    still references it. The fix is to delete the phantom entry from the
    manifest constant — or, if the field is still live on an older release
    inside the supported range, to declare it in ``HF_LEGACY_FIELDS``.
    """
    upstream = _all_hf_field_names()
    covered = _opake_buckets_for_hf()
    phantoms = covered - upstream - HF_LEGACY_FIELDS
    assert not phantoms, (
        f"opake argument manifest references HF fields that no longer exist "
        f"upstream:\n  {sorted(phantoms)}\n\n"
        f"This means HF removed a field that opake's manifest still "
        f"tracks. Remove the phantom entry from the corresponding constant "
        f"in ``opake.api.transformers.trainer._hf_convert``."
    )
