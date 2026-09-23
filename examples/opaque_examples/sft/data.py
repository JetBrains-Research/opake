"""Deterministic dataset preparation for completion-only SFT."""

# ruff: noqa: INP001, TRY003

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from datasets import Dataset


@dataclass(frozen=True)
class DatasetManifest:
    """JSON-safe provenance and cardinalities for a prepared dataset."""

    source_id: str
    source_revision: str | None
    original_fingerprint: str
    prompt_field: str
    completion_field: str
    split_seed: int
    eval_fraction: float
    train_cap: int | None
    eval_cap: int | None
    source_count: int
    usable_count: int
    rejected_count: int
    split_train_count: int
    split_eval_count: int
    final_train_count: int
    final_eval_count: int

    @property
    def field_mapping(self) -> dict[str, str]:
        return {
            "prompt": self.prompt_field,
            "completion": self.completion_field,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation containing no dataset rows."""
        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "original_fingerprint": self.original_fingerprint,
            "field_mapping": self.field_mapping,
            "split_seed": self.split_seed,
            "eval_fraction": self.eval_fraction,
            "train_cap": self.train_cap,
            "eval_cap": self.eval_cap,
            "source_count": self.source_count,
            "usable_count": self.usable_count,
            "rejected_count": self.rejected_count,
            "split_train_count": self.split_train_count,
            "split_eval_count": self.split_eval_count,
            "final_train_count": self.final_train_count,
            "final_eval_count": self.final_eval_count,
        }

    def to_json(self) -> str:
        """Serialize the manifest without including row content."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class PreparedDataset:
    """Completion-only SFT datasets and their resolved privacy delta."""

    train_dataset: Dataset
    eval_dataset: Dataset
    delta: float
    manifest: DatasetManifest


def _is_usable_record(
    record: dict[str, object], *, prompt_field: str, completion_field: str
) -> bool:
    return isinstance(record.get(prompt_field), str) and isinstance(
        record.get(completion_field), str
    )


def _to_completion_record(
    record: dict[str, object], *, prompt_field: str, completion_field: str
) -> dict[str, object]:
    return {
        "prompt": record[prompt_field],
        "completion": record[completion_field],
    }


def _validate_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_cap(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer or None")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _validate_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("eval_fraction must be a number strictly between 0 and 1")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("eval_fraction must be strictly between 0 and 1")
    return value


def _validate_delta(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("target_delta must be a number strictly between 0 and 1")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("target_delta must be strictly between 0 and 1")
    return value


def _apply_cap(dataset: Dataset, cap: int | None) -> Dataset:
    if cap is None or cap >= len(dataset):
        return dataset
    return dataset.select(range(cap))


def prepare_magicoder_dataset(
    dataset: Dataset,
    *,
    source_id: str,
    source_revision: str | None,
    prompt_field: str,
    completion_field: str,
    eval_fraction: float,
    split_seed: int,
    train_cap: int | None = None,
    eval_cap: int | None = None,
    target_delta: float | None = None,
) -> PreparedDataset:
    """Validate and deterministically prepare completion-only SFT datasets."""
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a non-streaming datasets.Dataset")

    source_id = _validate_nonempty_string(source_id, name="source_id")
    if source_revision is not None:
        source_revision = _validate_nonempty_string(
            source_revision, name="source_revision"
        )
    prompt_field = _validate_nonempty_string(prompt_field, name="prompt_field")
    completion_field = _validate_nonempty_string(
        completion_field, name="completion_field"
    )
    eval_fraction = _validate_fraction(eval_fraction)
    if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
        raise TypeError("split_seed must be an integer")
    split_seed = int(split_seed)
    train_cap = _validate_cap(train_cap, name="train_cap")
    eval_cap = _validate_cap(eval_cap, name="eval_cap")
    target_delta = _validate_delta(target_delta)

    source_count = len(dataset)
    original_fingerprint = dataset._fingerprint
    usable = dataset.filter(
        _is_usable_record,
        fn_kwargs={
            "prompt_field": prompt_field,
            "completion_field": completion_field,
        },
    )
    usable_count = len(usable)
    if usable_count == 0:
        raise ValueError("dataset contains no usable records")

    expected_eval_count = math.ceil(usable_count * eval_fraction)
    if usable_count - expected_eval_count == 0:
        raise ValueError("effective train dataset is empty after splitting")
    if expected_eval_count == 0:
        raise ValueError("effective eval dataset is empty after splitting")

    split = usable.train_test_split(
        test_size=eval_fraction,
        seed=split_seed,
        shuffle=True,
    )
    split_train = split["train"]
    split_eval = split["test"]
    split_train_count = len(split_train)
    split_eval_count = len(split_eval)

    effective_train = _apply_cap(split_train, train_cap)
    effective_eval = _apply_cap(split_eval, eval_cap)
    if len(effective_train) == 0:
        raise ValueError("effective train dataset is empty")
    if len(effective_eval) == 0:
        raise ValueError("effective eval dataset is empty")

    map_kwargs = {
        "prompt_field": prompt_field,
        "completion_field": completion_field,
    }
    train_dataset = effective_train.map(
        _to_completion_record,
        fn_kwargs=map_kwargs,
        remove_columns=effective_train.column_names,
    )
    eval_dataset = effective_eval.map(
        _to_completion_record,
        fn_kwargs=map_kwargs,
        remove_columns=effective_eval.column_names,
    )
    final_train_count = len(train_dataset)
    final_eval_count = len(eval_dataset)
    delta = final_train_count**-1.1 if target_delta is None else target_delta

    manifest = DatasetManifest(
        source_id=source_id,
        source_revision=source_revision,
        original_fingerprint=original_fingerprint,
        prompt_field=prompt_field,
        completion_field=completion_field,
        split_seed=split_seed,
        eval_fraction=eval_fraction,
        train_cap=train_cap,
        eval_cap=eval_cap,
        source_count=source_count,
        usable_count=usable_count,
        rejected_count=source_count - usable_count,
        split_train_count=split_train_count,
        split_eval_count=split_eval_count,
        final_train_count=final_train_count,
        final_eval_count=final_eval_count,
    )
    return PreparedDataset(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        delta=delta,
        manifest=manifest,
    )


__all__ = ["DatasetManifest", "PreparedDataset", "prepare_magicoder_dataset"]
