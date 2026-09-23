from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest
from datasets import Dataset, IterableDataset
from opaque_examples.sft.data import prepare_magicoder_dataset


def _source_dataset(size: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "problem": [f"problem-{index}" for index in range(size)],
            "solution": [f"solution-{index}" for index in range(size)],
            "metadata": list(range(size)),
        }
    )


def _prepare(dataset: Dataset, **overrides: object):
    arguments = {
        "source_id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "source_revision": "configured-revision",
        "prompt_field": "problem",
        "completion_field": "solution",
        "eval_fraction": 0.2,
        "split_seed": 42,
    }
    arguments.update(overrides)
    return prepare_magicoder_dataset(dataset, **arguments)


def test_rejects_missing_fields_before_splitting() -> None:
    source = Dataset.from_dict(
        {
            "problem": ["p0", None, "p2", "p3"],
            "solution": ["c0", "c1", None, "c3"],
        }
    )

    result = _prepare(source, eval_fraction=0.5)

    assert result.manifest.source_count == 4
    assert result.manifest.usable_count == 2
    assert result.manifest.rejected_count == 2
    assert len(result.train_dataset) == 1
    assert len(result.eval_dataset) == 1
    prompts = list(result.train_dataset["prompt"]) + list(result.eval_dataset["prompt"])
    assert set(prompts) == {
        "p0",
        "p3",
    }


@pytest.mark.parametrize("bad_field", ["problem", "solution"])
def test_rejects_non_string_fields(bad_field: str) -> None:
    columns: dict[str, list[object]] = {
        "problem": ["p0", "p1"],
        "solution": ["c0", "c1"],
    }
    columns[bad_field] = [1, 2]
    source = Dataset.from_dict(columns)

    with pytest.raises(ValueError, match="no usable records"):
        _prepare(source)


def test_split_membership_is_deterministic_and_seeded() -> None:
    source = _source_dataset(40)

    first = _prepare(source, split_seed=7)
    second = _prepare(source, split_seed=7)
    other_seed = _prepare(source, split_seed=8)

    first_train = first.train_dataset["prompt"]
    first_eval = first.eval_dataset["prompt"]
    assert first_train == second.train_dataset["prompt"]
    assert first_eval == second.eval_dataset["prompt"]
    assert set(first_train).isdisjoint(first_eval)
    assert set(list(first_train) + list(first_eval)) == set(source["problem"])
    assert first_eval != other_seed.eval_dataset["prompt"]


def test_magicoder_75197_count_math() -> None:
    size = 75_197
    source = Dataset.from_dict(
        {
            "problem": ["p"] * size,
            "solution": ["c"] * size,
        }
    )

    result = _prepare(source, eval_fraction=0.01, split_seed=42)

    assert len(result.train_dataset) == 74_445
    assert len(result.eval_dataset) == 752
    assert result.manifest.final_train_count == 74_445
    assert result.manifest.final_eval_count == 752


def test_maps_exact_columns_and_records_manifest() -> None:
    source = _source_dataset(10)
    original_fingerprint = source._fingerprint

    result = _prepare(source)

    assert result.train_dataset.column_names == ["prompt", "completion"]
    assert result.eval_dataset.column_names == ["prompt", "completion"]
    assert result.manifest.source_id == "ise-uiuc/Magicoder-OSS-Instruct-75K"
    assert result.manifest.source_revision == "configured-revision"
    assert result.manifest.original_fingerprint == original_fingerprint
    assert result.manifest.field_mapping == {
        "prompt": "problem",
        "completion": "solution",
    }
    assert result.manifest.split_seed == 42
    assert result.manifest.eval_fraction == 0.2
    assert result.manifest.source_count == 10
    assert result.manifest.usable_count == 10
    assert result.manifest.rejected_count == 0
    assert result.manifest.split_train_count == 8
    assert result.manifest.split_eval_count == 2
    assert result.manifest.final_train_count == 8
    assert result.manifest.final_eval_count == 2
    with pytest.raises(FrozenInstanceError):
        result.manifest.source_count = 0  # type: ignore[misc]


def test_delta_is_derived_after_caps() -> None:
    result = _prepare(_source_dataset(), train_cap=3, eval_cap=2)

    assert result.delta == pytest.approx(3**-1.1)
    assert result.manifest.final_train_count == 3
    assert result.manifest.final_eval_count == 2


def test_explicit_delta_is_preserved() -> None:
    result = _prepare(_source_dataset(), target_delta=1e-6)

    assert result.delta == 1e-6


def test_caps_select_from_the_already_split_datasets() -> None:
    source = _source_dataset(20)
    full = _prepare(source)
    capped = _prepare(source, train_cap=4, eval_cap=2)

    assert capped.train_dataset["prompt"] == full.train_dataset["prompt"][:4]
    assert capped.eval_dataset["prompt"] == full.eval_dataset["prompt"][:2]
    assert capped.manifest.split_train_count == 16
    assert capped.manifest.split_eval_count == 4
    assert capped.manifest.final_train_count == 4
    assert capped.manifest.final_eval_count == 2


def test_manifest_and_output_never_contain_row_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_prompt = "SECRET-PROMPT-ROW-CONTENT"
    secret_completion = "SECRET-COMPLETION-ROW-CONTENT"
    source = Dataset.from_dict(
        {
            "problem": [secret_prompt, "another prompt"],
            "solution": [secret_completion, "another completion"],
        }
    )

    result = _prepare(source, eval_fraction=0.5)
    serialized = result.manifest.to_json()
    captured = capsys.readouterr()

    assert json.loads(serialized) == result.manifest.to_dict()
    assert secret_prompt not in serialized
    assert secret_completion not in serialized
    assert secret_prompt not in captured.out + captured.err
    assert secret_completion not in captured.out + captured.err


@pytest.mark.parametrize("target_delta", [0.0, 1.0, -1e-6, math.inf, math.nan])
def test_invalid_explicit_delta_fails(target_delta: float) -> None:
    with pytest.raises(ValueError, match="target_delta"):
        _prepare(_source_dataset(), target_delta=target_delta)


@pytest.mark.parametrize("eval_fraction", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_invalid_eval_fraction_fails(eval_fraction: float) -> None:
    with pytest.raises(ValueError, match="eval_fraction"):
        _prepare(_source_dataset(), eval_fraction=eval_fraction)


@pytest.mark.parametrize(
    ("caps", "message"),
    [
        ({"train_cap": 0}, "effective train dataset is empty"),
        ({"eval_cap": 0}, "effective eval dataset is empty"),
    ],
)
def test_empty_effective_population_fails(caps: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare(_source_dataset(), **caps)


@pytest.mark.parametrize("cap_name", ["train_cap", "eval_cap"])
@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_invalid_caps_fail(cap_name: str, cap: object) -> None:
    with pytest.raises((TypeError, ValueError), match=cap_name):
        _prepare(_source_dataset(), **{cap_name: cap})


def test_empty_and_fully_rejected_sources_fail() -> None:
    empty = Dataset.from_dict({"problem": [], "solution": []})
    missing_fields = Dataset.from_dict({"other": ["value", "value"]})

    with pytest.raises(ValueError, match="no usable records"):
        _prepare(empty)
    with pytest.raises(ValueError, match="no usable records"):
        _prepare(missing_fields)


def test_streaming_dataset_is_rejected() -> None:
    source = IterableDataset.from_generator(
        lambda: iter([{"problem": "p", "solution": "c"}])
    )

    with pytest.raises(TypeError, match=r"non-streaming datasets\.Dataset"):
        _prepare(source)  # type: ignore[arg-type]
