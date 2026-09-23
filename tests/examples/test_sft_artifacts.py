from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from opaque_examples.sft import artifacts
from opaque_examples.sft.data import DatasetManifest

MANIFEST_NAMES = {
    "dataset.json",
    "metrics.json",
    "privacy.json",
    "provenance.json",
    "run_config.json",
}
SOURCE_COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeSaver:
    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content = content
        self.saved_to: Path | None = None

    def save_pretrained(self, target: str | Path) -> None:
        self.saved_to = Path(target)
        (self.saved_to / self.filename).write_text(self.content, encoding="utf-8")


class FailingSaver:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def save_pretrained(self, target: str | Path) -> None:
        (Path(target) / self.filename).write_text("partial", encoding="utf-8")
        raise RuntimeError("save failed")


def _metadata() -> dict[str, Any]:
    return {
        "config": {"epochs": 3, "learning_rate": 1e-5, "report_to": ["wandb"]},
        "dataset_manifest": {
            "name": "owner/dataset",
            "fingerprint": "dataset-fingerprint",
            "counts": {"train": 120, "eval": 30},
            "field_mapping": {"prompt": "question", "completion": "answer"},
        },
        "summary": {"status": "completed", "steps": 12},
        "train_metrics": {"loss": 0.25},
        "eval_metrics": {"loss": 0.5},
        "privacy": {"epsilon": 2.0, "delta": 1e-5},
        "source_commit_sha": SOURCE_COMMIT_SHA,
        "run_references": {
            "wandb": {"run_id": "wandb-id", "url": "https://wandb.invalid/run"},
            "zenml": {"run_id": "zenml-id"},
        },
    }


def _write_bundle(
    target: Path,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    **overrides: Any,
) -> Path:
    metadata = _metadata()
    metadata.update(overrides)
    return artifacts.write_final_bundle(
        target,
        model=model or FakeSaver("adapter_model.safetensors", "adapter"),
        tokenizer=tokenizer or FakeSaver("tokenizer.json", "tokenizer"),
        **metadata,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_final_bundle_writes_inference_assets_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeSaver("adapter_model.safetensors", "adapter")
    tokenizer = FakeSaver("tokenizer.json", "tokenizer")
    monkeypatch.setattr(artifacts, "_dependency_versions", lambda: {"peft": "1.2.3"})

    bundle = _write_bundle(tmp_path / "bundle", model=model, tokenizer=tokenizer)

    assert bundle == tmp_path / "bundle"
    assert model.saved_to == bundle
    assert tokenizer.saved_to == bundle
    assert (bundle / "adapter_model.safetensors").read_text(
        encoding="utf-8"
    ) == "adapter"
    assert (bundle / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer"
    assert {path.name for path in bundle.glob("*.json")} - {
        "tokenizer.json"
    } == MANIFEST_NAMES
    assert _read_json(bundle / "run_config.json") == _metadata()["config"]
    assert _read_json(bundle / "dataset.json") == _metadata()["dataset_manifest"]
    assert _read_json(bundle / "metrics.json") == {
        "eval_metrics": {"loss": 0.5},
        "summary": {"status": "completed", "steps": 12},
        "train_metrics": {"loss": 0.25},
    }
    assert _read_json(bundle / "privacy.json") == _metadata()["privacy"]
    provenance = _read_json(bundle / "provenance.json")
    assert provenance["source_commit_sha"] == SOURCE_COMMIT_SHA
    assert provenance["runtime"]["python"]
    assert provenance["dependencies"] == {"peft": "1.2.3"}
    assert provenance["run_references"] == _metadata()["run_references"]


def test_json_output_is_deterministic_utf8(tmp_path: Path) -> None:
    first = _write_bundle(tmp_path / "first")
    second = _write_bundle(tmp_path / "second")

    for name in MANIFEST_NAMES:
        first_bytes = (first / name).read_bytes()
        assert first_bytes == (second / name).read_bytes()
        assert first_bytes.decode("utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("summary", {"loss": math.nan}),
        ("train_metrics", {"loss": math.inf}),
        ("config", {"api_key": "do-not-write"}),
        ("privacy", {"accountant_state": {"step": 1}}),
        ("dataset_manifest", {"rows": ["private prompt", "private completion"]}),
        (
            "dataset_manifest",
            {
                "payload": [
                    {"prompt": "private prompt", "completion": "private completion"}
                ]
            },
        ),
        ("run_references", {"wandb": {"token": "do-not-write"}}),
        ("source_commit_sha", "../../unsafe"),
    ],
)
def test_unsafe_metadata_is_rejected_without_row_or_secret_leakage(
    tmp_path: Path, argument: str, value: Any
) -> None:
    target = tmp_path / "bundle"

    with pytest.raises((TypeError, ValueError)):
        _write_bundle(target, **{argument: value})

    assert not target.exists()


@pytest.mark.parametrize("failing_component", ["model", "tokenizer"])
def test_saver_failure_removes_new_target(
    tmp_path: Path, failing_component: str
) -> None:
    target = tmp_path / "bundle"
    model: Any = FakeSaver("adapter_model.safetensors", "adapter")
    tokenizer: Any = FakeSaver("tokenizer.json", "tokenizer")
    if failing_component == "model":
        model = FailingSaver("partial-model")
    else:
        tokenizer = FailingSaver("partial-tokenizer")

    with pytest.raises(RuntimeError, match="save failed"):
        _write_bundle(target, model=model, tokenizer=tokenizer)

    assert not target.exists()


def test_manifest_failure_cleans_created_files_but_preserves_preexisting_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "bundle"
    target.mkdir()

    def fail_manifest_write(path: Path, value: Any) -> None:
        path.write_text("partial", encoding="utf-8")
        raise OSError("manifest failed")

    monkeypatch.setattr(artifacts, "_write_json", fail_manifest_write)

    with pytest.raises(OSError, match="manifest failed"):
        _write_bundle(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_nonempty_target_is_rejected_without_modification(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    existing = target / "keep.txt"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write_bundle(target)

    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(target.iterdir()) == [existing]


def test_real_dataset_manifest_and_checkpoint_references_are_aggregate_only(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        source_id="owner/dataset",
        source_revision="main",
        original_fingerprint="fingerprint",
        prompt_field="problem",
        completion_field="solution",
        split_seed=42,
        eval_fraction=0.01,
        train_cap=None,
        eval_cap=None,
        source_count=100,
        usable_count=100,
        rejected_count=0,
        split_train_count=99,
        split_eval_count=1,
        final_train_count=99,
        final_eval_count=1,
    )

    bundle = _write_bundle(
        tmp_path / "bundle",
        dataset_manifest=manifest,
        summary={"checkpoint_artifact_ids": ["artifact-version-id"]},
        source_commit_sha="unknown",
    )

    dataset = _read_json(bundle / "dataset.json")
    assert dataset["field_mapping"] == {
        "completion": "solution",
        "prompt": "problem",
    }
    metrics = _read_json(bundle / "metrics.json")
    assert metrics["summary"]["checkpoint_artifact_ids"] == ["artifact-version-id"]


def test_dp_checkpoint_state_from_model_saver_is_rejected(tmp_path: Path) -> None:
    saver = FakeSaver("accountant.json", "private accountant state")

    with pytest.raises(ValueError, match="training state"):
        _write_bundle(tmp_path / "bundle", model=saver)

    assert not (tmp_path / "bundle").exists()


def test_tokenizer_special_token_config_is_not_treated_as_a_credential(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path / "bundle",
        config={"eos_token": "<eos>", "pad_token": "<pad>"},
    )

    assert _read_json(bundle / "run_config.json") == {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
