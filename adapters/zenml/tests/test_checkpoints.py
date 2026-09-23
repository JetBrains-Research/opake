from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("zenml")
pytest.importorskip("transformers")

checkpoints = pytest.importorskip("adapters.zenml.checkpoints")


ARTIFACT_ID = UUID("12345678-1234-5678-1234-567812345678")
PIPELINE_RUN_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_id": "acme/tiny-model",
        "model_revision": "model-rev",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "output_dir": "/worker/local-output",
        "privacy": {
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            "noise_multiplier": 1.25,
        },
    }
    config.update(updates)
    return config


def _dataset_manifest(**updates: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "dataset_id": "acme/tiny-dataset",
        "dataset_revision": "dataset-rev",
        "original_hf_fingerprint": "fingerprint-123",
    }
    manifest.update(updates)
    return manifest


def _bind(
    callback: checkpoints.ZenMLCheckpointCallback,
    *,
    config: dict[str, object] | None = None,
    dataset_manifest: object | None = None,
    resolved_delta: float = 1e-5,
    source_commit_sha: str = "0123456789abcdef",
) -> None:
    callback.bind_sft_run(
        config=_config() if config is None else config,
        dataset_manifest=(
            _dataset_manifest() if dataset_manifest is None else dataset_manifest
        ),
        resolved_delta=resolved_delta,
        source_commit_sha=source_commit_sha,
        run_references={"dataset_artifact_id": "dataset-version-id"},
    )


def _complete_checkpoint(output_dir: Path, step: int = 12) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "training_args.bin",
        "dp_optimizer.pt",
        "dp_state.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"tiny-state")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (checkpoint / "accountant.json").write_text(
        json.dumps({"steps": step}), encoding="utf-8"
    )
    return checkpoint


def _save(
    callback: checkpoints.ZenMLCheckpointCallback,
    output_dir: Path,
    step: int = 12,
) -> object:
    control = object()
    returned = callback.on_save(
        SimpleNamespace(output_dir=str(output_dir)),
        SimpleNamespace(global_step=step),
        control,
    )
    assert returned is control
    return control


def test_on_save_persists_complete_checkpoint_and_captures_returned_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _complete_checkpoint(tmp_path)
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(callback)
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_save_artifact(path: Path, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return SimpleNamespace(id=ARTIFACT_ID, version="7")

    monkeypatch.setattr(checkpoints, "_save_artifact", fake_save_artifact)

    _save(callback, tmp_path)

    assert callback.checkpoint_artifact_ids == [str(ARTIFACT_ID)]
    assert calls[0][0] == checkpoint
    assert calls[0][1]["name"] == "opaque-sft-checkpoint-parent-123"
    metadata = calls[0][1]["user_metadata"]
    assert isinstance(metadata, dict)
    assert metadata == {
        "schema_version": checkpoints.MANIFEST_SCHEMA_VERSION,
        "parent_run_id": "parent-123",
        "global_step": 12,
        "source_commit_sha": "0123456789abcdef",
        "target_epsilon": 8.0,
        "noise_multiplier": 1.25,
        "target_delta": 1e-5,
    }
    assert all(not isinstance(value, (dict, list)) for value in metadata.values())

    manifest = json.loads(
        (checkpoint / checkpoints.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest == {
        "schema_version": checkpoints.MANIFEST_SCHEMA_VERSION,
        "parent_run_id": "parent-123",
        "global_step": 12,
        "source_commit_sha": "0123456789abcdef",
        "run_references": {"dataset_artifact_id": "dataset-version-id"},
        "privacy": {
            "target_epsilon": 8.0,
            "noise_multiplier": 1.25,
            "target_delta": 1e-5,
        },
    }
    serialized = json.dumps(manifest)
    for forbidden in (
        "compatibility",
        "fingerprint",
        "model-rev",
        "tiny-model",
        "rows",
        "secret",
        "token",
    ):
        assert forbidden not in serialized.lower()


def test_on_save_accepts_string_artifact_id_and_rank_zero_rng_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _complete_checkpoint(tmp_path)
    (checkpoint / "rng_state.pth").rename(checkpoint / "rng_state_0.pth")
    callback = checkpoints.ZenMLCheckpointCallback(
        parent_run_id="parent-123", artifact_name="explicit-checkpoints"
    )
    _bind(callback)
    monkeypatch.setattr(checkpoints, "_save_artifact", lambda *args, **kwargs: "v8")

    _save(callback, tmp_path)

    assert callback.checkpoint_artifact_ids == ["v8"]


@pytest.mark.parametrize(
    "missing",
    [
        "model.safetensors",
        "trainer_state.json",
        "training_args.bin",
        "dp_optimizer.pt",
        "dp_state.pt",
        "accountant.json",
        "rng_state.pth",
    ],
)
def test_on_save_rejects_incomplete_checkpoint_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    checkpoint = _complete_checkpoint(tmp_path)
    (checkpoint / missing).unlink()
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(callback)
    called = False

    def fake_save_artifact(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return "not-reached"

    monkeypatch.setattr(checkpoints, "_save_artifact", fake_save_artifact)

    with pytest.raises(ValueError, match="incomplete"):
        _save(callback, tmp_path)

    assert not called
    assert not (checkpoint / checkpoints.MANIFEST_FILENAME).exists()


def test_materialize_explicit_name_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = _complete_checkpoint(source_root)
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(callback)
    monkeypatch.setattr(checkpoints, "_save_artifact", lambda *args, **kwargs: "id")
    _save(callback, source_root)
    loads: list[tuple[str, str | None]] = []

    def fake_load_artifact(name_or_id: str, *, version: str | None = None) -> Path:
        loads.append((name_or_id, version))
        return source

    monkeypatch.setattr(checkpoints, "_load_artifact", fake_load_artifact)
    destination = tmp_path / "workspace" / "resume"

    materialized = checkpoints.materialize_checkpoint(
        "explicit-checkpoints@7", destination
    )

    assert loads == [("explicit-checkpoints", "7")]
    assert materialized == destination
    assert (destination / "dp_optimizer.pt").read_bytes() == b"tiny-state"
    assert (destination / checkpoints.MANIFEST_FILENAME).is_file()


def test_materialize_explicit_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = _complete_checkpoint(source_root)
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(callback)
    monkeypatch.setattr(checkpoints, "_save_artifact", lambda *args, **kwargs: "id")
    _save(callback, source_root)
    loads: list[tuple[str, str | None]] = []

    def fake_load_artifact(name_or_id: str, *, version: str | None = None) -> Path:
        loads.append((name_or_id, version))
        return source

    monkeypatch.setattr(checkpoints, "_load_artifact", fake_load_artifact)

    checkpoints.materialize_checkpoint(str(ARTIFACT_ID), tmp_path / "resume")

    assert loads == [(str(ARTIFACT_ID), None)]


@pytest.mark.parametrize(
    "reference",
    [
        "checkpoint-name",
        "checkpoint-name@latest",
        "checkpoint-name@",
        "@7",
        "unsafe/name@7",
        "name@7@extra",
    ],
)
def test_materialize_rejects_ambiguous_or_unsafe_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    monkeypatch.setattr(
        checkpoints,
        "_load_artifact",
        lambda *args, **kwargs: pytest.fail("load must not be called"),
    )

    with pytest.raises(ValueError, match="reference"):
        checkpoints.materialize_checkpoint(reference, tmp_path / "resume")


def test_materialize_rejects_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "resume"
    destination.mkdir()
    monkeypatch.setattr(
        checkpoints,
        "_load_artifact",
        lambda *args, **kwargs: pytest.fail("load must not be called"),
    )

    with pytest.raises(FileExistsError):
        checkpoints.materialize_checkpoint(str(ARTIFACT_ID), destination)


def test_materialize_rejects_incomplete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _complete_checkpoint(tmp_path / "source")
    (source / "dp_state.pt").unlink()
    monkeypatch.setattr(checkpoints, "_load_artifact", lambda *args, **kwargs: source)

    with pytest.raises(ValueError, match="incomplete"):
        checkpoints.materialize_checkpoint(str(ARTIFACT_ID), tmp_path / "resume")

    assert not (tmp_path / "resume").exists()


def test_materialize_rejects_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _complete_checkpoint(tmp_path / "source")
    (source / checkpoints.MANIFEST_FILENAME).write_text(
        json.dumps({"rows": [{"prompt": "private prompt"}]}), encoding="utf-8"
    )
    monkeypatch.setattr(checkpoints, "_load_artifact", lambda *args, **kwargs: source)

    with pytest.raises(ValueError, match="manifest"):
        checkpoints.materialize_checkpoint(str(ARTIFACT_ID), tmp_path / "resume")

    assert not (tmp_path / "resume").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_run_id", ""),
        ("source_commit_sha", ""),
        ("global_step", True),
        ("global_step", -1),
    ],
)
def test_materialize_rejects_invalid_manifest_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source = _complete_checkpoint(tmp_path / "source")
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(callback)
    monkeypatch.setattr(checkpoints, "_save_artifact", lambda *args, **kwargs: "id")
    _save(callback, source.parent)
    manifest_path = source / checkpoints.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(checkpoints, "_load_artifact", lambda *args, **kwargs: source)

    with pytest.raises(ValueError, match="manifest"):
        checkpoints.materialize_checkpoint(str(ARTIFACT_ID), tmp_path / "resume")


def test_bind_allows_cross_config_source_and_privacy_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _complete_checkpoint(tmp_path / "first")
    first = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")
    _bind(first)
    monkeypatch.setattr(checkpoints, "_save_artifact", lambda *args, **kwargs: "id")
    _save(first, checkpoint.parent)

    resumed = checkpoints.ZenMLCheckpointCallback(
        parent_run_id="parent-456", resume_checkpoint=checkpoint
    )
    _bind(
        resumed,
        config=_config(
            model_id="other/model",
            model_revision="other-revision",
            learning_rate=3e-5,
            privacy={
                "target_epsilon": 4.0,
                "target_delta": 2e-5,
                "noise_multiplier": 2.0,
            },
        ),
        dataset_manifest=_dataset_manifest(
            dataset_revision="other-dataset-revision",
            original_hf_fingerprint="other-fingerprint",
        ),
        resolved_delta=2e-5,
        source_commit_sha="fedcba9876543210",
    )
    resumed_checkpoint = _complete_checkpoint(tmp_path / "second")
    _save(resumed, resumed_checkpoint.parent)

    resumed_manifest = json.loads(
        (resumed_checkpoint / checkpoints.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert resumed_manifest["parent_run_id"] == "parent-456"
    assert resumed_manifest["source_commit_sha"] == "fedcba9876543210"
    assert resumed_manifest["privacy"] == {
        "target_epsilon": 4.0,
        "target_delta": 2e-5,
        "noise_multiplier": 2.0,
    }


def _run_metadata(**values: object) -> dict[str, object]:
    return {key: SimpleNamespace(value=value) for key, value in values.items()}


def test_find_latest_checkpoint_is_run_scoped_and_uses_highest_step() -> None:
    lower_id = UUID("11111111-1111-1111-1111-111111111111")
    higher_id = UUID("22222222-2222-2222-2222-222222222222")
    versions = [
        SimpleNamespace(
            id=ARTIFACT_ID,
            run_metadata=_run_metadata(
                schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
                parent_run_id=str(PIPELINE_RUN_ID),
                global_step=True,
            ),
        ),
        SimpleNamespace(
            id=lower_id,
            run_metadata=_run_metadata(
                schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
                parent_run_id=str(PIPELINE_RUN_ID),
                global_step=7,
            ),
        ),
        SimpleNamespace(
            id=higher_id,
            run_metadata=_run_metadata(
                schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
                parent_run_id=str(PIPELINE_RUN_ID),
                global_step=20,
            ),
        ),
        SimpleNamespace(
            id=UUID("33333333-3333-3333-3333-333333333333"),
            run_metadata=_run_metadata(
                schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
                parent_run_id="another-run",
                global_step=999,
            ),
        ),
        SimpleNamespace(id="malformed", run_metadata=None),
    ]
    calls: list[dict[str, object]] = []

    class Client:
        def get_pipeline_run(self, run_id: str) -> object:
            assert run_id == "named-run"
            return SimpleNamespace(id=PIPELINE_RUN_ID)

        def list_artifact_versions(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(items=versions, total=len(versions))

    assert checkpoints.find_latest_checkpoint(Client(), "named-run") == str(higher_id)
    assert calls == [
        {
            "pipeline_run": str(PIPELINE_RUN_ID),
            "hydrate": True,
            "size": 100,
            "sort_by": "desc:created",
        }
    ]


def test_find_latest_checkpoint_supports_hydrated_zenml_metadata() -> None:
    checkpoint_id = UUID("44444444-4444-4444-4444-444444444444")
    version = SimpleNamespace(
        id=checkpoint_id,
        metadata=SimpleNamespace(
            run_metadata={
                "schema_version": checkpoints.MANIFEST_SCHEMA_VERSION,
                "parent_run_id": str(PIPELINE_RUN_ID),
                "global_step": 2,
            }
        ),
    )
    client = SimpleNamespace(
        get_pipeline_run=lambda run_id: SimpleNamespace(id=PIPELINE_RUN_ID),
        list_artifact_versions=lambda **kwargs: SimpleNamespace(
            items=[version], total=1
        ),
    )

    assert checkpoints.find_latest_checkpoint(client, "named-run") == str(checkpoint_id)


@pytest.mark.parametrize(
    "run_metadata",
    [
        None,
        {},
        _run_metadata(
            schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
            parent_run_id=str(PIPELINE_RUN_ID),
            global_step=True,
        ),
        _run_metadata(
            schema_version=checkpoints.MANIFEST_SCHEMA_VERSION,
            parent_run_id=str(PIPELINE_RUN_ID),
            global_step=-1,
        ),
    ],
)
def test_find_latest_checkpoint_rejects_no_match_and_malformed_metadata(
    run_metadata: object,
) -> None:
    client = SimpleNamespace(
        get_pipeline_run=lambda run_id: SimpleNamespace(id=PIPELINE_RUN_ID),
        list_artifact_versions=lambda **kwargs: SimpleNamespace(
            items=[SimpleNamespace(id=ARTIFACT_ID, run_metadata=run_metadata)],
            total=1,
        ),
    )

    with pytest.raises(ValueError, match=r"No checkpoint artifacts.*pipeline run"):
        checkpoints.find_latest_checkpoint(client, "named-run")


@pytest.mark.parametrize(
    ("dataset_update", "run_references"),
    [
        ({"rows": [{"prompt": "write a secret poem"}]}, {}),
        ({"api_token": "super-secret-token"}, {}),
        ({}, {"environment": {"ACCESS_TOKEN": "super-secret-token"}}),
    ],
)
def test_bind_rejects_rows_credentials_and_environment_dumps(
    dataset_update: dict[str, object], run_references: dict[str, object]
) -> None:
    callback = checkpoints.ZenMLCheckpointCallback(parent_run_id="parent-123")

    with pytest.raises(ValueError, match=r"(?i)sensitive|row-bearing"):
        callback.bind_sft_run(
            config=_config(),
            dataset_manifest=_dataset_manifest(**dataset_update),
            resolved_delta=1e-5,
            source_commit_sha="0123456789abcdef",
            run_references=run_references,
        )


@pytest.mark.parametrize("name", ["bad/name", "bad@name", "../name", " name"])
def test_callback_rejects_unsafe_artifact_name(name: str) -> None:
    with pytest.raises(ValueError, match="artifact name"):
        checkpoints.ZenMLCheckpointCallback(
            parent_run_id="parent-123", artifact_name=name
        )
