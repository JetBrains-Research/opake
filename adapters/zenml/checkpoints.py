"""Versioned ZenML transport for complete, resumable Trainer checkpoints."""

# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - exercised only without the optional extra

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base that keeps this transport importable without Transformers."""


MANIFEST_FILENAME = "zenml-checkpoint.json"
MANIFEST_SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 1_000_000

_ARTIFACT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIRECT_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
_WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_REQUIRED_STATE_FILES = (
    "trainer_state.json",
    "training_args.bin",
    "dp_optimizer.pt",
    "dp_state.pt",
    "accountant.json",
)
_RANK_ZERO_RNG_FILES = ("rng_state.pth", "rng_state_0.pth")
_ROW_BEARING_KEYS = {
    "data",
    "dataset_rows",
    "example",
    "examples",
    "raw_data",
    "raw_rows",
    "record",
    "records",
    "row",
    "rows",
    "sample",
    "samples",
}
_CONTENT_KEYS = {"completion", "completions", "prompt", "prompts"}
_FIELD_MAPPING_KEYS = {"column_mapping", "field_mapping"}
_SECRET_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "client_secret",
    "cookie",
    "credential",
    "credentials",
    "hf_token",
    "huggingface_token",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "token",
}
_TOKENIZER_CONFIG_KEYS = {
    "bos_token",
    "cls_token",
    "decoder_start_token",
    "eos_token",
    "mask_token",
    "pad_token",
    "sep_token",
    "unk_token",
}
_ENVIRONMENT_KEYS = {
    "env",
    "environ",
    "environment",
    "environment_variables",
    "env_vars",
}


def _save_artifact(data: Path, **kwargs: object) -> object:
    from zenml import save_artifact

    return save_artifact(data, **kwargs)


def _load_artifact(name_or_id: str, *, version: str | None = None) -> object:
    from zenml import load_artifact

    return load_artifact(name_or_id, version=version)


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _plain_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Enum):
        return _plain_value(value.value, path=path)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = _plain_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            _plain_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _plain_value(dumped, path=path)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain_value(to_dict(), path=path)

    if is_dataclass(value) and not isinstance(value, type):
        return _plain_value(asdict(value), path=path)

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        public_attributes = {
            key: item for key, item in attributes.items() if not key.startswith("_")
        }
        return _plain_value(public_attributes, path=path)

    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _as_mapping(value: object, *, path: str) -> dict[str, object]:
    plain = _plain_value(value, path=path)
    if not isinstance(plain, dict):
        raise ValueError(f"{path} must be a mapping or configuration object")
    return plain


def _assert_no_sensitive_payload(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(str(key))
            current_path = (*path, normalized)
            if normalized in _ROW_BEARING_KEYS:
                raise ValueError(
                    f"Row-bearing value is forbidden at {'.'.join(current_path)}"
                )
            if normalized in _ENVIRONMENT_KEYS:
                raise ValueError(
                    f"Sensitive environment dump is forbidden at {'.'.join(current_path)}"
                )
            is_credential = normalized in _SECRET_KEYS or normalized.endswith(
                (
                    "_api_key",
                    "_credential",
                    "_credentials",
                    "_password",
                    "_secret",
                    "_token",
                )
            )
            if is_credential and normalized not in _TOKENIZER_CONFIG_KEYS:
                raise ValueError(
                    f"Sensitive credential is forbidden at {'.'.join(current_path)}"
                )
            if normalized in _CONTENT_KEYS and not any(
                ancestor in _FIELD_MAPPING_KEYS for ancestor in path
            ):
                raise ValueError(
                    f"Row-bearing prompt/completion content is forbidden at "
                    f"{'.'.join(current_path)}"
                )
            _assert_no_sensitive_payload(item, path=current_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_sensitive_payload(item, path=(*path, str(index)))


def _first_value(
    mapping: Mapping[str, object], aliases: set[str], *, recursive: bool = True
) -> object | None:
    for key, value in mapping.items():
        if _normalized_key(key) in aliases and value is not None:
            return value
    if recursive:
        for value in mapping.values():
            if isinstance(value, Mapping):
                found = _first_value(value, aliases)
                if found is not None:
                    return found
    return None


def _required_string(value: object | None, *, field: str) -> str:
    if not isinstance(value, (str, Path, UUID)):
        raise ValueError(f"{field} must be a non-empty string")
    result = str(value).strip()
    if not result or any(character in result for character in "\r\n\0"):
        raise ValueError(f"{field} must be a non-empty single-line string")
    return result


def _privacy_number(
    value: object,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{field} must be a finite positive number")
    return result


def _privacy_metadata(
    config: Mapping[str, object], resolved_delta: float
) -> dict[str, float]:
    target = _first_value(
        config,
        {
            "differential_privacy",
            "dp",
            "dp_config",
            "privacy",
            "privacy_config",
            "privacy_target",
            "target_privacy",
        },
        recursive=False,
    )
    if target is None:
        target = config
    if not isinstance(target, Mapping):
        raise ValueError("config privacy target must be a mapping")

    metadata: dict[str, float] = {}
    aliases = {
        "target_epsilon": {
            "epsilon",
            "privacy_target_epsilon",
            "target_epsilon",
        },
        "noise_multiplier": {
            "noise_multiplier",
            "noise_target",
            "sigma",
            "target_noise",
            "target_noise_multiplier",
        },
    }
    for metadata_key, field_aliases in aliases.items():
        value = _first_value(target, field_aliases)
        if value is not None:
            metadata[metadata_key] = _privacy_number(
                value,
                field=metadata_key,
                allow_zero=metadata_key == "noise_multiplier",
            )

    delta = _privacy_number(
        resolved_delta,
        field="resolved_delta",
        allow_zero=False,
    )
    if delta >= 1:
        raise ValueError("resolved_delta must be a number between zero and one")
    metadata["target_delta"] = delta
    return metadata


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _build_manifest(
    *,
    config: object,
    dataset_manifest: object,
    resolved_delta: float,
    source_commit_sha: str,
    parent_run_id: str,
    run_references: object,
) -> dict[str, object]:
    config_mapping = _as_mapping(config, path="config")
    dataset_mapping = _as_mapping(dataset_manifest, path="dataset_manifest")
    references_mapping = (
        {}
        if run_references is None
        else _as_mapping(run_references, path="run_references")
    )
    _assert_no_sensitive_payload(config_mapping, path=("config",))
    _assert_no_sensitive_payload(dataset_mapping, path=("dataset_manifest",))
    _assert_no_sensitive_payload(references_mapping, path=("run_references",))

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "parent_run_id": _required_string(parent_run_id, field="parent run ID"),
        "source_commit_sha": _required_string(
            source_commit_sha, field="source commit SHA"
        ),
        "run_references": references_mapping,
        "privacy": _privacy_metadata(config_mapping, resolved_delta),
    }
    _assert_no_sensitive_payload(manifest)
    return manifest


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_manifest_data(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "parent_run_id",
        "global_step",
        "source_commit_sha",
        "run_references",
        "privacy",
    }
    if set(value) != expected_keys:
        raise ValueError("Checkpoint manifest has an invalid structure")

    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Checkpoint manifest has an unsupported schema version")
    _required_string(value["parent_run_id"], field="parent run ID")
    _required_string(value["source_commit_sha"], field="source commit SHA")

    global_step = value["global_step"]
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("Checkpoint manifest has an invalid global step")
    if not isinstance(value["run_references"], dict):
        raise ValueError("Checkpoint manifest has invalid run references")

    privacy = value["privacy"]
    allowed_privacy_keys = {"target_epsilon", "noise_multiplier", "target_delta"}
    if (
        not isinstance(privacy, dict)
        or "target_delta" not in privacy
        or not set(privacy).issubset(allowed_privacy_keys)
    ):
        raise ValueError("Checkpoint manifest has invalid privacy metadata")
    for field, field_value in privacy.items():
        number = _privacy_number(
            field_value,
            field=field,
            allow_zero=field == "noise_multiplier",
        )
        if field == "target_delta" and number >= 1:
            raise ValueError("Checkpoint manifest has an invalid target delta")

    _assert_no_sensitive_payload(value)
    return value


def _read_manifest(checkpoint: Path) -> dict[str, object]:
    path = checkpoint / MANIFEST_FILENAME
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint manifest is missing")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Checkpoint manifest is unreasonably large")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_safe_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Checkpoint manifest is malformed") from error
    try:
        return _validate_manifest_data(value)
    except ValueError as error:
        raise ValueError(f"Checkpoint manifest is invalid: {error}") from error


def _nonempty_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _has_complete_weights(checkpoint: Path) -> bool:
    if any(_nonempty_regular_file(checkpoint / name) for name in _DIRECT_WEIGHT_FILES):
        return True
    for index_name in _WEIGHT_INDEX_FILES:
        index_path = checkpoint / index_name
        if not _nonempty_regular_file(index_path):
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shard_names = set(weight_map.values())
        if all(
            isinstance(name, str)
            and Path(name).name == name
            and _nonempty_regular_file(checkpoint / name)
            for name in shard_names
        ):
            return True
    return False


def _validate_checkpoint(checkpoint: Path, *, require_manifest: bool) -> None:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    missing = [
        filename
        for filename in _REQUIRED_STATE_FILES
        if not _nonempty_regular_file(checkpoint / filename)
    ]
    if not _has_complete_weights(checkpoint):
        missing.append("model or adapter weights")
    if not any(
        _nonempty_regular_file(checkpoint / filename)
        for filename in _RANK_ZERO_RNG_FILES
    ):
        missing.append("rank-0 RNG state")
    if missing:
        raise ValueError(
            f"Checkpoint is incomplete; missing or empty: {', '.join(missing)}"
        )

    for json_filename in ("trainer_state.json", "accountant.json"):
        try:
            parsed = json.loads(
                (checkpoint / json_filename).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Checkpoint is incomplete; {json_filename} is not valid JSON"
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Checkpoint is incomplete; {json_filename} must contain an object"
            )
    if require_manifest:
        _read_manifest(checkpoint)


def _atomic_write_manifest(checkpoint: Path, manifest: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=checkpoint,
        prefix=f".{MANIFEST_FILENAME}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(_canonical_json(manifest))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(checkpoint / MANIFEST_FILENAME)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_artifact_name(name: str) -> str:
    if _ARTIFACT_NAME_PATTERN.fullmatch(name) is None or name.lower() == "latest":
        raise ValueError(f"Unsafe ZenML artifact name: {name!r}")
    return name


def _default_artifact_name(parent_run_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", parent_run_id).strip("._-")
    if not slug:
        slug = hashlib.sha256(parent_run_id.encode("utf-8")).hexdigest()[:16]
    prefix = "opaque-sft-checkpoint-"
    maximum_slug_length = 128 - len(prefix)
    if len(slug) > maximum_slug_length:
        suffix = hashlib.sha256(parent_run_id.encode("utf-8")).hexdigest()[:12]
        slug = f"{slug[: maximum_slug_length - len(suffix) - 1]}-{suffix}"
    return _validate_artifact_name(f"{prefix}{slug}")


def _parse_reference(reference: str) -> tuple[str, str | None]:
    if (
        not isinstance(reference, str)
        or not reference
        or reference != reference.strip()
    ):
        raise ValueError(
            "ZenML checkpoint reference must be an explicit UUID or name@version"
        )
    try:
        artifact_id = UUID(reference)
    except ValueError:
        artifact_id = None
    if artifact_id is not None and str(artifact_id) == reference.lower():
        return str(artifact_id), None

    if reference.count("@") != 1:
        raise ValueError(
            "ZenML checkpoint reference must be an explicit UUID or name@version"
        )
    name, version = reference.split("@", maxsplit=1)
    try:
        _validate_artifact_name(name)
    except ValueError as error:
        raise ValueError(
            "ZenML checkpoint reference contains an unsafe name"
        ) from error
    if _VERSION_PATTERN.fullmatch(version) is None or version.lower() == "latest":
        raise ValueError("ZenML checkpoint reference must contain an explicit version")
    return name, version


def materialize_checkpoint(reference: str, destination: Path) -> Path:
    """Load one explicit ZenML checkpoint version into a new local directory."""

    name_or_id, version = _parse_reference(reference)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Checkpoint destination already exists: {destination}")

    loaded = _load_artifact(name_or_id, version=version)
    try:
        source = Path(loaded)
    except TypeError as error:
        raise ValueError(
            "ZenML checkpoint artifact did not materialize as a path"
        ) from error
    _validate_checkpoint(source, require_manifest=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.materializing-{uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        _validate_checkpoint(staging, require_manifest=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Checkpoint destination already exists: {destination}"
            )
        staging.rename(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _run_metadata_value(run_metadata: Mapping[str, object], key: str) -> object:
    entry = run_metadata.get(key)
    if isinstance(entry, Mapping):
        return entry.get("value")
    return getattr(entry, "value", None)


def find_latest_checkpoint(client: object, run_id: str) -> str:
    """Return the immutable ID of the highest-step checkpoint for one run."""

    requested_run_id = _required_string(run_id, field="pipeline run ID")
    get_pipeline_run = getattr(client, "get_pipeline_run", None)
    list_artifact_versions = getattr(client, "list_artifact_versions", None)
    if not callable(get_pipeline_run) or not callable(list_artifact_versions):
        raise TypeError("client must be a ZenML Client")

    pipeline_run = get_pipeline_run(requested_run_id)
    pipeline_run_id = _required_string(
        getattr(pipeline_run, "id", None), field="resolved pipeline run ID"
    )
    response = list_artifact_versions(
        pipeline_run=pipeline_run_id,
        hydrate=True,
        size=100,
        sort_by="desc:created",
    )
    versions = getattr(response, "items", None)
    if not isinstance(versions, Sequence) or isinstance(versions, (str, bytes)):
        raise RuntimeError("ZenML returned a malformed artifact version response")

    candidates: list[tuple[int, str]] = []
    for version in versions:
        run_metadata = getattr(version, "run_metadata", None)
        if not isinstance(run_metadata, Mapping):
            continue
        schema_version = _run_metadata_value(run_metadata, "schema_version")
        parent_run_id = _run_metadata_value(run_metadata, "parent_run_id")
        global_step = _run_metadata_value(run_metadata, "global_step")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != MANIFEST_SCHEMA_VERSION
            or parent_run_id != pipeline_run_id
            or isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            continue
        artifact_id = getattr(version, "id", None)
        if not isinstance(artifact_id, (str, UUID)) or not str(artifact_id):
            continue
        candidates.append((global_step, str(artifact_id)))

    if not candidates:
        raise ValueError(
            f"No checkpoint artifacts found for pipeline run {pipeline_run_id}"
        )
    return max(candidates, key=lambda candidate: candidate[0])[1]


class ZenMLCheckpointCallback(TrainerCallback):
    """Save complete Trainer checkpoints as explicitly versioned ZenML artifacts."""

    def __init__(
        self,
        parent_run_id: str,
        resume_checkpoint: Path | None = None,
        artifact_name: str | None = None,
    ) -> None:
        super().__init__()
        self.parent_run_id = _required_string(parent_run_id, field="parent run ID")
        self.resume_checkpoint = (
            Path(resume_checkpoint) if resume_checkpoint is not None else None
        )
        self.artifact_name = (
            _validate_artifact_name(artifact_name)
            if artifact_name is not None
            else _default_artifact_name(self.parent_run_id)
        )
        self.checkpoint_artifact_ids: list[str] = []
        self._manifest: dict[str, object] | None = None

    @staticmethod
    def materialize_checkpoint(reference: str, destination: Path) -> Path:
        return materialize_checkpoint(reference, destination)

    def bind_sft_run(
        self,
        *,
        config: object,
        dataset_manifest: object,
        resolved_delta: float,
        source_commit_sha: str,
        run_references: object,
    ) -> None:
        manifest = _build_manifest(
            config=config,
            dataset_manifest=dataset_manifest,
            resolved_delta=resolved_delta,
            source_commit_sha=source_commit_sha,
            parent_run_id=self.parent_run_id,
            run_references=run_references,
        )
        if self.resume_checkpoint is not None:
            _validate_checkpoint(self.resume_checkpoint, require_manifest=True)
        self._manifest = manifest

    def on_save(
        self,
        args: object,
        state: object,
        control: object,
        **kwargs: object,
    ) -> object:
        del kwargs
        if self._manifest is None:
            raise RuntimeError("bind_sft_run must be called before saving checkpoints")
        global_step = getattr(state, "global_step", None)
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise ValueError("Trainer global_step must be a non-negative integer")
        if global_step < 0:
            raise ValueError("Trainer global_step must be a non-negative integer")
        output_dir_value = getattr(args, "output_dir", None)
        if not isinstance(output_dir_value, (str, os.PathLike)):
            raise ValueError("Trainer output_dir must be a filesystem path")
        output_dir = Path(output_dir_value)
        checkpoint = output_dir / f"checkpoint-{global_step}"
        if checkpoint.name != f"checkpoint-{global_step}":
            raise ValueError(
                "Checkpoint directory must have an atomic checkpoint-step name"
            )
        _validate_checkpoint(checkpoint, require_manifest=False)
        manifest = {**self._manifest, "global_step": global_step}
        _atomic_write_manifest(checkpoint, manifest)
        _validate_checkpoint(checkpoint, require_manifest=True)

        privacy = manifest["privacy"]
        assert isinstance(privacy, dict)
        metadata: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "parent_run_id": self.parent_run_id,
            "global_step": global_step,
            "source_commit_sha": manifest["source_commit_sha"],
        }
        metadata.update(privacy)
        _assert_no_sensitive_payload(metadata)
        artifact = _save_artifact(
            checkpoint,
            name=self.artifact_name,
            user_metadata=metadata,
        )
        artifact_id = (
            artifact
            if isinstance(artifact, (str, UUID))
            else getattr(artifact, "id", None)
        )
        if not isinstance(artifact_id, (str, UUID)) or not str(artifact_id):
            raise RuntimeError(
                "ZenML save_artifact did not return an artifact version ID"
            )
        self.checkpoint_artifact_ids.append(str(artifact_id))
        return control


__all__ = [
    "ZenMLCheckpointCallback",
    "find_latest_checkpoint",
    "materialize_checkpoint",
]
