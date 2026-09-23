"""Write sanitized final SFT bundles."""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_DEPENDENCY_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "opaque",
    "peft",
    "torch",
    "transformers",
    "trl",
    "wandb",
    "zenml",
)
_MANIFEST_NAMES = (
    "run_config.json",
    "dataset.json",
    "metrics.json",
    "privacy.json",
    "provenance.json",
)
_SOURCE_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")
_SECRET_WORDS = {"credential", "credentials", "password", "secret"}
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
_ROW_WORDS = {
    "example",
    "examples",
    "record",
    "records",
    "row",
    "rows",
    "sample",
    "samples",
}
_AGGREGATE_WORDS = {"count", "counts", "num", "number", "size", "sizes"}
_STATE_OWNERS = {"accountant", "callback", "callbacks", "optimizer", "trainer"}


def write_final_bundle(
    bundle_dir: str | Path,
    *,
    model: Any,
    tokenizer: Any,
    config: Any,
    dataset_manifest: Any,
    summary: Any,
    train_metrics: Any,
    eval_metrics: Any,
    privacy: Any,
    source_commit_sha: str,
    run_references: Any,
) -> Path:
    """Write sanitized inference assets and aggregate run manifests."""
    save_model = _save_method(model, "model")
    save_tokenizer = _save_method(tokenizer, "tokenizer")
    manifests = _build_manifests(
        config=config,
        dataset_manifest=dataset_manifest,
        summary=summary,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        privacy=privacy,
        source_commit_sha=source_commit_sha,
        run_references=run_references,
    )

    target = Path(bundle_dir)
    target_created = _prepare_target(target)
    try:
        save_model(target)
        save_tokenizer(target)
        _reject_non_inference_assets(target)
        for name in _MANIFEST_NAMES:
            _write_json(target / name, manifests[name])
    except BaseException as error:
        try:
            _cleanup_target(target, target_created=target_created)
        except OSError as cleanup_error:
            error.add_note(f"Bundle cleanup also failed: {cleanup_error}")
        raise

    return target


def _save_method(value: Any, label: str) -> Any:
    save_pretrained = getattr(value, "save_pretrained", None)
    if not callable(save_pretrained):
        _type_error(f"{label} must provide a callable save_pretrained method")
    return save_pretrained


def _build_manifests(
    *,
    config: Any,
    dataset_manifest: Any,
    summary: Any,
    train_metrics: Any,
    eval_metrics: Any,
    privacy: Any,
    source_commit_sha: str,
    run_references: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(source_commit_sha, str) or (
        source_commit_sha != "unknown"
        and not _SOURCE_SHA_PATTERN.fullmatch(source_commit_sha)
    ):
        _value_error(
            "source_commit_sha must be 'unknown' or a 7-64 character hexadecimal Git SHA"
        )

    sanitized_config = _sanitize_mapping(_config_fields(config), "config")
    sanitized_dataset = _sanitize_mapping(dataset_manifest, "dataset_manifest")
    sanitized_summary = _sanitize_mapping(summary, "summary")
    sanitized_train_metrics = _sanitize_mapping(train_metrics, "train_metrics")
    sanitized_eval_metrics = _sanitize_mapping(eval_metrics, "eval_metrics")
    sanitized_privacy = _sanitize_mapping(privacy, "privacy")
    sanitized_references = _sanitize_run_references(run_references)
    dependencies = _sanitize_mapping(_dependency_versions(), "dependencies")

    return {
        "run_config.json": sanitized_config,
        "dataset.json": sanitized_dataset,
        "metrics.json": {
            "summary": sanitized_summary,
            "train_metrics": sanitized_train_metrics,
            "eval_metrics": sanitized_eval_metrics,
        },
        "privacy.json": sanitized_privacy,
        "provenance.json": {
            "source_commit_sha": source_commit_sha,
            "runtime": {"python": platform.python_version()},
            "dependencies": dependencies,
            "run_references": sanitized_references,
        },
    }


def _config_fields(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: getattr(config, field.name) for field in fields(config)}

    for method_name in ("to_dict", "model_dump"):
        method = getattr(config, method_name, None)
        if callable(method):
            value = method()
            if not isinstance(value, Mapping):
                _type_error(f"config.{method_name}() must return a mapping")
            return value
    _type_error("config must be a mapping, dataclass, or expose to_dict/model_dump")


def _sanitize_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        for method_name in ("to_dict", "model_dump"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
                break
        else:
            if is_dataclass(value) and not isinstance(value, type):
                value = {
                    field.name: getattr(value, field.name) for field in fields(value)
                }
    if not isinstance(value, Mapping):
        _type_error(
            f"{label} must be a mapping, dataclass, or expose to_dict/model_dump"
        )
    sanitized = _sanitize(value, path=label, active_ids=set())
    if not isinstance(sanitized, dict):
        _type_error(f"{label} must be a mapping")
    return sanitized


def _sanitize_run_references(value: Any) -> dict[str, Any]:
    references = _sanitize_mapping(value, "run_references")
    for key in references:
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if not normalized.startswith(("wandb", "weightsandbiases", "zenml")):
            _value_error("run_references may contain only W&B and ZenML references")
    return references


def _sanitize(value: Any, *, path: str, active_ids: set[int]) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        if isinstance(value, str):
            _validate_string(value, path)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _value_error(f"{path} contains a non-finite number")
        return value

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active_ids:
            _value_error(f"{path} contains a reference cycle")
        active_ids.add(value_id)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    _type_error(f"{path} contains a non-string object key")
                _validate_key(key, item, path)
                result[key] = _sanitize(
                    item, path=f"{path}.{key}", active_ids=active_ids
                )
            return result
        finally:
            active_ids.remove(value_id)

    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in active_ids:
            _value_error(f"{path} contains a reference cycle")
        if any(isinstance(item, (Mapping, list, tuple)) for item in value):
            _value_error(f"{path} contains an unsafe row-like payload collection")
        active_ids.add(value_id)
        try:
            return [
                _sanitize(item, path=f"{path}[{index}]", active_ids=active_ids)
                for index, item in enumerate(value)
            ]
        finally:
            active_ids.remove(value_id)

    _type_error(f"{path} contains non-JSON-safe value {type(value).__name__}")


def _validate_key(key: str, value: Any, path: str) -> None:
    words = _key_words(key)
    normalized = "".join(words)
    if _SECRET_WORDS.intersection(words) or "apikey" in normalized:
        _value_error(f"{path} contains secret-bearing key {key!r}")
    if (
        "token" in words
        and key.lower() not in _TOKENIZER_CONFIG_KEYS
        and not _AGGREGATE_WORDS.intersection(words)
    ):
        _value_error(f"{path} contains secret-bearing key {key!r}")
    checkpoint_reference = {"artifact", "id"}.issubset(words) or {
        "artifact",
        "ids",
    }.issubset(words)
    if ("checkpoint" in words or "checkpoints" in words) and not (
        checkpoint_reference
        and isinstance(value, (list, tuple))
        and all(isinstance(item, str) for item in value)
    ):
        _value_error(f"{path} contains checkpoint metadata key {key!r}")
    if {"env", "environment"}.intersection(words):
        _value_error(f"{path} contains environment metadata key {key!r}")
    if "state" in words and _STATE_OWNERS.intersection(words):
        _value_error(f"{path} contains training state key {key!r}")

    is_collection = isinstance(value, (Mapping, list, tuple))
    is_aggregate = bool(_AGGREGATE_WORDS.intersection(words))
    if is_collection and _ROW_WORDS.intersection(words) and not is_aggregate:
        _value_error(f"{path} contains row payload key {key!r}")
    if is_collection and (
        {"raw", "tokenized"}.intersection(words) or key in {"data", "payload"}
    ):
        _value_error(f"{path} contains row payload key {key!r}")
    if is_collection and {"prompt", "completion"}.intersection(words):
        _value_error(f"{path} contains row payload key {key!r}")


def _key_words(key: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return tuple(re.findall(r"[a-z0-9]+", separated.lower()))


def _validate_string(value: str, path: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        return
    if parsed.username is not None or parsed.password is not None:
        _value_error(f"{path} contains credentials in a URL")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        _validate_key(key, None, path)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _prepare_target(target: Path) -> bool:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            _file_exists_error(
                f"bundle target must be a new or empty directory: {target}"
            )
        if next(target.iterdir(), None) is not None:
            _file_exists_error(f"bundle target must be empty: {target}")
        return False
    target.mkdir()
    return True


def _reject_non_inference_assets(target: Path) -> None:
    forbidden_files = {
        "accountant.json",
        "dp_optimizer.pt",
        "dp_state.pt",
        "optimizer.pt",
        "optimizer.bin",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    for path in target.rglob("*"):
        if path.is_symlink():
            _value_error(
                f"save_pretrained created an unsafe symbolic link: {path.name}"
            )
        words = _key_words(path.name)
        if path.is_dir() and {"checkpoint", "checkpoints"}.intersection(words):
            _value_error(f"save_pretrained created a checkpoint directory: {path.name}")
        if path.is_file() and (
            path.name.lower() in forbidden_files
            or ("state" in words and _STATE_OWNERS.intersection(words))
        ):
            _value_error(f"save_pretrained created training state: {path.name}")


def _type_error(message: str) -> None:
    raise TypeError(message)


def _value_error(message: str) -> None:
    raise ValueError(message)


def _file_exists_error(message: str) -> None:
    raise FileExistsError(message)


def _write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _cleanup_target(target: Path, *, target_created: bool) -> None:
    if target_created:
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.exists():
            shutil.rmtree(target)
        return

    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
        target.mkdir()
        return
    if not target.exists():
        target.mkdir()
        return
    for child in target.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
