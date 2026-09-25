"""Inert, flat generation artifacts for a separate Docker scoring host.

The portable directory contains ``generation-manifest.json``,
``generation_receipt.json``, ``config.json``, checkpoint metadata (not weights),
and, per benchmark, ``NAME.jsonl``, ``NAME-generation.json``,
``NAME-samples.jsonl`` and ``NAME-raw_samples.jsonl``. The manifest binds every
file. Its byte SHA256 must be carried independently to the scoring host; hashes
alone are not authentication. Validation never imports a trainer or executes
benchmark/sample code. At most 1 GiB of regular, non-symlink files is accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
from pathlib import Path

from examples.moe_privacy import code_eval

MAX_GENERATION_BYTES = 1024**3
MANIFEST = "generation-manifest.json"
RECEIPT = "generation_receipt.json"
ARMS = (
    "base",
    "reference",
    "reference_aux",
    "dp",
    "dp_aux",
    "trl_reference",
    "trl_reference_aux",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CHECKPOINT_FILES = {
    "summary.json": "checkpoint-summary.json",
    "trainable/adapter_spec.json": "adapter_spec.json",
    "training_receipt.json": "checkpoint-training_receipt.json",
    "wandb_run.json": "checkpoint-wandb_run.json",
}
_WEIGHTS = "trainable/trainable.safetensors"
_SAMPLE_FILES = ("generation.json", "samples.jsonl", "raw_samples.jsonl")


def json_bytes(value) -> bytes:
    """Canonical UTF-8 JSON for the handoff (including one final newline)."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha256(data: bytes) -> str:
    """Hash inert bytes without interpreting them."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    """Create canonical metadata without replacing an existing artifact."""
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def file_record(path: Path) -> dict:
    """Return a bounded regular file's byte length and SHA256."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_GENERATION_BYTES:
        message = f"Unsafe or oversized handoff file: {path.name}"
        raise ValueError(message)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > MAX_GENERATION_BYTES:
                message = "Handoff file exceeds the 1 GiB limit"
                raise ValueError(message)
            digest.update(chunk)
    if size != info.st_size:
        message = "Handoff file changed while hashing"
        raise ValueError(message)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            message = f"Duplicate JSON key: {key}"
            raise ValueError(message)
        result[key] = value
    return result


def read_json(path: Path):
    """Read bounded JSON metadata, rejecting duplicate keys and nonfinite values."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
        message = f"Unsafe or oversized JSON metadata: {path.name}"
        raise ValueError(message)
    value = json.loads(path.read_bytes(), object_pairs_hook=_object)
    json_bytes(value)
    return value


def copy_file(source: Path, destination: Path) -> None:
    """Exclusive copy of inert bytes; never reuse a caller's output file."""
    before = file_record(source)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if file_record(destination) != before:
        message = "Handoff source changed while copying"
        raise ValueError(message)


def _checkpoint_identity(config, arm, seed, summary, files):
    if (
        not isinstance(config, dict)
        or arm not in ARMS
        or type(seed) is not int
        or seed < 0
        or not isinstance(config.get("model_id"), str)
        or not config["model_id"]
        or not isinstance(config.get("model_revision"), str)
        or not _REVISION.fullmatch(config["model_revision"])
    ):
        message = "Invalid checkpoint configuration, arm, seed or model revision"
        raise ValueError(message)
    backend = "pretrained"
    if arm != "base":
        if (
            not isinstance(summary, dict)
            or summary.get("status") != "completed"
            or summary.get("arm") != arm
            or summary.get("model_seed") != seed
            or json_bytes(summary.get("config")) != json_bytes(config)
        ):
            message = (
                "the evaluation configuration must match the completed training run"
            )
            raise ValueError(message)
        native = arm.startswith("trl_")
        backend = "trl.SFTTrainer" if native else "opaque.transformers.DPTrainer"
        experiment = "native_trl_moe_sft" if native else "pretrained_moe_sft"
        if (
            summary.get("experiment", experiment) != experiment
            or summary.get("execution", {}).get("trainer_backend", backend) != backend
        ):
            message = "Checkpoint backend does not match the training arm"
            raise ValueError(message)
    return {
        "model": {"id": config["model_id"], "revision": config["model_revision"]},
        "config_sha256": sha256(json_bytes(config)),
        "arm": arm,
        "seed": seed,
        "backend": backend,
        "files": files,
    }


def checkpoint_payload(config, checkpoint_dir, arm, seed) -> tuple[dict, dict]:
    """Read and fingerprint a completed export without loading any model weights."""
    payload = {"config.json": json_bytes(config)}
    files = {}
    summary = None
    if arm != "base":
        if checkpoint_dir is None:
            message = "A completed --checkpoint-dir is required for adapted arms"
            raise ValueError(message)
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "summary.json").is_file():
            message = "evaluate only a completed checkpoint"
            raise ValueError(message)
        summary = read_json(checkpoint_dir / "summary.json")
        code_eval.read_adapter_spec(checkpoint_dir / "trainable")
        for source, target in _CHECKPOINT_FILES.items():
            path = checkpoint_dir / source
            if (
                source in ("training_receipt.json", "wandb_run.json")
                and not path.exists()
            ):
                continue
            record = file_record(path)
            raw = path.read_bytes()
            if sha256(raw) != record["sha256"]:
                message = "Checkpoint metadata changed while reading"
                raise ValueError(message)
            files[source] = record
            payload[target] = raw
        files[_WEIGHTS] = file_record(checkpoint_dir / _WEIGHTS)
    identity = _checkpoint_identity(config, arm, seed, summary, files)
    return identity, payload


def _inventory(directory: Path) -> dict:
    if not stat.S_ISDIR(directory.lstat().st_mode):
        message = "Generation must be a regular directory, not a symlink"
        raise ValueError(message)
    paths = sorted(directory.iterdir())
    size = 0
    for path in paths:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            message = "Generation must contain only flat regular files"
            raise ValueError(message)
        size += info.st_size
        if size > MAX_GENERATION_BYTES:
            message = "Generation exceeds the 1 GiB limit"
            raise ValueError(message)
    return {path.name: file_record(path) for path in paths if path.name != MANIFEST}


def _rebuild_manifest(directory: Path) -> dict:
    inventory = _inventory(directory)
    receipt = read_json(directory / RECEIPT)
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version",
            "stage",
            "status",
            "checkpoint",
            "benchmarks",
            "decoding",
            "smoke_limit",
        }
        or receipt["schema_version"] != 1
        or receipt["stage"] != "generation"
        or receipt["status"] != "completed"
    ):
        message = "A completed generation receipt is required"
        raise ValueError(message)
    names = receipt["benchmarks"]
    if (
        not isinstance(names, list)
        or not names
        or any(name not in code_eval.BENCHMARKS for name in names)
        or len(set(names)) != len(names)
    ):
        message = "Require distinct supported benchmarks"
        raise ValueError(message)
    decoding = receipt["decoding"]
    if decoding != code_eval.generation_settings(decoding["max_new_tokens"]):
        message = "Generation receipt differs from the fixed decoding protocol"
        raise ValueError(message)
    config = read_json(directory / "config.json")
    claimed = receipt["checkpoint"]
    files = {}
    required = {"config.json", RECEIPT}
    summary, spec = None, None
    if claimed["arm"] != "base":
        for source, target in _CHECKPOINT_FILES.items():
            if source not in claimed["files"]:
                if source in ("summary.json", "trainable/adapter_spec.json"):
                    message = "Missing checkpoint metadata"
                    raise ValueError(message)
                continue
            files[source] = inventory[target]
            required.add(target)
        summary = read_json(directory / "checkpoint-summary.json")
        spec = read_json(directory / "adapter_spec.json")
        weights = claimed["files"][_WEIGHTS]
        if (
            set(weights) != {"sha256", "size_bytes"}
            or not _SHA256.fullmatch(weights["sha256"])
            or type(weights["size_bytes"]) is not int
            or not 0 < weights["size_bytes"] <= MAX_GENERATION_BYTES
            or spec["config"]["rank"] != config["lora_rank"]
            or spec["config"]["alpha"] != config["lora_alpha"]
        ):
            message = "Invalid trainable export identity"
            raise ValueError(message)
        files[_WEIGHTS] = weights
    checkpoint = _checkpoint_identity(
        config, claimed["arm"], claimed["seed"], summary, files
    )
    if checkpoint != claimed:
        message = "Checkpoint receipt differs from saved checkpoint metadata"
        raise ValueError(message)
    benchmarks = {}
    model_info = None
    for name in names:
        required.update({f"{name}.jsonl", *(f"{name}-{s}" for s in _SAMPLE_FILES)})
        tasks, metadata = code_eval.load_benchmark(name, directory / f"{name}.jsonl")
        generation = read_json(directory / f"{name}-generation.json")
        if not isinstance(generation, dict) or set(generation) != {
            "format_version",
            "evalplus_version",
            "mode",
            "smoke_limit",
            "benchmark",
            "model",
            "generation",
            "task_ids",
            "tasks",
            "samples_sha256",
            "raw_samples_sha256",
        }:
            message = "Unexpected generation metadata fields"
            raise ValueError(message)
        code_eval.validate_generation(
            tasks,
            metadata,
            (directory / f"{name}-samples.jsonl").read_bytes(),
            generation,
            raw_samples=(directory / f"{name}-raw_samples.jsonl").read_bytes(),
        )
        model = generation["model"]
        if (
            set(model)
            != {
                "id",
                "revision",
                "arm",
                "adapter_config",
                "router_count",
                "merged_for_evaluation",
            }
            or {key: model[key] for key in ("id", "revision")} != checkpoint["model"]
            or model["arm"] != checkpoint["arm"]
            or model["merged_for_evaluation"] is not True
            or generation["smoke_limit"] != receipt["smoke_limit"]
            or generation["generation"]
            != {**decoding, "eos_token_ids": generation["generation"]["eos_token_ids"]}
            or (spec is not None and model["adapter_config"] != spec["config"])
            or (
                spec is not None
                and model["router_count"] != spec["metadata"]["router_count"]
            )
            or (model_info is not None and model != model_info)
        ):
            message = "Generation differs from checkpoint or decoding receipt"
            raise ValueError(message)
        model_info = model
        benchmarks[name] = {**generation, "expected_task_ids": list(tasks)}
    if set(inventory) != required:
        message = "Missing or unexpected generation artifact files"
        raise ValueError(message)
    return {
        "schema_version": 1,
        "kind": "moe-code-generation",
        "checkpoint": checkpoint,
        "benchmarks": benchmarks,
        "files": inventory,
    }


def seal_generation(directory, *, checkpoint, benchmarks, max_new_tokens, smoke_limit):
    """Seal only validated payloads; no caller-supplied manifest metadata overrides."""
    directory = Path(directory)
    write_json(
        directory / RECEIPT,
        {
            "schema_version": 1,
            "stage": "generation",
            "status": "completed",
            "checkpoint": checkpoint,
            "benchmarks": benchmarks,
            "decoding": code_eval.generation_settings(max_new_tokens),
            "smoke_limit": smoke_limit,
        },
    )
    manifest = _rebuild_manifest(directory)
    raw = json_bytes(manifest)
    if (
        sum(item["size_bytes"] for item in manifest["files"].values()) + len(raw)
        > MAX_GENERATION_BYTES
    ):
        message = "Generation exceeds the 1 GiB limit"
        raise ValueError(message)
    with (directory / MANIFEST).open("xb") as stream:
        stream.write(raw)
    return {**manifest, "manifest_sha256": sha256(raw)}


def validate_generation(directory, expected_manifest_sha256=None) -> dict:
    """Reconstruct the manifest from inert payloads and receipts, then compare it.

    The optional expected hash is useful for local inspection, but is mandatory
    in ``campaign.score_checkpoint``. The returned ``manifest_sha256`` is computed
    from the manifest file's exact bytes; it is not a self-referential file field.
    """
    directory = Path(directory)
    try:
        if expected_manifest_sha256 is not None and (
            not isinstance(expected_manifest_sha256, str)
            or not _SHA256.fullmatch(expected_manifest_sha256)
        ):
            message = "Expected a 64-hex manifest SHA256"
            raise ValueError(message)
        file_record(directory / MANIFEST)
        manifest = read_json(directory / MANIFEST)
        raw = (directory / MANIFEST).read_bytes()
        digest = sha256(raw)
        if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
            message = "Generation manifest SHA256 differs from the expected root"
            raise ValueError(message)
        expected = _rebuild_manifest(directory)
        if manifest != expected or raw != json_bytes(expected):
            message = "Generation manifest differs from saved payloads and receipts"
            raise ValueError(message)
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        message = "Malformed or missing generation handoff artifact"
        raise ValueError(message) from exc
    return {**expected, "manifest_sha256": digest}


def materialize_samples(directory, benchmark, output_dir) -> Path:
    """Make a fresh legacy-shaped input view for the unchanged Docker evaluator."""
    if benchmark not in code_eval.BENCHMARKS:
        message = "Unknown benchmark"
        raise ValueError(message)
    directory, output_dir = Path(directory), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename in _SAMPLE_FILES:
        copy_file(directory / f"{benchmark}-{filename}", output_dir / filename)
    return output_dir
