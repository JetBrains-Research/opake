"""Bounded weights-only transport for validated current or recovered SFT exports."""

import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from .bundle import validate_weights
from .errors import refuse
from .source import digest_json, sha256

LIMIT = 1024**3
ENTRIES = 256
JSON_LIMIT = 1024**2
FULL_STEPS = 256
PAYLOAD = {
    "summary.json",
    "trainable/adapter_spec.json",
    "trainable/trainable.safetensors",
}
RECOVERY_MANIFEST = "checkpoint-manifest.json"
ARCHIVE_FILES = PAYLOAD | {
    RECOVERY_MANIFEST,
    "bundle-manifest.json",
    "metrics.jsonl",
    "trainability.json",
    "execution.json",
    "wandb_run.json",
    "tracking_failure.json",
    "runtime.json",
    "prefetch.json",
    "adapter_reload.json",
    "stage_receipt.json",
    "deployment.json",
    "test-records.jsonl",
}


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > JSON_LIMIT:
        refuse("Checkpoint metadata is missing, unsafe or oversized.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        refuse("Expected checkpoint metadata object.")
    json.dumps(value, allow_nan=False)
    return value


def validate_export(directory: Path, *, config: dict, arm: str, seed: int) -> dict:
    """Require actual completed training; a W&B state alone cannot establish this."""
    for name in PAYLOAD:
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            refuse("Expected regular weights-only checkpoint files.")
    if sum((directory / name).stat().st_size for name in PAYLOAD) > LIMIT:
        refuse("Checkpoint exceeds the 1Gi allowance.")
    summary = _read_json(directory / "summary.json")
    if (
        summary.get("status") != "completed"
        or summary.get("config") != config
        or summary.get("arm") != arm
        or summary.get("model_seed") != seed
        or summary.get("privacy", {}).get("steps") != FULL_STEPS
        or not str(summary.get("execution", {}).get("device", "")).startswith("cuda")
    ):
        refuse(
            "Checkpoint lacks matching full completed CUDA training evidence; do not retrain automatically."
        )
    validate_weights(directory, summary)
    privacy = summary["privacy"]
    private = arm in {"dp", "dp_aux"}
    if privacy.get("private") is not private or privacy.get("load_release") is not (
        arm in {"dp_aux", "reference_aux"}
    ):
        refuse("Checkpoint privacy mechanism differs from its arm.")
    if private and (
        type(privacy.get("epsilon")) not in (int, float)
        or not 0 < privacy["epsilon"] <= config["target_epsilon"]
        or privacy.get("delta") != config["delta"]
        or privacy.get("target_epsilon") != config["target_epsilon"]
        or type(privacy.get("noise_multiplier")) not in (int, float)
        or privacy["noise_multiplier"] <= 0
        or (
            arm == "dp_aux"
            and privacy.get("load_noise_ratio") != config["load_noise_ratio"]
        )
    ):
        refuse("Checkpoint does not satisfy its reported privacy budget.")
    if arm.startswith("trl_") and (
        summary["execution"].get("opaque_patches") is not False
        or privacy.get("epsilon") is not None
        or privacy.get("noise_multiplier") != 0
    ):
        refuse("Native checkpoint is not a genuine non-private TRL export.")
    from .gpu_runner import PARTITIONS

    for name, expected in PARTITIONS.items():
        if summary.get("data", {}).get(name, {}).get("sha256") != expected:
            refuse(f"Recovered checkpoint has a different {name} partition.")
    return {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": sha256(directory / name),
        }
        for name in sorted(PAYLOAD)
    }


def package_export(
    directory: Path, destination: Path, *, config: dict, arm: str, seed: int = 0
) -> dict:
    """Preserve original receipts; create a separate, checksummed transport artifact."""
    files = validate_export(directory, config=config, arm=arm, seed=seed)
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(PAYLOAD):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / name, path)
        if sha256(path) != files[name]["sha256"]:
            refuse(
                "Checkpoint changed during transport packaging; preserve the originals."
            )
    manifest = {
        "schema": 1,
        "kind": "recovered-weights-only",
        "files": files,
        "config_sha256": digest_json(config),
        "arm": arm,
        "seed": seed,
    }
    with (destination / RECOVERY_MANIFEST).open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def extract_archive(archive: Path, destination: Path, *, expected_sha256: str) -> None:
    """Extract bounded regular files, never links/devices or paths supplied by a tar."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or archive.stat().st_size > LIMIT
        or sha256(archive) != expected_sha256
    ):
        refuse("Checkpoint archive checksum or size is invalid.")
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    size = 0
    with tarfile.open(archive, "r|gz") as stream:
        for index, item in enumerate(stream):
            path = PurePosixPath(item.name)
            name = path.as_posix()
            if (
                index >= ENTRIES
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or name in seen
            ):
                refuse("Unsafe or duplicate checkpoint archive entry.")
            seen.add(name)
            if item.isdir() and name in {".", "trainable"}:
                continue
            if (
                name not in ARCHIVE_FILES
                or not item.isfile()
                or item.sparse
                or item.size < 0
            ):
                refuse("Checkpoint archive contains unsupported files or links.")
            size += item.size
            if size > LIMIT or (name.endswith(".json") and item.size > JSON_LIMIT):
                refuse("Unpacked checkpoint exceeds its bounded allowance.")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    if not seen >= PAYLOAD:
        refuse("Checkpoint archive is incomplete.")


def download_object(reference: dict, destination: Path, *, limit: int = LIMIT) -> Path:
    """Download one fixed object with a streaming size cap and independent checksum."""
    from zenml.io import fileio

    if not re.fullmatch(
        r"gs://gke-dev-dws-jbr-zenml/[A-Za-z0-9_./%-]+", reference["uri"]
    ):
        refuse(
            "Checkpoint must be an object in the approved artifact bucket, without URL credentials."
        )
    size = 0
    with (
        fileio.open(reference["uri"], "rb") as source,
        destination.open("xb") as output,
    ):
        while chunk := source.read(1024**2):
            size += len(chunk)
            if size > limit:
                refuse("Artifact download exceeds its fixed allowance.")
            output.write(chunk)
    if sha256(destination) != reference["sha256"]:
        refuse("Downloaded artifact checksum differs from the approved input.")
    return destination


def download_checkpoint(plan: dict, directory: Path) -> Path:
    """Download on the trusted pipeline side; pass only local weights to generation."""
    reference = plan["checkpoint"]
    archive = download_object(reference, directory / "checkpoint.tar.gz")
    destination = directory / "checkpoint"
    extract_archive(archive, destination, expected_sha256=reference["sha256"])
    if (destination / RECOVERY_MANIFEST).exists():
        manifest = _read_json(destination / RECOVERY_MANIFEST)
        config = _read_json(destination / "summary.json")["config"]
        actual = validate_export(
            destination, config=config, arm=plan["arm"], seed=plan["seed"]
        )
        if manifest != {
            "schema": 1,
            "kind": "recovered-weights-only",
            "files": actual,
            "config_sha256": digest_json(config),
            "arm": plan["arm"],
            "seed": plan["seed"],
        }:
            refuse("Recovered checkpoint manifest differs from its verified files.")
    else:
        from .bundle import validate_bundle

        validate_bundle(destination, "sft-train-v1")
    return destination


def download_benchmark(plan: dict, directory: Path) -> Path:
    """Reuse the frozen benchmark bytes instead of resolving a mutable release URL."""
    from examples.moe_privacy.code_eval import load_benchmark

    directory.mkdir(exist_ok=False)
    path = download_object(
        plan["benchmark_input"],
        directory / f"{plan['benchmark']}.jsonl",
        limit=128 * 1024**2,
    )
    load_benchmark(plan["benchmark"], path)
    return directory
