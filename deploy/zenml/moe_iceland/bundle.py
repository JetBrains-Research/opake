"""Bounded experiment outputs, without checkpoint or RNG-state materialization."""

import json
import math
import os
import re
import stat
import struct
from pathlib import Path
from urllib.parse import urlsplit

from .errors import refuse
from .settings import ARTIFACT_STORE, BUNDLE_ENTRY_LIMIT, BUNDLE_LIMITS, validate_image
from .source import sha256

_MANIFEST = "bundle-manifest.json"
_SPEC = "trainable/adapter_spec.json"
_WEIGHTS = "trainable/trainable.safetensors"
_GPU_LIMITS = {
    "gpu-probe-v1": 16 * 1024**2,
    "sft-smoke-v1": 1024**3,
    "sft-train-v1": 1024**3,
    "generate-v1": 1024**3,
}
_SFT_STEPS = {"sft-smoke-v1": 2, "sft-train-v1": 256}
_COMMON_FILES = {"runtime.json", "stage_receipt.json"}
_SFT_FILES = {
    "summary.json",
    "metrics.jsonl",
    "trainability.json",
    "execution.json",
    "wandb_run.json",
    "prefetch.json",
    "adapter_reload.json",
    _SPEC,
    _WEIGHTS,
}
_GEN_FILES = {"prefetch.json", "wandb_run.json"}
_GEN_JSON_LIMIT = 16 * 1024**2
_GEN_JSONL_LIMIT = 128 * 1024**2
_JSON_LIMIT = 1024**2
_JSONL_LIMIT = 16 * 1024**2
_JSONL_ROWS = 16384
_JSON_DEPTH = 32
_TRACKING_VALUE_LIMIT = 128
_MATRIX_NDIM = 2
_EXPERT_NDIM = 3
_HEADER_PREFIX_BYTES = struct.calcsize("<Q")
_OFFSET_COUNT = 2
_ADAPTER_KIND = "static_parametrization_expert_lora"
_ARMS = {
    "reference",
    "reference_aux",
    "dp",
    "dp_aux",
    "trl_reference",
    "trl_reference_aux",
}


def validate_bundle(
    directory: Path, profile: str, *, require_manifest: bool = True
) -> dict:
    """Validate a CPU export or a sealed GPU inventory, without loading training code.

    GPU inventories exclude the manifest itself. ``require_manifest=False`` permits
    pre-seal validation before ``deployment.json`` is added; an existing manifest
    is always verified. CPU smoke/pilot validation does not use manifests.
    Completed SFT and generation exports remain releasable without verified tracking
    uploads; probes still require successful W&B verification. Generation validates
    one full inert handoff, not code correctness or scoring-host isolation.
    """
    if profile in _GPU_LIMITS:
        try:
            return _validate_gpu_bundle(directory, profile, require_manifest)
        except OSError as error:
            refuse(f"Cannot inspect GPU output bundle: {error}")
    if profile not in BUNDLE_LIMITS:
        refuse(f"Unsupported output bundle profile: {profile}")
    if not directory.is_dir() or directory.is_symlink():
        refuse("The runner must produce a regular output directory.")
    files = {}
    total = 0
    entries = 0
    for current, dirs, names in os.walk(directory, followlinks=False):
        for name in [*dirs, *names]:
            entries += 1
            if entries > BUNDLE_ENTRY_LIMIT:
                refuse("Output bundle exceeds its entry cap.")
            path = Path(current) / name
            relative = path.relative_to(directory)
            if path.is_symlink():
                refuse(f"Symlink in output bundle: {relative}")
            if path.is_dir():
                if relative.parts[0] != "model":
                    refuse(f"Unexpected output directory: {relative}")
                continue
            if not path.is_file():
                refuse(f"Non-regular output: {relative}")
            if len(relative.parts) == 1:
                allowed = name in {
                    "metrics.jsonl",
                    "summary.json",
                    "checks.json",
                    "deployment.json",
                }
            else:
                allowed = relative.parts[0] == "model" and path.suffix in {
                    ".safetensors",
                    ".pt",
                    ".bin",
                    ".json",
                }
                allowed = allowed and not any(
                    token in str(relative).casefold()
                    for token in ("rng", "random", "optim", "checkpoint", "scheduler")
                )
            if not allowed:
                refuse(f"Not a weights-only/public output: {relative}")
            total += path.stat().st_size
            if total > BUNDLE_LIMITS[profile]:
                refuse("Output bundle exceeds its byte cap.")
            files[relative.as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    required = {"metrics.jsonl", "summary.json", "checks.json"}
    if not required <= files.keys() or not any(
        name.startswith("model/") and not name.endswith(".json") for name in files
    ):
        refuse("Missing metrics, summary, checks, or exported model weights.")
    for name in ("summary.json", "checks.json"):
        if files[name]["bytes"] > 1024**2:
            refuse(f"Oversized JSON metadata: {name}")
        value = json.loads((directory / name).read_text())
        if not isinstance(value, dict) or not value:
            refuse(f"Expected nonempty JSON object: {name}")
    with (directory / "metrics.jsonl").open() as stream:
        count = 0
        while line := stream.readline(1024**2 + 1):
            if len(line) > 1024**2 or not isinstance(json.loads(line), dict):
                refuse("Expected bounded JSON objects in metrics.jsonl.")
            count += 1
        if not count:
            refuse("The gate must perform an update, not just import packages.")
    return {"bytes": total, "files": files}


def seal_bundle(directory: Path, profile: str) -> dict:
    """Seal a complete GPU bundle with a format-1, SHA-256 payload inventory.

    Add the deployment receipt first. Repeating this operation verifies the seal;
    it never replaces an existing manifest or blesses changed contents.
    """
    if profile not in _GPU_LIMITS:
        refuse(f"Sealing is supported only for GPU artifact profiles: {profile}")
    inventory = validate_bundle(directory, profile, require_manifest=False)
    if "deployment.json" not in inventory["files"]:
        refuse("Missing deployment.json before sealing the GPU bundle.")
    path = directory / _MANIFEST
    if path.exists():
        return inventory
    raw = (
        json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > _JSON_LIMIT or inventory["bytes"] + len(raw) > _GPU_LIMITS[profile]:
        refuse("Output bundle including its manifest exceeds its byte cap.")
    directories = int(profile in _SFT_STEPS or profile == "generate-v1")
    if len(inventory["files"]) + directories + 1 > BUNDLE_ENTRY_LIMIT:
        refuse("Output bundle including its manifest exceeds its entry cap.")
    with path.open("xb") as stream:
        stream.write(raw)
    return validate_bundle(directory, profile)


def _gpu_inventory(directory: Path, profile: str, allowed: set[str]) -> dict:
    if not directory.is_dir() or directory.is_symlink():
        refuse("The runner must produce a regular output directory.")
    generation = profile == "generate-v1"
    allowed_directory = (
        "answers" if generation else "trainable" if profile in _SFT_STEPS else None
    )
    sizes = {}
    entries = total = 0

    def walk_error(error):
        raise error

    for current, dirs, names in os.walk(
        directory, followlinks=False, onerror=walk_error
    ):
        for name in sorted([*dirs, *names]):
            entries += 1
            if entries > BUNDLE_ENTRY_LIMIT:
                refuse("Output bundle exceeds its entry cap.")
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                refuse(f"Symlink in output bundle: {relative}")
            if stat.S_ISDIR(info.st_mode):
                if relative != allowed_directory:
                    refuse(f"Unexpected output directory: {relative}")
                continue
            if not stat.S_ISREG(info.st_mode):
                refuse(f"Non-regular output: {relative}")
            if info.st_nlink != 1:
                refuse(f"Hard link in output bundle: {relative}")
            answer = generation and relative.startswith("answers/")
            if relative not in allowed and not answer:
                refuse(
                    f"Unexpected GPU output file (not weights-only/public): {relative}"
                )
            total += info.st_size
            if total > _GPU_LIMITS[profile]:
                refuse("Output bundle exceeds its byte cap.")
            limit = _JSONL_LIMIT if relative.endswith(".jsonl") else _JSON_LIMIT
            if answer:
                limit = (
                    _GEN_JSONL_LIMIT if relative.endswith(".jsonl") else _GEN_JSON_LIMIT
                )
            if relative != _WEIGHTS and info.st_size > limit:
                refuse(f"Oversized JSON metadata: {relative}")
            sizes[relative] = info.st_size
    return {
        name: {"bytes": size, "sha256": sha256(directory / name)}
        for name, size in sorted(sizes.items())
    }


def _json_object(raw: bytes, name: str) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                refuse(f"Duplicate JSON key in {name}: {key}")
            value[key] = item
        return value

    def constant(value):
        refuse(f"Nonfinite JSON value in {name}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        refuse(f"Invalid JSON in {name}: {error}")
    _object(value, name)
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > _JSON_DEPTH:
            refuse(f"JSON nesting depth exceeds {_JSON_DEPTH}: {name}")
        if type(item) in (int, float):
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            if not finite:
                refuse(f"Nonfinite JSON metadata: {name}")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return value


def _read_json(directory: Path, name: str, *, limit: int = _JSON_LIMIT) -> dict:
    with (directory / name).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        refuse(f"Oversized JSON metadata: {name}")
    return _json_object(raw, name)


def _json_lines(directory: Path, name: str):
    with (directory / name).open("rb") as stream:
        count = total = 0
        while raw := stream.readline(_JSON_LIMIT + 1):
            count += 1
            total += len(raw)
            if len(raw) > _JSON_LIMIT or total > _JSONL_LIMIT or count > _JSONL_ROWS:
                refuse(f"Expected bounded JSON objects in {name}.")
            yield _json_object(raw, f"{name}:{count}")


def _object(value, name: str) -> dict:
    if not isinstance(value, dict) or not value:
        refuse(f"Expected nonempty JSON object: {name}")
    return value


def _expect(value: dict, key: str, expected, name: str) -> None:
    if key not in value or not _equal(value[key], expected):
        refuse(f"{name}: {key} must be {expected!r}.")


def _equal(value, expected) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and value.keys() == expected.keys()
            and all(_equal(value[key], item) for key, item in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _equal(left, right) for left, right in zip(value, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        return type(value) in (int, float) and value == expected
    return type(value) is type(expected) and value == expected


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        refuse(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value, name: str, *, positive: bool = False):
    if (
        type(value) not in (int, float)
        or (isinstance(value, float) and not math.isfinite(value))
        or (value <= 0 if positive else value < 0)
    ):
        refuse(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}."
        )
    return value


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_gpu_bundle(directory: Path, profile: str, require_manifest: bool) -> dict:
    probe = profile == "gpu-probe-v1"
    generation = profile == "generate-v1"
    required = _COMMON_FILES | (
        {"probe.json"} if probe else _GEN_FILES if generation else _SFT_FILES
    )
    allowed = required | {_MANIFEST, "deployment.json"}
    if not probe:
        allowed |= {"tracking_failure.json"}
    if profile in _SFT_STEPS:
        allowed |= {"test-records.jsonl"}
    files = _gpu_inventory(directory, profile, allowed)
    manifest_present = _MANIFEST in files
    if require_manifest and not manifest_present:
        refuse(f"Missing {_MANIFEST}; seal the GPU bundle before materialization.")
    if require_manifest or manifest_present:
        required |= {"deployment.json"}
    missing = required - files.keys()
    if missing:
        refuse(f"Missing required GPU output files: {', '.join(sorted(missing))}")
    files.pop(_MANIFEST, None)
    inventory = {
        "schema": 1,
        "profile": profile,
        "bytes": sum(item["bytes"] for item in files.values()),
        "files": files,
    }
    if manifest_present:
        manifest = _read_json(directory, _MANIFEST)
        recorded = _object(manifest.get("files"), _MANIFEST)
        if (
            type(manifest.get("schema")) is not int
            or type(manifest.get("bytes")) is not int
            or any(
                not isinstance(item, dict) or type(item.get("bytes")) is not int
                for item in recorded.values()
            )
            or manifest != inventory
        ):
            refuse(
                "Bundle manifest inventory/checksum mismatch; do not reseal changed files."
            )
    documents = {
        name: _read_json(
            directory,
            name,
            limit=_GEN_JSON_LIMIT
            if generation and name.startswith("answers/")
            else _JSON_LIMIT,
        )
        for name in files
        if name.endswith(".json")
    }
    stage = documents["stage_receipt.json"]
    for key, expected in {
        "profile": profile,
        "status": "completed",
        "cuda_verified": True,
    }.items():
        _expect(stage, key, expected, "stage_receipt.json")
    runtime = documents["runtime.json"]
    if "cuda" in runtime:
        cuda = _object(runtime["cuda"], "runtime.json cuda")
        _expect(cuda, "verified", True, "runtime.json cuda")
    if probe:
        _expect(stage, "wandb_verified", True, "stage_receipt.json")
        receipt = documents["probe.json"]
        _expect(receipt, "status", "completed", "probe.json")
        _expect(receipt, "wandb_verified", True, "probe.json")
        _expect(
            _object(receipt.get("cuda"), "probe.json cuda"),
            "verified",
            True,
            "probe.json cuda",
        )
    elif generation:
        _validate_generation(directory, documents)
    else:
        _validate_training(directory, profile, documents, files)
    if "deployment.json" in documents:
        _validate_deployment(documents, profile)
    return inventory


def _validate_deployment(documents: dict, profile: str) -> None:
    receipt = documents["deployment.json"]
    for key, expected in {
        "schema": 2,
        "runner_returncode": 0,
        "cuda_verified": True,
        "wandb_verified": documents["stage_receipt.json"]["wandb_verified"],
    }.items():
        _expect(receipt, key, expected, "deployment.json")
    plan = _object(receipt.get("plan"), "deployment.json plan")
    _expect(plan, "profile", profile, "deployment.json plan")
    if profile in _SFT_STEPS:
        summary = documents["summary.json"]
        _expect(receipt, "adapter_reload_verified", True, "deployment.json")
        _expect(plan, "arm", summary["arm"], "deployment.json plan")
        _expect(plan, "seed", summary["model_seed"], "deployment.json plan")
        _expect(
            plan,
            "backend",
            documents["stage_receipt.json"]["backend"],
            "deployment.json plan",
        )
    elif profile == "generate-v1":
        stage = documents["stage_receipt.json"]
        for key in (
            "arm",
            "backend",
            "benchmark",
            "config_sha256",
            "source_sha256",
            "image",
        ):
            _expect(plan, key, stage[key], "deployment.json plan")
        _expect(plan, "seed", stage["model_seed"], "deployment.json plan")
        _expect(receipt, "zenml_run_id", stage["zenml_run_id"], "deployment.json")
        _expect(
            _object(receipt.get("source"), "deployment.json source"),
            "source_sha256",
            stage["source_sha256"],
            "deployment.json source",
        )
        for key in ("checkpoint", "benchmark_input"):
            label = f"deployment.json plan {key}"
            reference = _object(plan.get(key), label)
            if (
                set(reference) != {"uri", "sha256"}
                or not _hash(reference.get("sha256"))
                or not isinstance(reference.get("uri"), str)
                or not re.fullmatch(
                    re.escape(ARTIFACT_STORE) + r"/[A-Za-z0-9_./%-]+", reference["uri"]
                )
            ):
                refuse(
                    f"{label}: require a checksummed reference in the approved artifact store."
                )
        benchmark = documents[f"answers/{stage['benchmark']}-generation.json"][
            "benchmark"
        ]
        _expect(
            plan["benchmark_input"],
            "sha256",
            benchmark["sha256"],
            "deployment.json plan benchmark_input",
        )


def _validate_generation(directory: Path, documents: dict) -> None:
    from examples.moe_privacy import code_eval, handoff

    stage = documents["stage_receipt.json"]
    label = "stage_receipt.json"
    for key, expected in {"stage": "generation", "model_seed": 0, "steps": 0}.items():
        _expect(stage, key, expected, label)
    arm = stage.get("arm")
    if not isinstance(arm, str) or arm not in _ARMS:
        refuse(f"{label}: unknown generation arm.")
    backend = "trl" if arm.startswith("trl_") else "opaque"
    _expect(stage, "backend", backend, label)
    for key in ("config_sha256", "source_sha256", "generation_manifest_sha256"):
        if not _hash(stage.get(key)):
            refuse(f"{label}: require a SHA-256 {key}.")
    for key in ("image", "zenml_run_id"):
        if not isinstance(stage.get(key), str) or not stage[key].strip():
            refuse(f"{label}: require {key}.")
    validate_image(stage["image"])
    benchmark = stage.get("benchmark")
    if not isinstance(benchmark, str) or benchmark not in code_eval.BENCHMARKS:
        refuse(f"{label}: require one supported generation benchmark.")
    runtime = documents["runtime.json"]
    _expect(
        _object(runtime.get("cuda"), "runtime.json cuda"),
        "verified",
        True,
        "runtime.json cuda",
    )
    if "backend" in runtime:
        _expect(runtime, "backend", backend, "runtime.json")
    try:
        generated = handoff.validate_generation(
            directory / "answers",
            expected_manifest_sha256=stage["generation_manifest_sha256"],
        )
    except (ValueError, RecursionError) as error:
        refuse(f"Invalid generation handoff: {error}")
    if set(generated["benchmarks"]) != {benchmark}:
        refuse("Generation must contain exactly the one planned benchmark.")
    generation = generated["benchmarks"][benchmark]
    _expect(generation, "mode", "full", "generation benchmark")
    _expect(generation, "smoke_limit", None, "generation benchmark")
    _expect(generation["generation"], "max_new_tokens", 512, "generation decoding")
    if len(generation["task_ids"]) != code_eval.BENCHMARKS[benchmark]["count"]:
        refuse("Generation must cover the full benchmark task set.")
    for key, expected in {
        "arm": arm,
        "seed": stage["model_seed"],
        "backend": "trl.SFTTrainer"
        if backend == "trl"
        else "opaque.transformers.DPTrainer",
    }.items():
        _expect(generated["checkpoint"], key, expected, "generation checkpoint")
    config = documents["answers/config.json"]
    prefetch = documents["prefetch.json"]
    _expect(prefetch, "status", "completed", "prefetch.json")
    for kind in ("model", "dataset"):
        identifier, revision = config.get(f"{kind}_id"), config.get(f"{kind}_revision")
        if not isinstance(identifier, str) or not identifier.strip():
            refuse(f"answers/config.json: require {kind}_id.")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            refuse(f"answers/config.json: require pinned {kind}_revision.")
        for key in (f"{kind}_id", f"{kind}_revision"):
            _expect(prefetch, key, config[key], "prefetch.json")
    _validate_tracking(documents)


def _validate_training(
    directory: Path, profile: str, documents: dict, files: dict
) -> None:
    summary = documents["summary.json"]
    _expect(summary, "schema_version", 1, "summary.json")
    _expect(summary, "status", "completed", "summary.json")
    arm = summary.get("arm")
    if not isinstance(arm, str) or arm not in _ARMS:
        refuse("summary.json: unknown SFT arm.")
    seed = _integer(summary.get("model_seed"), "summary.json model_seed")
    if seed >= 2**32:
        refuse("summary.json model_seed must be below 2**32.")
    native = arm.startswith("trl_")
    backend = "trl" if native else "opaque"
    _expect(
        summary,
        "experiment",
        "native_trl_moe_sft" if native else "pretrained_moe_sft",
        "summary.json",
    )
    config = _object(summary.get("config"), "summary.json config")
    steps = _SFT_STEPS[profile]
    _expect(config, "steps", steps, "summary.json config")
    stage = documents["stage_receipt.json"]
    for key, expected in {
        "backend": backend,
        "arm": arm,
        "model_seed": seed,
        "steps": steps,
    }.items():
        _expect(stage, key, expected, "stage_receipt.json")
    _validate_data(summary, config)
    _validate_privacy(summary, config, arm, native, steps)
    _validate_execution(summary, config, documents["execution.json"], arm, native)
    spec = _validate_adapter_spec(summary, documents)
    _validate_safetensors(directory / _WEIGHTS, spec, files[_WEIGHTS]["bytes"])
    _validate_metrics(directory, summary, steps)
    prefetch = documents["prefetch.json"]
    _expect(prefetch, "status", "completed", "prefetch.json")
    for key in ("model_id", "model_revision", "dataset_id", "dataset_revision"):
        _expect(prefetch, key, config[key], "prefetch.json")
    reload = documents["adapter_reload.json"]
    _expect(reload, "status", "completed", "adapter_reload.json")
    _expect(reload, "verified", True, "adapter_reload.json")
    _expect(reload, "tensor_count", len(spec["tensors"]), "adapter_reload.json")
    _validate_tracking(documents)
    if "test-records.jsonl" in files:
        _validate_test_records(directory, summary)


def _validate_data(summary: dict, config: dict) -> None:
    data = _object(summary.get("data"), "summary.json data")
    _expect(data, "kind", "public_code_instruction_completion", "summary.json data")
    _expect(data, "packing", False, "summary.json data")
    for kind in ("model", "dataset"):
        identifier, revision = config.get(f"{kind}_id"), config.get(f"{kind}_revision")
        if not isinstance(identifier, str) or not identifier.strip():
            refuse(f"summary.json config: require {kind}_id.")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            refuse(f"summary.json config: require pinned {kind}_revision.")
    for key in ("dataset_id", "dataset_revision"):
        _expect(data, key, config[key], "summary.json data")
    seen = set()
    for name in ("train", "validation", "test", "diagnostic"):
        label = f"summary.json data partition {name}"
        part = _object(data.get(name), label)
        count = _integer(
            config.get(f"{name}_sequences"), label, int(name in {"train", "validation"})
        )
        _expect(part, "records", count, label)
        if not _hash(part.get("sha256")):
            refuse(f"{label}: require a SHA-256 partition digest.")
        _expect(data, f"{name}_sha256", part["sha256"], label)
        hashes = part.get("prompt_hashes")
        if (
            not isinstance(hashes, list)
            or len(hashes) != count
            or not all(_hash(value) for value in hashes)
            or len(set(hashes)) != count
            or seen.intersection(hashes)
        ):
            refuse(f"{label}: require unique, disjoint public prompt hashes.")
        seen.update(hashes)


def _validate_privacy(
    summary: dict, config: dict, arm: str, native: bool, steps: int
) -> None:
    privacy = _object(summary.get("privacy"), "summary.json privacy")
    label = "summary.json privacy"
    private = arm in {"dp", "dp_aux"}
    _expect(privacy, "private", private, label)
    _expect(privacy, "steps", steps, label)
    _expect(privacy, "raw_training_telemetry_released", False, label)
    noise = _number(
        privacy.get("noise_multiplier"), f"{label} noise_multiplier", positive=private
    )
    if private:
        epsilon = _number(privacy.get("epsilon"), f"{label} epsilon")
        target = _number(
            config.get("target_epsilon"), f"{label} target_epsilon", positive=True
        )
        delta = _number(config.get("delta"), f"{label} delta", positive=True)
        _expect(privacy, "target_epsilon", target, label)
        _expect(privacy, "delta", delta, label)
        if delta >= 1 or epsilon > target:
            refuse(
                "summary.json privacy: delta must be below 1 and epsilon <= target_epsilon."
            )
        _expect(privacy, "unit", "one_preprocessed_prompt_answer_pair", label)
        _expect(privacy, "adjacency", "add_remove", label)
    else:
        for key in ("epsilon", "delta", "target_epsilon"):
            _expect(privacy, key, None, label)
        if noise != 0:
            refuse("summary.json privacy: nonprivate noise_multiplier must be zero.")
    batch = _integer(
        config.get("expected_batch_size"), f"{label} expected_batch_size", 1
    )
    if batch > config["train_sequences"]:
        refuse("summary.json privacy: sample_rate exceeds one.")
    if native:
        _expect(privacy, "sample_rate", None, label)
    else:
        _number(privacy.get("sample_rate"), f"{label} sample_rate", positive=True)
        _expect(privacy, "sample_rate", batch / config["train_sequences"], label)
    _expect(privacy, "load_release", arm in {"dp_aux", "reference_aux"}, label)
    if arm == "dp_aux":
        ratio = _number(
            config.get("load_noise_ratio"), f"{label} load_noise_ratio", positive=True
        )
        _number(
            privacy.get("load_noise_ratio"), f"{label} load_noise_ratio", positive=True
        )
        _expect(privacy, "load_noise_ratio", ratio, label)
    else:
        _expect(privacy, "load_noise_ratio", None, label)
    _expect(
        privacy,
        "balancing",
        {
            "dp_aux": "lagged_noisy",
            "reference_aux": "lagged_unnoised",
            "trl_reference_aux": "native_current_batch",
        }.get(arm, "off"),
        label,
    )


def _validate_execution(
    summary: dict, config: dict, receipt: dict, arm: str, native: bool
) -> None:
    execution = _object(summary.get("execution"), "summary.json execution")
    device = execution.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        refuse("summary.json execution.device must record CUDA training.")
    if any(
        key not in execution or not _equal(execution[key], value)
        for key, value in receipt.items()
    ):
        refuse("execution.json differs from summary.json execution metadata.")
    if not native:
        if not _equal(receipt, execution):
            refuse(
                "execution.json must preserve the complete private/opaque execution metadata."
            )
        return
    label = "execution.json native TRL"
    batch = _integer(config.get("microbatch_size"), f"{label} microbatch_size", 1)
    effective = config["expected_batch_size"]
    if effective % batch:
        refuse("execution.json requires integral native gradient accumulation.")
    coefficient = _number(
        config.get("router_aux_loss_coef"), f"{label} router_aux_loss_coef"
    )
    expected = {
        "trainer_backend": "trl.SFTTrainer",
        "opaque_patches": False,
        "physical_batch_size": batch,
        "effective_batch_size": effective,
        "gradient_accumulation_steps": effective // batch,
        "sampling_mode": "shuffled_without_replacement",
        "router_aux_loss_coef": coefficient if arm == "trl_reference_aux" else 0.0,
        "aux_loss_scope": "native_current_physical_batch",
        "aux_loss_accumulation": "mean_of_physical_batch_aux_losses",
        "global_batch_aux_equivalent": False,
        "task_loss_reduction": "mean_of_physical_batch_completion_token_means",
    }
    for key, value in expected.items():
        _expect(receipt, key, value, label)


def _names(value, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(name, str) and name for name in value)
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        refuse(f"{label}: require a sorted, nonempty list of unique names.")
    return value


def _shape(value, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= _EXPERT_NDIM
        or any(type(size) is not int or size <= 0 for size in value)
        or math.prod(value) > 1024**3 // 4
    ):
        refuse(f"Invalid bounded tensor shape: {label}")
    return value


def _validate_adapter_spec(summary: dict, documents: dict) -> dict:
    spec = documents[_SPEC]
    _expect(spec, "format_version", 1, _SPEC)
    _expect(spec, "adapter_kind", _ADAPTER_KIND, _SPEC)
    config = _object(spec.get("config"), f"{_SPEC} config")
    metadata = _object(spec.get("metadata"), f"{_SPEC} metadata")
    if not _equal(metadata, documents["trainability.json"]) or not _equal(
        metadata, summary.get("trainability")
    ):
        refuse(
            "Adapter metadata differs from trainability.json or summary.json trainability."
        )
    _expect(metadata, "adapter_kind", _ADAPTER_KIND, _SPEC)
    rank = _integer(config.get("rank"), "adapter rank", 1)
    alpha = _number(config.get("alpha"), "adapter alpha", positive=True)
    for key, value in {"rank": rank, "alpha": alpha}.items():
        _expect(metadata, key, value, "adapter metadata")
        _expect(summary["config"], f"lora_{key}", value, "summary.json config")
    geometry = _object(metadata.get("geometry"), "adapter geometry")
    layers = _integer(geometry.get("num_layers"), "adapter num_layers", 1)
    experts = _integer(geometry.get("num_experts"), "adapter num_experts", 1)
    top_k = _integer(geometry.get("top_k"), "adapter top_k", 1)
    if top_k > experts:
        refuse("Adapter top_k exceeds num_experts.")
    targets = _names(config.get("target_parameters"), "adapter target_parameters")
    _expect(metadata, "target_parameters", targets, "adapter metadata")
    if len(targets) != 2 * layers:
        refuse("Adapter target_parameters differ from the expert-LoRA export scope.")
    prefixes = [f"model.layers.{index}.mlp" for index in range(layers)]
    expected_targets = sorted(
        f"{prefix}.experts.{projection}"
        for prefix in prefixes
        for projection in ("gate_up_proj", "down_proj")
    )
    if targets != expected_targets:
        refuse(
            "Adapter target_parameters include tensors outside the expert-LoRA scope."
        )
    routers = _names(metadata.get("router_names"), "adapter router_names")
    adapters = _names(
        metadata.get("expert_adapter_names"), "adapter expert_adapter_names"
    )
    expected_adapters = sorted(
        f"{module}.parametrizations.{projection}.0.{factor}"
        for target in targets
        for module, projection in [target.rsplit(".", 1)]
        for factor in ("lora_A", "lora_B")
    )
    if (
        routers != sorted(f"{prefix}.gate.weight" for prefix in prefixes)
        or adapters != expected_adapters
    ):
        refuse(
            "Adapter/router tensor names differ from the expert-LoRA/router export scope."
        )
    _expect(metadata, "expert_adapter_count", len(adapters), "adapter metadata")
    _expect(metadata, "router_count", len(routers), "adapter metadata")
    tensors = _object(spec.get("tensors"), "adapter tensors")
    if tensors.keys() != set(routers + adapters):
        refuse("Adapter tensor keys differ from the declared trainable scope.")
    for name, tensor in tensors.items():
        _object(tensor, f"adapter tensor {name}")
        _expect(tensor, "dtype", "torch.float32", f"adapter tensor {name}")
        _shape(tensor.get("shape"), name)
    for prefix in prefixes:
        router = tensors[f"{prefix}.gate.weight"]["shape"]
        down = f"{prefix}.experts.parametrizations.down_proj.0"
        gate = f"{prefix}.experts.parametrizations.gate_up_proj.0"
        down_a = tensors[f"{down}.lora_A"]["shape"]
        if len(router) != _MATRIX_NDIM or len(down_a) != _EXPERT_NDIM:
            refuse("Adapter/router tensors have invalid dimensions.")
        hidden, intermediate = router[1], down_a[2]
        expected = {
            f"{prefix}.gate.weight": [experts, hidden],
            f"{gate}.lora_A": [experts, rank, hidden],
            f"{gate}.lora_B": [experts, 2 * intermediate, rank],
            f"{down}.lora_A": [experts, rank, intermediate],
            f"{down}.lora_B": [experts, hidden, rank],
        }
        for name, shape in expected.items():
            _expect(tensors[name], "shape", shape, f"adapter tensor {name}")
    parameters = sum(math.prod(tensor["shape"]) for tensor in tensors.values())
    _expect(metadata, "trainable_parameters", parameters, "adapter metadata")
    _integer(metadata.get("total_parameters"), "adapter total_parameters", parameters)
    return spec


def validate_weights(directory: Path, summary: dict) -> dict:
    """Validate a recovered weights-only export without fabricating new GPU receipts."""
    spec_path = directory / _SPEC
    weight_path = directory / _WEIGHTS
    for path in (spec_path, weight_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            refuse("Expected regular weights-only export files.")
    if spec_path.stat().st_size > _JSON_LIMIT or weight_path.stat().st_size > 1024**3:
        refuse("Weights-only export exceeds its bounded allowance.")
    spec = _json_object(spec_path.read_bytes(), _SPEC)
    spec = _validate_adapter_spec(
        summary, {_SPEC: spec, "trainability.json": summary.get("trainability")}
    )
    _validate_safetensors(weight_path, spec, weight_path.stat().st_size)
    return spec


def _validate_safetensors(path: Path, spec: dict, size: int) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(_HEADER_PREFIX_BYTES)
        if len(prefix) != _HEADER_PREFIX_BYTES:
            refuse("Truncated safetensors header length.")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= _JSON_LIMIT or _HEADER_PREFIX_BYTES + length > size:
            refuse("Invalid or oversized safetensors header.")
        raw = stream.read(length)
    if len(raw) != length or raw[:1] != b"{":
        refuse("Invalid safetensors JSON header.")
    header = _json_object(raw, _WEIGHTS)
    if "__metadata__" in header:
        metadata = header.pop("__metadata__")
        if not isinstance(metadata, dict) or any(
            not isinstance(value, str) for value in metadata.values()
        ):
            refuse("Invalid safetensors string metadata.")
    if header.keys() != spec["tensors"].keys():
        refuse("safetensors tensor keys differ from adapter_spec.json.")
    spans = []
    for name, tensor in header.items():
        _object(tensor, f"safetensors tensor {name}")
        if set(tensor) != {"dtype", "shape", "data_offsets"}:
            refuse(f"Invalid safetensors tensor header: {name}")
        _expect(tensor, "dtype", "F32", f"safetensors tensor {name}")
        shape = _shape(tensor.get("shape"), f"safetensors {name}")
        _expect(
            tensor,
            "shape",
            spec["tensors"][name]["shape"],
            f"safetensors tensor {name}",
        )
        offsets = tensor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != _OFFSET_COUNT
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * 4
        ):
            refuse(f"Invalid safetensors tensor byte offsets: {name}")
        spans.append(tuple(offsets))
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            refuse("safetensors tensor layout contains gaps or overlaps.")
        end = stop
    if end != size - _HEADER_PREFIX_BYTES - length:
        refuse("safetensors tensor data is truncated or contains trailing bytes.")


def _validate_metrics(directory: Path, summary: dict, steps: int) -> None:
    first = last = None
    previous = -1
    for row in _json_lines(directory, "metrics.jsonl"):
        step = _integer(row.get("step"), "metrics.jsonl step")
        if not previous < step <= steps:
            refuse(
                "metrics.jsonl steps must increase through the full training horizon."
            )
        _number(row.get("eval_loss"), "metrics.jsonl eval_loss")
        _number(row.get("elapsed_seconds"), "metrics.jsonl elapsed_seconds")
        if first is None:
            first = row
        last, previous = row, step
    if first is None or first["step"] != 0 or previous != steps:
        refuse(
            "metrics.jsonl must include initial evaluation and all completed training steps."
        )
    if not _equal(first, summary.get("initial")) or not _equal(
        last, summary.get("final")
    ):
        refuse("metrics.jsonl initial/final rows differ from summary.json metrics.")


def _validate_tracking(documents: dict) -> None:
    uploaded = _validate_tracker(
        documents["wandb_run.json"], documents.get("tracking_failure.json")
    )
    verified = documents["stage_receipt.json"].get("wandb_verified")
    if type(verified) is not bool:
        refuse("stage_receipt.json: wandb_verified must be boolean.")
    if verified and not uploaded:
        refuse(
            "stage_receipt.json: wandb_verified requires a completed W&B upload "
            "without tracking failure."
        )


def _validate_tracking_failure(receipt: dict) -> None:
    label = "tracking_failure.json"
    if set(receipt) != {"operation", "exception_type"}:
        refuse(f"{label}: require only operation and exception_type.")
    for key, value in receipt.items():
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > _TRACKING_VALUE_LIMIT
        ):
            refuse(
                f"{label}: {key} must be a nonempty string of at most 128 characters."
            )
    if receipt["operation"] not in {
        "init",
        "log_metrics",
        "log_progress",
        "complete",
        "finish",
    }:
        refuse(f"{label}: unknown tracking operation.")


def _validate_tracker(receipt: dict, failure: dict | None = None) -> bool:
    label = "wandb_run.json"
    _expect(receipt, "status", "completed", label)
    upload_status = receipt.get("upload_status")
    if not isinstance(upload_status, str) or upload_status not in {
        "pending",
        "completed",
        "offline",
        "failed",
    }:
        refuse(f"{label}: require a valid upload_status.")
    if failure is not None:
        _validate_tracking_failure(failure)
    if (upload_status == "failed") != (failure is not None):
        refuse(f"{label}: upload_status and tracking_failure.json must agree.")
    for key in ("entity", "project", "run_id"):
        value = receipt.get(key)
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", value
        ):
            refuse(f"{label}: require a public {key}.")
    base = receipt.get("base_url")
    if not isinstance(base, str):
        refuse(f"{label}: require an HTTPS base_url.")
    parsed = urlsplit(base)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        refuse(f"{label}: require a credential-free HTTPS base_url.")
    expected = f"{base.rstrip('/')}/{receipt['entity']}/{receipt['project']}/runs/{receipt['run_id']}"
    if receipt.get("url") is None and upload_status == "failed" and failure is not None:
        return False
    _expect(receipt, "url", expected, label)
    return upload_status == "completed" and failure is None


def _validate_test_records(directory: Path, summary: dict) -> None:
    label = "test-records.jsonl"
    _expect(summary, "test_evaluated", True, label)
    count = summary["data"]["test"]["records"]
    test = _object(summary.get("test"), f"{label} summary.test")
    _number(test.get("eval_records"), f"{label} eval_records", positive=True)
    if test["eval_records"] != count:
        refuse(f"{label}: summary test count differs from the public partition.")
    fields = {
        "record_index",
        "nll_sum",
        "nll",
        "supervised_tokens",
        "attended_tokens",
        "teacher_forced_correct_tokens",
    }
    length = _integer(
        summary["config"].get("sequence_length"), f"{label} sequence_length", 1
    )
    rows = 0
    for index, row in enumerate(_json_lines(directory, label)):
        if set(row) != fields:
            refuse(f"{label}: unexpected public evaluation record fields.")
        _expect(row, "record_index", index, label)
        tokens = _integer(row["supervised_tokens"], f"{label} supervised_tokens", 1)
        attended = _integer(row["attended_tokens"], f"{label} attended_tokens", tokens)
        correct = _integer(
            row["teacher_forced_correct_tokens"], f"{label} correct tokens"
        )
        loss = _number(row["nll_sum"], f"{label} nll_sum")
        mean = _number(row["nll"], f"{label} nll")
        if (
            correct > tokens
            or attended > length
            or not math.isclose(mean, loss / tokens, rel_tol=1e-12, abs_tol=1e-12)
        ):
            refuse(f"{label}: inconsistent public evaluation statistics.")
        rows += 1
    if rows != count or rows == 0:
        refuse(f"{label}: missing public test records.")
