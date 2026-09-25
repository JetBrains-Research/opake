"""Versioned, fail-closed policy for the Iceland coefficient experiment."""

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .errors import refuse

PROJECT_NAME = "models-rd"
PROJECT_ID = "357b8eb4-6c65-40d1-a4de-ea48a3279288"
DEPLOYMENT_REFERENCE = "0e42b111d621256cc84d82f2b8000a939315db91"
GPU_PROFILES = {
    # Include scheduling headroom: Iceland may take several minutes to place a GPU.
    "gpu-probe-v1": 1800,
    "sft-smoke-v1": 3600,
    "sft-train-v1": 8 * 3600,
    "generate-v1": 4 * 3600,
}
CONFIGS = {
    "sft_balancing.json": "opaque",
    "sft_balancing_aux02.json": "opaque",
    "sft_balancing_aux0001_ratio1.json": "opaque",
    "sft_balancing_aux0001_ratio01.json": "opaque",
    "sft_balancing_aux02_ratio01.json": "opaque",
    "sft_trl_baselines.json": "trl",
    "sft_trl_aux0001.json": "trl",
}
ARMS = {
    "opaque": {"reference", "reference_aux", "dp", "dp_aux"},
    "trl": {"trl_reference", "trl_reference_aux"},
}
RESOURCE_PROFILE = "one-80gb-gpu-v1"
SCHEDULER = "gpu-binpack-scheduler"
TOLERATIONS = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
SCHEDULING_SECONDS = 900
MIN_TIMEOUT_SECONDS = 120
MIN_GPU_MEMORY = 80_000_000_000
# CLUSTER.md allows up to 100Gi ephemeral per pod. Keep below that ceiling.
SCRATCH_BYTES = 64 * 1024**3
CACHE_BYTES = 48 * 1024**3


def gpu_resources(profile: str = "sft-train-v1") -> dict:
    """Reserve one accelerator, with bounded host memory and node-backed storage."""
    # Probe only checks CUDA/W&B and does not prefetch Mellum; keep scratch small so
    # scheduling is less likely to fail on ephemeral-storage when GPUs are scarce.
    if profile == "gpu-probe-v1":
        ephemeral = ("16Gi", "24Gi")
    elif profile == "sft-smoke-v1":
        ephemeral = ("48Gi", "64Gi")
    else:
        ephemeral = ("64Gi", "100Gi")
    return {
        "requests": {
            "cpu": "4",
            "memory": "64Gi",
            "nvidia.com/gpu": "1",
            "ephemeral-storage": ephemeral[0],
        },
        "limits": {
            "cpu": "4",
            "memory": "64Gi",
            "nvidia.com/gpu": "1",
            "ephemeral-storage": ephemeral[1],
        },
    }


def utc_deadline(value: str) -> datetime:
    """Require an explicit timezone instead of interpreting a local clock silently."""
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        refuse("Provide a timezone-qualified deadline_utc.")
    if result.tzinfo is None:
        refuse("Provide a timezone-qualified deadline_utc.")
    return result.astimezone(UTC)


def validate_authorization(plan: dict, *, now: datetime | None = None) -> None:
    """Cap a single serial reservation by both aggregate GPU time and wall time."""
    authorization = plan.get("authorization", {})
    if set(authorization) != {"id", "deadline_utc", "gpu_seconds"}:
        refuse("GPU execution requires a renewed aggregate authorization and deadline.")
    try:
        UUID(authorization["id"])
    except (TypeError, ValueError, AttributeError):
        refuse("A valid renewed authorization UUID is required.")
    budget = authorization["gpu_seconds"]
    seconds = plan.get("timeout_seconds")
    if type(budget) is not int or budget <= 0:
        refuse("The aggregate GPU-second budget must be a positive integer.")
    if (
        type(seconds) is not int
        or not MIN_TIMEOUT_SECONDS <= seconds <= GPU_PROFILES[plan["profile"]]
    ):
        refuse("Provide an explicit bounded profile timeout_seconds (at least 120).")
    wall_deadline = utc_deadline(authorization["deadline_utc"])
    required = seconds
    if plan.get("run_deadline_utc"):
        run_deadline = utc_deadline(plan["run_deadline_utc"])
        if run_deadline > wall_deadline:
            refuse("The run deadline cannot extend aggregate authorization.")
        wall_deadline, required = run_deadline, 1
    if (
        seconds > budget
        or (wall_deadline - (now or datetime.now(UTC))).total_seconds() < required
    ):
        refuse(
            "The job does not fit the remaining authorization/deadline; do not submit."
        )


def validate_gpu_plan(plan: dict) -> None:
    """Reject configuration, backend, resources, secret, and execution drift."""
    if (plan.get("project_name"), plan.get("project_id")) != (PROJECT_NAME, PROJECT_ID):
        refuse("GPU experiments require the confirmed models-rd project and UUID.")
    if (
        plan.get("accelerator") != "cuda"
        or plan.get("resource_profile") != RESOURCE_PROFILE
    ):
        refuse("GPU profiles require the versioned CUDA resource contract.")
    if (
        plan.get("resources") != gpu_resources(plan["profile"])
        or plan.get("resources_acknowledged") is not True
    ):
        refuse(
            "Explicit resource confirmation is required for the fixed one-GPU profile."
        )
    backend = plan.get("backend")
    if plan.get("service_account_verification", "inspect") not in {
        "inspect",
        "admission",
    }:
        refuse(
            "Explicit inspection or registered-account admission verification is required."
        )
    probe = plan["profile"] == "gpu-probe-v1"
    if probe:
        if plan.get("config_name") is not None or plan.get("config_sha256") is not None:
            refuse("GPU probe must not carry a training configuration.")
        if backend != "opaque" or plan.get("arm") not in ARMS.get("opaque", set()):
            refuse("GPU probe uses the opaque backend with an allowlisted arm label.")
    elif CONFIGS.get(plan.get("config_name")) != backend or plan.get(
        "arm"
    ) not in ARMS.get(backend, set()):
        refuse("Allowlisted configuration/backend/arm combination required.")
    if plan["profile"] == "sft-train-v1" and (plan["config_name"], plan["arm"]) not in {
        ("sft_balancing_aux0001_ratio1.json", "dp_aux"),
        ("sft_balancing_aux0001_ratio01.json", "dp_aux"),
        ("sft_balancing_aux02_ratio01.json", "dp_aux"),
        ("sft_trl_aux0001.json", "trl_reference_aux"),
    }:
        refuse(
            "Full training is restricted to the four missing follow-ups; recover existing controls."
        )
    if plan["profile"] == "sft-smoke-v1" and plan["arm"] not in {
        "dp_aux",
        "trl_reference_aux",
    }:
        refuse("The pretrained smoke must exercise the auxiliary objective.")
    if plan.get("seed") != 0:
        refuse("This coefficient round uses public seed zero, without repetitions.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(plan.get("source_sha256", ""))):
        refuse("An explicit source_sha256 is required for GPU execution.")
    if not probe and not re.fullmatch(
        r"[0-9a-f]{64}", str(plan.get("config_sha256", ""))
    ):
        refuse("An explicit config_sha256 is required for GPU execution.")
    secrets = plan.get("image_pull_secrets")
    if (
        not isinstance(secrets, list)
        or not secrets
        or len(set(secrets)) != len(secrets)
    ):
        refuse("Confirm nonempty, distinct existing image-pull secret references.")
    secret = plan.get("wandb_secret", {})
    if set(secret) != {"name", "key"} or not re.fullmatch(
        r"[A-Za-z0-9._-]+", secret.get("key", "")
    ):
        refuse(
            "Provide only the existing W&B secret name/key, never a credential value."
        )
    for name in [*secrets, secret.get("name", "")]:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", name):
            refuse("Invalid Kubernetes secret reference.")
    if plan["profile"] != "gpu-probe-v1":
        if not plan.get("gate_run_id"):
            refuse("GPU training/generation requires a completed matching gate.")
        UUID(plan["gate_run_id"])
    if plan["profile"] == "generate-v1":
        for key in ("checkpoint", "benchmark_input"):
            reference = plan.get(key) or {}
            if (
                set(reference) != {"uri", "sha256"}
                or not re.fullmatch(
                    r"gs://gke-dev-dws-jbr-zenml/[A-Za-z0-9_./%-]+",
                    str(reference["uri"]),
                )
                or not re.fullmatch(r"[0-9a-f]{64}", str(reference["sha256"]))
            ):
                refuse(
                    f"Generation requires a checksummed {key} in the approved GCS bucket."
                )
        if plan.get("benchmark") not in {"humaneval", "mbpp"}:
            refuse("Choose one fixed benchmark for each generation job.")
    elif plan.get("checkpoint") or plan.get("benchmark") or plan.get("benchmark_input"):
        refuse("Training cannot accept checkpoint/resume or benchmark overrides.")
    validate_authorization(plan)


def validate_config(root: Path, plan: dict) -> dict:
    """Verify the approved bytes before staging, submitting, or loading a config."""
    if plan.get("profile") == "gpu-probe-v1":
        if plan.get("config_name") is not None or plan.get("config_sha256") is not None:
            refuse("GPU probe must not carry a training configuration.")
        return {}
    if CONFIGS.get(plan.get("config_name")) != plan.get("backend"):
        refuse("Unknown configuration/backend.")
    path = root / "examples/moe_privacy/configs" / plan["config_name"]
    if (
        path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest() != plan["config_sha256"]
    ):
        refuse("Configuration hash mismatch; never change training settings to fit.")
    return json.loads(path.read_text())


def reserve_budget(directory: Path, plan: dict) -> Path:
    """Persist conservative allocations; caller holds the deployment submission lock.

    Reservations are not refunded after a failure or lost client. This prevents an
    uncertain submission from being retried against apparently unspent GPU time.
    """
    validate_authorization(plan)
    authorization = plan["authorization"]
    path = directory / f"{authorization['id']}.json"
    receipt = {"authorization": authorization, "reservations": {}}
    if path.exists():
        receipt = json.loads(path.read_text())
    if receipt["authorization"] != authorization:
        refuse(
            "An existing authorization cannot be expanded or assigned a new deadline."
        )
    reservations = receipt["reservations"]
    if plan["run_name"] in reservations:
        refuse("This run already has a reservation; inspect it instead of retrying.")
    used = sum(reservations.values())
    if (
        not math.isfinite(used)
        or used + plan["timeout_seconds"] > authorization["gpu_seconds"]
    ):
        refuse("Aggregate GPU allocation exhausted; obtain new authorization.")
    reservations[plan["run_name"]] = plan["timeout_seconds"]
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    temporary.replace(path)
    return path
