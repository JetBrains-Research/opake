"""ZenML steps that validate infrastructure without exposing sensitive data."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
from adapters.zenml.checkpoints import (
    ZenMLCheckpointCallback,
    materialize_checkpoint,
)

from zenml import get_step_context, step
from zenml.artifacts.artifact_config import ArtifactConfig
from zenml.config import ResourceSettings

_EXAMPLES_ROOT = str(Path(__file__).resolve().parents[2] / "examples")
if _EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLES_ROOT)

if TYPE_CHECKING:
    from types import ModuleType

HUGGINGFACE_API_URL = "https://huggingface.co/api/models?limit=1"
REPORT_FILENAME = "probe-report.json"
CHECKSUM_FILENAME = "probe-report.sha256"
_HTTP_SUCCESS_MIN = 200
_HTTP_REDIRECT_LIMIT = 400
_CUDA_EXPECTED_RESULT = 14.0
_SHA256_HEXDIGEST_LENGTH = hashlib.sha256().digest_size * 2
_MAX_REPORT_BYTES = 64 * 1024
_NOT_JSON_SAFE_ERROR = "Probe report is not JSON-safe"
_INVALID_ENV_NAME_ERROR = (
    "Required environment variable names must be non-empty strings"
)
_MISSING_ENV_ERROR = "Required environment variables are unavailable: "
_HUGGINGFACE_ACCESS_ERROR = "Hugging Face API probe failed"
_HUGGINGFACE_STATUS_ERROR = "Hugging Face API probe returned an unsuccessful status"
_CUDA_UNAVAILABLE_ERROR = "A CUDA GPU is required but CUDA is unavailable"
_CUDA_RESULT_ERROR = "CUDA probe computation returned an unexpected result"
_NOT_DIRECTORY_ERROR = "Probe artifact must be a directory"
_INCOMPLETE_ARTIFACT_ERROR = "Probe artifact is incomplete"
_REPORT_SIZE_ERROR = "Probe report exceeds the size limit"
_MALFORMED_CHECKSUM_ERROR = "Probe artifact checksum is malformed"
_CHECKSUM_MISMATCH_ERROR = "Probe artifact checksum mismatch"
_INVALID_JSON_ERROR = "Probe artifact report is not valid JSON"
_INVALID_REPORT_TYPE_ERROR = "Probe artifact report must be a JSON object"
_NONCANONICAL_JSON_ERROR = "Probe artifact report is not canonical JSON"
_CONTENT_MISMATCH_ERROR = "Probe artifact content does not match the report artifact"
ProbeReport = dict[str, Any]


def resolve_sft_config(config: dict[str, object]) -> object:
    """Load the optional SFT example config resolver when training starts."""
    from opaque_examples.sft import resolve_sft_config as resolve

    return resolve(config)


def _load_sft_runner() -> ModuleType:
    """Load the optional SFT runner only inside the training step."""
    from opaque_examples.sft import runner

    return runner


def _report_sft_stage(stage: str) -> None:
    """Emit a sanitized, immediately flushed remote progress marker."""
    print(f"Opaque SFT stage: {stage}", flush=True)


def _prepare_runtime_directories() -> None:
    """Create configured writable cache directories before ML imports."""
    for variable in ("HF_HOME", "WANDB_DIR"):
        configured = os.environ.get(variable)
        if configured:
            Path(configured).mkdir(parents=True, exist_ok=True)


def _get_current_run_id() -> str:
    """Return the active ZenML pipeline run ID."""
    return str(get_step_context().pipeline_run.id)


class _ProbeRuntimeError(RuntimeError):
    """Raised when a required infrastructure capability is unavailable."""


class _ProbeValidationError(ValueError):
    """Raised when probe inputs or artifact contents are invalid."""


def _canonical_json(report: ProbeReport) -> bytes:
    """Serialize a report in a stable, compact, JSON-safe representation."""
    try:
        serialized = json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise _ProbeValidationError(_NOT_JSON_SAFE_ERROR) from error
    return f"{serialized}\n".encode()


def _required_environment_status(names: list[str]) -> dict[str, bool]:
    """Return availability flags without reading values into the report."""
    if any(not isinstance(name, str) or not name for name in names):
        raise _ProbeValidationError(_INVALID_ENV_NAME_ERROR)

    availability = {name: bool(os.environ.get(name)) for name in sorted(set(names))}
    missing = [name for name, available in availability.items() if not available]
    if missing:
        raise _ProbeRuntimeError(_MISSING_ENV_ERROR + ", ".join(missing))
    return availability


def _check_huggingface_access(timeout_seconds: float = 10.0) -> None:
    """Check the public Hugging Face API without sending authentication data."""
    request = Request(
        HUGGINGFACE_API_URL,
        headers={"User-Agent": "opaque-infrastructure-probe/1"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.getcode()
    except (OSError, URLError):
        raise _ProbeRuntimeError(_HUGGINGFACE_ACCESS_ERROR) from None

    if status is None or not _HTTP_SUCCESS_MIN <= status < _HTTP_REDIRECT_LIMIT:
        raise _ProbeRuntimeError(_HUGGINGFACE_STATUS_ERROR)


def _cuda_details(require_gpu: bool) -> dict[str, Any]:
    """Collect non-identifying CUDA details and optionally run a computation."""
    available = bool(torch.cuda.is_available())
    if require_gpu and not available:
        raise _ProbeRuntimeError(_CUDA_UNAVAILABLE_ERROR)

    computation_succeeded = False
    if require_gpu:
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = tensor.square().sum().item()
        if result != _CUDA_EXPECTED_RESULT:
            raise _ProbeRuntimeError(_CUDA_RESULT_ERROR)
        computation_succeeded = True

    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "probe_computation_succeeded": computation_succeeded,
        "required": require_gpu,
        "runtime_version": str(torch.version.cuda) if torch.version.cuda else None,
    }


def _build_probe_report(
    require_gpu: bool,
    required_secret_env: list[str],
    check_huggingface: bool,
) -> ProbeReport:
    """Run the probes and build a report containing no environment values."""
    environment_status = _required_environment_status(required_secret_env)
    if check_huggingface:
        _check_huggingface_access()

    return {
        "cuda": _cuda_details(require_gpu),
        "huggingface": {
            "checked": check_huggingface,
            "reachable": check_huggingface,
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "required_environment": environment_status,
        "torch": {"version": str(torch.__version__)},
    }


def _write_probe_artifact(
    report: ProbeReport, output_directory: Path | None = None
) -> Path:
    """Write canonical report bytes and their checksum to a directory."""
    if output_directory is None:
        output_directory = Path(tempfile.mkdtemp(prefix="opaque-zenml-probe-"))
    else:
        output_directory.mkdir(parents=True, exist_ok=False)

    report_bytes = _canonical_json(report)
    checksum = hashlib.sha256(report_bytes).hexdigest()
    (output_directory / REPORT_FILENAME).write_bytes(report_bytes)
    (output_directory / CHECKSUM_FILENAME).write_text(f"{checksum}\n", encoding="ascii")
    return output_directory


def _verify_artifact_contents(
    artifact_directory: Path, expected_report: ProbeReport
) -> ProbeReport:
    """Validate artifact structure, checksum, canonical JSON, and report content."""
    if not artifact_directory.is_dir():
        raise _ProbeValidationError(_NOT_DIRECTORY_ERROR)

    report_path = artifact_directory / REPORT_FILENAME
    checksum_path = artifact_directory / CHECKSUM_FILENAME
    if not report_path.is_file() or not checksum_path.is_file():
        raise _ProbeValidationError(_INCOMPLETE_ARTIFACT_ERROR)

    report_bytes = report_path.read_bytes()
    if len(report_bytes) > _MAX_REPORT_BYTES:
        raise _ProbeValidationError(_REPORT_SIZE_ERROR)

    expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    if len(expected_checksum) != _SHA256_HEXDIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in expected_checksum
    ):
        raise _ProbeValidationError(_MALFORMED_CHECKSUM_ERROR)

    actual_checksum = hashlib.sha256(report_bytes).hexdigest()
    if actual_checksum != expected_checksum:
        raise _ProbeValidationError(_CHECKSUM_MISMATCH_ERROR)

    try:
        materialized_report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _ProbeValidationError(_INVALID_JSON_ERROR) from None
    if not isinstance(materialized_report, dict):
        raise _ProbeValidationError(_INVALID_REPORT_TYPE_ERROR)
    if _canonical_json(materialized_report) != report_bytes:
        raise _ProbeValidationError(_NONCANONICAL_JSON_ERROR)
    if materialized_report != expected_report:
        raise _ProbeValidationError(_CONTENT_MISMATCH_ERROR)

    return {
        "report_matches": True,
        "sha256": actual_checksum,
        "verified": True,
    }


@step(enable_cache=False)
def probe_environment(
    require_gpu: bool,
    required_secret_env: list[str],
    check_huggingface: bool,
) -> tuple[
    Annotated[Path, ArtifactConfig(name="probe_artifact")],
    Annotated[ProbeReport, ArtifactConfig(name="probe_report")],
]:
    """Probe runtime capabilities and create a checksummed directory artifact."""
    report = _build_probe_report(
        require_gpu=require_gpu,
        required_secret_env=required_secret_env,
        check_huggingface=check_huggingface,
    )
    return _write_probe_artifact(report), report


@step(enable_cache=False)
def verify_probe_artifact(
    probe_artifact: Path,
    expected_report: ProbeReport,
) -> Annotated[
    ProbeReport,
    ArtifactConfig(name="probe_verification_report"),
]:
    """Verify a probe directory after ZenML materializes it downstream."""
    return _verify_artifact_contents(probe_artifact, expected_report)


@step(
    enable_cache=False,
    settings={"resources": ResourceSettings(gpu_count=1)},
)
def train_sft_step(
    config: dict[str, object],
    source_commit_sha: str,
    resume_checkpoint: str | None = None,
) -> tuple[
    Annotated[Path, "model_bundle"],
    Annotated[dict[str, object], "run_summary"],
]:
    """Run one fully configured SFT job in a GPU-backed ZenML step."""
    _report_sft_stage("resolve-config")
    resolved_config = resolve_sft_config(config)
    _report_sft_stage("load-runner")
    runner_module = _load_sft_runner()
    _report_sft_stage("resolve-run-context")
    parent_run_id = _get_current_run_id()
    workspace = Path(tempfile.mkdtemp(prefix="opaque-zenml-sft-"))
    output_directory = workspace / "model-bundle"
    resume_directory = workspace / "resume-checkpoint"
    completed = False

    try:
        _report_sft_stage("materialize-resume")
        _prepare_runtime_directories()
        local_checkpoint = (
            materialize_checkpoint(resume_checkpoint, resume_directory)
            if resume_checkpoint is not None
            else None
        )
        callback = ZenMLCheckpointCallback(
            parent_run_id=parent_run_id,
            resume_checkpoint=local_checkpoint,
        )

        previous_source_commit = os.environ.get("OPAQUE_SOURCE_COMMIT_SHA")
        os.environ["OPAQUE_SOURCE_COMMIT_SHA"] = source_commit_sha
        try:
            _report_sft_stage("run-training")
            summary = runner_module.run_sft(
                resolved_config,
                output_directory,
                callbacks=(callback,),
                resume_from_checkpoint=local_checkpoint,
                source_commit_sha=source_commit_sha,
            )
        finally:
            if previous_source_commit is None:
                os.environ.pop("OPAQUE_SOURCE_COMMIT_SHA", None)
            else:
                os.environ["OPAQUE_SOURCE_COMMIT_SHA"] = previous_source_commit

        run_summary: dict[str, object] = summary.to_dict()
        _canonical_json(run_summary)
        _report_sft_stage("complete")
        completed = True
        return output_directory, run_summary
    finally:
        if resume_directory.exists():
            shutil.rmtree(resume_directory)
        if not completed:
            shutil.rmtree(workspace, ignore_errors=True)
