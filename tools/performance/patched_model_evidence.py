"""Record and compare reproducible, staged patched-model evidence.

This repository tool deliberately does not impose wall-clock thresholds on PR
CI. It records raw observations and correctness results in a stable schema so a
controlled benchmark host can compare the same workload across two commits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Never

import torch

from opaque.api.engine.device import device_capabilities
from opaque.profiling import get_memory_stats, step_perf

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

SCHEMA = "opaque.patched-model-evidence.v1"
COMPARISON_SCHEMA = "opaque.patched-model-comparison.v1"
_ROOT = Path(__file__).resolve().parents[2]
_GIB = 1024**3
_FULL_COMMIT_LENGTH = 40


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be recorded or compared."""


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is not None:
        raise EvidenceError(message) from cause
    raise EvidenceError(message)


def _invalid(message: str) -> Never:
    raise ValueError(message)


@dataclass(frozen=True)
class CheckResult:
    """One numerical or gradient correctness result."""

    name: str
    kind: Literal["numerical", "gradient"]
    passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    details: str = ""


@dataclass(frozen=True)
class Stage:
    """A separately measured workload stage.

    ``prepare`` runs outside the measurement window. Its return value is passed
    to ``run``. This lets a backward stage construct its graph without charging
    forward work to the backward sample.
    """

    name: str
    unit: str
    units: int
    backend: str | Callable[[], str]
    prepare: Callable[[], Any]
    run: Callable[[Any], Any]
    cleanup: Callable[[Any, Any], None] | None = None

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.unit
            or (not callable(self.backend) and not self.backend)
        ):
            _invalid("stage name, unit, and backend must be non-empty")
        if self.units <= 0:
            _invalid("stage units must be > 0")


@dataclass(frozen=True)
class Workload:
    """A patched-model workload supplied by a benchmark plug-in."""

    workload_id: str
    config: Mapping[str, Any]
    stages: Sequence[Stage]
    checks: Callable[[], Sequence[CheckResult]]
    counters: Callable[[], Mapping[str, int | float | str]] | None = None

    def __post_init__(self) -> None:
        names = [stage.name for stage in self.stages]
        if not self.workload_id:
            _invalid("workload_id must be non-empty")
        if not names or len(names) != len(set(names)):
            _invalid("workload stages must have unique, non-empty names")


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = _run_git("rev-parse", "HEAD")
        status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("evidence must be recorded inside a Git worktree", error)
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_paths": [line[3:] for line in status.splitlines()],
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "opaque-engine",
        "opaque-patches",
        "opaque-transformers",
        "peft",
        "torch",
        "transformers",
        "triton",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` consistently as CUDA, then MPS, then CPU."""

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        _fail("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        _fail("MPS was requested but is unavailable")
    return device


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "type": device.type,
        "index": device.index,
        "peak_memory_trackable": device_capabilities(device).peak_memory_trackable,
    }
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            name=properties.name,
            total_memory_bytes=properties.total_memory,
            capability=list(torch.cuda.get_device_capability(index)),
            runtime=torch.version.cuda,
        )
    elif device.type == "mps":
        metadata.update(
            name=platform.processor() or platform.machine(),
            total_memory_bytes=int(torch.mps.recommended_max_memory()),
            runtime=platform.mac_ver()[0],
        )
    else:
        metadata.update(name=platform.processor() or platform.machine())
    return metadata


def _measurement_settings() -> dict[str, Any]:
    environment_names = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCHINDUCTOR_CACHE_DIR",
    )
    settings: dict[str, Any] = {
        "environment": {
            name: os.environ[name] for name in environment_names if name in os.environ
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if torch.cuda.is_available():
        settings["cuda"] = {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    return settings


def _source_digest(factory: Callable[..., Workload]) -> tuple[str, list[str]]:
    paths = [Path(__file__).resolve()]
    source = inspect.getsourcefile(factory)
    if source is not None:
        paths.append(Path(source).resolve())
    digest = hashlib.sha256()
    relative_paths = []
    for path in sorted(set(paths)):
        relative = (
            path.relative_to(_ROOT).as_posix()
            if path.is_relative_to(_ROOT)
            else str(path)
        )
        relative_paths.append(relative)
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), relative_paths


def _counter_snapshot() -> dict[str, int]:
    counters: dict[str, int] = {}
    try:
        from torch._dynamo.utils import counters as dynamo_counters
    except ImportError:
        dynamo_counters = {}
    for group, values in dynamo_counters.items():
        for name, value in values.items():
            if isinstance(value, int):
                counters[f"dynamo.{group}.{name}"] = value
    try:
        from torch._inductor import metrics as inductor_metrics
    except ImportError:
        inductor_metrics = None
    if inductor_metrics is not None:
        for name in ("generated_kernel_count", "generated_cpp_vec_kernel_count"):
            value = getattr(inductor_metrics, name, None)
            if isinstance(value, int):
                counters[f"inductor.{name}"] = value
    return counters


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    return {
        name: value - before.get(name, 0)
        for name, value in sorted(after.items())
        if value - before.get(name, 0) != 0
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        _invalid("cannot summarize an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        _fail("samples must contain finite values")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _execute(stage: Stage) -> None:
    prepared = stage.prepare()
    result = stage.run(prepared)
    if stage.cleanup is not None:
        stage.cleanup(prepared, result)
    del result, prepared


def _measure_stage(
    stage: Stage, device: torch.device, *, warmup: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmup):
        _execute(stage)
        _synchronize(device)

    samples = []
    exact_peak = device_capabilities(device).peak_memory_trackable
    for index in range(repeats):
        prepared = stage.prepare()
        baseline = get_memory_stats(device)
        with step_perf(device, batch_size=stage.units) as timer:
            result = stage.run(prepared)
        perf = timer.perf
        selected_backend = stage.backend() if callable(stage.backend) else stage.backend
        if not selected_backend:
            _fail(f"stage {stage.name!r} reported an empty backend")
        peak_reserved_bytes = (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else round(perf.memory_reserved_gb * _GIB)
        )
        sample = {
            "index": index,
            "duration_sec": perf.step_time_sec,
            "throughput": perf.samples_per_second,
            "selected_backend": selected_backend,
            "baseline_allocated_bytes": round(baseline.allocated_gb * _GIB),
            "peak_allocated_bytes": round(perf.memory_peak_gb * _GIB),
            "peak_reserved_bytes": peak_reserved_bytes,
            "incremental_peak_allocated_bytes": round(
                max(perf.memory_peak_gb - baseline.allocated_gb, 0.0) * _GIB
            ),
            "end_allocated_bytes": round(perf.memory_allocated_gb * _GIB),
            "end_reserved_bytes": round(perf.memory_reserved_gb * _GIB),
            "peak_exact": exact_peak,
            "reserved_peak_exact": device.type == "cuda",
        }
        samples.append(sample)
        if stage.cleanup is not None:
            stage.cleanup(prepared, result)
        del result, prepared

    durations = [sample["duration_sec"] for sample in samples]
    throughputs = [sample["throughput"] for sample in samples]
    peaks = [sample["peak_allocated_bytes"] for sample in samples]
    reserved_peaks = [sample["peak_reserved_bytes"] for sample in samples]
    incremental_peaks = [
        sample["incremental_peak_allocated_bytes"] for sample in samples
    ]
    selected_backends = sorted({sample["selected_backend"] for sample in samples})
    return {
        "name": stage.name,
        "unit": stage.unit,
        "units_per_sample": stage.units,
        "selected_backend": (
            selected_backends[0] if len(selected_backends) == 1 else "mixed"
        ),
        "selected_backends": selected_backends,
        "samples": samples,
        "summary": {
            "duration_sec": _summarize(durations),
            "throughput": _summarize(throughputs),
            "peak_allocated_bytes": _summarize(peaks),
            "peak_reserved_bytes": _summarize(reserved_peaks),
            "incremental_peak_allocated_bytes": _summarize(incremental_peaks),
        },
    }


def _fingerprint_payload(
    *,
    workload: Workload,
    device: Mapping[str, Any],
    packages: Mapping[str, str],
    seed: int,
    warmup: int,
    repeats: int,
    measurement_settings: Mapping[str, Any],
) -> dict[str, Any]:
    dependency_versions = {
        name: version
        for name, version in packages.items()
        if name in {"peft", "torch", "transformers", "triton"}
    }
    return {
        "workload_id": workload.workload_id,
        "config": workload.config,
        "device": device,
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "dependencies": dependency_versions,
        "measurement": {
            "seed": seed,
            "warmup": warmup,
            "repeats": repeats,
            "settings": measurement_settings,
        },
        "stages": [
            {"name": stage.name, "unit": stage.unit, "units": stage.units}
            for stage in workload.stages
        ],
    }


def record_run(
    factory: Callable[[torch.device, Mapping[str, Any], int], Workload],
    *,
    variant: str,
    device: torch.device,
    config: Mapping[str, Any],
    seed: int,
    warmup: int,
    repeats: int,
    command: Sequence[str] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Record one baseline or candidate artifact."""

    if not variant:
        _invalid("variant must be non-empty")
    if warmup < 0 or repeats <= 0:
        _invalid("warmup must be >= 0 and repeats must be > 0")
    git = _git_metadata()
    if git["dirty"] and not allow_dirty:
        _fail(
            "worktree is dirty; commit the benchmarked implementation or pass "
            "--allow-dirty for a diagnostic artifact"
        )
    workload = factory(device, config, seed)
    packages = _package_versions()
    device_metadata = _device_metadata(device)
    measurement_settings = _measurement_settings()
    fingerprint_payload = _fingerprint_payload(
        workload=workload,
        device=device_metadata,
        packages=packages,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
        measurement_settings=measurement_settings,
    )
    source_digest, source_paths = _source_digest(factory)
    before = _counter_snapshot()
    stages = [
        _measure_stage(stage, device, warmup=warmup, repeats=repeats)
        for stage in workload.stages
    ]
    compiler_counters = _counter_delta(before, _counter_snapshot())
    workload_counters = dict(workload.counters()) if workload.counters else {}
    checks = [asdict(check) for check in workload.checks()]
    run = {
        "schema": SCHEMA,
        "variant": variant,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "git": git,
            "command": list(command if command is not None else sys.argv),
            "seed": seed,
            "warmup": warmup,
            "repeats": repeats,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "packages": packages,
            "device": device_metadata,
            "source_digest": source_digest,
            "source_paths": source_paths,
            "measurement_settings": measurement_settings,
        },
        "comparison_fingerprint": hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "workload": {
            "id": workload.workload_id,
            "config": dict(workload.config),
        },
        "stages": stages,
        "compiler_counters": compiler_counters,
        "workload_counters": workload_counters,
        "correctness": checks,
    }
    validate_run(run)
    return run


def _require(mapping: Mapping[str, Any], key: str, expected: type, path: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        _fail(f"{path}.{key} must be {expected.__name__}")
    return value


def validate_run(run: Mapping[str, Any]) -> None:
    """Validate the stable schema and semantic completeness of one artifact."""

    if run.get("schema") != SCHEMA:
        _fail(f"schema must be {SCHEMA!r}")
    _require(run, "variant", str, "$")
    fingerprint = _require(run, "comparison_fingerprint", str, "$")
    metadata = _require(run, "metadata", dict, "$")
    git = _require(metadata, "git", dict, "$.metadata")
    commit = _require(git, "commit", str, "$.metadata.git")
    if len(commit) != _FULL_COMMIT_LENGTH:
        _fail("$.metadata.git.commit must be a full commit hash")
    _require(metadata, "command", list, "$.metadata")
    _require(metadata, "packages", dict, "$.metadata")
    _require(metadata, "device", dict, "$.metadata")
    _require(metadata, "platform", dict, "$.metadata")
    _require(metadata, "seed", int, "$.metadata")
    _require(metadata, "warmup", int, "$.metadata")
    _require(metadata, "repeats", int, "$.metadata")
    _require(metadata, "measurement_settings", dict, "$.metadata")
    workload = _require(run, "workload", dict, "$")
    _require(workload, "id", str, "$.workload")
    _require(workload, "config", dict, "$.workload")
    stages = _require(run, "stages", list, "$")
    if not stages:
        _fail("$.stages must not be empty")
    names = set()
    for stage_index, stage in enumerate(stages):
        path = f"$.stages[{stage_index}]"
        if not isinstance(stage, dict):
            _fail(f"{path} must be an object")
        name = _require(stage, "name", str, path)
        if name in names:
            _fail(f"{path}.name is duplicated")
        names.add(name)
        _require(stage, "unit", str, path)
        _require(stage, "units_per_sample", int, path)
        selected_backend = _require(stage, "selected_backend", str, path)
        selected_backends = _require(stage, "selected_backends", list, path)
        samples = _require(stage, "samples", list, path)
        if not samples:
            _fail(f"{path}.samples must not be empty")
        for sample_index, sample in enumerate(samples):
            sample_path = f"{path}.samples[{sample_index}]"
            if not isinstance(sample, dict):
                _fail(f"{sample_path} must be an object")
            _require(sample, "selected_backend", str, sample_path)
            for field in (
                "duration_sec",
                "throughput",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "incremental_peak_allocated_bytes",
            ):
                value = sample.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    _fail(f"{sample_path}.{field} must be finite")
                if value < 0:
                    _fail(f"{sample_path}.{field} must be >= 0")
        raw_backends = sorted({sample["selected_backend"] for sample in samples})
        if selected_backends != raw_backends:
            _fail(f"{path}.selected_backends must match raw samples")
        expected_backend = raw_backends[0] if len(raw_backends) == 1 else "mixed"
        if selected_backend != expected_backend:
            _fail(f"{path}.selected_backend must summarize raw samples")
        summary = _require(stage, "summary", dict, path)
        raw_fields = {
            "duration_sec": "duration_sec",
            "throughput": "throughput",
            "peak_allocated_bytes": "peak_allocated_bytes",
            "peak_reserved_bytes": "peak_reserved_bytes",
            "incremental_peak_allocated_bytes": "incremental_peak_allocated_bytes",
        }
        for field in (
            "duration_sec",
            "throughput",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "incremental_peak_allocated_bytes",
        ):
            metric = _require(summary, field, dict, f"{path}.summary")
            if metric.get("count") != len(samples):
                _fail(f"{path}.summary.{field}.count must match raw samples")
            expected_summary = _summarize(
                [float(sample[raw_fields[field]]) for sample in samples]
            )
            for statistic in ("median", "p95", "min", "max"):
                value = metric.get(statistic)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    _fail(f"{path}.summary.{field}.{statistic} must be a finite number")
                if not math.isclose(
                    value,
                    float(expected_summary[statistic]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    _fail(
                        f"{path}.summary.{field}.{statistic} does not match raw samples"
                    )
    _require(run, "compiler_counters", dict, "$")
    _require(run, "workload_counters", dict, "$")
    correctness = _require(run, "correctness", list, "$")
    if not correctness:
        _fail("$.correctness must contain numerical or gradient checks")
    kinds = set()
    for index, check in enumerate(correctness):
        path = f"$.correctness[{index}]"
        if not isinstance(check, dict):
            _fail(f"{path} must be an object")
        kinds.add(_require(check, "kind", str, path))
        _require(check, "passed", bool, path)
    if not {"numerical", "gradient"}.issubset(kinds):
        _fail("$.correctness must include numerical and gradient checks")
    packages = metadata["packages"]
    fingerprint_payload = {
        "workload_id": workload["id"],
        "config": workload["config"],
        "device": metadata["device"],
        "platform": metadata["platform"],
        "dependencies": {
            name: version
            for name, version in packages.items()
            if name in {"peft", "torch", "transformers", "triton"}
        },
        "measurement": {
            "seed": metadata["seed"],
            "warmup": metadata["warmup"],
            "repeats": metadata["repeats"],
            "settings": metadata["measurement_settings"],
        },
        "stages": [
            {
                "name": stage["name"],
                "unit": stage["unit"],
                "units": stage["units_per_sample"],
            }
            for stage in stages
        ],
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        _fail("$.comparison_fingerprint does not match artifact contents")


def _ratio(candidate: float, baseline: float) -> float | None:
    return candidate / baseline if baseline > 0 else None


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_throughput_regression: float = 0.05,
    max_memory_regression: float = 0.0,
) -> dict[str, Any]:
    """Compare two compatible artifacts with independent performance policies."""

    validate_run(baseline)
    validate_run(candidate)
    if baseline["metadata"]["git"]["dirty"] or candidate["metadata"]["git"]["dirty"]:
        _fail("dirty diagnostic artifacts cannot be used for revision comparisons")
    if (
        not math.isfinite(max_throughput_regression)
        or not 0 <= max_throughput_regression < 1
    ):
        _invalid("max_throughput_regression must be in [0, 1)")
    if not math.isfinite(max_memory_regression) or max_memory_regression < 0:
        _invalid("max_memory_regression must be >= 0")
    if baseline["comparison_fingerprint"] != candidate["comparison_fingerprint"]:
        _fail("baseline and candidate environments/configurations are not comparable")
    candidate_stages = {stage["name"]: stage for stage in candidate["stages"]}
    rows = []
    for baseline_stage in baseline["stages"]:
        candidate_stage = candidate_stages.get(baseline_stage["name"])
        if candidate_stage is None:
            _fail(f"candidate is missing stage {baseline_stage['name']!r}")
        base_summary = baseline_stage["summary"]
        cand_summary = candidate_stage["summary"]
        throughput_ratio = _ratio(
            cand_summary["throughput"]["median"],
            base_summary["throughput"]["median"],
        )
        memory_ratio = _ratio(
            cand_summary["peak_allocated_bytes"]["max"],
            base_summary["peak_allocated_bytes"]["max"],
        )
        incremental_memory_ratio = _ratio(
            cand_summary["incremental_peak_allocated_bytes"]["max"],
            base_summary["incremental_peak_allocated_bytes"]["max"],
        )
        memory_peak_exact = all(
            sample["peak_exact"]
            for sample in (*baseline_stage["samples"], *candidate_stage["samples"])
        )
        throughput_status = (
            "unavailable"
            if throughput_ratio is None
            else (
                "regression"
                if throughput_ratio < 1 - max_throughput_regression
                else "pass"
            )
        )
        available_memory_ratios = [
            ratio
            for ratio in (memory_ratio, incremental_memory_ratio)
            if ratio is not None and memory_peak_exact
        ]
        memory_status = (
            "unavailable"
            if not available_memory_ratios
            else (
                "regression"
                if any(
                    ratio > 1 + max_memory_regression
                    for ratio in available_memory_ratios
                )
                else "pass"
            )
        )
        rows.append(
            {
                "name": baseline_stage["name"],
                "baseline_backend": baseline_stage["selected_backend"],
                "candidate_backend": candidate_stage["selected_backend"],
                "throughput_ratio": throughput_ratio,
                "throughput_status": throughput_status,
                "peak_memory_ratio": memory_ratio,
                "incremental_peak_memory_ratio": incremental_memory_ratio,
                "memory_peak_exact": memory_peak_exact,
                "memory_status": memory_status,
            }
        )
    baseline_correctness_passed = all(
        check["passed"] for check in baseline["correctness"]
    )
    candidate_correctness_passed = all(
        check["passed"] for check in candidate["correctness"]
    )
    correctness_passed = baseline_correctness_passed and candidate_correctness_passed
    performance_passed = all(
        row["throughput_status"] != "regression"
        and row["memory_status"] != "regression"
        for row in rows
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "baseline": {
            "variant": baseline["variant"],
            "commit": baseline["metadata"]["git"]["commit"],
        },
        "candidate": {
            "variant": candidate["variant"],
            "commit": candidate["metadata"]["git"]["commit"],
        },
        "policy": {
            "max_throughput_regression": max_throughput_regression,
            "max_memory_regression": max_memory_regression,
            "timing_enforcement": "opt-in",
        },
        "baseline_correctness_passed": baseline_correctness_passed,
        "candidate_correctness_passed": candidate_correctness_passed,
        "correctness_passed": correctness_passed,
        "performance_passed": performance_passed,
        "stages": rows,
    }


def _load_factory(
    spec: str,
) -> Callable[[torch.device, Mapping[str, Any], int], Workload]:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        _fail("workload must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        _fail(f"{spec!r} does not resolve to a callable")
    return factory


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot read evidence from {path}: {error}", error)
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="record one revision")
    record.add_argument(
        "--workload",
        default="tools.performance.workloads.tiny_llama:create_workload",
    )
    record.add_argument("--variant", required=True)
    record.add_argument("--device", default="auto")
    record.add_argument("--seed", type=int, default=956)
    record.add_argument("--warmup", type=int, default=3)
    record.add_argument("--repeats", type=int, default=10)
    record.add_argument("--config-json", default="{}")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record a diagnostic artifact from an uncommitted worktree",
    )

    compare = subparsers.add_parser("compare", help="compare two revisions")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--max-throughput-regression", type=float, default=0.05)
    compare.add_argument("--max-memory-regression", type=float, default=0.0)
    compare.add_argument(
        "--enforce-performance",
        action="store_true",
        help="return non-zero for timing/memory regression (off by default)",
    )

    validate = subparsers.add_parser("validate", help="validate run artifacts")
    validate.add_argument("artifacts", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evidence record, validate, or compare command."""

    args = _parser().parse_args(argv)
    if args.command == "record":
        try:
            config = json.loads(args.config_json)
        except json.JSONDecodeError as error:
            _fail(f"--config-json is invalid: {error}", error)
        if not isinstance(config, dict):
            _fail("--config-json must be a JSON object")
        run = record_run(
            _load_factory(args.workload),
            variant=args.variant,
            device=resolve_device(args.device),
            config=config,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            command=[
                sys.executable,
                "-m",
                "tools.performance.patched_model_evidence",
                *sys.argv[1:],
            ],
            allow_dirty=args.allow_dirty,
        )
        _write_json(args.output, run)
        return 0
    if args.command == "validate":
        for artifact in args.artifacts:
            validate_run(_read_json(artifact))
        return 0

    report = compare_runs(
        _read_json(args.baseline),
        _read_json(args.candidate),
        max_throughput_regression=args.max_throughput_regression,
        max_memory_regression=args.max_memory_regression,
    )
    if args.output is not None:
        _write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not report["correctness_passed"]:
        return 2
    if args.enforce_performance and not report["performance_passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
