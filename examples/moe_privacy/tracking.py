"""Optional W&B logging of public experiment metrics, not raw trainer state."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    import argparse

_CONFIG_KEYS = frozenset(
    [
        "name",
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_revision",
        "dataset_format",
        "dataset_languages",
        "require_full_answers",
        "train_sequences",
        "validation_sequences",
        "test_sequences",
        "diagnostic_sequences",
        "detailed_metrics",
        "sequence_length",
        "expected_batch_size",
        "microbatch_size",
        "eval_batch_size",
        "steps",
        "eval_every",
        "learning_rate",
        "clipping_norm",
        "target_epsilon",
        "delta",
        "router_aux_loss_coef",
        "load_noise_ratio",
        "filter_beta",
        "lora_rank",
        "lora_alpha",
        "data_seed",
        "validation_seed",
        "gradient_checkpointing",
        "trainer_backend",
        "physical_batch_size",
        "gradient_accumulation_steps",
        "batch_clipping_norm",
        "sampling_mode",
        "task_loss_reduction",
        "transition_probability",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "top_k",
        "moe_intermediate_size",
    ]
)
_EVAL_KEYS = frozenset(
    [
        "loss",
        "load_cv",
        "max_expert_share",
        "routing_entropy",
        "active_experts_mean",
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "memory_allocated_gb",
        "memory_peak_gb",
        "memory_reserved_gb",
        "finalization_time_sec",
        "gather_time_sec",
        "metric_time_sec",
        "model_time_sec",
        "step_time_sec",
        "transfer_bytes",
        "transfer_overlap_ratio",
        "transfer_overlap_sec",
        "transfer_time_sec",
        "nll_sum",
        "nll_token_mean",
        "nll_example_mean",
        "teacher_forced_token_accuracy",
        "supervised_tokens",
        "attended_tokens",
        "records",
        "batches",
        "batch_size",
        "last_batch_size",
        "router_layers",
        "experts",
        "top_k",
        "routed_assignments",
        "elapsed_seconds",
        "attended_tokens_per_second",
        "supervised_tokens_per_second",
        "global_load_cv_mean",
        "global_load_cv_worst",
        "global_load_cv_p90",
        "global_max_to_mean_load",
        "global_routing_entropy_mean",
        "global_effective_experts_mean",
        "global_underused_expert_fraction",
        "batch_load_cv_mean",
        "batch_load_cv_p95",
        "batch_max_to_mean_load_mean",
        "batch_max_to_mean_load_p95",
        "pooled_load_cv",
        "pooled_max_to_mean_load",
        "pooled_effective_experts",
        "pooled_routing_entropy",
    ]
)
_PRIVACY_KEYS = (
    "private",
    "unit",
    "adjacency",
    "epsilon",
    "delta",
    "target_epsilon",
    "noise_multiplier",
    "steps",
    "sample_rate",
    "load_release",
    "load_noise_ratio",
    "normalization",
    "max_tokens",
    "mean_tokens",
    "scope",
)
_UUID_PATTERN = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
_DOMAIN_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_IMAGE_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_MAX_CONTEXT_VALUE_LENGTH = 512
_MAX_REGISTRY_PORT = 65535
_PUBLIC_ENV_PATTERNS = {
    "OPAQUE_ZENML_RUN_ID": _UUID_PATTERN,
    "OPAQUE_ZENML_PROJECT_ID": _UUID_PATTERN,
    "OPAQUE_ZENML_STACK_ID": _UUID_PATTERN,
    "OPAQUE_DEPLOYMENT_SOURCE_SHA256": r"[0-9a-fA-F]{64}",
    "OPAQUE_DEPLOYMENT_IMAGE": (
        rf"{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})*(?::[0-9]{{1,5}})?/"
        rf"(?:{_IMAGE_COMPONENT}/)*{_IMAGE_COMPONENT}@sha256:[0-9a-f]{{64}}"
    ),
    "OPAQUE_DEPLOYMENT_RESOURCE_PROFILE": r"one-80gb-gpu-v1",
}


def _public_run_context() -> dict:
    context = {}
    for key, pattern in _PUBLIC_ENV_PATTERNS.items():
        value = os.getenv(key)
        if value is None:
            continue
        if (
            len(value) > _MAX_CONTEXT_VALUE_LENGTH
            or re.fullmatch(pattern, value) is None
        ):
            message = f"invalid public tracking environment variable: {key}"
            raise ValueError(message)
        if key == "OPAQUE_DEPLOYMENT_IMAGE":
            registry = value.split("/", 1)[0]
            if (
                ":" in registry
                and not 0 < int(registry.rsplit(":", 1)[1]) <= _MAX_REGISTRY_PORT
            ):
                message = f"invalid public tracking environment variable: {key}"
                raise ValueError(message)
        context[key.removeprefix("OPAQUE_").lower()] = value
    return context


@dataclass(frozen=True)
class TrackingOptions:
    """Explicit tracking; fail_open tolerates only online communication failures."""

    mode: str = field(default_factory=lambda: os.getenv("WANDB_MODE", "auto"))
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "WANDB_BASE_URL", "https://jetbrains.wandb.io"
        )
    )
    entity: str = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY", "federated-compute")
    )
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "opaque"))
    group: str | None = field(default_factory=lambda: os.getenv("WANDB_RUN_GROUP"))
    name: str | None = field(default_factory=lambda: os.getenv("WANDB_NAME"))
    fail_open: bool = False

    def __post_init__(self):
        if type(self.fail_open) is not bool:
            message = "W&B fail_open must be a boolean"
            raise TypeError(message)
        if self.mode not in {"auto", "online", "offline", "disabled"}:
            message = "W&B mode must be auto, online, offline, or disabled"
            raise ValueError(message)
        url = urlparse(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
        ):
            message = "W&B base URL must be an HTTP(S) server URL without credentials"
            raise ValueError(message)
        if not self.entity or not self.project:
            message = "W&B entity and project must be explicit and nonempty"
            raise ValueError(message)


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Share tracking CLI options across trainers and the external mirror."""
    defaults = TrackingOptions()
    group = parser.add_argument_group("W&B tracking (public metrics only)")
    group.add_argument(
        "--wandb-mode",
        choices=("auto", "online", "offline", "disabled"),
        default=defaults.mode,
    )
    group.add_argument("--wandb-base-url", default=defaults.base_url)
    group.add_argument("--wandb-entity", default=defaults.entity)
    group.add_argument("--wandb-project", default=defaults.project)
    group.add_argument("--wandb-group", default=defaults.group)
    group.add_argument("--wandb-name", default=defaults.name)
    group.add_argument(
        "--wandb-fail-open",
        action="store_true",
        help="Continue with local outputs after online W&B communication failures",
    )


def options_from_args(args) -> TrackingOptions:
    """Read only tracking options, excluding potentially sensitive trainer args."""
    return TrackingOptions(
        fail_open=getattr(args, "wandb_fail_open", False),
        **{
            name: getattr(args, f"wandb_{name}")
            for name in ("mode", "base_url", "entity", "project", "group", "name")
        },
    )


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def public_metrics(row: dict) -> dict:
    """Allowlist held-out scalars; never forward arbitrary Trainer log keys."""
    result = {}
    for key, value in row.items():
        if not _finite(value):
            continue
        if key == "elapsed_seconds":
            result["runtime/evaluation_elapsed_seconds"] = value
        elif key.startswith("eval_"):
            suffix = key.removeprefix("eval_")
            if suffix in _EVAL_KEYS or re.fullmatch(
                r"layer_\d+_(?:expert_\d+_share|load_cv|max_to_mean_load|routing_entropy)",
                suffix,
            ):
                result[f"eval/{suffix}"] = value
    return result


def _metric_payload(row: dict) -> dict:
    step = row.get("step")
    if type(step) is not int or step < 0:
        message = "public metric rows require a nonnegative integer step"
        raise ValueError(message)
    metrics = public_metrics(row)
    return {"trainer/step": step, **metrics} if metrics else {}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class ExperimentTracker:
    """A run per arm/seed, with stable identity for a restartable external mirror.

    Only the public config dataclass should be passed, never TrainingArguments:
    the latter contains unpublished noise and sampling seeds. W&B console, code,
    Git and automatic machine metadata capture are disabled deliberately.
    A fail-open outage disables further uploads except for a final cleanup attempt.
    The receipt separates training status from upload_status; tracking_failure.json
    records only the first failed operation and a bounded exception type name.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        config: dict,
        arm: str,
        seed: int,
        experiment: str,
        options: TrackingOptions | None = None,
        resume: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.options = options
        self.resume = resume
        self.run = None
        self.completed = False
        self.status = "pending"
        self._mode = None
        self._run_id = None
        self._run_url = None
        self._upload_status = "pending"
        self._tracking_failed = False
        self._communication_errors = ()
        self._context = (
            _public_run_context()
            if options is not None and options.mode != "disabled"
            else {}
        )
        self.config = {
            key: value for key, value in config.items() if key in _CONFIG_KEYS
        }
        self.config.update(self._context)
        self.config.update(
            arm=arm,
            model_seed=seed,
            experiment=experiment,
            private=arm in ("dp", "dp_aux"),
            balancing_enabled=arm in ("reference_aux", "dp_aux", "trl_reference_aux"),
            balancing_kind="native_current_batch"
            if arm == "trl_reference_aux"
            else "lagged_noisy"
            if arm == "dp_aux"
            else "lagged_unnoised"
            if arm == "reference_aux"
            else "off",
            nonprivate_control="matched_per_record_clipping"
            if arm in ("reference", "reference_aux")
            else "native_trl_batch_gradient_clipping"
            if arm in ("trl_reference", "trl_reference_aux")
            else None,
            telemetry_scope="public evaluation and operational diagnostics only",
        )
        if arm not in ("dp", "dp_aux"):
            self.config["target_epsilon"] = None
        if arm not in ("reference_aux", "dp_aux", "trl_reference_aux"):
            self.config["router_aux_loss_coef"] = 0.0
        if arm != "dp_aux":
            self.config["load_noise_ratio"] = None

    def __enter__(self):
        return self

    def start(self) -> None:
        options = self.options
        if (
            options is None
            or options.mode == "disabled"
            or self.run is not None
            or self._tracking_failed
        ):
            return
        if options.fail_open:
            json.dumps(self.config, allow_nan=False)
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as error:
            message = "W&B requires the optional SDK: uv pip install wandb==0.30.0"
            if options.mode == "auto":
                warnings.warn(
                    message + "; continuing with local JSON logs", stacklevel=2
                )
                return
            raise RuntimeError(message) from error
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout

        mode = options.mode
        if mode == "auto":
            mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        self._mode = mode
        self._upload_status = "pending" if mode == "online" else "offline"
        self._communication_errors = (
            wandb.errors.CommError,
            ConnectionError,
            TimeoutError,
            RequestsConnectionError,
            RequestsTimeout,
        )
        directory = self.output_dir / "wandb"
        directory.mkdir(parents=True, exist_ok=True)
        identity = f"{options.entity}/{options.project}/{self.output_dir.resolve()}"
        self._run_id = hashlib.sha256(identity.encode()).hexdigest()[:8]
        group = options.group or self.config.get("name", self.config["experiment"])
        settings = wandb.Settings(
            base_url=options.base_url.rstrip("/"),
            console="off",
            disable_git=True,
            save_code=False,
            x_disable_stats=True,
            x_disable_meta=True,
            init_timeout=30,
        )
        with self._sdk_operation("init"):
            self.run = wandb.init(
                entity=options.entity,
                project=options.project,
                group=group,
                name=options.name
                or f"{group}-{self.config['arm']}-s{self.config['model_seed']}",
                id=self._run_id,
                resume="allow" if self.resume and mode == "online" else None,
                job_type=self.config["experiment"],
                tags=["moe", self.config["arm"], "public-data"],
                config=self.config,
                dir=str(directory),
                mode=mode,
                reinit="create_new",
                settings=settings,
            )
            self._run_id = self.run.id
            self._run_url = self.run.url
            self.run.define_metric("trainer/step")
            self.run.define_metric("*", step_metric="trainer/step", step_sync=False)
            self.run.summary["status"] = "running"
        self.status = "running"
        self._receipt()

    @contextmanager
    def _sdk_operation(self, operation):
        try:
            yield
        except self._communication_errors as error:
            if not self.options.fail_open or self._mode != "online":
                raise
            if not self._tracking_failed:
                self._tracking_failed = True
                self._upload_status = "failed"
                _write_json(
                    self.output_dir / "tracking_failure.json",
                    {
                        "operation": operation,
                        "exception_type": type(error).__name__[:128],
                    },
                )
            self._receipt()

    def _receipt(self):
        if self._run_id is not None:
            _write_json(
                self.output_dir / "wandb_run.json",
                {
                    "run_id": self._run_id,
                    "url": self._run_url,
                    "status": self.status,
                    "upload_status": self._upload_status,
                    "entity": self.options.entity,
                    "project": self.options.project,
                    "base_url": self.options.base_url.rstrip("/"),
                    **self._context,
                },
            )

    def log_metrics(self, row: dict) -> None:
        if self.run is None and not self._tracking_failed:
            return
        payload = _metric_payload(row)
        if payload and not self._tracking_failed:
            with self._sdk_operation("log_metrics"):
                self.run.log(payload)

    def log_progress(self, step: int, total: int, seconds: float) -> None:
        if self.run is None and not self._tracking_failed:
            return
        if (
            type(step) is not int
            or type(total) is not int
            or not 0 <= step <= total
            or total <= 0
        ):
            message = "invalid optimizer progress"
            raise ValueError(message)
        if not _finite(seconds) or seconds < 0:
            message = "elapsed training time must be finite and nonnegative"
            raise ValueError(message)
        if not self._tracking_failed:
            with self._sdk_operation("log_progress"):
                self.run.log(
                    {
                        "trainer/step": step,
                        "progress/completed_steps": step,
                        "progress/fraction": step / total,
                        "runtime/train_seconds": seconds,
                    }
                )
                self.run.summary["progress/completed_steps"] = step

    def complete(self, summary: dict) -> None:
        if summary.get("status") != "completed":
            message = "only a completed experiment summary can finish a run"
            raise ValueError(message)
        if self.run is None and not self._tracking_failed:
            self.completed = True
            return
        actual_config = {
            key: value
            for key, value in summary.get("config", {}).items()
            if key in _CONFIG_KEYS
        }
        if not self.config["private"]:
            actual_config["target_epsilon"] = None
        if not self.config["balancing_enabled"]:
            actual_config["router_aux_loss_coef"] = 0.0
        if self.config["arm"] != "dp_aux":
            actual_config["load_noise_ratio"] = None
        history = []
        public_summary = {}
        for phase in ("initial", "final"):
            row = summary.get(phase, {})
            if row:
                payload = _metric_payload(row)
                if payload:
                    history.append(payload)
                public_summary.update(
                    {
                        f"{phase}/{key}": value
                        for key, value in public_metrics(row).items()
                    }
                )
        privacy = summary.get("privacy", {})
        test_metrics = {
            key.replace("eval/", "test/", 1): value
            for key, value in public_metrics(summary.get("test", {})).items()
        }
        if test_metrics:
            step = summary["config"]["steps"]
            _metric_payload({"step": step})
            history.append({"trainer/step": step, **test_metrics})
            public_summary.update(test_metrics)
        public_summary.update(
            {f"privacy/{key}": privacy[key] for key in _PRIVACY_KEYS if key in privacy}
        )
        for key in (
            "training_seconds",
            "seconds_per_update",
            "relative_loss_improvement",
            "relative_ce_improvement",
            "learning_target",
            "learning_target_met",
            "peak_rss_mib",
            "parameters",
            "checks_run",
        ):
            if key in summary:
                public_summary[key] = summary[key]
        execution = summary.get("execution", {})
        for key in (
            "device",
            "device_name",
            "cuda_version",
            "peak_cuda_memory_allocated_bytes",
        ):
            if key in execution:
                public_summary[f"execution/{key}"] = execution[key]
        trainability = summary.get("trainability", {})
        for key in (
            "adapter_kind",
            "trainable_parameters",
            "total_parameters",
            "router_count",
            "expert_adapter_count",
        ):
            if key in trainability:
                public_summary[f"trainability/{key}"] = trainability[key]
        source = summary.get("provenance", {})
        provenance = {
            key: source[key]
            for key in (
                "pr_revision",
                "code_revision",
                "source_revision",
                "runner_sha256",
                "sft_sources_sha256",
                "python",
                "dependencies",
            )
            if key in source
        }
        if self.options.fail_open:
            json.dumps(
                [actual_config, history, public_summary, provenance], allow_nan=False
            )
        if not self._tracking_failed:
            with self._sdk_operation("complete"):
                self.run.config.update(actual_config)
                for payload in history:
                    self.run.log(payload)
                self.run.summary.update(public_summary)
                if provenance:
                    self.run.config.update({"provenance": provenance})
                self.run.summary["status"] = "completed"
        self.completed = True
        self.status = "completed"
        self._receipt()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None or self._tracking_failed:
            self.status = (
                "failed"
                if exc_type
                else "completed"
                if self.completed
                else "incomplete"
            )
            if self.run is not None and not self._tracking_failed:
                with self._sdk_operation("finish"):
                    self.run.summary["status"] = self.status
            self._receipt()
            if self.run is not None:
                with self._sdk_operation("finish"):
                    self.run.finish(exit_code=0 if self.status == "completed" else 1)
                    if not self._tracking_failed and self._mode == "online":
                        self._upload_status = "completed"
            self._receipt()
        return False
