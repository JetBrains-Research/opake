"""Launch Opaque pipelines on the managed Federated Compute ZenML stack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, ClassVar

import yaml
from jetbrains.mlops.zenml import (
    MountConfiguration,
    PodConfiguration,
    ZenMLEnv,
    attach_secret_ref,
    get_step_settings,
)

from zenml.client import Client
from zenml.config import DockerSettings
from zenml.enums import StackComponentType

ADAPTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_PROJECT = "learn"
DEFAULT_STACK = "jbr-cmk-dev-jbr-eu-iceland1-fed-comp"

PROFILE_PATHS = {
    "probe": ADAPTER_ROOT / "configs" / "probe.yaml",
    "probe-gpu": ADAPTER_ROOT / "configs" / "probe-gpu.yaml",
    "smoke": ADAPTER_ROOT / "configs" / "smoke.yaml",
    "production": ADAPTER_ROOT / "configs" / "production.yaml",
}
PROFILE_RESOURCES = {
    "probe": {"cpu_count": 0.1, "memory_gb": 2.0},
    "probe-gpu": {"cpu_count": 4.0, "memory_gb": 16.0, "gpu_count": 1},
    "smoke": {"cpu_count": 16.0, "memory_gb": 64.0, "gpu_count": 1},
    "production": {"cpu_count": 16.0, "memory_gb": 64.0, "gpu_count": 1},
}
GPU_PROFILES = {"probe-gpu", "smoke", "production"}


class OpaqueRuntimeEnv(ZenMLEnv):
    """Runtime-only environment and Kubernetes secret references."""

    BASIC_ENVS: ClassVar[list[dict[str, object]]] = [
        attach_secret_ref("HF_TOKEN", "jbr-fed", "HF_TOKEN"),
        attach_secret_ref("WANDB_API_KEY", "jbr-fed", "WANDB_API_KEY"),
        {"name": "ZENML_STORE_URL", "value": "https://zenml-external.labs.jb.gg"},
        {"name": "WANDB_BASE_URL", "value": "https://jetbrains.wandb.io"},
        {"name": "WANDB_ENTITY", "value": "federated-compute"},
        {"name": "WANDB_PROJECT", "value": "opaque"},
        {"name": "HF_HOME", "value": "/tmp/huggingface"},
        {"name": "WANDB_DIR", "value": "/tmp/wandb"},
        {"name": "TMPDIR", "value": "/tmp"},
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILE_PATHS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--stack", default=DEFAULT_STACK)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-checkpoint",
        help="Explicit ZenML checkpoint artifact UUID or name@version.",
    )
    resume_group.add_argument(
        "--resume-run",
        help="Resume from the highest-step checkpoint produced by this ZenML run.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "Override any SFTJobConfig field for a training profile; VALUE uses "
            "YAML syntax and the option may be repeated."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the profile without connecting or launching.",
    )
    return parser


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    """Parse and validate repeated trainer configuration overrides."""
    overrides: dict[str, Any] = {}
    for value in values:
        field, separator, raw_value = value.partition("=")
        if not separator or not field.strip():
            message = f"Invalid --set value {value!r}; expected FIELD=VALUE"
            raise ValueError(message)
        field = field.strip()
        if field in overrides:
            message = f"SFT configuration field {field!r} was overridden twice"
            raise ValueError(message)
        overrides[field] = yaml.safe_load(raw_value)
    return overrides


def _load_sft_config_contract():
    """Load the example-owned SFT contract without publishing it as a package API."""
    examples_root = str(PROJECT_ROOT / "examples")
    sys.path.insert(0, examples_root)
    try:
        from opaque_examples.sft.config import SFTJobConfig, resolve_sft_config
    finally:
        sys.path.remove(examples_root)
    return SFTJobConfig, resolve_sft_config


def _load_sft_config_type():
    """Return the typed SFT job contract for profile validation and tests."""
    config_type, _ = _load_sft_config_contract()
    return config_type


def _validate_training_overrides(
    profile: str, overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge and validate a training profile with explicit field overrides."""
    profile_data = yaml.safe_load(PROFILE_PATHS[profile].read_text(encoding="utf-8"))
    configured = profile_data.get("parameters", {}).get("config")
    if not isinstance(configured, dict):
        message = f"Training profile {profile!r} must define parameters.config"
        raise ValueError(message)

    merged = dict(configured)
    merged.update(overrides)
    config_type, resolve_sft_config = _load_sft_config_contract()
    resolved = resolve_sft_config(merged)

    effective: dict[str, Any] = {}
    for field in fields(config_type):
        value = getattr(resolved, field.name)
        effective[field.name] = list(value) if isinstance(value, tuple) else value
    json.dumps(effective, allow_nan=False)
    return effective


def _find_latest_checkpoint_for_run(client: Client, run_id: str) -> str:
    """Resolve a prior run to one immutable checkpoint artifact version ID."""
    from adapters.zenml.checkpoints import find_latest_checkpoint

    return find_latest_checkpoint(client, run_id)


def _configured_pipeline(profile: str):
    from zenml.utils.source_utils import set_custom_source_root

    set_custom_source_root(str(PROJECT_ROOT))
    from adapters.zenml.pipelines import infrastructure_probe_pipeline, sft_pipeline

    config_path = PROFILE_PATHS[profile]
    if not config_path.is_file():
        message = f"ZenML profile does not exist: {config_path}"
        raise FileNotFoundError(message)
    pipeline = (
        infrastructure_probe_pipeline
        if profile in {"probe", "probe-gpu"}
        else sft_pipeline
    )
    configured = pipeline.with_options(config_path=str(config_path))
    docker_settings = DockerSettings.model_validate(
        configured.configuration.settings["docker"].model_dump()
    )
    additional_pod_spec_args: dict[str, object] = {}
    if profile in GPU_PROFILES:
        additional_pod_spec_args = {
            "scheduler_name": "gpu-binpack-scheduler",
            "tolerations": [
                {
                    "effect": "NoSchedule",
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                }
            ],
        }
    pod_configuration = PodConfiguration(
        env=OpaqueRuntimeEnv(),
        additional_pod_spec_args=additional_pod_spec_args,
        auto_mount_shm_on_gpu=profile in GPU_PROFILES,
        storage_configuration=(
            [MountConfiguration(mount_path="/tmp", size_gb=200)]
            if profile in {"smoke", "production"}
            else []
        ),
        **PROFILE_RESOURCES[profile],
    )
    settings = get_step_settings(
        docker_settings=docker_settings,
        pod_configuration=pod_configuration,
    )
    return pipeline.with_options(config_path=str(config_path), settings=settings)


def _resolve_source_commit_sha() -> str:
    configured = os.environ.get("OPAQUE_SOURCE_COMMIT_SHA", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _component_summary(stack) -> dict[str, list[dict[str, str]]]:
    return {
        component_type.value: [
            {"name": component.name, "flavor": component.flavor.name}
            for component in components
        ]
        for component_type, components in stack.components.items()
    }


def _validate_external_stack(client: Client, stack_name: str):
    stack = client.get_stack(stack_name)
    required = {
        StackComponentType.ORCHESTRATOR,
        StackComponentType.IMAGE_BUILDER,
        StackComponentType.ARTIFACT_STORE,
        StackComponentType.CONTAINER_REGISTRY,
    }
    missing = sorted(
        component.value for component in required if not stack.components.get(component)
    )
    if missing:
        message = f"stack {stack_name!r} is missing components: {', '.join(missing)}"
        raise RuntimeError(message)

    orchestrator = stack.components[StackComponentType.ORCHESTRATOR][0]
    artifact_store = stack.components[StackComponentType.ARTIFACT_STORE][0]
    if orchestrator.flavor.name == "local":
        message = f"stack {stack_name!r} does not have a remote orchestrator"
        raise RuntimeError(message)
    if artifact_store.flavor.name == "local":
        message = f"stack {stack_name!r} does not have a durable remote artifact store"
        raise RuntimeError(message)
    return stack


def _validate_local_builder(stack) -> None:
    image_builder = stack.components[StackComponentType.IMAGE_BUILDER][0]
    if image_builder.flavor.name != "local":
        return
    if shutil.which("docker") is None and shutil.which("podman") is None:
        message = (
            f"stack image builder {image_builder.name!r} is local, but neither Docker "
            "nor Podman is available. Install a container engine or launch with a "
            "prebuilt image configuration."
        )
        raise RuntimeError(message)


def launch(
    profile: str,
    *,
    project: str = DEFAULT_PROJECT,
    stack_name: str = DEFAULT_STACK,
    resume_checkpoint: str | None = None,
    resume_run: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Validate external infrastructure and launch one configured pipeline."""
    client = Client()
    client.set_active_project(project)
    stack = _validate_external_stack(client, stack_name)
    _validate_local_builder(stack)
    client.activate_stack(stack_name)
    print(json.dumps({"stack": stack.name, "components": _component_summary(stack)}))

    configured = _configured_pipeline(profile)
    effective_overrides = dict(overrides or {})
    if profile in {"smoke", "production"}:
        effective_config = _validate_training_overrides(profile, effective_overrides)
        selected_checkpoint = resume_checkpoint
        if resume_run is not None:
            selected_checkpoint = _find_latest_checkpoint_for_run(client, resume_run)
            print(
                json.dumps(
                    {
                        "resume_run": resume_run,
                        "resume_checkpoint": selected_checkpoint,
                    }
                )
            )
        configured(
            config=effective_config,
            source_commit_sha=_resolve_source_commit_sha(),
            resume_checkpoint=selected_checkpoint,
        )
    else:
        if resume_checkpoint is not None or resume_run is not None:
            message = "checkpoint resume options are valid only for training profiles"
            raise ValueError(message)
        if effective_overrides:
            message = "--set is valid only for training profiles"
            raise ValueError(message)
        configured()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    overrides = _parse_overrides(args.set)
    configured = _configured_pipeline(args.profile)
    if args.profile in {"smoke", "production"}:
        _validate_training_overrides(args.profile, overrides)
    else:
        if args.resume_checkpoint is not None or args.resume_run is not None:
            message = "checkpoint resume options are valid only for training profiles"
            raise ValueError(message)
        if overrides:
            message = "--set is valid only for training profiles"
            raise ValueError(message)
    if args.validate_only:
        print(f"Validated ZenML profile: {PROFILE_PATHS[args.profile]}")
        return 0

    del configured
    launch(
        args.profile,
        project=args.project,
        stack_name=args.stack,
        resume_checkpoint=args.resume_checkpoint,
        resume_run=args.resume_run,
        overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
