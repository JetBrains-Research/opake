"""Offline deployment policy; importing this module never initializes ZenML."""

import re
from decimal import Decimal
from uuid import UUID

from .errors import refuse
from .gpu_policy import GPU_PROFILES

PARENT_REVISION = "bde08c91247cc2fe127e776bc72e402d8e7fef83"
STACK_NAME = "jbr-cmk-dev-jbr-eu-iceland1-fed-comp"
STACK_ID = "28320977-dcbd-423d-b52e-a59888d8a24f"
NAMESPACE = "federated-compute"
CLIENT_URL = "https://zenml.labs.jb.gg"
POD_URL = "https://zenml-external.labs.jb.gg"
ARTIFACT_STORE = "gs://gke-dev-dws-jbr-zenml"
DEPLOY_PATH = "deploy/zenml/moe_iceland"
SOURCE_LIMIT = 32 * 1024**2
SOURCE_FILE_LIMIT = 4096
BUNDLE_LIMITS = {"smoke": 16 * 1024**2, "pilot": 128 * 1024**2}
DEADLINES = {"smoke": 900, "pilot": 3600}
CPU_DEFAULTS = ("0.2", "0.2", "512Mi", "512Mi")
MAX_CPU = Decimal(8)
MAX_MEMORY = 16 * 1024**3
BUNDLE_ENTRY_LIMIT = 256


def validate_image(image: str) -> None:
    """Require a digest at a confirmed European Artifact Registry path."""
    if not re.fullmatch(
        r"europe(?:-west4)?-docker\.pkg\.dev/[a-z0-9][a-z0-9-]*/"
        r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9/._-]*@sha256:[0-9a-f]{64}",
        image,
    ):
        refuse(
            "Provide the confirmed europe-docker.pkg.dev or "
            "europe-west4-docker.pkg.dev project/repository/"
            "image@sha256:DIGEST; tags and registry-cache rewrites are not accepted."
        )


def validate_project(name: str) -> None:
    """Exclude training/TRACE sandbox projects from this public-data deployment."""
    if not name.strip() or name.casefold() == "learn" or "trace" in name.casefold():
        refuse("Explicitly confirm a non-learn, non-TRACE project.")


def cpu_quantity(value: str) -> Decimal:
    """Parse and bound a Kubernetes CPU quantity."""
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|[0-9]+m)", value):
        refuse(f"Invalid CPU quantity: {value!r}")
    result = Decimal(value[:-1]) / 1000 if value.endswith("m") else Decimal(value)
    if not 0 < result <= MAX_CPU:
        refuse("CPU must be positive and at most 8 cores.")
    return result


def memory_quantity(value: str) -> int:
    """Parse a bounded, explicit Mi/Gi memory quantity."""
    match = re.fullmatch(r"([1-9][0-9]*)(Mi|Gi)", value)
    if not match:
        refuse("Memory must be an explicit positive integer Mi or Gi quantity.")
    result = int(match[1]) * 1024 ** (2 if match[2] == "Mi" else 3)
    if result > MAX_MEMORY:
        refuse("This isolated CPU deployment caps memory at 16Gi per pod.")
    return result


def resources(values: tuple[str, str, str, str]) -> dict:
    """Build explicit requests/limits with a small fixed ephemeral-storage envelope."""
    cpu_request, cpu_limit, memory_request, memory_limit = values
    if cpu_quantity(cpu_request) > cpu_quantity(cpu_limit):
        refuse("CPU request exceeds its limit.")
    if memory_quantity(memory_request) > memory_quantity(memory_limit):
        refuse("Memory request exceeds its limit.")
    return {
        "requests": {
            "cpu": cpu_request,
            "memory": memory_request,
            "ephemeral-storage": "1Gi",
        },
        "limits": {
            "cpu": cpu_limit,
            "memory": memory_limit,
            "ephemeral-storage": "4Gi",
        },
    }


def runner_command(profile: str, arm: str, seed: int, output: str) -> list[str]:
    """Construct exactly the parent runner's checked, single-arm invocation."""
    if profile in GPU_PROFILES:
        return [
            "/opt/venv/bin/python",
            "-m",
            "deploy.zenml.moe_iceland.gpu_runner",
            "--plan",
            output.rsplit("/", 1)[0] + "/execution-plan.json",
        ]
    return [
        "/opt/venv/bin/python",
        "-m",
        "examples.moe_privacy.run",
        "--config",
        f"examples/moe_privacy/configs/{profile}.json",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--output-dir",
        output,
        "--checks",
    ]


def validate_execution_plan(plan: dict) -> None:
    """Revalidate serialized plans at both submission and execution boundaries."""
    fixed = {
        "schema": 1,
        "workspace": "prod",
        "client_url": CLIENT_URL,
        "stack_id": STACK_ID,
        "stack_name": STACK_NAME,
        "namespace": NAMESPACE,
        "artifact_store": ARTIFACT_STORE,
        "parent_revision": PARENT_REVISION,
    }
    if any(plan.get(key) != value for key, value in fixed.items()):
        refuse("Plan target differs from the isolated Iceland deployment.")
    if (
        plan.get("project_confirmed") is not True
        or plan.get("stack_confirmed") is not True
    ):
        refuse("The serialized plan lacks explicit project/stack confirmation.")
    validate_project(plan["project_name"])
    UUID(plan["project_id"])
    validate_image(plan["image"])
    gpu = plan["profile"] in GPU_PROFILES
    if gpu:
        from .gpu_policy import validate_gpu_plan

        validate_gpu_plan(plan)
    elif plan["profile"] not in DEADLINES or plan["arm"] not in {
        "reference",
        "dp",
        "dp_aux",
    }:
        refuse("Unknown experiment profile or arm.")
    if type(plan["seed"]) is not int or not 0 <= plan["seed"] < 2**32:
        refuse("Invalid public generation seed.")
    requested = plan["resources"]
    values = tuple(
        requested[bound][kind]
        for kind in ("cpu", "memory")
        for bound in ("requests", "limits")
    )
    if not gpu and requested != resources(values):
        refuse(
            "Unexpected resource keys or ephemeral allocation; no GPU profile exists."
        )
    if (
        plan["profile"] == "pilot" or requested != resources(CPU_DEFAULTS)
    ) and plan.get("resources_acknowledged") is not True:
        refuse("Explicit resource confirmation is required in the serialized plan.")
    if plan["profile"] == "pilot":
        if not plan.get("gate_run_id"):
            refuse("Pilot requires a successful smoke gate.")
        UUID(plan["gate_run_id"])
    for key in ("service_account", "step_service_account"):
        if not plan.get(key) or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", plan[key]
        ):
            refuse("Confirm existing service accounts in the execution plan.")
    if not re.fullmatch(r"/tmp/moe-iceland-[0-9a-f]{32}/output", plan["output_dir"]):
        refuse("Expected a fresh, bounded runtime output directory.")
    expected = runner_command(
        plan["profile"], plan["arm"], plan["seed"], plan["output_dir"]
    )
    if plan["runner_command"] != expected:
        refuse("Runner command differs from the checked parent-runner contract.")


def pod_settings(pod_resources: dict, deadline: int) -> dict:
    """Build the settings shared by step and orchestrator pods."""
    environment = {
        "ZENML_STORE_URL": POD_URL,
        "ZENML_CONFIG_PATH": "/tmp/moe-zenml",
        "ZENML_ANALYTICS_OPT_IN": "false",
        "ZENML_ENABLE_REPO_INIT_WARNINGS": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    security = {"runAsUser": 1000, "runAsGroup": 1000, "runAsNonRoot": True}
    return {
        "resources": pod_resources,
        "node_selectors": {"kubernetes.io/arch": "amd64"},
        "env": [{"name": name, "value": value} for name, value in environment.items()],
        "env_from": [],
        "volumes": [],
        "volume_mounts": [],
        "image_pull_secrets": [],
        "host_ipc": False,
        "container_security_context": {
            **security,
            "allowPrivilegeEscalation": False,
            "privileged": False,
            "capabilities": {"drop": ["ALL"]},
        },
        # ZenML expects V1PodSpec attribute names here, unlike the nested K8s dicts.
        "additional_pod_spec_args": {
            "security_context": {**security, "fsGroup": 1000},
            "active_deadline_seconds": deadline,
            "termination_grace_period_seconds": 30,
        },
    }


def pipeline_settings(plan: dict) -> dict:
    """Produce native Kubernetes settings and disable all automatic image builds."""
    gpu = plan["profile"] in GPU_PROFILES
    deadline = plan["timeout_seconds"] if gpu else DEADLINES[plan["profile"]]
    result = {
        "orchestrator.kubernetes": {
            "pod_settings": pod_settings(plan["resources"], deadline),
            "orchestrator_pod_settings": pod_settings(
                resources(CPU_DEFAULTS), deadline
            ),
            "service_account_name": plan.get("service_account"),
            "step_pod_service_account_name": plan.get("step_service_account"),
            "synchronous": False,
            "max_parallelism": 1,
            "active_deadline_seconds": deadline,
            "ttl_seconds_after_finished": 3600,
            "orchestrator_job_backoff_limit": 0,
            "backoff_limit_margin": 0,
            "pod_failure_policy": {
                "rules": [
                    {
                        "action": "FailJob",
                        "onExitCodes": {
                            "containerName": "main",
                            "operator": "NotIn",
                            "values": [0],
                        },
                    },
                    {
                        "action": "FailJob",
                        "onPodConditions": [
                            {"type": "DisruptionTarget", "status": "True"},
                        ],
                    },
                ]
            },
            "api_request_timeout": 30,
            "max_api_retries": 1,
            "job_monitoring_interval": 5,
            "pod_stop_grace_period": 30,
            "prevent_orchestrator_pod_caching": True,
            "job_name_prefix": "moe-iceland",
        },
        "docker": {
            "parent_image": plan.get("image"),
            "skip_build": True,
            "install_stack_requirements": False,
            "install_deployment_requirements": False,
            "disable_automatic_requirements_detection": True,
            "allow_including_files_in_images": False,
            "allow_download_from_code_repository": False,
            "allow_download_from_artifact_store": True,
        },
    }
    if gpu:
        from .gpu_policy import SCHEDULER, SCHEDULING_SECONDS, TOLERATIONS

        orchestrator = result["orchestrator.kubernetes"]
        # New Jobs expire quickly unless the observing client sees them running.
        orchestrator["active_deadline_seconds"] = SCHEDULING_SECONDS
        for role in ("pod_settings", "orchestrator_pod_settings"):
            pod = orchestrator[role]
            pod["image_pull_secrets"] = plan["image_pull_secrets"]
            pod["labels"] = {"opaque-run": plan["run_name"]}
            pod["additional_pod_spec_args"]["service_account_name"] = plan[
                "step_service_account" if role == "pod_settings" else "service_account"
            ]
        pod = orchestrator["pod_settings"]
        pod["scheduler_name"] = SCHEDULER
        pod["tolerations"] = TOLERATIONS
        scratch = plan["resources"]["requests"]["ephemeral-storage"]
        pod["volumes"] = [{"name": "scratch", "emptyDir": {"sizeLimit": scratch}}]
        pod["volume_mounts"] = [{"name": "scratch", "mountPath": "/scratch"}]
        replaced = {
            "CUDA_VISIBLE_DEVICES",
            "WANDB_MODE",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }
        pod["env"] = [entry for entry in pod["env"] if entry["name"] not in replaced]
        pod["env"].extend(
            {"name": name, "value": value}
            for name, value in {
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": "/scratch/hf",
                "TMPDIR": "/scratch",
                "WANDB_DIR": "/scratch/wandb",
                "WANDB_MODE": "online",
                "WANDB_BASE_URL": "https://jetbrains.wandb.io",
                "WANDB_ENTITY": "federated-compute",
                "WANDB_PROJECT": "opaque",
            }.items()
        )
        pod["env"].append(
            {
                "name": "WANDB_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {**plan["wandb_secret"], "optional": False},
                },
            }
        )
    return result


def validate_native_settings(plan: dict) -> dict:
    """Validate against the pinned API, without constructing a Client."""
    from importlib.metadata import version

    if version("zenml") != "0.96.4":
        refuse("Install ZenML 0.96.4 in the isolated deployment environment.")
    from zenml.config import DockerSettings
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorSettings,
    )

    settings = pipeline_settings(plan)
    KubernetesOrchestratorSettings.model_validate(settings["orchestrator.kubernetes"])
    DockerSettings.model_validate(settings["docker"])
    return settings
