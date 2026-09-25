"""Authenticated checks, reachable only through an explicit submission process."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from .errors import refuse
from .gpu_policy import GPU_PROFILES, validate_config
from .settings import (
    ARTIFACT_STORE,
    CLIENT_URL,
    DEADLINES,
    NAMESPACE,
    STACK_ID,
    STACK_NAME,
    validate_execution_plan,
    validate_native_settings,
    validate_project,
)


def validate_target(client, plan: dict) -> None:
    """Discover and validate accessible prod targets without changing active config."""
    from zenml.enums import StackComponentType
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorConfig,
    )

    if str(client.zen_store.config.url).rstrip("/") != CLIENT_URL:
        refuse("The submitting client must remain on the prod laptop URL.")
    info = client.zen_store.get_store_info()
    if info.version != "0.96.4":
        refuse("Server/client version mismatch; confirm the supported deployment.")
    if info.pro_workspace_name and info.pro_workspace_name not in {"prod", "jcp-prod"}:
        refuse("The server reports a non-prod workspace; do not submit.")
    # Discover actual access rather than trusting a briefing or a cached active name.
    projects = client.list_projects(id=UUID(plan["project_id"]), hydrate=True)
    stacks = client.list_stacks(id=UUID(STACK_ID), hydrate=True)
    if len(projects.items) != 1 or len(stacks.items) != 1:
        refuse("The confirmed project/stack is not uniquely accessible.")
    project, stack = projects.items[0], stacks.items[0]
    validate_project(project.name)
    if str(project.id) != plan["project_id"] or project.name != plan["project_name"]:
        refuse("Actual project name/ID differs from the explicit confirmation.")
    if str(stack.id) != STACK_ID or stack.name != STACK_NAME:
        refuse("Actual stack name/ID differs from the Iceland confirmation.")
    if (
        client.active_project.id != project.id
        or client.active_stack_model.id != stack.id
    ):
        refuse("Per-invocation project/stack selection was not honored; aborting.")
    gpu = plan["profile"] in GPU_PROFILES
    if not gpu and (stack.secrets or stack.environment):
        refuse("Unreviewed stack secrets/environment; ask the stack owner to review.")
    components = stack.components
    orchestrators = components.get(StackComponentType.ORCHESTRATOR, [])
    stores = components.get(StackComponentType.ARTIFACT_STORE, [])
    if len(orchestrators) != 1 or len(stores) != 1:
        refuse("Expected one native Kubernetes orchestrator and GCS artifact store.")
    orchestrator, store = orchestrators[0], stores[0]
    if orchestrator.flavor_name != "kubernetes" or store.flavor_name != "gcp":
        refuse("The actual stack is not native Kubernetes + GCS.")
    registries = components.get(StackComponentType.CONTAINER_REGISTRY, [])
    if len(registries) != 1 or registries[0].flavor_name != "gcp":
        refuse("Expected the stack's registered GCP container repository.")
    registry_uri = str(registries[0].configuration.get("uri", "")).rstrip("/")
    if not registry_uri or not plan["image"].startswith(registry_uri + "/"):
        refuse("The image must belong to the stack's actual registered repository.")
    config = KubernetesOrchestratorConfig.model_validate(orchestrator.configuration)
    if config.kubernetes_namespace != NAMESPACE or config.local:
        refuse("Actual cluster namespace/local settings do not match Iceland.")
    if config.pass_zenml_token_as_secret and not gpu:
        refuse("Stack uses Kubernetes token secrets; owner approval/change required.")
    if str(store.configuration.get("path", "")).rstrip("/") != ARTIFACT_STORE:
        refuse("Actual artifact-store bucket differs from the confirmed bucket.")
    for component_list in components.values():
        for component in component_list:
            if not gpu and (component.secrets or component.environment):
                refuse("Unreviewed component secrets/environment; aborting.")
    # Do not inherit hidden volumes, GPU requests, credentials, selectors or sidecars.
    for pod in (config.pod_settings, config.orchestrator_pod_settings):
        if not gpu and pod and any(pod.model_dump(exclude_defaults=True).values()):
            refuse(
                "Stored pod settings require explicit review. This launcher will not "
                "merge them with its isolated profile or mutate the shared stack."
            )
    if gpu:
        from kubernetes.client import CoreV1Api

        from .cluster import validate_cluster_references, validate_inherited

        validate_inherited(stack, config, plan)
        validate_cluster_references(
            CoreV1Api(client.active_stack.orchestrator.get_kube_client()), plan, config
        )
    for key in ("service_account", "step_service_account"):
        if not plan.get(key):
            refuse("Confirm existing orchestrator and step service accounts.")
    candidates = client.list_pipeline_runs(
        project=project.id,
        stack_id=stack.id,
        pipeline_name="moe_iceland",
        in_progress=True,
        size=20,
    )
    # ZenML's in_progress filter can retain terminal historical rows; trust status.
    running = [
        item
        for item in candidates.items
        if not getattr(item.status, "is_finished", False)
    ]
    if running and gpu:
        # A stale status is not permission to duplicate a possibly running job.
        # Refresh only our own run when Kubernetes still has terminal evidence.
        from zenml.models import PipelineRunUpdate

        for item in running:
            old = client.get_pipeline_run(item.id, project=project.id)
            if not old.name.startswith("moe-iceland-"):
                break
            if getattr(old.status, "is_finished", False):
                continue
            status, _ = client.active_stack.orchestrator.fetch_status(old)
            if status and status.is_finished and status.value != "completed":
                client.zen_store.update_run(old.id, PipelineRunUpdate(status=status))
        candidates = client.list_pipeline_runs(
            project=project.id,
            stack_id=stack.id,
            pipeline_name="moe_iceland",
            in_progress=True,
            size=20,
        )
        running = [
            item
            for item in candidates.items
            if not getattr(item.status, "is_finished", False)
        ]
    if running:
        refuse("An Iceland MoE job is still in progress; finish/cancel it first.")


def completed_bundle(client, run_id: str, plan: dict):
    """Check status before accessing step metadata or loading any artifacts."""
    from zenml.enums import ExecutionStatus

    run = client.get_pipeline_run(UUID(run_id), project=UUID(plan["project_id"]))
    if run.status != ExecutionStatus.COMPLETED:
        refuse("The smoke gate/run is not completed successfully.")
    if str(run.project_id) != plan["project_id"] or not run.stack:
        refuse("Gate project/stack is missing or mismatched.")
    if (
        str(run.stack.id) != STACK_ID
        or not run.pipeline
        or run.pipeline.name != "moe_iceland"
    ):
        refuse("Gate is not an Iceland MoE run on the confirmed stack.")
    if len(run.steps) != 1:
        refuse("Expected exactly one step, with no fan-out.")
    step = next(iter(run.steps.values()))
    if step.status != ExecutionStatus.COMPLETED:
        refuse("The gate step did not complete (cached/skipped is not a gate).")
    outputs = step.outputs.get("bundle", [])
    if len(outputs) != 1:
        refuse("The completed run has no unique materialized output bundle.")
    artifact = outputs[0]
    if not artifact.uri.startswith(ARTIFACT_STORE + "/"):
        refuse("The bundle is not in the confirmed GCS artifact store.")
    if (
        artifact.materializer.import_path
        != "zenml.materializers.path_materializer.PathMaterializer"
    ):
        refuse("Gate did not use the expected directory materializer.")
    return run, artifact


def reject_duplicate_training(client, plan: dict) -> None:
    """Do not rerun a recorded private/native configuration, even after failure."""
    if plan["profile"] != "sft-train-v1":
        return
    for page in range(1, 21):
        runs = client.list_pipeline_runs(
            project=UUID(plan["project_id"]),
            stack_id=UUID(STACK_ID),
            pipeline_name="moe_iceland",
            size=50,
            page=page,
        )
        for item in runs.items:
            old = client.get_pipeline_run(item.id, project=UUID(plan["project_id"]))
            previous = old.config.parameters.get("plan", {})
            status = getattr(old.status, "value", str(old.status))
            # Only completed successful runs block reuse. Failed probes/smokes may be
            # corrected (for example after an image fix) without fabricating success.
            if status != "completed":
                continue
            if all(
                previous.get(key) == plan[key]
                for key in ("profile", "backend", "config_sha256", "arm", "seed")
            ):
                refuse(
                    f"Configuration already submitted as {old.id}; recover evidence, never auto-retrain."
                )
        if page * 50 >= runs.total:
            return
    refuse(
        "Run history exceeds the bounded duplicate check; reconcile it before submitting."
    )


def validate_gate(client, plan: dict) -> None:
    """Require a successful checked smoke run and its actual uploaded bundle."""
    from .bundle import validate_bundle

    run, artifact = completed_bundle(client, plan["gate_run_id"], plan)
    gate_plan = run.config.parameters.get("plan", {})
    for key in ("source_sha256", "image", "image_contract", "project_id"):
        if gate_plan.get(key) != plan.get(key):
            refuse(
                f"Smoke gate {key} differs; rerun smoke for this exact source/image."
            )
    if plan["profile"] in GPU_PROFILES:
        for key in (
            "resource_profile",
            "service_account",
            "step_service_account",
            "service_account_verification",
            "image_pull_secrets",
            "wandb_secret",
            "reviewed_target_sha256",
        ):
            if gate_plan.get(key) != plan.get(key):
                refuse(f"GPU gate {key} differs from the requested execution.")
        # Scratch may grow from probe → smoke → train; keep accelerator/host compute fixed.
        gate_resources = gate_plan.get("resources") or {}
        plan_resources = plan.get("resources") or {}
        for side in ("requests", "limits"):
            gate_side = dict(gate_resources.get(side) or {})
            plan_side = dict(plan_resources.get(side) or {})
            gate_side.pop("ephemeral-storage", None)
            plan_side.pop("ephemeral-storage", None)
            if gate_side != plan_side:
                refuse(
                    "GPU gate CPU/memory/GPU allocation differs from the requested execution."
                )
        expected = (
            "gpu-probe-v1" if plan["profile"] == "sft-smoke-v1" else "sft-smoke-v1"
        )
        if gate_plan.get("profile") != expected:
            refuse(
                "Require the GPU/artifact probe before smoke and real pretrained smoke before full jobs."
            )
        if expected != "gpu-probe-v1" and gate_plan.get("backend") != plan["backend"]:
            refuse("GPU training gate backend differs from the requested execution.")
        if expected == "sft-smoke-v1" and gate_plan.get("arm") != (
            "dp_aux" if plan["backend"] == "opaque" else "trl_reference_aux"
        ):
            refuse(
                "The pretrained gate must exercise the backend's auxiliary objective."
            )
        directory = artifact.load(disable_cache=True)
        if not isinstance(directory, Path):
            refuse("GPU gate artifact did not materialize a directory.")
        validate_bundle(directory, expected)
        receipt = json.loads((directory / "deployment.json").read_text())
        if receipt.get("plan") != gate_plan or receipt.get("runner_returncode") != 0:
            refuse("GPU gate receipt does not match the completed run.")
        if not receipt.get("cuda_verified") or not receipt.get("wandb_verified"):
            refuse(
                "GPU gate must verify real CUDA computation and public W&B connectivity."
            )
        if expected == "gpu-probe-v1":
            probe = json.loads((directory / "probe.json").read_text())
            if plan["backend"] not in probe.get("verified_backends", []):
                refuse(
                    "The infrastructure probe did not verify this backend in a fresh process."
                )
        if expected == "sft-smoke-v1" and not receipt.get("adapter_reload_verified"):
            refuse("The pretrained gate must reload its materialized trainables.")
        return
    if gate_plan.get("profile") != "smoke" or gate_plan.get("arm") != "dp_aux":
        refuse("Pilot requires a completed dp_aux smoke gate with --checks.")
    directory = artifact.load(disable_cache=True)
    if not isinstance(directory, Path):
        refuse("Gate artifact did not materialize a directory.")
    validate_bundle(directory, "smoke")
    receipt = json.loads((directory / "deployment.json").read_text())
    if receipt.get("plan") != gate_plan or receipt.get("runner_returncode") != 0:
        refuse("Smoke artifact receipt does not match the completed run.")
    if receipt.get("checks_requested") is not True:
        refuse("The smoke runner did not request the real checks.")


def submit_one(root: Path, plan: dict) -> dict:
    """Submit once, wait for completion with a deadline, and report the artifact URI."""
    validate_execution_plan(plan)
    from zenml.client import Client
    from zenml.utils import source_utils

    source_utils.set_custom_source_root(str(root.resolve(strict=True)))
    from .pipeline import moe_iceland

    settings = validate_native_settings(plan)
    client = Client()
    validate_target(client, plan)
    gpu = plan["profile"] in GPU_PROFILES
    if gpu:
        validate_config(root, plan)
        reject_duplicate_training(client, plan)
    if plan["profile"] == "pilot" or (gpu and plan["profile"] != "gpu-probe-v1"):
        validate_gate(client, plan)
    if gpu:
        plan = {
            **plan,
            "run_deadline_utc": (
                datetime.now(UTC) + timedelta(seconds=plan["timeout_seconds"])
            ).isoformat(),
        }
        validate_execution_plan(plan)
        (root.parent / "execution-plan.json").write_text(json.dumps(plan, indent=2))
    run = moe_iceland.with_options(
        run_name=plan["run_name"],
        settings=settings,
    )(plan=plan)
    if run is None:
        message = "ZenML returned no run; do not retry without checking the server."
        raise RuntimeError(message)
    print(
        json.dumps({"submitted_run_id": str(run.id), "run_name": plan["run_name"]}),
        flush=True,
    )
    timeout = plan["timeout_seconds"] if gpu else DEADLINES[plan["profile"]]
    observer = None
    if gpu:
        from kubernetes.client import BatchV1Api, CoreV1Api

        from .cluster import RunObserver

        api = client.active_stack.orchestrator.get_kube_client()
        observer = RunObserver(
            CoreV1Api(api), BatchV1Api(api), str(run.id), plan, root.parent / "evidence"
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = client.get_pipeline_run(run.id, project=UUID(plan["project_id"]))
        if observer:
            observer.inspect()
        if current.status.is_finished:
            _, artifact = completed_bundle(client, str(run.id), plan)
            if gpu:
                from .bundle import validate_bundle

                directory = artifact.load(disable_cache=True)
                validate_bundle(directory, plan["profile"])
            return {
                "run_id": str(run.id),
                "artifact_uri": artifact.uri,
                "status": "completed",
            }
        time.sleep(5)
    if observer:
        observer.stop()
    message = (
        f"Run {run.id} exceeded the local wait limit. Its Kubernetes Jobs have an "
        "active deadline; inspect/cancel them before retrying."
    )
    raise TimeoutError(message)
