"""Checks and bounded observation of this launcher's own Kubernetes workloads."""

import json
from datetime import UTC, datetime
from pathlib import Path

from .errors import refuse
from .gpu_policy import SCHEDULING_SECONDS
from .settings import NAMESPACE, POD_URL, pipeline_settings
from .source import digest_json


def target_fingerprint(stack, config) -> str:
    """Hash inherited settings for explicit review without printing secret values."""
    return digest_json(
        {
            "stack": str(stack.id),
            "secrets": stack.secrets,
            "environment": stack.environment,
            "orchestrator": config.model_dump(mode="json"),
            "components": {
                str(component.id): {
                    "configuration": component.configuration,
                    "secrets": component.secrets,
                    "environment": component.environment,
                }
                for components in stack.components.values()
                for component in components
            },
        }
    )


def _subset(actual, expected) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return all(
            key in expected and _subset(value, expected[key])
            for key, value in actual.items()
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return all(value in expected for value in actual)
    return actual == expected


def validate_inherited(stack, config, plan: dict) -> None:
    """Review inherited settings, allowing only a subset of the explicit safe pods."""
    from zenml.integrations.kubernetes.pod_settings import KubernetesPodSettings

    if plan.get("reviewed_target_sha256") != target_fingerprint(stack, config):
        refuse(
            "Review inherited stack settings and confirm reviewed_target_sha256 before submission."
        )
    allowed_environment = {"ZENML_STORE_URL": POD_URL}
    for item in [
        stack,
        *(component for values in stack.components.values() for component in values),
    ]:
        if item.secrets or not _subset(item.environment or {}, allowed_environment):
            refuse(
                "Inherited stack/component secrets or environment exceed the approved public allowlist."
            )
    settings = pipeline_settings(plan)["orchestrator.kubernetes"]
    for key in ("pod_settings", "orchestrator_pod_settings"):
        actual = getattr(config, key)
        if actual is None:
            continue
        expected = KubernetesPodSettings.model_validate(settings[key])
        if not _subset(
            actual.model_dump(exclude_defaults=True, exclude_none=True),
            expected.model_dump(exclude_defaults=True, exclude_none=True),
        ):
            refuse(
                "Stored pod settings exceed the explicit profile; review with the stack owner."
            )


def validate_cluster_references(core, plan: dict, config=None) -> None:
    """Check namespace-local references without logging credential contents."""
    if plan.get("service_account_verification", "inspect") == "admission":
        expected = {
            "service_account": getattr(config, "service_account_name", None),
            "step_service_account": getattr(
                config, "step_pod_service_account_name", None
            ),
        }
        if any(not name or plan[key] != name for key, name in expected.items()):
            refuse(
                "Admission verification requires exactly the registered existing service accounts."
            )
    else:
        for key in ("service_account", "step_service_account"):
            core.read_namespaced_service_account(
                plan[key], NAMESPACE, _request_timeout=30
            )
    for name in plan["image_pull_secrets"]:
        secret = core.read_namespaced_secret(name, NAMESPACE, _request_timeout=30)
        if secret.type != "kubernetes.io/dockerconfigjson" or not (
            secret.data or {}
        ).get(".dockerconfigjson"):
            refuse(
                "An approved image-pull secret lacks a Docker registry configuration."
            )
    reference = plan["wandb_secret"]
    secret = core.read_namespaced_secret(
        reference["name"], NAMESPACE, _request_timeout=30
    )
    if not (secret.data or {}).get(reference["key"]):
        refuse(
            "The required W&B runtime secret/key is unavailable in federated-compute."
        )


class RunObserver:
    """Retain bounded diagnostics and enforce startup limits on this run only."""

    def __init__(self, core, batch, run_id: str, plan: dict, directory: Path):
        self.core, self.batch = core, batch
        self.selector = f"run_id={run_id}"
        self.plan = plan
        self.directory = directory
        self.started = datetime.now(UTC)
        self.extended = set()
        self.run_id = run_id

    def inspect(self) -> list:
        """Persist status/log tails before deleted pods lose their evidence."""
        pods = self.core.list_namespaced_pod(
            NAMESPACE, label_selector=self.selector, _request_timeout=30
        ).items
        jobs = self.batch.list_namespaced_job(
            NAMESPACE, label_selector=self.selector, _request_timeout=30
        ).items
        self.directory.mkdir(parents=True, exist_ok=True)
        evidence = {
            "run_id": self.run_id,
            "observed_at": datetime.now(UTC).isoformat(),
            "pods": [],
        }
        for pod in pods:
            if (pod.metadata.labels or {}).get("run_id") != self.run_id:
                refuse("Refusing to observe a pod outside this run.")
            statuses = pod.status.container_statuses or []
            waiting = [
                item.state.waiting.reason
                for item in statuses
                if item.state and item.state.waiting
            ]
            evidence["pods"].append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "waiting": waiting,
                }
            )
            events = self.core.list_namespaced_event(
                NAMESPACE,
                field_selector=f"involvedObject.uid={pod.metadata.uid}",
                _request_timeout=30,
            )
            (self.directory / f"{pod.metadata.name}.events.json").write_text(
                json.dumps(
                    [
                        {"reason": event.reason, "message": event.message}
                        for event in events.items[-50:]
                    ],
                    indent=2,
                )
            )
            if any(
                item.state and (item.state.running or item.state.terminated)
                for item in statuses
            ):
                log = self.core.read_namespaced_pod_log(
                    pod.metadata.name,
                    NAMESPACE,
                    container="main",
                    tail_lines=80,
                    limit_bytes=32768,
                    _request_timeout=30,
                )
                # Logs stay local; pod specifications/environments/secrets are never serialized.
                (self.directory / f"{pod.metadata.name}.log").write_text(log)
        (self.directory / "status.json").write_text(json.dumps(evidence, indent=2))
        now = datetime.now(UTC)
        for job in jobs:
            if (job.metadata.labels or {}).get("run_id") != self.run_id:
                refuse("Refusing to modify a job outside this run.")
            job_pods = [
                pod
                for pod in pods
                if (pod.metadata.labels or {}).get("job-name") == job.metadata.name
            ]
            running = any(
                pod.status.phase in {"Running", "Succeeded"} for pod in job_pods
            )
            origin = (
                job.status.start_time or job.metadata.creation_timestamp or self.started
            )
            if not running and (now - origin).total_seconds() >= SCHEDULING_SECONDS:
                self.stop(jobs)
                refuse(
                    "GPU image-pull/scheduling exceeded 15 minutes; evidence retained, no retry."
                )
            if running and job.metadata.name not in self.extended:
                remaining = self.plan["timeout_seconds"] - int(
                    (origin - self.started).total_seconds()
                )
                self.batch.patch_namespaced_job(
                    job.metadata.name,
                    NAMESPACE,
                    {"spec": {"activeDeadlineSeconds": max(1, remaining)}},
                    _request_timeout=30,
                )
                self.extended.add(job.metadata.name)
        return jobs

    def stop(self, jobs: list | None = None) -> None:
        """Stop only Jobs carrying this run's exact ID; never alter shared settings."""
        if jobs is None:
            jobs = self.batch.list_namespaced_job(
                NAMESPACE, label_selector=self.selector, _request_timeout=30
            ).items
        for job in jobs:
            if (job.metadata.labels or {}).get("run_id") != self.run_id:
                refuse("Refusing to stop a job outside this run.")
            self.batch.delete_namespaced_job(
                job.metadata.name,
                NAMESPACE,
                propagation_policy="Foreground",
                _request_timeout=30,
            )
