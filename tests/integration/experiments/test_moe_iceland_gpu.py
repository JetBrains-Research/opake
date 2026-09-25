"""GPU policy checks against real ZenML/Kubernetes serialization, without a cluster."""

import copy
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy.zenml.moe_iceland import gpu_policy, settings
from deploy.zenml.moe_iceland.cluster import RunObserver, validate_cluster_references
from deploy.zenml.moe_iceland.launch import make_plan, parser, submission_context
from deploy.zenml.moe_iceland.source import sha256

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "europe-west4-docker.pkg.dev/test-project/test-repo/moe@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("GPU adapter tests must not contact a server")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")


@pytest.fixture
def gpu_plan():
    config = "sft_balancing_aux0001_ratio1.json"
    return make_plan(
        parser().parse_args(
            [
                "--profile",
                "gpu-probe-v1",
                "--project-name",
                gpu_policy.PROJECT_NAME,
                "--project-id",
                gpu_policy.PROJECT_ID,
                "--confirm-project",
                gpu_policy.PROJECT_NAME,
                "--confirm-stack-id",
                settings.STACK_ID,
                "--image",
                IMAGE,
                "--service-account",
                "test-orchestrator",
                "--step-service-account",
                "test-step",
                "--acknowledge-confirmed-resources",
                "--config-name",
                config,
                "--config-sha256",
                sha256(ROOT / "examples/moe_privacy/configs" / config),
                "--source-sha256",
                "c" * 64,
                "--authorization-id",
                str(uuid4()),
                "--deadline-utc",
                (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "--gpu-seconds",
                "7200",
                "--timeout-seconds",
                "900",
                "--image-pull-secret",
                "test-registry",
                "--wandb-secret-name",
                "test-tracking",
                "--wandb-secret-key",
                "WANDB_API_KEY",
            ]
        )
    )


def test_gpu_plan_and_exact_config(gpu_plan):
    settings.validate_execution_plan(gpu_plan)
    config = gpu_policy.validate_config(ROOT, gpu_plan)
    assert config["steps"] == 256
    assert config["router_aux_loss_coef"] == 0.001
    assert gpu_plan["runner_command"][2] == "deploy.zenml.moe_iceland.gpu_runner"
    gpu_plan["config_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Configuration hash"):
        gpu_policy.validate_config(ROOT, gpu_plan)


def test_generation_requires_frozen_checkpoint_and_benchmark(gpu_plan):
    gpu_plan.update(
        profile="generate-v1",
        gate_run_id=str(uuid4()),
        benchmark="humaneval",
        checkpoint={
            "uri": "gs://gke-dev-dws-jbr-zenml/weights/data.tar.gz",
            "sha256": "d" * 64,
        },
        benchmark_input={
            "uri": "gs://gke-dev-dws-jbr-zenml/benchmarks/humaneval.jsonl",
            "sha256": "e" * 64,
        },
    )
    settings.validate_execution_plan(gpu_plan)
    gpu_plan["benchmark_input"]["sha256"] = "mutable"
    with pytest.raises(ValueError, match="checksummed benchmark_input"):
        settings.validate_execution_plan(gpu_plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("project_name", "learn"),
        ("project_id", str(uuid4())),
        ("accelerator", "cpu"),
        ("backend", "trl"),
        ("arm", "trl_reference_aux"),
        ("config_name", "../arbitrary.json"),
        ("seed", 1),
        ("config_sha256", ""),
        ("source_sha256", ""),
        ("image_pull_secrets", []),
        ("wandb_secret", {"value": "not-allowed"}),
        ("resources_acknowledged", False),
        ("timeout_seconds", 901),
        ("checkpoint", {"uri": "arbitrary-resume"}),
    ],
)
def test_gpu_plan_rejects_drift(gpu_plan, key, value):
    gpu_plan[key] = value
    messages = {
        "project_name": "non-learn",
        "project_id": "models-rd",
        "accelerator": "CUDA",
        "backend": "Allowlisted",
        "arm": "Allowlisted",
        "config_name": "Allowlisted",
        "seed": "public seed",
        "config_sha256": "explicit",
        "source_sha256": "explicit",
        "image_pull_secrets": "nonempty",
        "wandb_secret": "secret name/key",
        "resources_acknowledged": "resource confirmation",
        "timeout_seconds": "profile timeout",
        "checkpoint": "Training cannot",
    }
    with pytest.raises(ValueError, match=messages[key]):
        settings.validate_execution_plan(gpu_plan)


def test_gpu_plan_rejects_commands_and_oversized_resources(gpu_plan):
    gpu_plan["runner_command"].extend(["--command", "arbitrary"])
    with pytest.raises(ValueError, match="Runner command"):
        settings.validate_execution_plan(gpu_plan)
    gpu_plan["runner_command"] = settings.runner_command(
        gpu_plan["profile"], "dp_aux", 0, gpu_plan["output_dir"]
    )
    gpu_plan["resources"]["limits"]["ephemeral-storage"] = "200Gi"
    with pytest.raises(ValueError, match="resource confirmation"):
        settings.validate_execution_plan(gpu_plan)


def test_authorization_is_not_renewed_by_migration(gpu_plan, tmp_path):
    gpu_plan["authorization"]["deadline_utc"] = "2026-09-22T20:10:00+00:00"
    with pytest.raises(ValueError, match="authorization/deadline"):
        gpu_policy.validate_authorization(
            gpu_plan, now=datetime(2026, 9, 23, tzinfo=UTC)
        )
    gpu_plan["authorization"]["deadline_utc"] = (
        datetime.now(UTC) + timedelta(days=1)
    ).isoformat()
    gpu_plan["authorization"]["gpu_seconds"] = 900
    receipt = gpu_policy.reserve_budget(tmp_path, gpu_plan)
    assert json.loads(receipt.read_text())["reservations"][gpu_plan["run_name"]] == 900
    with pytest.raises(ValueError, match="already has a reservation"):
        gpu_policy.reserve_budget(tmp_path, gpu_plan)
    gpu_plan["run_name"] += "-next"
    with pytest.raises(ValueError, match="exhausted"):
        gpu_policy.reserve_budget(tmp_path, gpu_plan)


def test_serialized_gpu_and_orchestrator_pods(gpu_plan):
    from kubernetes.client import ApiClient
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorSettings,
    )
    from zenml.integrations.kubernetes.manifest_utils import build_pod_manifest

    native = KubernetesOrchestratorSettings.model_validate(
        settings.validate_native_settings(gpu_plan)["orchestrator.kubernetes"]
    )
    assert native.active_deadline_seconds == 900
    assert native.orchestrator_job_backoff_limit == 0
    for gpu, pod_settings in [
        (True, native.pod_settings),
        (False, native.orchestrator_pod_settings),
    ]:
        pod = build_pod_manifest(
            pod_name="moe-test",
            image_name=IMAGE,
            command=["python"],
            args=[],
            privileged=False,
            pod_settings=pod_settings,
            service_account_name="test-existing",
            env={"ZENML_STORE_URL": settings.CLIENT_URL},
        )
        spec = ApiClient().sanitize_for_serialization(pod)["spec"]
        container = spec["containers"][0]
        env = {item["name"]: item for item in container["env"]}
        assert env["ZENML_STORE_URL"]["value"] == settings.POD_URL
        assert spec["imagePullSecrets"] == [{"name": "test-registry"}]
        assert not spec.get("hostIPC")
        assert not container["securityContext"]["privileged"]
        if gpu:
            expected = gpu_policy.gpu_resources(gpu_plan["profile"])
            assert container["resources"] == expected
            assert spec["schedulerName"] == gpu_policy.SCHEDULER
            assert spec["tolerations"] == gpu_policy.TOLERATIONS
            assert spec["volumes"] == [
                {
                    "name": "scratch",
                    "emptyDir": {
                        "sizeLimit": expected["requests"]["ephemeral-storage"]
                    },
                }
            ]
            assert "CUDA_VISIBLE_DEVICES" not in env
            assert env["WANDB_API_KEY"]["valueFrom"]["secretKeyRef"] == {
                "name": "test-tracking",
                "key": "WANDB_API_KEY",
                "optional": False,
            }
        else:
            assert container["resources"] == settings.resources(settings.CPU_DEFAULTS)
            assert not spec.get("tolerations")
            assert not spec.get("volumes")
            assert "WANDB_API_KEY" not in env
            assert env["CUDA_VISIBLE_DEVICES"]["value"] == ""


def test_actual_zenml_job_rendering_uses_each_registered_account(gpu_plan):
    from kubernetes.client import ApiClient
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorSettings,
    )
    from zenml.integrations.kubernetes.orchestrators.kubernetes_orchestrator import (
        KubernetesOrchestrator,
    )

    native = KubernetesOrchestratorSettings.model_validate(
        settings.validate_native_settings(gpu_plan)["orchestrator.kubernetes"]
    )
    adapter = SimpleNamespace(
        _get_service_account_name=lambda _: gpu_plan["service_account"],
        config=SimpleNamespace(is_local=False),
    )
    run_id = str(uuid4())
    for role, account in [
        ("pod_settings", "step_service_account"),
        ("orchestrator_pod_settings", "service_account"),
    ]:
        manifest = KubernetesOrchestrator._prepare_job_manifest(
            adapter,
            name="moe-test",
            command=["python"],
            args=[],
            image=IMAGE,
            environment={"ZENML_STORE_URL": settings.CLIENT_URL},
            labels={"run_id": run_id},
            annotations={},
            settings=native,
            pod_settings=getattr(native, role),
            backoff_limit=0,
        )
        serialized = ApiClient().sanitize_for_serialization(manifest)
        assert serialized["metadata"]["labels"]["run_id"] == run_id
        assert serialized["spec"]["template"]["metadata"]["labels"]["run_id"] == run_id
        assert (
            serialized["spec"]["template"]["spec"]["serviceAccountName"]
            == gpu_plan[account]
        )
        assert serialized["spec"]["activeDeadlineSeconds"] == 900
        assert serialized["spec"]["backoffLimit"] == 0


def test_references_check_keys_without_releasing_values(gpu_plan):
    core = Mock()
    core.read_namespaced_secret.side_effect = [
        SimpleNamespace(
            type="kubernetes.io/dockerconfigjson", data={".dockerconfigjson": "secret"}
        ),
        SimpleNamespace(data={"WANDB_API_KEY": "secret"}),
    ]
    validate_cluster_references(core, gpu_plan)
    assert core.read_namespaced_service_account.call_count == 2
    core.read_namespaced_secret.side_effect = [SimpleNamespace(type="Opaque", data={})]
    with pytest.raises(ValueError, match="registry configuration"):
        validate_cluster_references(core, gpu_plan)


def test_approved_admission_uses_only_registered_accounts_and_still_checks_secrets(
    gpu_plan,
):
    gpu_plan["service_account_verification"] = "admission"
    config = SimpleNamespace(
        service_account_name=gpu_plan["service_account"],
        step_pod_service_account_name=gpu_plan["step_service_account"],
    )
    core = Mock()
    core.read_namespaced_secret.side_effect = [
        SimpleNamespace(
            type="kubernetes.io/dockerconfigjson",
            data={".dockerconfigjson": "not-disclosed"},
        ),
        SimpleNamespace(data={"WANDB_API_KEY": "not-disclosed"}),
    ]
    validate_cluster_references(core, gpu_plan, config)
    core.read_namespaced_service_account.assert_not_called()
    assert core.read_namespaced_secret.call_count == 2
    gpu_plan["step_service_account"] = "unapproved-account"
    with pytest.raises(ValueError, match="registered existing service accounts"):
        validate_cluster_references(core, gpu_plan, config)


def test_gpu_submission_evidence_survives_exception(gpu_plan, tmp_path):
    directory = tmp_path / "submissions" / gpu_plan["run_name"]

    def fail():
        with submission_context(tmp_path, gpu_plan) as current:
            assert current == directory
            (directory / "evidence.json").write_text('{"status":"failed"}')
            raise RuntimeError("a failed submission must retain evidence")

    with pytest.raises(RuntimeError, match="retain evidence"):
        fail()
    assert (directory / "evidence.json").exists()


def test_pending_gpu_job_is_stopped_once_not_retried(gpu_plan, tmp_path):
    core, batch = Mock(), Mock()
    run_id = str(uuid4())
    job = SimpleNamespace(
        metadata=SimpleNamespace(
            labels={"run_id": run_id}, name="own-job", creation_timestamp=None
        ),
        status=SimpleNamespace(start_time=datetime.now(UTC) - timedelta(seconds=901)),
    )
    core.list_namespaced_pod.return_value.items = []
    batch.list_namespaced_job.return_value.items = [job]
    observer = RunObserver(core, batch, run_id, gpu_plan, tmp_path)
    with pytest.raises(ValueError, match="15 minutes"):
        observer.inspect()
    batch.delete_namespaced_job.assert_called_once()
    batch.patch_namespaced_job.assert_not_called()
    assert (tmp_path / "status.json").exists()


def test_inherited_gpu_orchestrator_is_rejected_even_with_review(gpu_plan):
    from deploy.zenml.moe_iceland.cluster import target_fingerprint, validate_inherited
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorConfig,
    )

    stack = SimpleNamespace(id="test-stack", environment={}, secrets={}, components={})
    config = KubernetesOrchestratorConfig(kubernetes_namespace=settings.NAMESPACE)
    gpu_plan["reviewed_target_sha256"] = target_fingerprint(stack, config)
    validate_inherited(stack, config, gpu_plan)
    config = KubernetesOrchestratorConfig(
        kubernetes_namespace=settings.NAMESPACE,
        orchestrator_pod_settings=copy.deepcopy(
            settings.validate_native_settings(gpu_plan)["orchestrator.kubernetes"][
                "pod_settings"
            ]
        ),
    )
    gpu_plan["reviewed_target_sha256"] = target_fingerprint(stack, config)
    with pytest.raises(ValueError, match="Stored pod settings"):
        validate_inherited(stack, config, gpu_plan)
