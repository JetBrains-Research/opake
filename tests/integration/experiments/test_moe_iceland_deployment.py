"""Offline deployment checks; no server, Docker daemon, or cluster is contacted."""

import builtins
import copy
import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec
from uuid import UUID, uuid4

import pytest
from deploy.zenml.moe_iceland import source
from deploy.zenml.moe_iceland.bundle import validate_bundle
from deploy.zenml.moe_iceland.launch import (
    authorize,
    image_inspection_command,
    main,
    make_plan,
    parser,
    submission_environment,
)
from deploy.zenml.moe_iceland.settings import (
    ARTIFACT_STORE,
    CLIENT_URL,
    CPU_DEFAULTS,
    DEADLINES,
    DEPLOY_PATH,
    NAMESPACE,
    POD_URL,
    STACK_ID,
    STACK_NAME,
    pipeline_settings,
    resources,
    validate_execution_plan,
    validate_image,
    validate_native_settings,
    validate_project,
)

# A syntactically valid test value, not a provisioned registry/project/image.
IMAGE = "europe-west4-docker.pkg.dev/test-project/test-repo/moe@sha256:" + "a" * 64
PROJECT_ID = "c49c2fb4-416b-413a-9d82-f8f736a202df"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny_network(*args, **kwargs):
        raise AssertionError("Deployment tests must never contact a server.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_ENABLE_REPO_INIT_WARNINGS", "false")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture
def zenml_api():
    zenml = pytest.importorskip("zenml")
    assert zenml.__version__ == "0.96.4"
    pytest.importorskip("kubernetes")
    return zenml


def plan(*arguments):
    return make_plan(parser().parse_args(list(arguments)))


def confirmed_arguments():
    return [
        "--submit",
        "--project-name",
        "moe-research",
        "--project-id",
        PROJECT_ID,
        "--confirm-project",
        "moe-research",
        "--confirm-stack-id",
        STACK_ID,
        "--image",
        IMAGE,
        "--service-account",
        "test-orchestrator",
        "--step-service-account",
        "test-step",
    ]


def output_bundle(tmp_path):
    directory = tmp_path / "output"
    (directory / "model").mkdir(parents=True)
    (directory / "model/weights.safetensors").write_bytes(b"artifact contents")
    (directory / "summary.json").write_text(
        json.dumps({"versions": {"torch": "2.14.0"}})
    )
    (directory / "checks.json").write_text(json.dumps({"passed": True}))
    (directory / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 0.5}) + "\n"
    )
    return directory


def sample_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in source.ROOT_FILES:
        (root / name).write_text("fixture\n")
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/opaque-sample"]\n'
    )
    files = {
        "examples/moe_privacy/__init__.py": "# experiment constants\n",
        "examples/moe_privacy/run.py": "# archived example\n",
        "examples/moe_privacy/checks.py": "# archived checks\n",
        "examples/moe_privacy/compare.py": "# comparison utility\n",
        "examples/moe_privacy/configs/smoke.json": "{}\n",
        "examples/moe_privacy/configs/pilot.json": "{}\n",
        f"{DEPLOY_PATH}/launch.py": "# archived launcher\n",
        "packages/opaque-sample/src/opaque/sample.py": "VALUE = 17\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_dry_run_without_zenml_import_or_auth(monkeypatch, capsys):
    original_import = builtins.__import__

    def no_zenml(name, *args, **kwargs):
        assert name.split(".")[0] != "zenml", "Dry-run imported ZenML"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_zenml)
    monkeypatch.delenv("ZENML_STORE_API_KEY", raising=False)
    before = dict(os.environ)
    assert main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["plan"]["client_url"] == CLIENT_URL
    assert result["plan"]["runner_command"][-1] == "--checks"
    assert "examples/moe_privacy/configs/smoke.json" in result["runner_command"]
    assert result["plan"]["project_id"] is None
    assert os.environ == before


def test_native_pods_and_jobs(zenml_api):
    from kubernetes.client import ApiClient
    from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
        KubernetesOrchestratorSettings,
    )
    from zenml.integrations.kubernetes.manifest_utils import (
        build_job_manifest,
        build_pod_manifest,
        pod_template_manifest_from_pod,
    )

    request = plan("--image", IMAGE)
    settings = validate_native_settings(request)
    native = KubernetesOrchestratorSettings.model_validate(
        settings["orchestrator.kubernetes"]
    )
    assert native.max_parallelism == 1
    for pod_settings in (native.pod_settings, native.orchestrator_pod_settings):
        pod = build_pod_manifest(
            pod_name="moe-test",
            image_name=IMAGE,
            command=["python"],
            args=[],
            privileged=False,
            pod_settings=pod_settings,
            service_account_name="test-existing",
            env={"ZENML_STORE_URL": CLIENT_URL},
        )
        job = build_job_manifest(
            "moe-test",
            pod_template_manifest_from_pod(pod),
            active_deadline_seconds=native.active_deadline_seconds,
            ttl_seconds_after_finished=native.ttl_seconds_after_finished,
            backoff_limit=native.orchestrator_job_backoff_limit,
        )
        spec = ApiClient().sanitize_for_serialization(job)["spec"]
        assert spec["activeDeadlineSeconds"] == 900
        assert spec["backoffLimit"] == 0
        assert spec["parallelism"] == 1
        pod_spec = spec["template"]["spec"]
        assert not pod_spec.get("volumes")
        assert not pod_spec.get("imagePullSecrets")
        assert pod_spec["securityContext"]["runAsUser"] == 1000
        assert pod_spec["securityContext"]["runAsGroup"] == 1000
        container = pod_spec["containers"][0]
        assert container["resources"] == resources(CPU_DEFAULTS)
        assert container["securityContext"]["runAsUser"] == 1000
        assert container["securityContext"]["runAsGroup"] == 1000
        assert not container.get("volumeMounts")
        assert not container.get("envFrom")
        assert all("valueFrom" not in entry for entry in container["env"])
        assert {entry["name"]: entry["value"] for entry in container["env"]}[
            "ZENML_STORE_URL"
        ] == POD_URL
        assert set(container["resources"]["limits"]) == {
            "cpu",
            "memory",
            "ephemeral-storage",
        }
    docker = settings["docker"]
    assert docker["skip_build"]
    assert docker["allow_download_from_artifact_store"]
    assert not docker["allow_including_files_in_images"]
    assert not docker["allow_download_from_code_repository"]


@pytest.mark.parametrize("name", ["learn", "LEARN", "trace", "trace-team", ""])
def test_reject_disallowed_project(name):
    with pytest.raises(ValueError, match="non-learn, non-TRACE"):
        validate_project(name)


@pytest.mark.parametrize(
    "image",
    [
        "europe-west4-docker.pkg.dev/test-project/test-repo/moe:latest",
        IMAGE.replace("europe-west4", "europe-west1"),
        IMAGE.replace("europe-west4-docker.pkg.dev", "registry-cache.example.org"),
        IMAGE[:-1],
    ],
)
def test_reject_unconfirmed_image(image):
    with pytest.raises(ValueError, match="europe-west4-docker"):
        validate_image(image)


def test_accept_registered_europe_registry():
    validate_image(
        "europe-docker.pkg.dev/grazie-development/zenml-generated/moe@sha256:"
        + "a" * 64
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--profile", "pilot"],
        ["--cpu-request", "1"],
        [
            "--cpu-request",
            "1",
            "--cpu-limit",
            "1",
            "--memory-request",
            "1Gi",
            "--memory-limit",
            "1Gi",
        ],
    ],
)
def test_resources_fail_closed(arguments):
    with pytest.raises(ValueError, match=r"explicit CPU/memory|all four|unconfirmed"):
        plan(*arguments)


def test_pilot_explicit_cpu_limits_and_gate():
    request = plan(
        "--profile",
        "pilot",
        "--gate-run-id",
        str(uuid4()),
        "--cpu-request",
        "1",
        "--cpu-limit",
        "2",
        "--memory-request",
        "1Gi",
        "--memory-limit",
        "2Gi",
        "--acknowledge-confirmed-resources",
    )
    settings = pipeline_settings(request)["orchestrator.kubernetes"]
    assert settings["active_deadline_seconds"] == DEADLINES["pilot"]
    assert settings["pod_settings"]["resources"]["limits"]["memory"] == "2Gi"
    assert settings["orchestrator_pod_settings"]["resources"] == resources(CPU_DEFAULTS)


@pytest.mark.parametrize(
    "values",
    [
        ("0", "1", "512Mi", "512Mi"),
        ("2", "1", "512Mi", "512Mi"),
        ("1", "1", "2Gi", "1Gi"),
        ("1", "1", "128Gi", "128Gi"),
        ("1", "1", "-1Mi", "512Mi"),
    ],
)
def test_invalid_resource_quantities(values):
    with pytest.raises(ValueError, match=r"positive|exceeds|caps memory|explicit"):
        resources(values)


@pytest.mark.parametrize("credential", ["ZENML_STORE_API_KEY", "ZENML_STORE_API_TOKEN"])
def test_confirmations_and_child_selection(monkeypatch, tmp_path, credential):
    args = parser().parse_args(confirmed_arguments())
    request = make_plan(args)
    authorize(args, request)
    monkeypatch.delenv("ZENML_STORE_API_KEY", raising=False)
    monkeypatch.delenv("ZENML_STORE_API_TOKEN", raising=False)
    monkeypatch.setenv(credential, "test-value-not-a-real-credential")
    monkeypatch.setenv("ZENML_STORE_URL", CLIENT_URL)
    before = dict(os.environ)
    env = submission_environment(request, tmp_path / "config", tmp_path / "source")
    assert env[credential] == "test-value-not-a-real-credential"
    assert env["ZENML_ACTIVE_PROJECT_ID"] == PROJECT_ID
    assert env["ZENML_ACTIVE_STACK_ID"] == STACK_ID
    assert env["ZENML_STORE_URL"] == CLIENT_URL
    assert env.get("HOME") == before.get("HOME")
    assert os.environ == before
    args.confirm_project = "different"
    with pytest.raises(ValueError, match="repeat"):
        authorize(args, request)
    monkeypatch.setenv("ZENML_STORE_URL", POD_URL)
    with pytest.raises(ValueError, match="laptop"):
        submission_environment(request, tmp_path / "config", tmp_path / "source")


@pytest.mark.parametrize("both", [False, True])
def test_submission_requires_one_credential(monkeypatch, tmp_path, both):
    monkeypatch.setenv("ZENML_STORE_URL", CLIENT_URL)
    for name in ("ZENML_STORE_API_KEY", "ZENML_STORE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
        if both:
            monkeypatch.setenv(name, "not-a-real-credential")
    with pytest.raises(ValueError, match="exactly one"):
        submission_environment(plan(), tmp_path / "config", tmp_path / "source")


def test_image_inspection_never_builds_pulls_or_mounts():
    command = image_inspection_command(IMAGE)
    assert command[:2] == ["docker", "run"]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--user=1000:1000" in command


def test_archive_excludes_envs_and_build_trees(tmp_path):
    root = sample_source(tmp_path)
    for name in (".venv", "vendor", "target", ".git", ".temp", ".worktrees", ".junie"):
        path = root / "examples/moe_privacy" / name / "secret.json"
        path.parent.mkdir()
        path.write_text("not source")
    files = source.source_files(root)
    assert "examples/moe_privacy/run.py" in files
    assert "examples/moe_privacy/__init__.py" in files
    assert "examples/moe_privacy/compare.py" in files
    assert all(not (set(Path(name).parts) & source.EXCLUDED) for name in files)
    manifest = source.inventory(root)
    (root / source.MANIFEST).write_text(json.dumps(manifest))
    assert source.verify_source(root) == manifest
    (root / "examples/moe_privacy/run.py").write_text("changed")
    with pytest.raises(ValueError, match="manifest"):
        source.verify_source(root)


def test_archive_rejects_links_and_overflow(tmp_path, monkeypatch):
    root = sample_source(tmp_path)
    link = root / "examples/moe_privacy/leak.py"
    link.symlink_to(root / "uv.lock")
    with pytest.raises(ValueError, match="Symlink"):
        source.source_files(root)
    link.unlink()
    monkeypatch.setattr(source, "SOURCE_LIMIT", 1)
    with pytest.raises(ValueError, match="32MiB"):
        source.source_files(root)


def test_archive_accepts_only_root_license_links(tmp_path):
    root = sample_source(tmp_path)
    license_path = root / "packages/opaque-sample/LICENSE"
    license_path.symlink_to("../../LICENSE")
    manifest = source.inventory(root)
    assert (
        manifest["files"]["packages/opaque-sample/LICENSE"]
        == manifest["files"]["LICENSE"]
    )
    license_path.unlink()
    outside = tmp_path / "outside-license"
    outside.write_text("not approved source")
    license_path.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        source.source_files(root)


def test_actual_zenml_code_archive(tmp_path, zenml_api):
    root = sample_source(tmp_path)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    manifest = source.inventory(root)
    (root / source.MANIFEST).write_text(json.dumps(manifest))
    archive_path = tmp_path / "code.tar.gz"
    receipt = source.validate_code_archive(root, archive_path)
    assert receipt["sha256"] == source.sha256(archive_path)
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {*manifest["files"], source.MANIFEST}
        with archive.extractfile(
            "packages/opaque-sample/src/opaque/sample.py"
        ) as stream:
            assert stream.read() == b"VALUE = 17\n"


def test_directory_materializer_persists_contents(tmp_path, zenml_api):
    from zenml.materializers.path_materializer import PathMaterializer

    directory = output_bundle(tmp_path)
    expected = validate_bundle(directory, "smoke")
    store = tmp_path / "artifact-store"
    store.mkdir()
    materializer = PathMaterializer(uri=str(store))
    materializer.save(directory)
    assert (store / "data.tar.gz").is_file()
    shutil.rmtree(directory)
    restored = materializer.load(Path)
    try:
        assert isinstance(restored, Path)
        assert validate_bundle(restored, "smoke") == expected
        assert (
            restored / "model/weights.safetensors"
        ).read_bytes() == b"artifact contents"
    finally:
        shutil.rmtree(restored)


@pytest.mark.parametrize(
    "name", ["rng_state.pt", "model/optimizer.pt", "model/checkpoint.bin"]
)
def test_bundle_refuses_resumable_state(tmp_path, name):
    directory = output_bundle(tmp_path)
    (directory / name).write_bytes(b"not a public output")
    with pytest.raises(ValueError, match="weights-only"):
        validate_bundle(directory, "smoke")


def test_bundle_refuses_missing_metrics_and_symlinks(tmp_path):
    directory = output_bundle(tmp_path)
    (directory / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="update"):
        validate_bundle(directory, "smoke")
    (directory / "model/link.pt").symlink_to(directory / "summary.json")
    with pytest.raises(ValueError, match="Symlink"):
        validate_bundle(directory, "smoke")


def target_client(request):
    from zenml.client import Client
    from zenml.models import (
        ComponentResponse,
        ProjectResponse,
        ServerModel,
        StackResponse,
    )

    timestamps = {"created": datetime.now(UTC), "updated": datetime.now(UTC)}
    project = ProjectResponse(
        id=UUID(PROJECT_ID),
        name=request["project_name"],
        body={**timestamps, "display_name": "MoE research"},
        metadata={},
    )
    orchestrator = ComponentResponse(
        id=uuid4(),
        name="test-orchestrator",
        body={**timestamps, "type": "orchestrator", "flavor_name": "kubernetes"},
        metadata={"configuration": {"kubernetes_namespace": NAMESPACE}},
    )
    store = ComponentResponse(
        id=uuid4(),
        name="test-store",
        body={**timestamps, "type": "artifact_store", "flavor_name": "gcp"},
        metadata={"configuration": {"path": ARTIFACT_STORE}},
    )
    registry = ComponentResponse(
        id=uuid4(),
        name="test-registry",
        body={**timestamps, "type": "container_registry", "flavor_name": "gcp"},
        metadata={"configuration": {"uri": IMAGE.rsplit("/", 1)[0]}},
    )
    stack = StackResponse(
        id=UUID(STACK_ID),
        name=STACK_NAME,
        body=timestamps,
        metadata={
            "components": {
                "orchestrator": [orchestrator],
                "artifact_store": [store],
                "container_registry": [registry],
            }
        },
    )
    client = create_autospec(Client, instance=True)
    client.zen_store.config.url = CLIENT_URL
    client.zen_store.get_store_info.return_value = ServerModel(
        version="0.96.4",
        auth_scheme="OAUTH2_PASSWORD_BEARER",
        pro_workspace_name="jcp-prod",
    )
    client.list_projects.return_value = SimpleNamespace(items=[project])
    client.list_stacks.return_value = SimpleNamespace(items=[stack])
    client.list_pipeline_runs.return_value = SimpleNamespace(total=0)
    client.active_project = project
    client.active_stack_model = stack
    return client, project, stack


def test_actual_target_ids_checked_before_pipeline(zenml_api):
    from deploy.zenml.moe_iceland.remote import validate_target

    request = plan(*confirmed_arguments())
    client, project, stack = target_client(request)
    validate_target(client, request)
    project.id = uuid4()
    with pytest.raises(ValueError, match="project name/ID"):
        validate_target(client, request)
    project.id = UUID(PROJECT_ID)
    stack.name = "not-iceland"
    with pytest.raises(ValueError, match="stack name/ID"):
        validate_target(client, request)
    client.activate_stack.assert_not_called()
    assert client.active_project is project
    assert client.active_stack_model is stack
    assert {call[0] for call in client.mock_calls} <= {
        "zen_store.get_store_info",
        "list_projects",
        "list_stacks",
        "list_pipeline_runs",
    }


def test_target_rejects_image_outside_registered_repository(zenml_api):
    from deploy.zenml.moe_iceland.remote import validate_target

    request = plan(*confirmed_arguments())
    client, _, _ = target_client(request)
    request["image"] = IMAGE.replace("test-repo", "different-repo")
    with pytest.raises(ValueError, match="registered repository"):
        validate_target(client, request)


@pytest.mark.parametrize(
    "configuration",
    [
        {"kubernetes_namespace": "other"},
        {"pass_zenml_token_as_secret": True},
        {"pod_settings": {"volumes": [{"name": "unapproved", "emptyDir": {}}]}},
        {
            "orchestrator_pod_settings": {
                "resources": {"limits": {"nvidia.com/gpu": "1"}}
            }
        },
    ],
)
def test_stored_stack_configuration_cannot_bypass_isolation(configuration, zenml_api):
    from deploy.zenml.moe_iceland.remote import validate_target

    request = plan(*confirmed_arguments())
    client, _, stack = target_client(request)
    stack.components["orchestrator"][0].configuration.update(
        copy.deepcopy(configuration)
    )
    with pytest.raises(
        ValueError, match=r"namespace|Kubernetes token secrets|Stored pod settings"
    ):
        validate_target(client, request)


def test_pipeline_has_real_path_materializer_and_no_retry(zenml_api):
    from deploy.zenml.moe_iceland.pipeline import execute_moe, moe_iceland

    assert execute_moe.configuration.retry is None
    assert execute_moe.configuration.enable_cache is False
    assert execute_moe.configuration.step_operator is False
    output = execute_moe.configuration.outputs["bundle"]
    assert (
        output.materializer_source[0].import_path
        == "zenml.materializers.path_materializer.PathMaterializer"
    )
    assert moe_iceland.configuration.retry is None
    assert moe_iceland.configuration.enable_cache is False


def test_pending_gate_rejected_before_step_or_artifact_access(zenml_api):
    from deploy.zenml.moe_iceland.remote import completed_bundle
    from zenml.enums import ExecutionStatus

    client = Mock()
    client.get_pipeline_run.return_value = SimpleNamespace(
        status=ExecutionStatus.QUEUED
    )
    with pytest.raises(ValueError, match="not completed"):
        completed_bundle(client, str(uuid4()), {"project_id": PROJECT_ID})


def test_serialized_plan_cannot_override_safe_command_or_resources():
    request = plan(*confirmed_arguments())
    validate_execution_plan(request)
    request["runner_command"].remove("--checks")
    with pytest.raises(ValueError, match="Runner command"):
        validate_execution_plan(request)
    request = plan(*confirmed_arguments())
    request["resources"]["limits"]["nvidia.com/gpu"] = "1"
    with pytest.raises(ValueError, match="no GPU"):
        validate_execution_plan(request)
    request = plan(*confirmed_arguments())
    request["project_confirmed"] = False
    with pytest.raises(ValueError, match="explicit project/stack"):
        validate_execution_plan(request)


def test_serialized_pilot_still_requires_resource_acknowledgment():
    request = plan(
        *confirmed_arguments(),
        "--profile",
        "pilot",
        "--gate-run-id",
        str(uuid4()),
        "--cpu-request",
        "1",
        "--cpu-limit",
        "1",
        "--memory-request",
        "1Gi",
        "--memory-limit",
        "1Gi",
        "--acknowledge-confirmed-resources",
    )
    validate_execution_plan(request)
    request["resources_acknowledged"] = False
    with pytest.raises(ValueError, match="resource confirmation"):
        validate_execution_plan(request)


def test_source_cannot_include_local_credentials_or_results(tmp_path):
    root = sample_source(tmp_path)
    for name in (
        f"{DEPLOY_PATH}/credentials.toml",
        f"{DEPLOY_PATH}/receipt.json",
        "examples/moe_privacy/metrics.jsonl",
        "examples/moe_privacy/summary.json",
    ):
        (root / name).write_text("not source")
    files = source.source_files(root)
    assert not any(
        name.endswith(
            ("credentials.toml", "receipt.json", "metrics.jsonl", "summary.json")
        )
        for name in files
    )


def test_bundle_limits_count_empty_directories(tmp_path):
    directory = output_bundle(tmp_path)
    for index in range(256):
        (directory / "model" / str(index)).mkdir()
    with pytest.raises(ValueError, match="entry cap"):
        validate_bundle(directory, "smoke")


def test_gcs_and_kubernetes_service_connectors_register(zenml_api):
    from zenml.integrations.gcp.artifact_stores.gcp_artifact_store import (
        GCPArtifactStore,
    )
    from zenml.service_connectors.service_connector_registry import (
        service_connector_registry,
    )

    assert GCPArtifactStore.__name__ == "GCPArtifactStore"
    for connector in ("gcp", "kubernetes"):
        registered = service_connector_registry.get_service_connector_type(connector)
        assert registered.connector_class is not None


def test_stack_rejects_active_job_and_non_prod_server(zenml_api):
    from deploy.zenml.moe_iceland.remote import validate_target

    request = plan(*confirmed_arguments())
    client, _, _ = target_client(request)
    client.list_pipeline_runs.return_value.total = 1
    with pytest.raises(ValueError, match="still in progress"):
        validate_target(client, request)
    client.list_pipeline_runs.return_value.total = 0
    client.zen_store.get_store_info.return_value.pro_workspace_name = "jcp-stgn"
    with pytest.raises(ValueError, match="non-prod"):
        validate_target(client, request)


def test_lock_refuses_unhashed_dependencies_and_private_urls(tmp_path):
    from deploy.zenml.moe_iceland.provenance import locked_versions

    lock = tmp_path / "requirements.lock"
    lock.write_text("zenml==0.96.4\n")
    with pytest.raises(ValueError, match="complete hash lock"):
        locked_versions(lock)
    lock.write_text(
        "zenml @ https://invalid.example/package.whl --hash=sha256:" + "a" * 64
    )
    with pytest.raises(ValueError, match="Only the pinned public CPU torch URL"):
        locked_versions(lock)


def test_lock_ignores_vendor_metadata_but_rejects_extra_installations(
    tmp_path, monkeypatch
):
    import sysconfig

    from deploy.zenml.moe_iceland import provenance

    site = tmp_path / "site-packages"
    vendor = site / "setuptools" / "_vendor"

    def distribution(root, name):
        metadata = root / f"{name.replace('-', '_')}-1.0.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
        )

    distribution(site, "moe-test-package")
    distribution(vendor, "moe-test-vendored")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(vendor))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(site))
    monkeypatch.setattr(provenance, "PINS", {"moe-test-package": "1.0"})
    lock = tmp_path / "requirements.lock"
    lock.write_text("moe-test-package==1.0 --hash=sha256:" + "a" * 64)

    assert provenance.locked_versions(lock) == {"moe-test-package": "1.0"}
    distribution(site, "moe-test-unreviewed")
    with pytest.raises(ValueError, match="moe-test-unreviewed"):
        provenance.locked_versions(lock)


def test_gate_validates_uploaded_receipt_and_exact_source(tmp_path, zenml_api):
    from deploy.zenml.moe_iceland.remote import validate_gate
    from zenml.enums import ExecutionStatus

    gate_plan = plan(*confirmed_arguments())
    gate_plan.update(source_sha256="a" * 64, image_contract={"schema": 1})
    directory = output_bundle(tmp_path)
    (directory / "deployment.json").write_text(
        json.dumps(
            {
                "plan": gate_plan,
                "runner_returncode": 0,
                "checks_requested": True,
            }
        )
    )
    artifact = SimpleNamespace(
        uri=ARTIFACT_STORE + "/test/bundle",
        materializer=SimpleNamespace(
            import_path="zenml.materializers.path_materializer.PathMaterializer"
        ),
        load=Mock(return_value=directory),
    )
    completed = SimpleNamespace(
        status=ExecutionStatus.COMPLETED,
        project_id=PROJECT_ID,
        stack=SimpleNamespace(id=UUID(STACK_ID)),
        pipeline=SimpleNamespace(name="moe_iceland"),
        steps={
            "execute_moe": SimpleNamespace(
                status=ExecutionStatus.COMPLETED, outputs={"bundle": [artifact]}
            )
        },
        config=SimpleNamespace(parameters={"plan": gate_plan}),
    )
    client = Mock()
    client.get_pipeline_run.return_value = completed
    request = {**gate_plan, "profile": "pilot", "gate_run_id": str(uuid4())}
    validate_gate(client, request)
    artifact.load.assert_called_once_with(disable_cache=True)
    request["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source_sha256 differs"):
        validate_gate(client, request)
    request["source_sha256"] = gate_plan["source_sha256"]
    (directory / "deployment.json").write_text(
        json.dumps(
            {
                "plan": gate_plan,
                "runner_returncode": 0,
                "checks_requested": False,
            }
        )
    )
    with pytest.raises(ValueError, match="real checks"):
        validate_gate(client, request)


def test_native_client_getters_honor_invocation_ids(monkeypatch, zenml_api):
    from zenml.client import Client

    client, project, stack = target_client(plan(*confirmed_arguments()))
    client._active_project = None
    client._active_stack = None
    client.get_project.return_value = project
    client.get_stack.return_value = stack
    monkeypatch.setenv("ZENML_ACTIVE_PROJECT_ID", PROJECT_ID)
    monkeypatch.setenv("ZENML_ACTIVE_STACK_ID", STACK_ID)
    assert Client.active_project.fget(client) is project
    assert Client.active_stack_model.fget(client) is stack
    client.get_project.assert_called_once_with(PROJECT_ID)
    client.get_stack.assert_called_once_with(STACK_ID)
    client.activate_stack.assert_not_called()


def test_internal_entry_point_requires_isolated_stage(tmp_path, capsys):
    assert main(["--submit", "--_submit-plan", str(tmp_path / "plan.json")]) == 2
    assert "launcher-created isolated stage" in capsys.readouterr().err
