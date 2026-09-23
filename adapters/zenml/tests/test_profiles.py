"""Contract tests for ZenML adapter profiles and launcher validation."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("zenml")
pytest.importorskip("jetbrains.mlops.zenml")

from adapters.zenml import run


@pytest.mark.parametrize("profile", run.PROFILE_PATHS)
def test_probe_profiles_load(profile):
    pipeline = run._configured_pipeline(profile)

    assert pipeline.configuration.enable_cache is False
    assert pipeline.configuration.enable_step_logs is True


def test_pipeline_source_root_includes_adapters_and_examples():
    from zenml.utils.source_utils import get_source_root

    run._configured_pipeline("smoke")

    assert Path(get_source_root()) == run.PROJECT_ROOT


def test_gpu_profile_requests_one_gpu():
    profile = yaml.safe_load(run.PROFILE_PATHS["probe-gpu"].read_text())

    resources = profile["steps"]["probe_environment"]["settings"]["resources"]
    assert resources["gpu_count"] == 1
    docker = profile["settings"]["docker"]
    assert docker["parent_image"] == ("pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime")
    assert docker["build_config"]["build_options"]["platform"] == "linux/amd64"
    assert docker["python_package_installer"] == "pip"
    assert docker["python_package_installer_args"] == {"break-system-packages": None}
    assert docker["requirements"] == "adapters/zenml/requirements.txt"
    assert docker["disable_automatic_requirements_detection"] is True
    assert docker["target_repository"] == "opaque-zenml"
    assert docker["user"] == "1000"


def test_probe_profiles_run_as_non_root():
    for profile_name in ("probe", "probe-gpu"):
        profile = yaml.safe_load(run.PROFILE_PATHS[profile_name].read_text())

        assert profile["settings"]["docker"]["user"] == "1000"


def test_profiles_use_external_zenml_endpoint():
    for profile_name in run.PROFILE_PATHS:
        pipeline = run._configured_pipeline(profile_name)
        environment = pipeline.configuration.settings["orchestrator"].pod_settings[
            "env"
        ]
        by_name = {entry["name"]: entry for entry in environment}

        assert by_name["ZENML_STORE_URL"]["value"] == (
            "https://zenml-external.labs.jb.gg"
        )
        assert by_name["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
            "name": "jbr-fed",
            "key": "HF_TOKEN",
        }
        assert by_name["WANDB_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": "jbr-fed",
            "key": "WANDB_API_KEY",
        }


def test_configured_profiles_include_jb_mlops_orchestrator_settings():
    pipeline = run._configured_pipeline("probe")

    settings = pipeline.configuration.settings
    assert set(settings) == {"docker", "orchestrator"}
    assert settings["docker"].parent_image == (
        "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
    )
    assert not hasattr(settings["docker"], "runtime_environment")
    assert settings["orchestrator"].pod_settings["annotations"] == {
        "cluster-autoscaler.kubernetes.io/safe-to-evict": "false"
    }
    assert settings["orchestrator"].pod_settings["resources"]["requests"] == {
        "cpu": "0.1",
        "ephemeral-storage": "1Gi",
        "memory": "2.0Gi",
    }


@pytest.mark.parametrize("profile", ["probe-gpu", "smoke", "production"])
def test_gpu_profiles_request_gpu_on_orchestrator(profile):
    pipeline = run._configured_pipeline(profile)

    resources = pipeline.configuration.settings["orchestrator"].pod_settings[
        "resources"
    ]
    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"
    pod_settings = pipeline.configuration.settings["orchestrator"].pod_settings
    assert pod_settings["additional_pod_spec_args"] == {
        "scheduler_name": "gpu-binpack-scheduler",
        "tolerations": [
            {
                "effect": "NoSchedule",
                "key": "nvidia.com/gpu",
                "operator": "Exists",
            }
        ],
    }


@pytest.mark.parametrize("profile", ["smoke", "production"])
def test_training_profiles_mount_large_temporary_workspace(profile):
    pipeline = run._configured_pipeline(profile)
    pod_settings = pipeline.configuration.settings["orchestrator"].pod_settings
    mounts = {
        mount["mountPath"]: mount["name"] for mount in pod_settings["volume_mounts"]
    }

    assert set(mounts) == {"/dev/shm", "/tmp"}
    tmp_volume = next(
        volume for volume in pod_settings["volumes"] if volume["name"] == mounts["/tmp"]
    )
    storage = tmp_volume["ephemeral"]["volumeClaimTemplate"]["spec"]["resources"][
        "requests"
    ]["storage"]
    assert storage == "200.00Gi"


def test_training_profiles_are_uncached_single_gpu_and_do_not_build_secrets():
    for profile_name in ("smoke", "production"):
        profile = yaml.safe_load(run.PROFILE_PATHS[profile_name].read_text())
        training = profile["steps"]["train_sft_step"]
        docker = profile["settings"]["docker"]

        assert profile["enable_cache"] is False
        assert training["enable_cache"] is False
        assert training["settings"]["resources"]["gpu_count"] == 1
        assert "runtime_environment" not in docker
        assert docker["parent_image"] == (
            "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime"
        )
        assert docker["requirements"] == "adapters/zenml/training-requirements.txt"
        assert docker["disable_automatic_requirements_detection"] is True
        assert docker["user"] == "1000"
        assert "dockerfile" not in docker
        assert set(profile["parameters"]) == {"config"}


def test_training_profiles_use_self_hosted_federated_compute_wandb():
    for profile_name in ("smoke", "production"):
        pipeline = run._configured_pipeline(profile_name)
        environment = pipeline.configuration.settings["orchestrator"].pod_settings[
            "env"
        ]
        values = {
            entry["name"]: entry.get("value")
            for entry in environment
            if "value" in entry
        }

        assert values["WANDB_BASE_URL"] == "https://jetbrains.wandb.io"
        assert values["WANDB_ENTITY"] == "federated-compute"
        assert values["WANDB_PROJECT"] == "opaque"
        assert values["HF_HOME"] == "/tmp/huggingface"
        assert values["WANDB_DIR"] == "/tmp/wandb"


def test_training_profiles_define_complete_mellum_magicoder_jobs():
    smoke = yaml.safe_load(run.PROFILE_PATHS["smoke"].read_text())["parameters"][
        "config"
    ]
    production = yaml.safe_load(run.PROFILE_PATHS["production"].read_text())[
        "parameters"
    ]["config"]

    expected_fields = {
        field.name
        for field in run._load_sft_config_type().__dataclass_fields__.values()
    }
    assert set(smoke) == expected_fields
    assert set(production) == expected_fields
    for config in (smoke, production):
        assert config["model_name_or_path"] == "JetBrains/Mellum2-12B-A2.5B-Base"
        assert config["dataset_name"] == "ise-uiuc/Magicoder-OSS-Instruct-75K"
        assert config["prompt_field"] == "problem"
        assert config["completion_field"] == "solution"
        assert config["max_length"] == 2048
        assert config["target_epsilon"] == 8.0
        assert config["target_delta"] is None
    assert smoke["max_steps"] == 2
    assert smoke["max_train_samples"] == 512
    assert production["max_steps"] is None
    assert production["max_train_samples"] is None


def test_no_profile_places_environment_values_in_the_docker_image():
    for path in run.PROFILE_PATHS.values():
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert "runtime_environment" not in profile["settings"]["docker"]


def test_local_builder_requires_container_engine(monkeypatch):
    class Flavor:
        name = "local"

    class Builder:
        name = "subprocess-builder"
        flavor = Flavor()

    class Stack:
        def __init__(self):
            self.components = {run.StackComponentType.IMAGE_BUILDER: [Builder()]}

    monkeypatch.setattr(shutil, "which", lambda executable: None)

    with pytest.raises(RuntimeError, match="neither Docker nor Podman"):
        run._validate_local_builder(Stack())


def test_training_runtime_uses_published_pinned_dependencies():
    requirements = (Path(__file__).parents[1] / "training-requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "opaque[all]==0.15.6rc1" in requirements
    assert "datasets==5.0.1" in requirements
    assert "peft==0.20.0" in requirements
    assert "transformers==5.16.1" in requirements
    assert "wandb==0.21.4" in requirements
    assert "source.tar.gz" not in requirements


def test_resume_checkpoint_cli_requires_an_explicit_reference():
    args = run._build_parser().parse_args(
        ["smoke", "--resume-checkpoint", "opaque-sft-checkpoint@7"]
    )

    assert args.resume_checkpoint == "opaque-sft-checkpoint@7"


def test_resume_run_and_exact_checkpoint_are_mutually_exclusive():
    parser = run._build_parser()
    args = parser.parse_args(["production", "--resume-run", "prior-run-id"])

    assert args.resume_run == "prior-run-id"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "production",
                "--resume-run",
                "prior-run-id",
                "--resume-checkpoint",
                "checkpoint@7",
            ]
        )


def test_launcher_accepts_typed_overrides_for_the_full_sft_contract():
    overrides = run._parse_overrides(
        [
            "max_steps=4",
            "target_epsilon=null",
            "noise_multiplier=1.25",
            "lora_target_modules=[q_proj, v_proj]",
            "no_wandb=true",
        ]
    )

    effective = run._validate_training_overrides("smoke", overrides)
    assert overrides == {
        "max_steps": 4,
        "target_epsilon": None,
        "noise_multiplier": 1.25,
        "lora_target_modules": ["q_proj", "v_proj"],
        "no_wandb": True,
    }
    assert effective["max_steps"] == 4
    assert effective["target_epsilon"] is None
    assert effective["noise_multiplier"] == 1.25
    assert effective["lora_target_modules"] == ["q_proj", "v_proj"]
    assert effective["no_wandb"] is True
    assert effective["model_name_or_path"] == "JetBrains/Mellum2-12B-A2.5B-Base"


@pytest.mark.parametrize("value", ["missing-separator", "=value"])
def test_launcher_rejects_malformed_overrides(value):
    with pytest.raises(ValueError, match="expected FIELD=VALUE"):
        run._parse_overrides([value])


def test_launcher_rejects_unknown_trainer_fields():
    with pytest.raises(ValueError, match="Unknown SFT configuration field"):
        run._validate_training_overrides("smoke", {"typo": True})


def test_launcher_resolves_resume_run_to_immutable_checkpoint(monkeypatch):
    client = SimpleNamespace(
        set_active_project=lambda project: None,
        activate_stack=lambda stack: None,
    )
    calls: dict[str, object] = {}

    def configured(**parameters):
        calls["parameters"] = parameters

    monkeypatch.setattr(run, "Client", lambda: client)
    monkeypatch.setattr(
        run,
        "_validate_external_stack",
        lambda received, stack_name: SimpleNamespace(name=stack_name, components={}),
    )
    monkeypatch.setattr(run, "_validate_local_builder", lambda stack: None)
    monkeypatch.setattr(run, "_component_summary", lambda stack: {})
    monkeypatch.setattr(run, "_configured_pipeline", lambda profile: configured)
    monkeypatch.setattr(
        run,
        "_validate_training_overrides",
        lambda profile, overrides: {"model_name_or_path": "model"},
    )
    monkeypatch.setattr(
        run,
        "_find_latest_checkpoint_for_run",
        lambda received, run_id: "12345678-1234-5678-1234-567812345678",
    )
    monkeypatch.setattr(run, "_resolve_source_commit_sha", lambda: "abcdef")

    run.launch("production", resume_run="prior-run-id")

    assert calls["parameters"] == {
        "config": {"model_name_or_path": "model"},
        "source_commit_sha": "abcdef",
        "resume_checkpoint": "12345678-1234-5678-1234-567812345678",
    }
