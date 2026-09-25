"""CUDA dispatch, data integrity and credential-boundary checks without a GPU job."""

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from deploy.zenml.moe_iceland import gpu_policy, gpu_runner
from deploy.zenml.moe_iceland.settings import STACK_ID

ROOT = Path(__file__).resolve().parents[3]


def test_staged_source_contains_explicit_trainer_dependencies():
    from deploy.zenml.moe_iceland.source import SFT_SOURCES, inventory

    source = inventory(ROOT)
    assert source["bytes"] < 32 * 1024**2
    assert {f"examples/moe_privacy/{name}" for name in SFT_SOURCES} <= source[
        "files"
    ].keys()
    assert "deploy/zenml/moe_iceland/gpu_runner.py" in source["files"]
    assert not any(".venv" in name or "results/" in name for name in source["files"])


@pytest.mark.parametrize(
    ("changed", "allowed"),
    [
        ("packages/opaque-dpsgd/tests/accounting/test_moe_aux.py", True),
        ("packages/opaque-dpsgd/src/opaque/api/dpsgd/noise/_gaussian.py", False),
        ("uv.lock", False),
    ],
)
def test_gpu_test_updates_never_authorize_library_or_dependency_drift(
    monkeypatch, changed, allowed
):
    from deploy.zenml.moe_iceland import source

    def git(command, **kwargs):
        if "diff" in command:
            return changed
        if "rev-parse" in command:
            return source.PARENT_REVISION
        return ""

    monkeypatch.setattr(source.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="drifted"):
        source.require_pinned_workspace(ROOT)
    if allowed:
        assert (
            source.require_pinned_workspace(ROOT, allow_test_changes=True)
            == source.PARENT_REVISION
        )
    else:
        with pytest.raises(ValueError, match="drifted"):
            source.require_pinned_workspace(ROOT, allow_test_changes=True)


@pytest.fixture
def config():
    return json.loads(
        (
            ROOT / "examples/moe_privacy/configs/sft_balancing_aux0001_ratio1.json"
        ).read_text()
    )


@pytest.fixture
def plan():
    return {
        "backend": "opaque",
        "arm": "dp_aux",
        "seed": 0,
        "profile": "sft-train-v1",
        "run_name": "moe-iceland-example-0123456789ab",
        "project_id": gpu_policy.PROJECT_ID,
        "stack_id": STACK_ID,
        "source_sha256": "a" * 64,
        "image": "europe-west4-docker.pkg.dev/test-project/test-repo/test@sha256:"
        + "b" * 64,
        "resource_profile": gpu_policy.RESOURCE_PROFILE,
    }


def test_reject_cpu_fallback_before_any_model_download(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CPU fallback"):
        gpu_runner.cuda_details()


def test_reject_undersized_gpu_before_allocation(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=40_000_000_000),
    )
    with pytest.raises(ValueError, match="80 GB"):
        gpu_runner.cuda_details()


def test_child_receives_only_required_secret_and_public_linkage(plan):
    run_id = str(uuid4())
    environ = {
        "WANDB_API_KEY": "required",
        "ZENML_STORE_API_KEY": "cluster-secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/gcp",
        "HF_TOKEN": "unneeded",
        "UNRELATED_SECRET": "not-allowed",
        "WANDB_MODE": "disabled",
        "CUDA_VISIBLE_DEVICES": "0",
        "OPAQUE_SKIP_TRANSFORMERS_PATCHES": "unreviewed",
        "HOME": os.environ["HOME"],
    }
    env = gpu_runner.child_environment(
        environ, pythonpath="/approved/source", plan=plan, run_id=run_id
    )
    assert env["WANDB_API_KEY"] == "required"
    assert env["HOME"] == environ["HOME"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["WANDB_MODE"] == "online"
    assert env["OPAQUE_ZENML_RUN_ID"] == run_id
    assert env["PYTHONUNBUFFERED"] == "1"
    for name in (
        "ZENML_STORE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "UNRELATED_SECRET",
        "OPAQUE_SKIP_TRANSFORMERS_PATCHES",
    ):
        assert name not in env
    offline = gpu_runner.offline_environment(env)
    assert (
        offline["HF_HUB_OFFLINE"]
        == offline["HF_DATASETS_OFFLINE"]
        == offline["TRANSFORMERS_OFFLINE"]
        == "1"
    )


@pytest.mark.parametrize(
    ("backend", "arm", "module", "ratio"),
    [
        ("opaque", "dp_aux", "sft_run", "1.0"),
        ("trl", "trl_reference_aux", "trl_run", "na"),
    ],
)
def test_explicit_native_private_cuda_commands(
    plan, config, tmp_path, backend, arm, module, ratio
):
    plan.update(backend=backend, arm=arm)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = gpu_runner.trainer_command(plan, path, tmp_path / "output")
    assert command[:3] == [sys.executable, "-m", f"examples.moe_privacy.{module}"]
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--arm") + 1] == arm
    assert f"-aux0.001-ratio{ratio}-" in command[command.index("--wandb-name") + 1]
    assert "--wandb-fail-open" in command


def test_smoke_changes_only_update_budget_not_data(config):
    from examples.moe_privacy.sft_run import SFTExperimentConfig

    gate = gpu_runner.execution_config(config, "sft-smoke-v1")
    assert gate == {**config, "steps": 2, "eval_every": 1}
    assert json.loads(json.dumps(asdict(SFTExperimentConfig(**gate)))) == gate
    assert gpu_runner.execution_config(config, "sft-train-v1") == config


def test_pinned_snapshot_allowlist_and_sizes():
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json", size=100),
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=2048),
            SimpleNamespace(rfilename="optimizer.pt", size=10**12),
        ]
    )
    names, size = gpu_runner.selected_files(info, gpu_runner.MODEL_FILES)
    assert names == ["config.json", "model-00001-of-00002.safetensors"]
    assert size == 2148


@pytest.mark.parametrize(
    ("name", "size", "message"),
    [
        ("../config.json", 5, "Unsafe file"),
        ("config.json", None, "unknown file sizes"),
        ("config.json", gpu_policy.CACHE_BYTES + 1, "allowance"),
    ],
)
def test_snapshot_refuses_unbounded_or_unsafe_download(name, size, message):
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, size=size)])
    with pytest.raises(ValueError, match=message):
        gpu_runner.selected_files(info, ("*.json",))


def completed_summary(config, plan):
    return {
        "status": "completed",
        "config": config,
        "arm": plan["arm"],
        "model_seed": 0,
        "execution": {"device": "cuda:0", "opaque_patches": False},
        "data": {
            name: {"sha256": digest} for name, digest in gpu_runner.PARTITIONS.items()
        },
        "privacy": {
            "steps": 256,
            "private": True,
            "load_release": True,
            "epsilon": 7.999,
            "delta": 1e-5,
            "load_noise_ratio": 1.0,
        },
    }


def test_validate_completion_and_audited_partitions(config, plan):
    summary = completed_summary(config, plan)
    gpu_runner.validate_training(summary, config, plan)
    for name in gpu_runner.PARTITIONS:
        changed = copy.deepcopy(summary)
        changed["data"][name]["sha256"] = "wrong"
        with pytest.raises(ValueError, match=f"audited {name}"):
            gpu_runner.validate_training(changed, config, plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [("epsilon", 8.01), ("load_release", False), ("load_noise_ratio", 0.1)],
)
def test_reject_accounting_mismatch(config, plan, key, value):
    summary = completed_summary(config, plan)
    summary["privacy"][key] = value
    with pytest.raises(ValueError, match="joint accounting"):
        gpu_runner.validate_training(summary, config, plan)


def test_reject_native_patch_or_cpu_receipt(config, plan):
    plan.update(backend="trl", arm="trl_reference_aux")
    summary = completed_summary(config, plan)
    summary["execution"]["opaque_patches"] = True
    with pytest.raises(ValueError, match="Native training"):
        gpu_runner.validate_training(summary, config, plan)
    summary["execution"]["device"] = "cpu"
    with pytest.raises(ValueError, match="did not execute on CUDA"):
        gpu_runner.validate_training(summary, config, plan)


def test_deadline_prevents_next_process(tmp_path):
    with pytest.raises(ValueError, match="deadline expired"):
        gpu_runner._run_child(
            [sys.executable, "-c", "raise AssertionError()"],
            root=tmp_path,
            env={},
            expires=time.monotonic() - 1,
        )


def test_public_artifact_selection_excludes_caches_and_private_state(tmp_path):
    from deploy.zenml.moe_iceland.pipeline import public_outputs

    raw, output = tmp_path / "raw", tmp_path / "output"
    raw.mkdir()
    (raw / "summary.json").write_text('{"status":"completed"}')
    (raw / "wandb").mkdir()
    (raw / "wandb" / "private-cache.json").write_text("{}")
    (raw / "optimizer.pt").write_bytes(b"not releasable")
    (raw / "rng.json").write_text("{}")
    public_outputs(raw, output)
    assert {path.name for path in output.iterdir()} == {"summary.json"}


def test_remaining_runtime_does_not_require_fresh_full_allocation():
    now = datetime.now(UTC)
    plan = {
        "profile": "sft-train-v1",
        "timeout_seconds": 7200,
        "authorization": {
            "id": str(uuid4()),
            "gpu_seconds": 7200,
            "deadline_utc": (now + timedelta(seconds=10)).isoformat(),
        },
        "run_deadline_utc": (now + timedelta(seconds=5)).isoformat(),
    }
    gpu_policy.validate_authorization(plan, now=now)
    with pytest.raises(ValueError, match="authorization/deadline"):
        gpu_policy.validate_authorization(plan, now=now + timedelta(seconds=6))


def test_fresh_native_import_rejects_no_opaque_patches():
    environment = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    environment["PYTHONPATH"] = (
        str(ROOT / ".temp/native-trl-test-deps")
        + os.pathsep
        + environment.get("PYTHONPATH", ".")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from examples.moe_privacy.trl_run import assert_native_model; assert_native_model()",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
