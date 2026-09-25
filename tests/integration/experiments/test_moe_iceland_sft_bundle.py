"""Offline GPU artifact contracts using synthetic receipts and real safetensors."""

import copy
import hashlib
import json
import math
import os
import shutil
import socket
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
from deploy.zenml.moe_iceland import bundle
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "bundle-manifest.json"
WEIGHTS = "trainable/trainable.safetensors"
SPEC = "trainable/adapter_spec.json"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("Artifact tests must not contact a server")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_ENABLE_REPO_INIT_WARNINGS", "false")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.mark.parametrize("profile", ["gpu-probe-v1", "sft-smoke-v1", "sft-train-v1"])
def test_pipeline_selects_a_complete_sealable_bundle(tmp_path, profile):
    import shutil

    from deploy.zenml.moe_iceland.pipeline import public_outputs

    raw = make_bundle(tmp_path, profile)
    (raw / "data.json").write_text("{}")
    (raw / "validation-records-000000.jsonl").write_text("{}\n")
    (raw / "wandb").mkdir()
    (raw / "wandb" / "excluded.log").write_text("SDK logs are not artifacts")
    destination = tmp_path / "selected"
    public_outputs(raw, destination)
    shutil.copyfile(raw / "deployment.json", destination / "deployment.json")
    expected = bundle.seal_bundle(destination, profile)
    assert bundle.validate_bundle(destination, profile) == expected
    assert not (destination / "data.json").exists()
    assert not (destination / "wandb").exists()


def adapter_spec(*, experts=2, hidden=8, intermediate=6):
    prefix = "model.layers.0.mlp"
    shapes = {
        f"{prefix}.experts.parametrizations.gate_up_proj.0.lora_A": [
            experts,
            4,
            hidden,
        ],
        f"{prefix}.experts.parametrizations.gate_up_proj.0.lora_B": [
            experts,
            2 * intermediate,
            4,
        ],
        f"{prefix}.experts.parametrizations.down_proj.0.lora_A": [
            experts,
            4,
            intermediate,
        ],
        f"{prefix}.experts.parametrizations.down_proj.0.lora_B": [experts, hidden, 4],
        f"{prefix}.gate.weight": [experts, hidden],
    }
    targets = sorted(
        f"{prefix}.experts.{name}" for name in ("gate_up_proj", "down_proj")
    )
    count = sum(math.prod(shape) for shape in shapes.values())
    metadata = {
        "adapter_kind": "static_parametrization_expert_lora",
        "rank": 4,
        "alpha": 8,
        "geometry": {"num_layers": 1, "num_experts": experts, "top_k": 1},
        "target_parameters": targets,
        "expert_adapter_names": sorted(name for name in shapes if "lora_" in name),
        "expert_adapter_count": 4,
        "router_names": [f"{prefix}.gate.weight"],
        "router_count": 1,
        "total_parameters": count + 1000,
        "trainable_parameters": count,
    }
    return {
        "format_version": 1,
        "adapter_kind": metadata["adapter_kind"],
        "config": {"rank": 4, "alpha": 8, "target_parameters": targets},
        "metadata": metadata,
        "tensors": {
            name: {"shape": shape, "dtype": "torch.float32"}
            for name, shape in sorted(shapes.items())
        },
    }


def sparse_weights(path, spec):
    header = {}
    offset = 0
    for name, tensor in spec["tensors"].items():
        size = math.prod(tensor["shape"]) * 4
        header[name] = {
            "dtype": "F32",
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)


def make_bundle(tmp_path, profile="sft-smoke-v1", arm="dp_aux", *, spec=None):
    directory = tmp_path / "output"
    directory.mkdir()
    probe = profile == "gpu-probe-v1"
    steps = 0 if probe else 2 if profile == "sft-smoke-v1" else 256
    native = arm.startswith("trl_")
    backend = "trl" if native else "opaque"
    stage = {
        "profile": profile,
        "status": "completed",
        "cuda_verified": True,
        "wandb_verified": True,
        "backend": backend,
        "arm": arm,
        "model_seed": 0,
        "steps": steps,
    }
    write_json(directory / "stage_receipt.json", stage)
    write_json(
        directory / "runtime.json",
        {"python": "3.12", "cuda": {"verified": True, "device": "cuda:0"}},
    )
    write_json(
        directory / "deployment.json",
        {
            "schema": 2,
            "plan": {"profile": profile, "backend": backend, "arm": arm, "seed": 0},
            "runner_returncode": 0,
            "cuda_verified": True,
            "wandb_verified": True,
            "adapter_reload_verified": not probe,
        },
    )
    if probe:
        write_json(
            directory / "probe.json",
            {
                "status": "completed",
                "cuda": {"verified": True},
                "wandb_verified": True,
            },
        )
        return directory

    config = read_json(
        ROOT / "examples/moe_privacy/configs/sft_balancing_aux0001_ratio1.json"
    )
    config.update(
        steps=steps,
        train_sequences=8,
        validation_sequences=2,
        test_sequences=2,
        diagnostic_sequences=1,
        expected_batch_size=2,
        microbatch_size=1,
    )
    data = {
        "kind": "public_code_instruction_completion",
        "dataset_id": config["dataset_id"],
        "dataset_revision": config["dataset_revision"],
        "packing": False,
    }
    for name in ("train", "validation", "test", "diagnostic"):
        count = config[f"{name}_sequences"]
        data[f"{name}_sha256"] = digest(name)
        data[name] = {
            "records": count,
            "sha256": digest(name),
            "prompt_hashes": [digest(f"{name}:{index}") for index in range(count)],
        }
    private = arm in {"dp", "dp_aux"}
    execution = {
        "device": "cuda:0",
        "device_name": "fixture GPU",
        "cuda_version": "12.8",
        "peak_cuda_memory_allocated_bytes": 1024,
    }
    settings = {
        "trainer_backend": "trl.SFTTrainer",
        "physical_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 2,
        "batch_clipping_norm": config["clipping_norm"],
        "sampling_mode": "shuffled_without_replacement",
        "shuffle_seed": 0,
        "task_loss_reduction": "mean_of_physical_batch_completion_token_means",
        "router_aux_loss_coef": config["router_aux_loss_coef"]
        if arm == "trl_reference_aux"
        else 0.0,
        "aux_loss_scope": "native_current_physical_batch",
        "aux_loss_accumulation": "mean_of_physical_batch_aux_losses",
        "global_batch_aux_equivalent": False,
        "router_precision": "fp32_native_forward_autocast_disabled",
        "expert_weight_cache": "per_expert_module_forward_parametrize_cached",
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optimizer": "sgd",
        "learning_rate": config["learning_rate"],
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "loss_type": "nll",
        "opaque_patches": False,
    }
    if native:
        execution.update(settings)
    initial = {"step": 0, "elapsed_seconds": 0.1, "eval_loss": 1.5}
    final = {"step": steps, "elapsed_seconds": 1.0, "eval_loss": 1.3}
    description = spec or adapter_spec()
    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "native_trl_moe_sft" if native else "pretrained_moe_sft",
        "config": config,
        "arm": arm,
        "model_seed": 0,
        "data": data,
        "trainability": description["metadata"],
        "execution": execution,
        "privacy": {
            "private": private,
            "unit": "one_preprocessed_prompt_answer_pair",
            "adjacency": "add_remove",
            "epsilon": 7.9 if private else None,
            "delta": config["delta"] if private else None,
            "target_epsilon": config["target_epsilon"] if private else None,
            "noise_multiplier": 1.2 if private else 0.0,
            "steps": steps,
            "sample_rate": None if native else 0.25,
            "load_release": arm in {"dp_aux", "reference_aux"},
            "load_noise_ratio": config["load_noise_ratio"] if arm == "dp_aux" else None,
            "balancing": {
                "dp_aux": "lagged_noisy",
                "reference_aux": "lagged_unnoised",
                "trl_reference_aux": "native_current_batch",
            }.get(arm, "off"),
            "raw_training_telemetry_released": False,
        },
        "initial": initial,
        "final": final,
        "test": {},
        "test_evaluated": False,
    }
    write_json(directory / "summary.json", summary)
    write_json(directory / "trainability.json", description["metadata"])
    write_json(directory / "execution.json", settings if native else execution)
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (initial, final))
    )
    (directory / "trainable").mkdir()
    write_json(directory / SPEC, description)
    if spec is None:
        save_file(
            {
                name: np.full(tensor["shape"], 0.25, dtype=np.float32)
                for name, tensor in description["tensors"].items()
            },
            directory / WEIGHTS,
        )
    else:
        sparse_weights(directory / WEIGHTS, spec)
    write_json(
        directory / "prefetch.json",
        {
            "status": "completed",
            **{
                key: config[key]
                for key in (
                    "model_id",
                    "model_revision",
                    "dataset_id",
                    "dataset_revision",
                )
            },
        },
    )
    write_json(
        directory / "adapter_reload.json",
        {
            "status": "completed",
            "verified": True,
            "tensor_count": len(description["tensors"]),
        },
    )
    write_json(
        directory / "wandb_run.json",
        {
            "status": "completed",
            "upload_status": "completed",
            "run_id": "fixture1",
            "entity": "federated-compute",
            "project": "moe-iceland",
            "base_url": "https://jetbrains.wandb.io",
            "url": "https://jetbrains.wandb.io/federated-compute/moe-iceland/runs/fixture1",
        },
    )
    return directory


def set_tracking_upload(directory, upload_status, *, operation=None):
    receipt = read_json(directory / "wandb_run.json")
    receipt["upload_status"] = upload_status
    write_json(directory / "wandb_run.json", receipt)
    if operation is not None:
        write_json(
            directory / "tracking_failure.json",
            {"operation": operation, "exception_type": "CommError"},
        )
    for name in ("stage_receipt.json", "deployment.json"):
        receipt = read_json(directory / name)
        receipt["wandb_verified"] = upload_status == "completed" and operation is None
        write_json(directory / name, receipt)


@pytest.mark.parametrize("profile", ["sft-smoke-v1", "sft-train-v1"])
@pytest.mark.parametrize(
    "arm",
    [
        "reference",
        "reference_aux",
        "dp",
        "dp_aux",
        "trl_reference",
        "trl_reference_aux",
    ],
)
def test_seal_real_safetensors_and_summary_formats(tmp_path, profile, arm):
    directory = make_bundle(tmp_path, profile, arm)
    expected = bundle.validate_bundle(directory, profile, require_manifest=False)
    with pytest.raises(ValueError, match="manifest"):
        bundle.validate_bundle(directory, profile)
    sealed = bundle.seal_bundle(directory, profile)
    assert sealed == expected == bundle.validate_bundle(directory, profile)
    assert sealed == read_json(directory / MANIFEST)
    assert sealed["schema"] == 1
    assert sealed["profile"] == profile
    assert MANIFEST not in sealed["files"]
    assert sealed["bytes"] == sum(file["bytes"] for file in sealed["files"].values())
    for name, file in sealed["files"].items():
        content = (directory / name).read_bytes()
        assert file == {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    assert bundle.seal_bundle(directory, profile) == sealed


def test_probe_and_predeployment_validation(tmp_path):
    directory = make_bundle(tmp_path, "gpu-probe-v1")
    deployment = read_json(directory / "deployment.json")
    (directory / "deployment.json").unlink()
    bundle.validate_bundle(directory, "gpu-probe-v1", require_manifest=False)
    with pytest.raises(ValueError, match=r"deployment\.json"):
        bundle.seal_bundle(directory, "gpu-probe-v1")
    write_json(directory / "deployment.json", deployment)
    assert bundle.seal_bundle(directory, "gpu-probe-v1") == bundle.validate_bundle(
        directory, "gpu-probe-v1"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("cuda", {"verified": False}),
        ("cuda", {"verified": 1}),
        ("wandb_verified", False),
    ],
)
def test_probe_requires_actual_success_receipts(tmp_path, field, value):
    directory = make_bundle(tmp_path, "gpu-probe-v1")
    receipt = read_json(directory / "probe.json")
    receipt[field] = value
    write_json(directory / "probe.json", receipt)
    with pytest.raises(ValueError, match="probe"):
        bundle.seal_bundle(directory, "gpu-probe-v1")


@pytest.mark.parametrize("profile", ["sft-smoke-v1", "sft-train-v1"])
@pytest.mark.parametrize("arm", ["dp_aux", "trl_reference"])
@pytest.mark.parametrize(
    "operation", ["init", "log_metrics", "log_progress", "complete", "finish"]
)
def test_sft_tracking_outage_keeps_completed_training_releasable(
    tmp_path, profile, arm, operation
):
    directory = make_bundle(tmp_path, profile, arm)
    summary = read_json(directory / "summary.json")
    set_tracking_upload(directory, "failed", operation=operation)
    if operation == "init":
        receipt = read_json(directory / "wandb_run.json")
        receipt["url"] = None
        write_json(directory / "wandb_run.json", receipt)
    expected = bundle.validate_bundle(directory, profile, require_manifest=False)
    assert bundle.seal_bundle(directory, profile) == expected
    assert bundle.validate_bundle(directory, profile) == expected
    assert read_json(directory / "summary.json") == summary
    assert read_json(directory / "wandb_run.json")["status"] == "completed"
    for name in ("stage_receipt.json", "deployment.json"):
        assert read_json(directory / name)["wandb_verified"] is False
    failure = directory / "tracking_failure.json"
    assert read_json(failure) == {"operation": operation, "exception_type": "CommError"}
    assert expected["files"][failure.name] == {
        "bytes": failure.stat().st_size,
        "sha256": hashlib.sha256(failure.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("profile", ["sft-smoke-v1", "sft-train-v1"])
@pytest.mark.parametrize("upload_status", ["pending", "offline"])
def test_sft_pending_and_offline_uploads_are_releasable_but_unverified(
    tmp_path, profile, upload_status
):
    directory = make_bundle(tmp_path, profile)
    set_tracking_upload(directory, upload_status)
    sealed = bundle.seal_bundle(directory, profile)
    assert bundle.validate_bundle(directory, profile) == sealed
    assert "tracking_failure.json" not in sealed["files"]
    for name in ("stage_receipt.json", "deployment.json"):
        assert read_json(directory / name)["wandb_verified"] is False


def test_sft_completed_upload_may_remain_conservatively_unverified(tmp_path):
    directory = make_bundle(tmp_path)
    for name in ("stage_receipt.json", "deployment.json"):
        receipt = read_json(directory / name)
        receipt["wandb_verified"] = False
        write_json(directory / name, receipt)
    sealed = bundle.seal_bundle(directory, "sft-smoke-v1")
    assert bundle.validate_bundle(directory, "sft-smoke-v1") == sealed


@pytest.mark.parametrize("name", ["stage_receipt.json", "deployment.json"])
def test_probe_still_requires_verified_uploads_in_every_receipt(tmp_path, name):
    directory = make_bundle(tmp_path, "gpu-probe-v1")
    receipt = read_json(directory / name)
    receipt["wandb_verified"] = False
    write_json(directory / name, receipt)
    with pytest.raises(ValueError, match="wandb_verified"):
        bundle.seal_bundle(directory, "gpu-probe-v1")


@pytest.mark.parametrize("profile", ["gpu-probe-v1", "sft-smoke-v1", "sft-train-v1"])
@pytest.mark.parametrize("schema", [1, "2", True, 2.0])
def test_gpu_deployment_requires_integer_schema_two(tmp_path, profile, schema):
    directory = make_bundle(tmp_path, profile)
    receipt = read_json(directory / "deployment.json")
    receipt["schema"] = schema
    write_json(directory / "deployment.json", receipt)
    with pytest.raises(ValueError, match=r"deployment\.json: schema"):
        bundle.seal_bundle(directory, profile)


@pytest.mark.parametrize("profile", ["gpu-probe-v1", "sft-smoke-v1"])
@pytest.mark.parametrize("name", ["stage_receipt.json", "deployment.json"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", "missing"])
def test_wandb_verification_flags_are_required_booleans(tmp_path, profile, name, value):
    directory = make_bundle(tmp_path, profile)
    receipt = read_json(directory / name)
    if value == "missing":
        receipt.pop("wandb_verified")
    else:
        receipt["wandb_verified"] = value
    write_json(directory / name, receipt)
    with pytest.raises(ValueError, match="wandb_verified"):
        bundle.seal_bundle(directory, profile)


@pytest.mark.parametrize("upload_status", ["pending", "offline", "failed"])
@pytest.mark.parametrize("claim", ["stage_receipt.json", "deployment.json", "both"])
def test_sft_unfinished_uploads_cannot_claim_verification(
    tmp_path, upload_status, claim
):
    directory = make_bundle(tmp_path)
    set_tracking_upload(
        directory,
        upload_status,
        operation="finish" if upload_status == "failed" else None,
    )
    names = ("stage_receipt.json", "deployment.json") if claim == "both" else (claim,)
    for name in names:
        receipt = read_json(directory / name)
        receipt["wandb_verified"] = True
        write_json(directory / name, receipt)
    with pytest.raises(ValueError, match="wandb_verified"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize(
    ("upload_status", "operation"),
    [
        ("completed", "finish"),
        ("pending", "init"),
        ("offline", "init"),
        ("failed", None),
    ],
)
def test_tracking_failure_must_match_failed_upload_status(
    tmp_path, upload_status, operation
):
    directory = make_bundle(tmp_path)
    set_tracking_upload(directory, upload_status, operation=operation)
    with pytest.raises(ValueError, match=r"upload_status|tracking_failure"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize(
    "upload_status", [None, True, [], "success", "running", "missing"]
)
def test_sft_upload_status_is_required_and_known(tmp_path, upload_status):
    directory = make_bundle(tmp_path)
    receipt = read_json(directory / "wandb_run.json")
    if upload_status == "missing":
        receipt.pop("upload_status")
    else:
        receipt["upload_status"] = upload_status
    write_json(directory / "wandb_run.json", receipt)
    with pytest.raises(ValueError, match="upload_status"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize(
    "failure",
    [
        {"operation": "init", "exception_type": "CommError", "message": "not public"},
        {"operation": "init"},
        {"exception_type": "CommError"},
        {"operation": "upload", "exception_type": "CommError"},
        {"operation": 1, "exception_type": "CommError"},
        {"operation": "x" * 129, "exception_type": "CommError"},
        {"operation": "init", "exception_type": None},
        {"operation": "init", "exception_type": 1},
        {"operation": "init", "exception_type": []},
        {"operation": "init", "exception_type": ""},
        {"operation": "init", "exception_type": " "},
        {"operation": "init", "exception_type": "E" * 129},
    ],
)
def test_tracking_failure_is_only_bounded_public_metadata(tmp_path, failure):
    directory = make_bundle(tmp_path)
    set_tracking_upload(directory, "failed", operation="init")
    write_json(directory / "tracking_failure.json", failure)
    with pytest.raises(ValueError, match=r"tracking_failure\.json"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


def test_init_failure_can_omit_url_with_bounded_failure_receipt(tmp_path):
    directory = make_bundle(tmp_path)
    set_tracking_upload(directory, "failed", operation="init")
    receipt = read_json(directory / "wandb_run.json")
    receipt.pop("url")
    write_json(directory / "wandb_run.json", receipt)
    write_json(
        directory / "tracking_failure.json",
        {"operation": "init", "exception_type": "E" * 128},
    )
    sealed = bundle.seal_bundle(directory, "sft-smoke-v1")
    assert bundle.validate_bundle(directory, "sft-smoke-v1") == sealed


@pytest.mark.parametrize("upload_status", ["completed", "pending", "offline", "failed"])
@pytest.mark.parametrize("omit_url", [False, True])
def test_missing_tracker_url_requires_matching_failure_receipt(
    tmp_path, upload_status, omit_url
):
    directory = make_bundle(tmp_path)
    set_tracking_upload(directory, upload_status)
    receipt = read_json(directory / "wandb_run.json")
    if omit_url:
        receipt.pop("url")
    else:
        receipt["url"] = None
    write_json(directory / "wandb_run.json", receipt)
    with pytest.raises(ValueError, match=r"url|tracking_failure"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize("upload_status", ["completed", "failed"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("url", "https://other.example/federated-compute/moe-iceland/runs/fixture1"),
        ("url", "https://jetbrains.wandb.io/federated-compute/moe-iceland/runs/other"),
        ("url", ""),
        ("entity", "other-team"),
        ("project", "other-project"),
        ("run_id", "other-run"),
        ("base_url", "http://jetbrains.wandb.io"),
        ("base_url", "https://user:password@jetbrains.wandb.io"),
    ],
)
def test_tracker_identity_and_https_remain_strict_during_outage(
    tmp_path, upload_status, key, value
):
    directory = make_bundle(tmp_path)
    set_tracking_upload(
        directory,
        upload_status,
        operation="finish" if upload_status == "failed" else None,
    )
    receipt = read_json(directory / "wandb_run.json")
    receipt[key] = value
    write_json(directory / "wandb_run.json", receipt)
    with pytest.raises(ValueError, match=r"wandb_run\.json"):
        bundle.seal_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize("name", ["wandb_run.json", "tracking_failure.json"])
def test_probe_excludes_sft_tracking_receipts(tmp_path, name):
    directory = make_bundle(tmp_path, "gpu-probe-v1")
    write_json(directory / name, {"status": "completed"})
    with pytest.raises(ValueError, match=r"Unexpected.*file"):
        bundle.seal_bundle(directory, "gpu-probe-v1")


@pytest.mark.parametrize(
    "name",
    [
        "summary.json",
        "metrics.jsonl",
        "trainability.json",
        "execution.json",
        SPEC,
        WEIGHTS,
        "runtime.json",
        "prefetch.json",
        "adapter_reload.json",
        "stage_receipt.json",
        "wandb_run.json",
        "tracking_failure.json",
        "deployment.json",
        MANIFEST,
    ],
)
def test_missing_sealed_files_are_rejected(tmp_path, name):
    directory = make_bundle(tmp_path)
    if name == "tracking_failure.json":
        set_tracking_upload(directory, "failed", operation="finish")
    bundle.seal_bundle(directory, "sft-smoke-v1")
    (directory / name).unlink()
    with pytest.raises(ValueError, match=r"Missing|manifest"):
        bundle.validate_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize(
    "name",
    [
        "data.json",
        "probe.json",
        "checks.json",
        "arbitrary.pt",
        "weights.bin",
        "optimizer.json",
        "rng_state.json",
        "credentials.json",
        "trainable/config.json",
        "trainable/optimizer.pt",
        "trainable/random.bin",
        "test-results.json",
    ],
)
def test_exact_gpu_file_allowlist(tmp_path, name):
    directory = make_bundle(tmp_path)
    (directory / name).write_text("{}")
    with pytest.raises(ValueError, match=r"Unexpected.*file|weights-only"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "name", ["wandb", "cache", "model", "trainable/wandb", "trainable/empty"]
)
def test_even_empty_unexpected_directories_are_rejected(tmp_path, name):
    directory = make_bundle(tmp_path)
    (directory / name).mkdir()
    with pytest.raises(ValueError, match=r"Unexpected.*directory"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "name", ["summary.json", "tracking_failure.json", "trainable", MANIFEST]
)
def test_file_directory_and_manifest_symlinks_are_rejected(tmp_path, name):
    directory = make_bundle(tmp_path)
    original = directory / name
    target = tmp_path / "target"
    if original.exists():
        original.rename(target)
    else:
        target.write_text("{}")
    original.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="Symlink"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_root_symlink_and_nonregular_files(tmp_path):
    directory = make_bundle(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="regular output directory"):
        bundle.validate_bundle(link, "sft-smoke-v1", require_manifest=False)
    (directory / "summary.json").unlink()
    os.mkfifo(directory / "summary.json")
    with pytest.raises(ValueError, match="Non-regular"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_hardlinks_are_not_materialized(tmp_path):
    directory = make_bundle(tmp_path)
    os.link(directory / "summary.json", tmp_path / "external-link")
    with pytest.raises(ValueError, match="Hard link"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "name", [WEIGHTS, "metrics.jsonl", "runtime.json", "tracking_failure.json"]
)
def test_checksums_detect_same_length_corruption_and_cannot_be_resealed(tmp_path, name):
    directory = make_bundle(tmp_path)
    if name == "tracking_failure.json":
        set_tracking_upload(directory, "failed", operation="finish")
    bundle.seal_bundle(directory, "sft-smoke-v1")
    path = directory / name
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    for verify in (
        lambda: bundle.validate_bundle(directory, "sft-smoke-v1"),
        lambda: bundle.validate_bundle(
            directory, "sft-smoke-v1", require_manifest=False
        ),
        lambda: bundle.seal_bundle(directory, "sft-smoke-v1"),
    ):
        with pytest.raises(ValueError, match=r"manifest|checksum"):
            verify()


@pytest.mark.parametrize(
    "problem", ["profile", "bytes", "digest", "missing", "extra", "self", "schema"]
)
def test_manifest_inventory_is_exact(tmp_path, problem):
    directory = make_bundle(tmp_path)
    manifest = bundle.seal_bundle(directory, "sft-smoke-v1")
    if problem == "profile":
        manifest["profile"] = "sft-train-v1"
    elif problem == "bytes":
        manifest["bytes"] += 1
    elif problem == "digest":
        manifest["files"][WEIGHTS]["sha256"] = "0" * 64
    elif problem == "missing":
        manifest["files"].pop("runtime.json")
    elif problem == "extra":
        manifest["files"]["../outside"] = {"bytes": 0, "sha256": "0" * 64}
    elif problem == "self":
        manifest["files"][MANIFEST] = {"bytes": 0, "sha256": "0" * 64}
    else:
        manifest["schema"] = True
    write_json(directory / MANIFEST, manifest)
    with pytest.raises(ValueError, match="manifest"):
        bundle.validate_bundle(directory, "sft-smoke-v1")


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("status",), "running", "completed"),
        (("model_seed",), True, "model_seed"),
        (("config", "steps"), 1, "steps"),
        (("final", "step"), 1, "steps|metrics"),
        (("execution", "device"), "cpu", "CUDA"),
        (("privacy", "private"), False, "privacy"),
        (("privacy", "epsilon"), 8.01, "epsilon"),
        (("privacy", "epsilon"), None, "epsilon"),
        (("privacy", "delta"), 0, "delta"),
        (("privacy", "target_epsilon"), 9, "epsilon"),
        (("privacy", "noise_multiplier"), 0, "noise_multiplier"),
        (("privacy", "steps"), 1, "steps"),
        (("privacy", "sample_rate"), 0.5, "sample_rate"),
        (("privacy", "load_release"), False, "load_release"),
        (("privacy", "load_noise_ratio"), 0.1, "load_noise_ratio"),
        (("privacy", "balancing"), "native_current_batch", "balancing"),
        (("privacy", "raw_training_telemetry_released"), True, "telemetry"),
        (("data", "train_sha256"), "a" * 64, "partition|data"),
        (("data", "train", "sha256"), "not-a-hash", "partition|data"),
        (("data", "train", "records"), 7, "partition|data"),
        (("trainability", "router_count"), 2, "trainability|metadata"),
    ],
)
def test_training_summary_invariants_before_sealing(tmp_path, keys, value, message):
    directory = make_bundle(tmp_path)
    summary = read_json(directory / "summary.json")
    current = summary
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    write_json(directory / "summary.json", summary)
    with pytest.raises(ValueError, match=message):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("stage_receipt.json", "status", "failed"),
        ("stage_receipt.json", "profile", "sft-train-v1"),
        ("stage_receipt.json", "cuda_verified", False),
        ("stage_receipt.json", "steps", 1),
        ("stage_receipt.json", "arm", "dp"),
        ("stage_receipt.json", "model_seed", 1),
        ("prefetch.json", "status", "failed"),
        ("prefetch.json", "model_revision", "b" * 40),
        ("adapter_reload.json", "verified", False),
        ("adapter_reload.json", "tensor_count", 4),
        ("wandb_run.json", "status", "running"),
        ("wandb_run.json", "url", "https://other.example/runs/fixture1"),
        ("deployment.json", "runner_returncode", 1),
        ("deployment.json", "cuda_verified", False),
    ],
)
def test_receipts_must_agree_with_successful_export(tmp_path, name, field, value):
    directory = make_bundle(tmp_path)
    receipt = read_json(directory / name)
    receipt[field] = value
    write_json(directory / name, receipt)
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("privacy", "epsilon", 0.0),
        ("privacy", "delta", 1e-5),
        ("privacy", "target_epsilon", 8.0),
        ("privacy", "noise_multiplier", 1.0),
        ("privacy", "sample_rate", 0.25),
        ("privacy", "load_release", True),
        ("privacy", "load_noise_ratio", 1.0),
        ("execution", "opaque_patches", True),
        ("execution", "sampling_mode", "poisson"),
        ("execution", "trainer_backend", "DPTrainer"),
        ("execution", "global_batch_aux_equivalent", True),
        ("execution", "router_aux_loss_coef", 0.2),
        ("execution", "gradient_accumulation_steps", 1),
    ],
)
def test_native_nonprivate_contract_is_not_dp_metadata(tmp_path, section, key, value):
    directory = make_bundle(tmp_path, arm="trl_reference_aux")
    summary = read_json(directory / "summary.json")
    summary[section][key] = value
    write_json(directory / "summary.json", summary)
    if section == "execution":
        execution = read_json(directory / "execution.json")
        execution[key] = value
        write_json(directory / "execution.json", execution)
    with pytest.raises(ValueError, match=r"privacy|execution"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "dtype"])
def test_real_safetensor_keys_shapes_and_dtypes_match_spec(tmp_path, problem):
    directory = make_bundle(tmp_path)
    tensors = load_file(directory / WEIGHTS)
    name = min(tensors)
    if problem == "missing":
        tensors.pop(name)
    elif problem == "extra":
        tensors["frozen.weight"] = np.zeros((2, 2), dtype=np.float32)
    elif problem == "shape":
        tensors[name] = tensors[name][:1].copy()
    else:
        tensors[name] = tensors[name].astype(np.float64)
    save_file(tensors, directory / WEIGHTS)
    with pytest.raises(ValueError, match=r"tensor|safetensors"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "problem", ["truncated", "header_cap", "gap", "overlap", "trailing", "duplicate"]
)
def test_safetensor_binary_layout_is_validated(tmp_path, problem):
    directory = make_bundle(tmp_path)
    path = directory / WEIGHTS
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    body = raw[8 + size :]
    if problem == "truncated":
        path.write_bytes(raw[:-1])
    elif problem == "header_cap":
        path.write_bytes(struct.pack("<Q", 1024**2 + 1))
    elif problem == "trailing":
        path.write_bytes(raw + b"\0")
    else:
        name = min(header)
        if problem == "gap":
            header[name]["data_offsets"] = [x + 4 for x in header[name]["data_offsets"]]
        elif problem == "overlap":
            name = max(header, key=lambda key: header[key]["data_offsets"][0])
            header[name]["data_offsets"] = [x - 4 for x in header[name]["data_offsets"]]
        encoded = json.dumps(header).encode()
        if problem == "duplicate":
            encoded = encoded[:-1] + b", " + json.dumps(name).encode() + b": {}}"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
    with pytest.raises(ValueError, match=r"safetensors|tensor|JSON"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_consistently_forged_frozen_parameter_scope_is_rejected(tmp_path):
    directory = make_bundle(tmp_path)
    spec = read_json(directory / SPEC)
    name = spec["metadata"]["router_names"][0]
    spec["metadata"]["router_names"] = ["model.embed_tokens.weight"]
    spec["tensors"]["model.embed_tokens.weight"] = spec["tensors"].pop(name)
    write_json(directory / SPEC, spec)
    write_json(directory / "trainability.json", spec["metadata"])
    summary = read_json(directory / "summary.json")
    summary["trainability"] = spec["metadata"]
    write_json(directory / "summary.json", summary)
    tensors = load_file(directory / WEIGHTS)
    tensors["model.embed_tokens.weight"] = tensors.pop(name)
    save_file(tensors, directory / WEIGHTS)
    with pytest.raises(ValueError, match=r"scope|router"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "name", ["summary.json", "runtime.json", "metrics.jsonl", "tracking_failure.json"]
)
@pytest.mark.parametrize(
    "raw", ['{"value": NaN}', '{"value": 1e999}', '{"value": 1, "value": 2}', "[]", "{"]
)
def test_all_metadata_is_strict_finite_json(tmp_path, name, raw):
    directory = make_bundle(tmp_path)
    (directory / name).write_text(raw + "\n")
    with pytest.raises(ValueError, match=r"JSON|finite|object"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize(
    "problem", ["empty", "initial_only", "out_of_order", "summary_mismatch"]
)
def test_metrics_prove_the_full_successful_horizon(tmp_path, problem):
    directory = make_bundle(tmp_path)
    summary = read_json(directory / "summary.json")
    rows = [summary["initial"], summary["final"]]
    if problem == "empty":
        rows = []
    elif problem == "initial_only":
        rows.pop()
    elif problem == "out_of_order":
        rows.reverse()
    else:
        rows[-1]["eval_loss"] += 0.1
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match=r"metrics|steps"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_optional_public_test_records(tmp_path):
    directory = make_bundle(tmp_path)
    summary = read_json(directory / "summary.json")
    summary["test_evaluated"] = True
    summary["test"] = {"eval_records": 2.0}
    write_json(directory / "summary.json", summary)
    rows = [
        {
            "record_index": index,
            "nll_sum": 1.0,
            "nll": 0.5,
            "supervised_tokens": 2,
            "attended_tokens": 3,
            "teacher_forced_correct_tokens": 1,
        }
        for index in range(2)
    ]
    (directory / "test-records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    bundle.seal_bundle(directory, "sft-smoke-v1")
    assert (
        "test-records.jsonl"
        in bundle.validate_bundle(directory, "sft-smoke-v1")["files"]
    )


def test_sft_artifact_above_cpu_pilot_cap_is_valid_and_bounded(tmp_path):
    spec = adapter_spec(experts=256, hidden=8192, intermediate=8192)
    directory = make_bundle(tmp_path, "sft-train-v1", spec=spec)
    with safe_open(directory / WEIGHTS, framework="np") as saved:
        assert set(saved.keys()) == spec["tensors"].keys()
    sealed = bundle.seal_bundle(directory, "sft-train-v1")
    assert 128 * 1024**2 < sealed["bytes"] < 1024**3
    assert bundle.validate_bundle(directory, "sft-train-v1") == sealed
    with (directory / WEIGHTS).open("r+b") as stream:
        stream.truncate(1024**3 + 1)
    with pytest.raises(ValueError, match="byte cap"):
        bundle.validate_bundle(directory, "sft-train-v1")


@pytest.mark.parametrize(
    "name", ["runtime.json", "metrics.jsonl", "tracking_failure.json"]
)
def test_metadata_byte_limits_apply_before_parsing(tmp_path, name):
    directory = make_bundle(tmp_path)
    with (directory / name).open("wb") as stream:
        stream.truncate(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match=r"JSON|metadata|byte cap"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_entry_limit_and_json_nesting_are_bounded(tmp_path, monkeypatch):
    directory = make_bundle(tmp_path)
    with monkeypatch.context() as local:
        local.setattr(bundle, "BUNDLE_ENTRY_LIMIT", 2)
        with pytest.raises(ValueError, match="entry cap"):
            bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)
    (directory / "runtime.json").write_text(
        '{"value":' + "[" * 100 + "0" + "]" * 100 + "}"
    )
    with pytest.raises(ValueError, match=r"depth|nesting"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


@pytest.mark.parametrize("limit", ["bytes", "entries"])
def test_sealing_reserves_space_for_manifest(tmp_path, monkeypatch, limit):
    directory = make_bundle(tmp_path)
    inventory = bundle.validate_bundle(
        directory, "sft-smoke-v1", require_manifest=False
    )
    if limit == "bytes":
        monkeypatch.setitem(bundle._GPU_LIMITS, "sft-smoke-v1", inventory["bytes"])
    else:
        monkeypatch.setattr(bundle, "BUNDLE_ENTRY_LIMIT", len(inventory["files"]) + 1)
    with pytest.raises(ValueError, match=r"manifest.*cap"):
        bundle.seal_bundle(directory, "sft-smoke-v1")
    assert not (directory / MANIFEST).exists()


def test_jsonl_line_and_row_caps(tmp_path, monkeypatch):
    directory = make_bundle(tmp_path)
    with monkeypatch.context() as local:
        local.setattr(bundle, "_JSONL_ROWS", 1)
        with pytest.raises(ValueError, match="bounded JSON"):
            bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)
    (directory / "metrics.jsonl").write_bytes(b" " * (1024**2 + 1))
    with pytest.raises(ValueError, match="bounded JSON"):
        bundle.validate_bundle(directory, "sft-smoke-v1", require_manifest=False)


def test_real_tiny_mellum_export_matches_static_validation(tmp_path):
    pytest.importorskip("transformers")
    import torch
    from examples.moe_privacy.sft_adapters import (
        configure_expert_lora,
        export_trainable,
    )
    from transformers import MellumConfig, MellumForCausalLM

    config = MellumConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=6,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, _ = configure_expert_lora(MellumForCausalLM(config), apply_patches=False)
    export = tmp_path / "real-export"
    export_trainable(model, export)
    spec = read_json(export / "adapter_spec.json")
    directory = make_bundle(tmp_path, spec=spec)
    shutil.copyfile(export / "trainable.safetensors", directory / WEIGHTS)
    sealed = bundle.seal_bundle(directory, "sft-smoke-v1")
    assert sealed == bundle.validate_bundle(directory, "sft-smoke-v1")


def test_sft_is_not_generation_and_cpu_sealing_is_out_of_scope(tmp_path):
    directory = make_bundle(tmp_path)
    with pytest.raises(
        ValueError, match=r"Unexpected GPU output|Unexpected output directory"
    ):
        bundle.validate_bundle(directory, "generate-v1")
    with pytest.raises(ValueError, match=r"profile|supported"):
        bundle.validate_bundle(directory, "unknown")
    with pytest.raises(ValueError, match=r"GPU|profile"):
        bundle.seal_bundle(directory, "smoke")


@pytest.mark.parametrize("tracking_failed", [False, True])
def test_real_path_materializer_roundtrip_preserves_seal(tmp_path, tracking_failed):
    pytest.importorskip("zenml")
    from zenml.materializers.path_materializer import PathMaterializer

    directory = make_bundle(tmp_path, arm="trl_reference_aux")
    if tracking_failed:
        set_tracking_upload(directory, "failed", operation="finish")
    expected = copy.deepcopy(bundle.seal_bundle(directory, "sft-smoke-v1"))
    store = tmp_path / "artifact-store"
    store.mkdir()
    materializer = PathMaterializer(uri=str(store))
    materializer.save(directory)
    shutil.rmtree(directory)
    restored = materializer.load(Path)
    try:
        assert bundle.validate_bundle(restored, "sft-smoke-v1") == expected
        with (restored / WEIGHTS).open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"\xff")
        with pytest.raises(ValueError, match=r"manifest|checksum"):
            bundle.validate_bundle(restored, "sft-smoke-v1")
    finally:
        shutil.rmtree(restored)
