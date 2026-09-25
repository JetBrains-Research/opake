"""Weights-only recovery/transport tests; synthetic receipts are never run evidence."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from deploy.zenml.moe_iceland import checkpoint, gpu_runner
from deploy.zenml.moe_iceland.source import sha256

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def exported(tmp_path):
    import torch
    from examples.moe_privacy.sft_adapters import (
        configure_expert_lora,
        export_trainable,
    )
    from transformers import MellumConfig, MellumForCausalLM

    torch.manual_seed(0)
    model = MellumForCausalLM(
        MellumConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=6,
            mlp_layer_types=["sparse"],
            use_cache=False,
        )
    )
    model, metadata = configure_expert_lora(model)
    raw = tmp_path / "original"
    raw.mkdir()
    export_trainable(model, raw / "trainable")
    config = json.loads(
        (
            ROOT / "examples/moe_privacy/configs/sft_balancing_aux0001_ratio1.json"
        ).read_text()
    )
    summary = {
        "status": "completed",
        "config": config,
        "arm": "dp_aux",
        "model_seed": 0,
        "trainability": metadata,
        "execution": {"device": "cuda:0"},
        "data": {
            name: {"sha256": digest} for name, digest in gpu_runner.PARTITIONS.items()
        },
        "privacy": {
            "private": True,
            "steps": 256,
            "load_release": True,
            "epsilon": 7.99,
            "delta": 1e-5,
            "target_epsilon": 8.0,
            "noise_multiplier": 0.6032,
            "load_noise_ratio": 1.0,
        },
    }
    (raw / "summary.json").write_text(json.dumps(summary))
    (raw / "wandb_run.json").write_text('{"status":"failed"}')
    return raw, config


def test_recovery_preserves_failed_tracking_and_original_weights(exported, tmp_path):
    raw, config = exported
    before = {
        name: sha256(raw / name) for name in checkpoint.PAYLOAD | {"wandb_run.json"}
    }
    destination = tmp_path / "transport"
    manifest = checkpoint.package_export(raw, destination, config=config, arm="dp_aux")
    assert manifest["kind"] == "recovered-weights-only"
    assert {name: sha256(raw / name) for name in before} == before
    assert not (destination / "wandb_run.json").exists()
    assert (
        checkpoint.validate_export(destination, config=config, arm="dp_aux", seed=0)
        == manifest["files"]
    )


def test_real_path_materializer_archive_is_safe_to_download(exported, tmp_path):
    from zenml.materializers.path_materializer import PathMaterializer

    raw, config = exported
    payload = tmp_path / "transport"
    checkpoint.package_export(raw, payload, config=config, arm="dp_aux")
    store = tmp_path / "store"
    store.mkdir()
    PathMaterializer(uri=str(store)).save(payload)
    archive = store / "data.tar.gz"
    restored = tmp_path / "downloaded"
    checkpoint.extract_archive(archive, restored, expected_sha256=sha256(archive))
    assert checkpoint.validate_export(restored, config=config, arm="dp_aux", seed=0)


@pytest.mark.parametrize(
    "kind", ["truncated", "unfinished", "configuration", "wrong_arm", "wrong_partition"]
)
def test_invalid_checkpoint_is_blocker_not_permission_to_train(exported, kind):
    raw, config = exported
    path = raw / "summary.json"
    summary = json.loads(path.read_text())
    if kind == "truncated":
        weights = raw / "trainable/trainable.safetensors"
        weights.write_bytes(weights.read_bytes()[:-4])
    elif kind == "unfinished":
        summary["status"] = "failed"
    elif kind == "configuration":
        summary["config"]["steps"] = 2
    elif kind == "wrong_arm":
        summary["arm"] = "dp"
    else:
        summary["data"]["test"]["sha256"] = "wrong"
    path.write_text(json.dumps(summary))
    with pytest.raises(
        ValueError, match=r"tensor data is truncated|completed CUDA|partition"
    ):
        checkpoint.validate_export(raw, config=config, arm="dp_aux", seed=0)


@pytest.mark.parametrize(
    ("name", "link"),
    [
        ("../escape", False),
        ("trainable/trainable.safetensors", True),
        ("optimizer.pt", False),
    ],
)
def test_archive_rejects_escape_links_and_private_state(tmp_path, name, link):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo(name)
        if link:
            item.type, item.linkname = tarfile.SYMTYPE, "/outside"
        else:
            item.size = 1
        stream.addfile(item, None if link else io.BytesIO(b"x"))
    with pytest.raises(ValueError, match=r"Unsafe|unsupported"):
        checkpoint.extract_archive(
            archive, tmp_path / "destination", expected_sha256=sha256(archive)
        )
    assert not (tmp_path / "escape").exists()


def test_download_is_bounded_and_checks_hash(tmp_path, monkeypatch):
    from zenml.io import fileio

    monkeypatch.setattr(fileio, "open", lambda *args, **kwargs: io.BytesIO(b"contents"))
    reference = {"uri": "gs://gke-dev-dws-jbr-zenml/test.jsonl", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="allowance"):
        checkpoint.download_object(reference, tmp_path / "large", limit=1)
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.download_object(reference, tmp_path / "wrong")


def test_generation_command_cannot_score_or_resume_training(tmp_path):
    plan = {"arm": "dp_aux", "seed": 0, "benchmark": "mbpp"}
    command = gpu_runner.generation_command(plan, tmp_path / "config.json", tmp_path)
    assert command[2:4] == ["examples.moe_privacy.campaign", "generate"]
    assert command[command.index("--benchmarks") + 1] == "mbpp"
    assert command[command.index("--max-new-tokens") + 1] == "512"
    assert "--eval-image" not in command
    assert "--resume" not in command
