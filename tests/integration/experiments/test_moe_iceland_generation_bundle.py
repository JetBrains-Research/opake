"""Offline generation artifact contracts; synthetic text is not model output or scores."""

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
from deploy.zenml.moe_iceland import bundle
from examples.moe_privacy import code_eval, handoff

PROFILE = "generate-v1"
MANIFEST = "bundle-manifest.json"
IMAGE = "europe-docker.pkg.dev/fixture/public/generation@sha256:" + "a" * 64
RUN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError(
            "Artifact fixtures must not contact a server or generate text"
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(code_eval, "generate_samples", deny)
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_ENABLE_REPO_INIT_WARNINGS", "false")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def write_json(path, value):
    path.write_bytes(handoff.json_bytes(value))


def read_json(path):
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_pipeline_handoff_selection_preserves_manifest_and_excludes_runtime_cache(
    tmp_path, benchmark
):
    from deploy.zenml.moe_iceland.pipeline import public_outputs

    raw = make_bundle(tmp_path, benchmark=benchmark)
    (raw / "wandb").mkdir()
    (raw / "wandb/cache.log").write_text("excluded SDK cache")
    (raw / "generation_execution.json").write_text("{}")
    selected = tmp_path / "selected"
    public_outputs(raw, selected)
    shutil.copyfile(raw / "deployment.json", selected / "deployment.json")
    expected = bundle.seal_bundle(selected, PROFILE)
    assert bundle.validate_bundle(selected, PROFILE) == expected
    assert (
        handoff.validate_generation(selected / "answers")["manifest_sha256"]
        == read_json(selected / "stage_receipt.json")["generation_manifest_sha256"]
    )
    assert not (selected / "wandb").exists()
    assert not (selected / "generation_execution.json").exists()


def write_benchmark(
    answers,
    name,
    *,
    smoke_limit,
    max_new_tokens,
    model,
    padding=0,
    generation_padding=0,
):
    spec = code_eval.BENCHMARKS[name]
    path = answers / f"{name}.jsonl"
    with path.open("wb") as stream:
        for index in reversed(range(spec["count"])):
            stream.write(
                handoff.json_bytes(
                    {
                        "task_id": f"{spec['prefix']}/{index}",
                        "prompt": f"def synthetic_{index}(x):\n",
                        "canonical_solution": "    raise RuntimeError('inert fixture')\n",
                        "entry_point": f"synthetic_{index}",
                        "contract": " " * padding if index == 0 else "",
                        "base_input": [[1]],
                        "plus_input": [[2]],
                        "atol": 0,
                    }
                )
            )
    tasks, metadata = code_eval.load_benchmark(name, path)
    ids = list(tasks)[:smoke_limit]
    samples, records = [], []
    for task_id in ids:
        prompt = tasks[task_id]["prompt"]
        solution = prompt + "    !!! synthetic fixture, not executable Python !!!\n"
        samples.append(handoff.json_bytes({"task_id": task_id, "solution": solution}))
        records.append(
            {
                "task_id": task_id,
                "prompt_sha256": handoff.sha256(prompt.encode()),
                "solution_sha256": handoff.sha256(solution.encode()),
                "raw_solution_sha256": handoff.sha256(solution.encode()),
                "completion_boundary": None,
                "prompt_tokens": 4,
                "latency_seconds": 0.25,
                "generated_tokens": 3,
                "token_limit_truncated": False,
                "stop_reason": "eos",
            }
        )
    if generation_padding:
        records[0]["fixture_metadata"] = "x" * generation_padding
    raw = b"".join(samples)
    for suffix in ("samples.jsonl", "raw_samples.jsonl"):
        (answers / f"{name}-{suffix}").write_bytes(raw)
    write_json(
        answers / f"{name}-generation.json",
        {
            "format_version": code_eval.GENERATION_FORMAT_VERSION,
            "evalplus_version": code_eval.EVALPLUS_VERSION,
            "mode": "full" if smoke_limit is None else "smoke",
            "smoke_limit": smoke_limit,
            "benchmark": metadata,
            "model": model,
            "generation": {
                **code_eval.generation_settings(max_new_tokens),
                "eos_token_ids": [0],
            },
            "task_ids": ids,
            "tasks": records,
            "samples_sha256": handoff.sha256(raw),
            "raw_samples_sha256": handoff.sha256(raw),
        },
    )


def make_bundle(
    tmp_path,
    benchmark="humaneval",
    arm="dp_aux",
    *,
    benchmarks=None,
    seed=0,
    smoke_limit=None,
    max_new_tokens=512,
    benchmark_padding=0,
    metadata_padding=0,
    generation_padding=0,
    checkpoint_receipts=False,
):
    directory = tmp_path / "output"
    answers = directory / "answers"
    answers.mkdir(parents=True)
    backend = "trl" if arm.startswith("trl_") else "opaque"
    config = {
        "model_id": code_eval.MODEL_ID,
        "model_revision": code_eval.MODEL_REVISION,
        "dataset_id": "fixture/public-code",
        "dataset_revision": "b" * 40,
        "lora_rank": 4,
        "lora_alpha": 8,
        "steps": 256,
    }
    checkpoint_dir = tmp_path / "checkpoint"
    trainable = checkpoint_dir / "trainable"
    trainable.mkdir(parents=True)
    write_json(
        checkpoint_dir / "summary.json",
        {
            "status": "completed",
            "experiment": "native_trl_moe_sft"
            if backend == "trl"
            else "pretrained_moe_sft",
            "arm": arm,
            "model_seed": seed,
            "config": config,
            "fixture_metadata": "x" * metadata_padding,
        },
    )
    spec = {
        "format_version": 1,
        "adapter_kind": "static_parametrization_expert_lora",
        "config": {"rank": 4, "alpha": 8, "target_parameters": ["expert"]},
        "metadata": {"router_count": 1},
        "tensors": {"expert": {"dtype": "torch.float32", "shape": [1]}},
    }
    write_json(trainable / "adapter_spec.json", spec)
    (trainable / "trainable.safetensors").write_bytes(
        b"Synthetic identity bytes, never loaded"
    )
    if checkpoint_receipts:
        write_json(
            checkpoint_dir / "training_receipt.json",
            {"stage": "training", "status": "completed"},
        )
        write_json(
            checkpoint_dir / "wandb_run.json",
            {
                "status": "completed",
                "upload_status": "failed",
                "run_id": "historical",
                "url": None,
            },
        )
    checkpoint, payload = handoff.checkpoint_payload(config, checkpoint_dir, arm, seed)
    for name, raw in payload.items():
        (answers / name).write_bytes(raw)
    benchmarks = [benchmark] if benchmarks is None else benchmarks
    for name in benchmarks:
        write_benchmark(
            answers,
            name,
            smoke_limit=smoke_limit,
            max_new_tokens=max_new_tokens,
            padding=benchmark_padding,
            generation_padding=generation_padding,
            model={
                "id": config["model_id"],
                "revision": config["model_revision"],
                "arm": arm,
                "adapter_config": spec["config"],
                "router_count": 1,
                "merged_for_evaluation": True,
            },
        )
    generated = handoff.seal_generation(
        answers,
        checkpoint=checkpoint,
        benchmarks=benchmarks,
        max_new_tokens=max_new_tokens,
        smoke_limit=smoke_limit,
    )
    stage = {
        "profile": PROFILE,
        "stage": "generation",
        "status": "completed",
        "cuda_verified": True,
        "wandb_verified": True,
        "backend": backend,
        "arm": arm,
        "model_seed": seed,
        "steps": 0,
        "config_sha256": handoff.sha256(json.dumps(config, indent=2).encode()),
        "source_sha256": "c" * 64,
        "image": IMAGE,
        "zenml_run_id": RUN_ID,
        "generation_manifest_sha256": generated["manifest_sha256"],
        "benchmark": benchmark,
    }
    write_json(directory / "stage_receipt.json", stage)
    write_json(
        directory / "runtime.json",
        {"cuda": {"verified": True, "device": "cuda:0"}, "backend": backend},
    )
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
        directory / "wandb_run.json",
        {
            "status": "completed",
            "upload_status": "completed",
            "run_id": "fixture",
            "entity": "fixture",
            "project": "public",
            "base_url": "https://wandb.invalid",
            "url": "https://wandb.invalid/fixture/public/runs/fixture",
        },
    )
    write_json(
        directory / "deployment.json",
        {
            "schema": 2,
            "plan": {
                **{
                    key: stage[key]
                    for key in (
                        "profile",
                        "backend",
                        "arm",
                        "benchmark",
                        "config_sha256",
                        "source_sha256",
                        "image",
                    )
                },
                "seed": seed,
                "checkpoint": {
                    "uri": "gs://gke-dev-dws-jbr-zenml/fixtures/checkpoint.tar",
                    "sha256": "d" * 64,
                },
                "benchmark_input": {
                    "uri": f"gs://gke-dev-dws-jbr-zenml/fixtures/{benchmark}.jsonl",
                    "sha256": handoff.file_record(answers / f"{benchmark}.jsonl")[
                        "sha256"
                    ],
                },
            },
            "source": {"source_sha256": stage["source_sha256"]},
            "runner_returncode": 0,
            "cuda_verified": True,
            "wandb_verified": True,
            "adapter_reload_verified": False,
            "zenml_run_id": RUN_ID,
        },
    )
    return directory


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
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
def test_full_generation_can_be_sealed_without_weights_or_scores(
    tmp_path, benchmark, arm
):
    directory = make_bundle(tmp_path, benchmark, arm)
    answers = directory / "answers"
    stage = read_json(directory / "stage_receipt.json")
    generated = handoff.validate_generation(
        answers, expected_manifest_sha256=stage["generation_manifest_sha256"]
    )
    assert stage["config_sha256"] != generated["checkpoint"]["config_sha256"]
    assert (
        len(generated["benchmarks"][benchmark]["task_ids"])
        == code_eval.BENCHMARKS[benchmark]["count"]
    )
    with pytest.raises(ValueError, match="Missing bundle-manifest"):
        bundle.validate_bundle(directory, PROFILE)
    expected = bundle.validate_bundle(directory, PROFILE, require_manifest=False)
    assert bundle.seal_bundle(directory, PROFILE) == expected
    assert bundle.validate_bundle(directory, PROFILE) == expected
    assert bundle.seal_bundle(directory, PROFILE) == expected
    assert MANIFEST not in expected["files"]
    assert "answers/generation-manifest.json" in expected["files"]
    assert expected["bytes"] == sum(
        record["bytes"] for record in expected["files"].values()
    )
    assert not any(
        name.endswith((".safetensors", ".pt", ".bin")) for name in expected["files"]
    )
    assert not (answers / "code_metrics.json").exists()


def test_generation_preseal_requires_deployment_only_when_sealing(tmp_path):
    directory = make_bundle(tmp_path)
    (directory / "deployment.json").unlink()
    bundle.validate_bundle(directory, PROFILE, require_manifest=False)
    with pytest.raises(ValueError, match=r"deployment\.json"):
        bundle.seal_bundle(directory, PROFILE)
    assert not (directory / MANIFEST).exists()


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
@pytest.mark.parametrize("tracking_failed", [False, True])
def test_real_generation_path_materializer_roundtrip(
    tmp_path, benchmark, tracking_failed
):
    pytest.importorskip("zenml")
    from zenml.materializers.path_materializer import PathMaterializer

    directory = make_bundle(tmp_path, benchmark)
    if tracking_failed:
        set_tracking(directory, "failed", operation="finish")
        change_json(directory, "wandb_run.json", "url", None)
    expected = bundle.seal_bundle(directory, PROFILE)
    store = tmp_path / "artifact-store"
    store.mkdir()
    materializer = PathMaterializer(uri=str(store))
    materializer.save(directory)
    shutil.rmtree(directory)
    restored = materializer.load(Path)
    try:
        assert bundle.validate_bundle(restored, PROFILE) == expected
        stage = read_json(restored / "stage_receipt.json")
        generated = handoff.validate_generation(
            restored / "answers",
            expected_manifest_sha256=stage["generation_manifest_sha256"],
        )
        assert generated["manifest_sha256"] == stage["generation_manifest_sha256"]
        if tracking_failed:
            assert read_json(restored / "tracking_failure.json") == {
                "operation": "finish",
                "exception_type": "FixtureUploadError",
            }
        with (restored / "answers" / f"{benchmark}-samples.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with pytest.raises(ValueError, match="inventory/checksum mismatch"):
            bundle.validate_bundle(restored, PROFILE)
    finally:
        shutil.rmtree(restored)


def change_json(directory, name, field, value):
    document = read_json(directory / name)
    target = document
    *parents, key = field.split(".")
    for parent in parents:
        target = target[parent]
    target[key] = value
    write_json(directory / name, document)


def set_tracking(directory, status, *, operation="complete"):
    change_json(directory, "wandb_run.json", "upload_status", status)
    for name in ("stage_receipt.json", "deployment.json"):
        change_json(directory, name, "wandb_verified", status == "completed")
    if status == "failed":
        write_json(
            directory / "tracking_failure.json",
            {"operation": operation, "exception_type": "FixtureUploadError"},
        )


def refresh_handoff_hashes(directory):
    answers = directory / "answers"
    manifest = read_json(answers / handoff.MANIFEST)
    manifest["files"] = {
        name: handoff.file_record(answers / name) for name in manifest["files"]
    }
    write_json(answers / handoff.MANIFEST, manifest)
    change_json(
        directory,
        "stage_receipt.json",
        "generation_manifest_sha256",
        handoff.file_record(answers / handoff.MANIFEST)["sha256"],
    )


def test_historical_checkpoint_tracking_is_not_generation_tracking(tmp_path):
    directory = make_bundle(tmp_path, checkpoint_receipts=True)
    result = bundle.seal_bundle(directory, PROFILE)
    assert "answers/checkpoint-wandb_run.json" in result["files"]
    assert "answers/checkpoint-training_receipt.json" in result["files"]
    assert read_json(directory / "stage_receipt.json")["wandb_verified"] is True


@pytest.mark.parametrize(
    "name",
    [
        "runtime.json",
        "prefetch.json",
        "stage_receipt.json",
        "wandb_run.json",
        "answers/generation-manifest.json",
        "answers/generation_receipt.json",
        "answers/config.json",
        "answers/checkpoint-summary.json",
        "answers/adapter_spec.json",
        "answers/humaneval.jsonl",
        "answers/humaneval-generation.json",
        "answers/humaneval-samples.jsonl",
        "answers/humaneval-raw_samples.jsonl",
    ],
)
def test_missing_generation_payload_is_rejected(tmp_path, name):
    directory = make_bundle(tmp_path)
    (directory / name).unlink()
    with pytest.raises(ValueError, match=r"Missing|missing|handoff"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


def test_missing_answers_directory_is_rejected(tmp_path):
    directory = make_bundle(tmp_path)
    shutil.rmtree(directory / "answers")
    with pytest.raises(ValueError, match="generation handoff"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


@pytest.mark.parametrize(
    "name",
    [
        "runtime.json",
        "stage_receipt.json",
        "deployment.json",
        "wandb_run.json",
        "answers/generation-manifest.json",
        "answers/humaneval-samples.jsonl",
        "answers/humaneval.jsonl",
    ],
)
def test_outer_manifest_detects_exact_byte_tampering_even_preseal(tmp_path, name):
    directory = make_bundle(tmp_path)
    bundle.seal_bundle(directory, PROFILE)
    before = (directory / MANIFEST).read_bytes()
    with (directory / name).open("ab") as stream:
        stream.write(b"\n")
    for require_manifest in (False, True):
        with pytest.raises(ValueError, match="inventory/checksum mismatch"):
            bundle.validate_bundle(
                directory, PROFILE, require_manifest=require_manifest
            )
    with pytest.raises(ValueError, match="inventory/checksum mismatch"):
        bundle.seal_bundle(directory, PROFILE)
    assert (directory / MANIFEST).read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 2),
        ("profile", "sft-train-v1"),
        ("bytes", True),
        ("files", {"../outside": {"bytes": 0, "sha256": "0" * 64}}),
    ],
)
def test_corrupt_outer_manifest_is_rejected(tmp_path, field, value):
    directory = make_bundle(tmp_path)
    bundle.seal_bundle(directory, PROFILE)
    change_json(directory, MANIFEST, field, value)
    with pytest.raises(ValueError, match="inventory/checksum mismatch"):
        bundle.validate_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "name",
    [
        "humaneval-samples.jsonl",
        "humaneval-raw_samples.jsonl",
        "humaneval.jsonl",
        "checkpoint-summary.json",
    ],
)
def test_inner_checksum_tampering_cannot_be_blessed_by_sealing(tmp_path, name):
    directory = make_bundle(tmp_path)
    with (directory / "answers" / name).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="generation handoff"):
        bundle.seal_bundle(directory, PROFILE)
    assert not (directory / MANIFEST).exists()


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
@pytest.mark.parametrize("smoke_limit", [2, "all"])
def test_valid_smoke_handoff_is_not_a_full_gpu_generation(
    tmp_path, benchmark, smoke_limit
):
    count = code_eval.BENCHMARKS[benchmark]["count"]
    directory = make_bundle(
        tmp_path, benchmark, smoke_limit=count if smoke_limit == "all" else smoke_limit
    )
    handoff.validate_generation(directory / "answers")
    with pytest.raises(ValueError, match="mode must be 'full'"):
        bundle.seal_bundle(directory, PROFILE)


def test_multiple_benchmarks_are_not_one_planned_generation(tmp_path):
    directory = make_bundle(tmp_path, benchmarks=["humaneval", "mbpp"])
    handoff.validate_generation(directory / "answers")
    with pytest.raises(ValueError, match="exactly the one planned benchmark"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize("max_new_tokens", [128, 1024])
def test_generation_requires_the_fixed_512_token_budget(tmp_path, max_new_tokens):
    directory = make_bundle(tmp_path, max_new_tokens=max_new_tokens)
    handoff.validate_generation(directory / "answers")
    with pytest.raises(ValueError, match="max_new_tokens must be 512"):
        bundle.seal_bundle(directory, PROFILE)


def test_generation_requires_public_seed_zero(tmp_path):
    directory = make_bundle(tmp_path, seed=1)
    handoff.validate_generation(directory / "answers")
    with pytest.raises(ValueError, match="model_seed must be 0"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
@pytest.mark.parametrize("kind", ["benchmark", "samples", "duplicate-samples"])
def test_partial_or_duplicate_tasks_cannot_be_hidden_by_updating_hashes(
    tmp_path, benchmark, kind
):
    directory = make_bundle(tmp_path, benchmark)
    answers = directory / "answers"
    generation_path = answers / f"{benchmark}-generation.json"
    generation = read_json(generation_path)
    if kind == "benchmark":
        path = answers / f"{benchmark}.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
    else:
        for suffix, key in (
            ("samples.jsonl", "samples_sha256"),
            ("raw_samples.jsonl", "raw_samples_sha256"),
        ):
            path = answers / f"{benchmark}-{suffix}"
            rows = path.read_bytes().splitlines(keepends=True)
            rows = rows[:-1] if kind == "samples" else [*rows[:-1], rows[0]]
            raw = b"".join(rows)
            path.write_bytes(raw)
            generation[key] = handoff.sha256(raw)
        generation["task_ids"] = generation["task_ids"][:-1]
        generation["tasks"] = generation["tasks"][:-1]
        write_json(generation_path, generation)
    refresh_handoff_hashes(directory)
    with pytest.raises(
        ValueError, match=r"generation handoff.*(requires all|protocol/data|task)"
    ):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("config.json", "model_revision", "e" * 40),
        ("checkpoint-summary.json", "config.model_id", "fixture/other-model"),
        ("checkpoint-summary.json", "arm", "reference"),
        ("checkpoint-summary.json", "status", "failed"),
        ("humaneval-generation.json", "model.revision", "e" * 40),
        ("humaneval-generation.json", "model.arm", "reference"),
        ("humaneval-generation.json", "model.merged_for_evaluation", False),
        ("adapter_spec.json", "config.rank", 8),
        ("generation_receipt.json", "checkpoint.backend", "trl.SFTTrainer"),
    ],
)
def test_handoff_recomputes_model_and_checkpoint_identity(tmp_path, name, field, value):
    directory = make_bundle(tmp_path)
    change_json(directory / "answers", name, field, value)
    refresh_handoff_hashes(directory)
    with pytest.raises(ValueError, match="generation handoff"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "sft-train-v1"),
        ("stage", "training"),
        ("status", "failed"),
        ("cuda_verified", False),
        ("cuda_verified", 1),
        ("wandb_verified", "true"),
        ("wandb_verified", 1),
        ("backend", "trl"),
        ("backend", []),
        ("arm", "base"),
        ("arm", "reference"),
        ("arm", []),
        ("model_seed", 1),
        ("model_seed", False),
        ("steps", 256),
        ("steps", False),
        ("benchmark", "mbpp"),
        ("benchmark", ["humaneval"]),
        ("benchmark", "unknown"),
        ("config_sha256", "invalid"),
        ("source_sha256", None),
        ("generation_manifest_sha256", "e" * 64),
        ("generation_manifest_sha256", "invalid"),
        ("image", "fixture:latest"),
        ("image", None),
        ("zenml_run_id", ""),
    ],
)
def test_invalid_stage_receipts_are_rejected(tmp_path, field, value):
    directory = make_bundle(tmp_path)
    change_json(directory, "stage_receipt.json", field, value)
    with pytest.raises(
        ValueError, match=r"stage_receipt|generation|benchmark|digest|registry|image"
    ):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "stage",
        "status",
        "cuda_verified",
        "wandb_verified",
        "backend",
        "arm",
        "model_seed",
        "steps",
        "benchmark",
        "config_sha256",
        "source_sha256",
        "generation_manifest_sha256",
        "image",
        "zenml_run_id",
    ],
)
def test_required_stage_identity_cannot_be_omitted(tmp_path, field):
    directory = make_bundle(tmp_path)
    stage = read_json(directory / "stage_receipt.json")
    stage.pop(field)
    write_json(directory / "stage_receipt.json", stage)
    with pytest.raises(ValueError, match="stage_receipt"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "runtime",
    [
        {"python": "3.12"},
        {"cuda": {}},
        {"cuda": None},
        {"cuda": {"verified": False}},
        {"cuda": {"verified": 1}},
        {"cuda": {"verified": True}, "backend": "trl"},
    ],
)
def test_missing_or_inconsistent_gpu_runtime_metadata_is_rejected(tmp_path, runtime):
    directory = make_bundle(tmp_path)
    write_json(directory / "runtime.json", runtime)
    with pytest.raises(ValueError, match=r"runtime\.json"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "pending"),
        ("model_id", "fixture/other"),
        ("model_revision", "e" * 40),
        ("dataset_id", "fixture/other"),
        ("dataset_revision", None),
    ],
)
def test_prefetch_must_match_the_handoff_configuration(tmp_path, field, value):
    directory = make_bundle(tmp_path)
    change_json(directory, "prefetch.json", field, value)
    with pytest.raises(ValueError, match=r"prefetch\.json"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 1),
        ("schema", True),
        ("runner_returncode", 1),
        ("runner_returncode", False),
        ("cuda_verified", False),
        ("wandb_verified", False),
        ("wandb_verified", 1),
        ("zenml_run_id", "other"),
        ("source.source_sha256", "e" * 64),
        ("plan.profile", "sft-train-v1"),
        ("plan.arm", "reference"),
        ("plan.backend", "trl"),
        ("plan.seed", 1),
        ("plan.seed", False),
        ("plan.benchmark", "mbpp"),
        ("plan.config_sha256", "e" * 64),
        ("plan.source_sha256", "e" * 64),
        ("plan.image", IMAGE.replace("a" * 64, "e" * 64)),
        ("plan.checkpoint", None),
        ("plan.checkpoint.sha256", "invalid"),
        ("plan.checkpoint.uri", "https://secret:token@example.invalid/checkpoint"),
        ("plan.benchmark_input", {}),
        ("plan.benchmark_input.sha256", "e" * 64),
        ("plan.benchmark_input.uri", "gs://other-bucket/benchmark.jsonl"),
        (
            "plan.benchmark_input.uri",
            "gs://gke-dev-dws-jbr-zenml/benchmark?token=secret",
        ),
    ],
)
def test_deployment_identity_and_input_bindings_cannot_drift(tmp_path, field, value):
    directory = make_bundle(tmp_path)
    change_json(directory, "deployment.json", field, value)
    with pytest.raises(ValueError, match=r"deployment\.json"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize("status", ["pending", "offline", "completed"])
def test_generation_release_is_independent_of_upload_verification(tmp_path, status):
    directory = make_bundle(tmp_path)
    set_tracking(directory, status)
    for name in ("stage_receipt.json", "deployment.json"):
        change_json(directory, name, "wandb_verified", False)
    bundle.seal_bundle(directory, PROFILE)
    assert read_json(directory / "wandb_run.json")["status"] == "completed"


@pytest.mark.parametrize(
    "operation", ["init", "log_metrics", "log_progress", "complete", "finish"]
)
@pytest.mark.parametrize("url_kind", ["present", "null", "absent"])
def test_completed_generation_survives_tracking_outages(tmp_path, operation, url_kind):
    directory = make_bundle(tmp_path)
    set_tracking(directory, "failed", operation=operation)
    receipt = read_json(directory / "wandb_run.json")
    if url_kind == "null":
        receipt["url"] = None
    elif url_kind == "absent":
        receipt.pop("url")
    write_json(directory / "wandb_run.json", receipt)
    sealed = bundle.seal_bundle(directory, PROFILE)
    assert "tracking_failure.json" in sealed["files"]
    assert read_json(directory / "stage_receipt.json")["wandb_verified"] is False


@pytest.mark.parametrize("status", ["pending", "offline", "failed"])
def test_unfinished_upload_cannot_be_claimed_as_verified(tmp_path, status):
    directory = make_bundle(tmp_path)
    set_tracking(directory, status)
    for name in ("stage_receipt.json", "deployment.json"):
        change_json(directory, name, "wandb_verified", True)
    with pytest.raises(ValueError, match="wandb_verified requires a completed"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize("status", ["pending", "offline", "completed"])
def test_missing_url_requires_a_matching_failed_upload(tmp_path, status):
    directory = make_bundle(tmp_path)
    set_tracking(directory, status)
    change_json(directory, "wandb_run.json", "url", None)
    with pytest.raises(ValueError, match="url must be"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "failure",
    [
        {"operation": "code_metrics", "exception_type": "Failure"},
        {"operation": "complete", "exception_type": ""},
        {"operation": "complete", "exception_type": "x" * 129},
        {"operation": "complete", "exception_type": "Failure", "message": "secret"},
    ],
)
def test_tracking_failure_is_bounded_and_uses_standard_operations(tmp_path, failure):
    directory = make_bundle(tmp_path)
    set_tracking(directory, "failed")
    write_json(directory / "tracking_failure.json", failure)
    with pytest.raises(ValueError, match=r"tracking_failure\.json"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "change", ["missing-failure", "wrong-status", "uncompleted-generation", "wrong-url"]
)
def test_tracking_status_and_identity_remain_strict(tmp_path, change):
    directory = make_bundle(tmp_path)
    set_tracking(directory, "failed")
    if change == "missing-failure":
        (directory / "tracking_failure.json").unlink()
    elif change == "wrong-status":
        change_json(directory, "wandb_run.json", "upload_status", "completed")
    elif change == "uncompleted-generation":
        change_json(directory, "wandb_run.json", "status", "failed")
    else:
        change_json(
            directory,
            "wandb_run.json",
            "url",
            "https://wandb.invalid/other/public/runs/fixture",
        )
    with pytest.raises(ValueError, match=r"wandb_run\.json"):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "name",
    [
        "trainable.safetensors",
        "optimizer.pt",
        "private_rng.bin",
        "credentials.json",
        "test-records.jsonl",
        "answers/trainable.safetensors",
        "answers/optimizer.pt",
        "answers/private_rng.json",
        "answers/.netrc",
        "answers/code_metrics.json",
    ],
)
def test_only_root_receipts_and_the_strict_handoff_inventory_are_allowed(
    tmp_path, name
):
    directory = make_bundle(tmp_path)
    write_json(directory / name, {"fixture": True})
    with pytest.raises(
        ValueError, match=r"Unexpected GPU output|unexpected generation artifact"
    ):
        bundle.seal_bundle(directory, PROFILE)


@pytest.mark.parametrize(
    "name", ["trainable", "wandb", ".cache", "answers/nested", "answers/trainable"]
)
def test_generation_does_not_allow_additional_directories(tmp_path, name):
    directory = make_bundle(tmp_path)
    (directory / name).mkdir()
    with pytest.raises(ValueError, match="Unexpected output directory"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


@pytest.mark.parametrize(
    "name",
    [
        "runtime.json",
        "answers/generation-manifest.json",
        "answers/humaneval-samples.jsonl",
        "answers",
    ],
)
def test_generation_rejects_symlinks(tmp_path, name):
    directory = make_bundle(tmp_path)
    path = directory / name
    external = tmp_path / "link-target"
    path.rename(external)
    path.symlink_to(external, target_is_directory=external.is_dir())
    with pytest.raises(ValueError, match="Symlink"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


def test_generation_rejects_symlinked_root(tmp_path):
    directory = make_bundle(tmp_path)
    link = tmp_path / "linked-root"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="regular output directory"):
        bundle.validate_bundle(link, PROFILE, require_manifest=False)


@pytest.mark.parametrize("name", ["runtime.json", "answers/humaneval.jsonl"])
def test_generation_rejects_hardlinks(tmp_path, name):
    directory = make_bundle(tmp_path)
    os.link(directory / name, tmp_path / "hardlink")
    with pytest.raises(ValueError, match="Hard link"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


def test_generation_rejects_nonregular_files_before_reading(tmp_path):
    directory = make_bundle(tmp_path)
    path = directory / "answers/humaneval.jsonl"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="Non-regular output"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


def test_handoff_metadata_above_one_mib_is_supported(tmp_path):
    directory = make_bundle(
        tmp_path, metadata_padding=1024**2, generation_padding=1024**2
    )
    for name in (
        "generation-manifest.json",
        "humaneval-generation.json",
        "checkpoint-summary.json",
    ):
        assert (directory / "answers" / name).stat().st_size > 1024**2
    expected = bundle.seal_bundle(directory, PROFILE)
    assert bundle.validate_bundle(directory, PROFILE) == expected


def test_benchmark_jsonl_does_not_inherit_sft_file_or_line_caps(tmp_path):
    directory = make_bundle(tmp_path, benchmark_padding=16 * 1024**2)
    assert (directory / "answers/humaneval.jsonl").stat().st_size > 16 * 1024**2
    expected = bundle.seal_bundle(directory, PROFILE)
    assert bundle.validate_bundle(directory, PROFILE) == expected


@pytest.mark.parametrize(
    ("name", "limit"),
    [
        ("runtime.json", 1024**2),
        ("stage_receipt.json", 1024**2),
        ("answers/generation-manifest.json", 16 * 1024**2),
        ("answers/checkpoint-summary.json", 16 * 1024**2),
        ("answers/humaneval.jsonl", 128 * 1024**2),
        ("answers/humaneval-samples.jsonl", 128 * 1024**2),
    ],
)
def test_generation_file_bounds_apply_before_parsing(tmp_path, name, limit):
    directory = make_bundle(tmp_path)
    with (directory / name).open("wb") as stream:
        stream.truncate(limit + 1)
    with pytest.raises(ValueError, match="Oversized JSON metadata"):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


@pytest.mark.parametrize("limit", ["entries", "bytes"])
def test_generation_inventory_is_bounded_before_hashing(tmp_path, monkeypatch, limit):
    directory = make_bundle(tmp_path)
    for index in range(257 if limit == "entries" else 9):
        with (directory / "answers" / f"extra-{index}.jsonl").open("wb") as stream:
            stream.truncate(1 if limit == "entries" else 128 * 1024**2)

    def unexpected_hash(path):
        raise AssertionError("Reject an oversized inventory before hashing payloads")

    monkeypatch.setattr(bundle, "sha256", unexpected_hash)
    with pytest.raises(
        ValueError, match="entry cap" if limit == "entries" else "byte cap"
    ):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)


@pytest.mark.parametrize("limit", ["entries", "bytes"])
def test_generation_seal_reserves_space_for_the_directory_and_manifest(
    tmp_path, monkeypatch, limit
):
    directory = make_bundle(tmp_path)
    inventory = bundle.validate_bundle(directory, PROFILE, require_manifest=False)
    if limit == "entries":
        monkeypatch.setattr(bundle, "BUNDLE_ENTRY_LIMIT", len(inventory["files"]) + 1)
    else:
        monkeypatch.setitem(bundle._GPU_LIMITS, PROFILE, inventory["bytes"])
    with pytest.raises(ValueError, match=r"including its manifest.*cap"):
        bundle.seal_bundle(directory, PROFILE)
    assert not (directory / MANIFEST).exists()


@pytest.mark.parametrize(
    "name",
    [
        "runtime.json",
        "answers/generation-manifest.json",
        "answers/checkpoint-summary.json",
    ],
)
@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b'{"value":NaN}', "Nonfinite"),
        (b'{"value":1e999}', "Nonfinite"),
        (b'{"value":1,"value":2}', "Duplicate"),
        (b'{"value":' + b"[" * 33 + b"0" + b"]" * 33 + b"}", "nesting depth"),
    ],
)
def test_generation_metadata_remains_finite_unambiguous_and_depth_bounded(
    tmp_path, name, raw, match
):
    directory = make_bundle(tmp_path)
    (directory / name).write_bytes(raw)
    with pytest.raises(ValueError, match=match):
        bundle.validate_bundle(directory, PROFILE, require_manifest=False)
