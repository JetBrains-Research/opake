"""Portable generation and isolated scoring; fixtures are not live benchmark scores."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.moe_privacy import (
    campaign,
    code_eval,
    handoff,
    sft_adapters,
    sft_run,
    tracking,
)
from transformers import AutoTokenizer


@pytest.fixture
def benchmark_dir(tmp_path):
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for name, spec in code_eval.BENCHMARKS.items():
        (directory / f"{name}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": f"{spec['prefix']}/{i}",
                        "prompt": f"def solve_{i}(x):\n",
                        "canonical_solution": "    raise RuntimeError('never execute on host')\n",
                        "entry_point": f"solve_{i}",
                        "contract": "",
                        "base_input": [[1]],
                        "plus_input": [[2]],
                        "atol": 0,
                    }
                )
                + "\n"
                for i in reversed(range(spec["count"]))
            )
        )
    return directory


@pytest.fixture
def checkpoint(tmp_path):
    config = asdict(sft_run.SFTExperimentConfig())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    directory = tmp_path / "training"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment": "pretrained_moe_sft",
                "arm": "reference",
                "model_seed": 0,
                "config": config,
            }
        )
    )
    (directory / "wandb_run.json").write_text(
        json.dumps({"status": "completed", "run_id": "training"})
    )
    export = directory / "trainable"
    export.mkdir()
    (export / "trainable.safetensors").write_bytes(b"inert test export; never loaded")
    (export / "adapter_spec.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "adapter_kind": "static_parametrization_expert_lora",
                "config": {"rank": 4, "alpha": 8, "target_parameters": ["expert"]},
                "metadata": {"router_count": 1},
                "tensors": {"expert": {"dtype": "torch.float32", "shape": [1]}},
            }
        )
    )
    return config_path, directory


def test_combined_generation_failure_never_resumes_training(
    tmp_path, checkpoint, benchmark_dir, monkeypatch
):
    config_path, directory = checkpoint
    original = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    sessions = []

    class Tracker:
        run = None

        def __init__(self, output_dir, **kwargs):
            sessions.append((output_dir, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def start(self):
            pass

    def fail_generation(*args, **kwargs):
        raise RuntimeError("generation fixture failure")

    monkeypatch.setattr(tracking, "ExperimentTracker", Tracker)
    monkeypatch.setattr(sft_run, "load_model", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(sft_adapters, "load_trainable", lambda *args: None)
    monkeypatch.setattr(code_eval, "merge_for_evaluation", lambda model: model)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(code_eval, "generate_samples", fail_generation)
    args = SimpleNamespace(
        config=config_path,
        output_dir=directory,
        arm="reference",
        seed=0,
        benchmarks=["humaneval"],
        benchmark_dir=benchmark_dir,
        max_new_tokens=512,
        smoke_limit=None,
        wandb_mode="online",
        group="test",
        image="sha256:" + "a" * 64,
        docker_sudo=False,
    )
    with pytest.raises(RuntimeError, match="generation fixture failure"):
        campaign.evaluate_checkpoint(args)
    assert all(path.read_bytes() == content for path, content in original.items())
    assert sessions
    assert all(not options.get("resume", False) for _, options in sessions)
    assert all(path != directory for path, _ in sessions)


IMAGE_ID = "sha256:" + "a" * 64


def write_json(path, value):
    path.write_bytes(handoff.json_bytes(value))


def snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def generation_args(tmp_path, checkpoint, benchmark_dir):
    config_path, checkpoint_dir = checkpoint
    return SimpleNamespace(
        config=config_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "generation",
        arm="reference",
        seed=0,
        benchmarks=["humaneval"],
        benchmark_dir=benchmark_dir,
        smoke_limit=2,
        max_new_tokens=512,
        wandb_mode="off",
    )


@pytest.fixture
def mock_generation(monkeypatch, checkpoint):
    _, checkpoint_dir = checkpoint
    spec = handoff.read_json(checkpoint_dir / "trainable" / "adapter_spec.json")
    calls = SimpleNamespace(load=[], restore=[], generate=[], tokenizers=[])
    model, tokenizer = object(), object()

    def load(config, seed):
        calls.load.append((config, seed))
        return model, {}

    def restore(actual, directory):
        assert actual is model
        calls.restore.append(directory)

    def load_tokenizer(*args, **kwargs):
        calls.tokenizers.append((args, kwargs))
        return tokenizer

    def generate(actual_model, actual_tokenizer, **kwargs):
        assert actual_model is model
        assert actual_tokenizer is tokenizer
        calls.generate.append(kwargs)
        tasks, metadata = code_eval.load_benchmark(
            kwargs["dataset"], kwargs["dataset_path"]
        )
        ids = list(tasks)[: kwargs["smoke_limit"]]
        rows, records = [], []
        for task_id in ids:
            prompt = tasks[task_id]["prompt"]
            solution = (
                prompt
                + "    raise RuntimeError('never execute generated code on host')\n"
            )
            rows.append({"task_id": task_id, "solution": solution})
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
        raw = b"".join(handoff.json_bytes(row) for row in rows)
        manifest = {
            "format_version": code_eval.GENERATION_FORMAT_VERSION,
            "evalplus_version": code_eval.EVALPLUS_VERSION,
            "mode": "full" if kwargs["smoke_limit"] is None else "smoke",
            "smoke_limit": kwargs["smoke_limit"],
            "benchmark": metadata,
            "model": {
                "id": kwargs["model_id"],
                "revision": kwargs["model_revision"],
                "arm": kwargs["arm"],
                "adapter_config": spec["config"],
                "router_count": 1,
                "merged_for_evaluation": True,
            },
            "generation": {
                **code_eval.generation_settings(kwargs["max_new_tokens"]),
                "eos_token_ids": [0],
            },
            "task_ids": ids,
            "tasks": records,
            "samples_sha256": handoff.sha256(raw),
            "raw_samples_sha256": handoff.sha256(raw),
        }
        output = kwargs["output_dir"]
        output.mkdir()
        (output / "samples.jsonl").write_bytes(raw)
        (output / "raw_samples.jsonl").write_bytes(raw)
        write_json(output / "generation.json", manifest)
        return manifest

    monkeypatch.setattr(sft_run, "load_model", load)
    monkeypatch.setattr(sft_adapters, "load_trainable", restore)
    monkeypatch.setattr(code_eval, "merge_for_evaluation", lambda model: model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(code_eval, "generate_samples", generate)
    return calls


@pytest.fixture
def generated(generation_args, mock_generation):
    manifest = campaign.generate_checkpoint(generation_args)
    return generation_args.output_dir, manifest


def score_args(generated, output_dir):
    directory, manifest = generated
    return SimpleNamespace(
        samples_dir=directory,
        output_dir=output_dir,
        eval_image=IMAGE_ID,
        expected_manifest_sha256=manifest["manifest_sha256"],
        wandb_mode="off",
    )


@pytest.fixture
def docker(monkeypatch):
    """Fake only Docker's process transport; exercise the real scoring boundary."""
    state = SimpleNamespace(commands=[], fail=None, corrupt_results=False)

    def run(command, **kwargs):
        state.commands.append(command)
        assert not kwargs.get("shell")
        args = command[3:] if command[:3] == ["sudo", "-n", "docker"] else command[1:]
        if args[:2] == ["image", "inspect"]:
            if state.fail == "inspect":
                raise FileNotFoundError("fixture Docker unavailable")
            info = {
                "Id": IMAGE_ID,
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {
                    "Labels": code_eval.IMAGE_LABELS,
                    "Entrypoint": code_eval.IMAGE_ENTRYPOINT,
                },
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(info), "")
        if args[0] == "version":
            info = {
                "Os": "linux",
                "Arch": "arm64",
                "Version": "fixture",
                "Platform": {"Name": "fixture engine"},
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(info), "")
        if args[:2] == ["rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        assert args[0] == "run", (
            "Only Docker inspection, execution and cleanup are permitted"
        )
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--cap-drop") + 1] == "ALL"
        assert args[args.index("--user") + 1] == "65534:65534"
        assert args[args.index("--pull") + 1] == "never"
        assert "--read-only" in args
        if state.fail == "timeout":
            raise subprocess.TimeoutExpired(command, 3600)
        if state.fail in ("exit", args[-2]):
            raise subprocess.CalledProcessError(1, command)
        mounts = {}
        for i, arg in enumerate(args):
            if arg == "--mount":
                fields = dict(
                    field.split("=", 1)
                    for field in args[i + 1].split(",")
                    if "=" in field
                )
                mounts[fields["target"]] = Path(fields["source"])
        output = mounts["/output"]
        samples = [
            json.loads(line)
            for line in mounts["/input/samples.jsonl"].read_bytes().splitlines()
        ]
        dataset = next(
            path
            for target, path in mounts.items()
            if target not in ("/output", "/input/samples.jsonl")
        )
        raw = dataset.read_bytes()
        ids = [row["task_id"] for row in samples]
        effective = raw
        if args[-1] == "smoke":
            effective = (
                b"\n".join(
                    line
                    for line in raw.splitlines()
                    if json.loads(line)["task_id"] in ids
                )
                + b"\n"
            )
        digest = hashlib.md5(effective, usedforsecurity=False).hexdigest()
        write_json(
            output / "execution.json",
            {
                "evalplus_version": code_eval.EVALPLUS_VERSION,
                "source_sha256": handoff.sha256(raw),
                "dataset_hash": digest,
                "mode": args[-1],
                "task_ids": ids,
                "syntax": dict.fromkeys(ids, True),
            },
        )
        outcomes = {
            row["task_id"]: [{**row, "base_status": "pass", "plus_status": "fail"}]
            for row in samples
        }
        if state.corrupt_results:
            outcomes.pop(ids[-1])
        write_json(
            output / "samples_eval_results.json", {"hash": digest, "eval": outcomes}
        )
        kwargs["stdout"].write(
            b"Synthetic Docker transport fixture, not a live evaluation.\n"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(code_eval.subprocess, "run", run)
    return state


def test_generation_needs_no_image_or_docker_and_preserves_common_loading(
    generation_args, mock_generation, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Generation must never invoke Docker or training")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(code_eval, "evaluate_samples", forbidden)
    monkeypatch.setattr(sft_run, "_run_experiment", forbidden)
    before = snapshot(generation_args.checkpoint_dir)
    result = campaign.generate_checkpoint(generation_args)
    assert result == handoff.validate_generation(
        generation_args.output_dir, result["manifest_sha256"]
    )
    assert all(path.is_file() for path in generation_args.output_dir.iterdir())
    assert result["checkpoint"]["config_sha256"] == handoff.sha256(
        handoff.json_bytes(asdict(mock_generation.load[0][0]))
    )
    assert result["checkpoint"]["files"]["summary.json"] == handoff.file_record(
        generation_args.checkpoint_dir / "summary.json"
    )
    assert mock_generation.restore == [generation_args.checkpoint_dir / "trainable"]
    assert mock_generation.load[0][1] == 0
    assert mock_generation.tokenizers[0][1] == {
        "revision": result["checkpoint"]["model"]["revision"],
        "local_files_only": True,
    }
    assert snapshot(generation_args.checkpoint_dir) == before


def test_manifest_is_deterministic_for_identical_saved_payloads(
    generation_args, mock_generation
):
    first = campaign.generate_checkpoint(generation_args)
    original = snapshot(generation_args.output_dir)
    generation_args.output_dir = generation_args.output_dir.with_name(
        "generation-second"
    )
    second = campaign.generate_checkpoint(generation_args)
    assert first == second
    assert original == snapshot(generation_args.output_dir)


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_full_generation_binds_complete_expected_task_coverage(
    generation_args, mock_generation, benchmark
):
    generation_args.benchmarks = [benchmark]
    generation_args.smoke_limit = None
    result = campaign.generate_checkpoint(generation_args)["benchmarks"][benchmark]
    assert result["mode"] == "full"
    assert len(result["tasks"]) == code_eval.BENCHMARKS[benchmark]["count"]
    assert result["task_ids"] == result["expected_task_ids"]


@pytest.mark.parametrize(
    "kind", ["corrupt", "missing", "duplicate", "partial", "reordered"]
)
def test_rejects_hash_corruption_and_invalid_sample_coverage(generated, kind):
    directory, _ = generated
    path = directory / "humaneval-samples.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if kind == "corrupt":
        rows[0]["solution"] += "modified"
    elif kind == "missing":
        rows[0].pop("task_id")
    elif kind == "duplicate":
        rows[1]["task_id"] = rows[0]["task_id"]
    elif kind == "partial":
        rows.pop()
    else:
        rows.reverse()
    path.write_bytes(b"".join(handoff.json_bytes(row) for row in rows))
    if kind != "corrupt":
        metadata = handoff.read_json(directory / "humaneval-generation.json")
        metadata["samples_sha256"] = handoff.sha256(path.read_bytes())
        write_json(directory / "humaneval-generation.json", metadata)
    with pytest.raises(ValueError, match=r"hash|task|samples|Samples|Generation"):
        handoff.validate_generation(directory)


@pytest.mark.parametrize(
    "kind", ["decoding", "checkpoint", "backend", "seed", "coverage", "override"]
)
def test_manifest_metadata_is_reconstructed_not_trusted(generated, kind):
    directory, _ = generated
    path = directory / handoff.MANIFEST
    manifest = handoff.read_json(path)
    if kind == "decoding":
        manifest["benchmarks"]["humaneval"]["generation"]["max_new_tokens"] = 128
    elif kind == "checkpoint":
        manifest["checkpoint"]["model"]["revision"] = "b" * 40
    elif kind == "backend":
        manifest["checkpoint"]["backend"] = "native"
    elif kind == "seed":
        manifest["checkpoint"]["seed"] = 1
    elif kind == "coverage":
        manifest["benchmarks"]["humaneval"]["expected_task_ids"].pop()
    else:
        manifest["metadata_override"] = {"status": "completed"}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="saved payloads and receipts"):
        handoff.validate_generation(directory)


@pytest.mark.parametrize(
    "kind", ["decoding", "model", "summary", "config", "extra_field"]
)
def test_payload_metadata_must_match_independent_checkpoint_and_decoding_receipts(
    generated, kind
):
    directory, _ = generated
    filename = {"summary": "checkpoint-summary.json", "config": "config.json"}.get(
        kind, "humaneval-generation.json"
    )
    path = directory / filename
    value = handoff.read_json(path)
    if kind == "decoding":
        value["generation"]["max_new_tokens"] = 256
    elif kind == "model":
        value["model"]["revision"] = "b" * 40
    elif kind == "summary":
        value["arm"] = "dp"
    elif kind == "config":
        value["steps"] += 1
    else:
        value["expected_task_ids"] = []
    write_json(path, value)
    # Updating a file checksum alone must not legitimize changed critical fields.
    manifest = handoff.read_json(directory / handoff.MANIFEST)
    manifest["files"][filename] = handoff.file_record(path)
    write_json(directory / handoff.MANIFEST, manifest)
    with pytest.raises(
        ValueError, match=r"checkpoint|Checkpoint|decoding|metadata|config"
    ):
        handoff.validate_generation(directory)


@pytest.mark.parametrize("kind", ["symlink", "directory", "extra", "missing"])
def test_handoff_is_flat_exact_and_does_not_follow_links(generated, tmp_path, kind):
    directory, _ = generated
    if kind == "symlink":
        (directory / "unexpected").symlink_to(tmp_path / "training" / "summary.json")
    elif kind == "directory":
        (directory / "unexpected").mkdir()
    elif kind == "extra":
        (directory / "unexpected").write_text("metadata override")
    else:
        (directory / "humaneval-raw_samples.jsonl").unlink()
    with pytest.raises(ValueError, match=r"flat regular files|missing|unexpected"):
        handoff.validate_generation(directory)


def test_generation_size_limit_includes_manifest_and_all_payloads(
    generated, monkeypatch
):
    directory, _ = generated
    total = sum(path.stat().st_size for path in directory.iterdir())
    monkeypatch.setattr(handoff, "MAX_GENERATION_BYTES", total - 1)
    with pytest.raises(ValueError, match="1 GiB"):
        handoff.validate_generation(directory)
    monkeypatch.setattr(handoff, "MAX_GENERATION_BYTES", total)
    handoff.validate_generation(directory)


@pytest.mark.parametrize("expected", [None, "", "bad", "f" * 64])
def test_scoring_requires_trusted_root_before_docker_or_output(
    generated, tmp_path, docker, expected
):
    args = score_args(generated, tmp_path / "score")
    args.expected_manifest_sha256 = expected
    with pytest.raises(ValueError, match=r"manifest|SHA256"):
        campaign.score_checkpoint(args)
    assert not docker.commands
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "image",
    [
        "trusted:latest",
        "--privileged",
        "sha256:bad",
        "repo@sha256:" + "a" * 64 + ";touch /tmp/no",
    ],
)
def test_scoring_rejects_unpinned_or_injected_images(
    generated, tmp_path, docker, image
):
    args = score_args(generated, tmp_path / "score")
    args.eval_image = image
    with pytest.raises(ValueError, match="pinned image"):
        campaign.score_checkpoint(args)
    assert not docker.commands
    assert not args.output_dir.exists()


def test_scoring_uses_unchanged_sandbox_and_preserves_every_historical_attempt(
    generated, tmp_path, docker, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Scoring must never load a model or regenerate")

    monkeypatch.setattr(sft_run, "load_model", forbidden)
    monkeypatch.setattr(code_eval, "generate_samples", forbidden)
    before = snapshot(generated[0])
    args = score_args(generated, tmp_path / "score-first")
    metrics = campaign.score_checkpoint(args)
    receipt = handoff.read_json(args.output_dir / "scoring_receipt.json")
    assert receipt["status"] == "completed"
    assert receipt["generation_manifest_sha256"] == generated[1]["manifest_sha256"]
    assert receipt["evaluator"]["image_id"] == IMAGE_ID
    assert receipt["evaluator"]["image_platform"] == {
        "os": "linux",
        "architecture": "amd64",
    }
    assert receipt["evaluator"]["docker_engine"]["architecture"] == "arm64"
    assert receipt["evaluator"]["host"]["os"]
    assert receipt["evaluator"]["timing_parity"]
    assert receipt["evaluator"]["limits"] == asdict(code_eval.SandboxLimits())
    assert metrics["humaneval"]["smoke"]
    assert metrics["humaneval"]["task_count"] == 2
    assert metrics == handoff.read_json(args.output_dir / "code_metrics.json")
    result = handoff.read_json(args.output_dir / "humaneval-execution" / "summary.json")
    assert metrics["humaneval"] == campaign.normalize_code_result(result)
    for filename, record in receipt["results"]["humaneval"]["files"].items():
        assert record == handoff.file_record(args.output_dir / filename)
    historical = snapshot(args.output_dir)
    assert (
        campaign.score_checkpoint(score_args(generated, tmp_path / "score-second"))
        == metrics
    )
    assert snapshot(args.output_dir) == historical
    assert snapshot(generated[0]) == before
    commands = len(docker.commands)
    with pytest.raises(FileExistsError):
        campaign.score_checkpoint(args)
    assert len(docker.commands) == commands
    assert snapshot(args.output_dir) == historical


@pytest.mark.parametrize("failure", ["inspect", "timeout", "exit", "results"])
def test_scoring_failure_isolated_from_training_generation_and_future_attempts(
    generated, checkpoint, tmp_path, docker, failure
):
    before_training = snapshot(checkpoint[1])
    before_generation = snapshot(generated[0])
    docker.fail = failure
    docker.corrupt_results = failure == "results"
    args = score_args(generated, tmp_path / "failed-score")
    with pytest.raises((RuntimeError, ValueError)):
        campaign.score_checkpoint(args)
    receipt = handoff.read_json(args.output_dir / "scoring_receipt.json")
    assert receipt["status"] == "failed"
    assert receipt["generation_manifest_sha256"] == generated[1]["manifest_sha256"]
    assert not (args.output_dir / "code_metrics.json").exists()
    failed_attempt = snapshot(args.output_dir)
    docker.fail, docker.corrupt_results = None, False
    campaign.score_checkpoint(score_args(generated, tmp_path / "retried-score"))
    assert snapshot(args.output_dir) == failed_attempt
    assert snapshot(checkpoint[1]) == before_training
    assert snapshot(generated[0]) == before_generation


def test_no_scoring_outputs_inside_generation(generated, docker):
    before = snapshot(generated[0])
    with pytest.raises(ValueError, match="generation directory"):
        campaign.score_checkpoint(score_args(generated, generated[0] / "score"))
    assert not docker.commands
    assert snapshot(generated[0]) == before


def test_explicit_smoke_remains_smoke_even_at_full_task_count(
    generation_args, mock_generation, tmp_path, docker
):
    generation_args.smoke_limit = 164
    manifest = campaign.generate_checkpoint(generation_args)
    metrics = campaign.score_checkpoint(
        score_args((generation_args.output_dir, manifest), tmp_path / "score")
    )
    assert metrics["humaneval"]["task_count"] == 164
    assert metrics["humaneval"]["smoke"]


def combined_args(generation_args):
    return SimpleNamespace(
        **{**vars(generation_args), "output_dir": generation_args.checkpoint_dir},
        image="trusted:local",
        docker_sudo=False,
    )


def test_combined_compatibility_and_append_preserve_previous_outputs(
    generation_args, mock_generation, docker
):
    args = combined_args(generation_args)
    training = snapshot(args.output_dir)
    first = campaign.evaluate_checkpoint(args)
    historical = snapshot(args.output_dir)
    args.benchmarks, args.append_benchmark = ["mbpp"], True
    combined = campaign.evaluate_checkpoint(args)
    assert combined["humaneval"] == first["humaneval"]
    assert set(combined) == {"humaneval", "mbpp"}
    assert combined == handoff.read_json(args.output_dir / "code_metrics.json")
    after = snapshot(args.output_dir)
    assert all(
        after[name] == raw
        for name, raw in historical.items()
        if name != "code_metrics.json"
    )
    assert all(after[name] == raw for name, raw in training.items())
    assert len(mock_generation.generate) == 2
    with pytest.raises(ValueError, match="only new benchmarks"):
        campaign.evaluate_checkpoint(args)


def test_combined_partial_failure_retains_completed_benchmark_and_all_samples(
    generation_args, mock_generation, docker, tmp_path
):
    args = combined_args(generation_args)
    args.benchmarks = ["humaneval", "mbpp"]
    training = snapshot(args.output_dir)
    docker.fail = "mbpp"
    with pytest.raises(RuntimeError):
        campaign.evaluate_checkpoint(args)
    assert set(handoff.read_json(args.output_dir / "code_metrics.json")) == {
        "humaneval"
    }
    assert (args.output_dir / "humaneval-execution" / "per_task.json").is_file()
    history = snapshot(args.output_dir)
    generation = args.output_dir / "generation-humaneval-mbpp"
    manifest = handoff.validate_generation(generation)
    docker.fail = None
    results = campaign.score_checkpoint(
        score_args((generation, manifest), tmp_path / "retry")
    )
    assert set(results) == {"humaneval", "mbpp"}
    assert len(mock_generation.generate) == 2
    assert snapshot(args.output_dir) == history
    assert all(history[name] == raw for name, raw in training.items())


def test_failed_append_keeps_prior_metrics_and_changed_export_cannot_append(
    generation_args, mock_generation, docker
):
    args = combined_args(generation_args)
    campaign.evaluate_checkpoint(args)
    first = snapshot(args.output_dir)
    args.benchmarks, args.append_benchmark = ["mbpp"], True
    weights = args.output_dir / "trainable" / "trainable.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="same completed checkpoint"):
        campaign.evaluate_checkpoint(args)
    weights.write_bytes(original)
    docker.fail = "mbpp"
    with pytest.raises(RuntimeError):
        campaign.evaluate_checkpoint(args)
    after = snapshot(args.output_dir)
    assert all(after[name] == raw for name, raw in first.items())


def test_cli_new_commands_and_legacy_evaluate_keep_distinct_required_inputs(
    tmp_path, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        campaign,
        "generate_checkpoint",
        lambda args: calls.append(args) or {"manifest_sha256": "a" * 64},
    )
    campaign.main(
        [
            "generate",
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint-dir",
            str(tmp_path / "train"),
            "--output-dir",
            str(tmp_path / "generation"),
            "--arm",
            "trl_reference",
            "--seed",
            "0",
            "--benchmark-dir",
            str(tmp_path / "benchmarks"),
            "--benchmarks",
            "humaneval",
            "--wandb-mode",
            "off",
        ]
    )
    assert calls[-1].checkpoint_dir == tmp_path / "train"
    assert not hasattr(calls[-1], "image")
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == "a" * 64
    monkeypatch.setattr(
        campaign, "score_checkpoint", lambda args: calls.append(args) or {}
    )
    campaign.main(
        [
            "score",
            "--samples-dir",
            str(tmp_path / "generation"),
            "--output-dir",
            str(tmp_path / "score"),
            "--eval-image",
            IMAGE_ID,
            "--expected-manifest-sha256",
            "a" * 64,
        ]
    )
    assert calls[-1].expected_manifest_sha256 == "a" * 64
    assert not hasattr(calls[-1], "config")
    monkeypatch.setattr(
        campaign, "evaluate_checkpoint", lambda args: calls.append(args) or {}
    )
    campaign.main(
        [
            "evaluate",
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(tmp_path / "train"),
            "--image",
            "trusted:local",
            "--benchmark-dir",
            str(tmp_path / "benchmarks"),
            "--group",
            "test",
            "--arm",
            "reference",
        ]
    )
    assert calls[-1].image == "trusted:local"


def test_score_cli_and_handoff_validation_import_without_ml_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "examples.moe_privacy.campaign",
            "score",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "--expected-manifest-sha256" in result.stdout


@pytest.mark.parametrize("arm", ["trl_reference", "trl_reference_aux"])
def test_native_export_uses_real_common_inference_path_without_training(
    generation_args, tmp_path, monkeypatch, arm
):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import MellumConfig, MellumForCausalLM, PreTrainedTokenizerFast

    def tiny_model():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            config = MellumConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                head_dim=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=128,
                num_experts=2,
                num_experts_per_tok=1,
                moe_intermediate_size=8,
                pad_token_id=1,
                bos_token_id=2,
                eos_token_id=3,
            )
            config._attn_implementation = "eager"
            return MellumForCausalLM(config)

    native = tmp_path / "native"
    native.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        trained, _ = sft_adapters.configure_expert_lora(tiny_model())
        sft_adapters.export_trainable(trained, native / "trainable")
    write_json(
        native / "summary.json",
        {
            "status": "completed",
            "experiment": "native_trl_moe_sft",
            "arm": arm,
            "model_seed": 0,
            "config": json.loads(generation_args.config.read_text()),
            "execution": {"trainer_backend": "trl.SFTTrainer"},
        },
    )
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    loads = []

    def pretrained(model_id, **kwargs):
        loads.append((model_id, kwargs))
        return tiny_model()

    def no_training(*args, **kwargs):
        pytest.fail("Native generation must not construct or run Opaque training")

    monkeypatch.setattr(sft_run.AutoModelForCausalLM, "from_pretrained", pretrained)
    monkeypatch.setattr(sft_run, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(sft_run.DPTrainer, "__init__", no_training)
    monkeypatch.setattr(sft_run, "_run_experiment", no_training)
    monkeypatch.setattr(code_eval, "evaluate_samples", no_training)
    generation_args.arm = arm
    generation_args.checkpoint_dir = native
    generation_args.max_new_tokens = 4
    before = snapshot(native)
    manifest = campaign.generate_checkpoint(generation_args)
    assert manifest["checkpoint"]["backend"] == "trl.SFTTrainer"
    assert manifest["benchmarks"]["humaneval"]["task_ids"] == [
        "HumanEval/0",
        "HumanEval/1",
    ]
    assert len(loads) == 1
    assert loads[0][1] == {
        "revision": manifest["checkpoint"]["model"]["revision"],
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "trust_remote_code": False,
    }
    assert handoff.validate_generation(generation_args.output_dir) == manifest
    assert snapshot(native) == before


@pytest.fixture
def wandb_sdk(monkeypatch):
    state = SimpleNamespace(runs=[], inits=[], outage=None)

    class Run:
        def __init__(self, options):
            self.id = options["id"]
            self.url = "https://example.invalid/runs/" + self.id
            self.config = options["config"]
            self.summary = {}
            self.logs = []
            self.exit_code = None

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, row):
            if state.outage == "log":
                raise TimeoutError("fixture communication outage")
            self.logs.append(row)

        def finish(self, *, exit_code):
            self.exit_code = exit_code
            if state.outage == "finish":
                raise TimeoutError("fixture communication outage")

    def init(**kwargs):
        state.inits.append(kwargs)
        if state.outage == "init":
            raise TimeoutError("fixture communication outage")
        run = Run(kwargs)
        state.runs.append(run)
        return run

    sdk = SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        errors=SimpleNamespace(CommError=ConnectionError),
        init=init,
    )
    original = tracking.importlib.import_module
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            sdk if name == "wandb" else original(name, *args, **kwargs)
        ),
    )
    return state


def test_online_stages_never_resume_training_or_publish_raw_training_losses(
    generation_args, mock_generation, wandb_sdk, docker, tmp_path
):
    summary_path = generation_args.checkpoint_dir / "summary.json"
    summary = handoff.read_json(summary_path)
    summary["raw_training_loss"] = 12345.0
    write_json(summary_path, summary)
    before = snapshot(generation_args.checkpoint_dir)
    generation_args.wandb_mode = "online"
    manifest = campaign.generate_checkpoint(generation_args)
    args = score_args((generation_args.output_dir, manifest), tmp_path / "score")
    args.wandb_mode = "online"
    docker.fail = "timeout"
    with pytest.raises(RuntimeError):
        campaign.score_checkpoint(args)
    assert [options["job_type"] for options in wandb_sdk.inits] == [
        "moe_code_generation",
        "moe_code_scoring",
    ]
    assert all(options["resume"] is None for options in wandb_sdk.inits)
    assert len({run.id for run in wandb_sdk.runs}) == 2
    assert [run.exit_code for run in wandb_sdk.runs] == [0, 1]
    assert [run.summary["status"] for run in wandb_sdk.runs] == ["completed", "failed"]
    for run in wandb_sdk.runs:
        assert "raw_training_loss" not in json.dumps(
            [run.config, run.summary, run.logs]
        )
    assert snapshot(generation_args.checkpoint_dir) == before
    assert all(path.is_file() for path in generation_args.output_dir.iterdir())
    assert (
        generation_args.output_dir.with_name("generation.tracking") / "wandb_run.json"
    ).is_file()


@pytest.mark.parametrize("outage", ["init", "log", "finish"])
def test_stage_tracking_fail_open_keeps_completed_scoring_artifacts(
    generated, wandb_sdk, docker, tmp_path, outage
):
    args = score_args(generated, tmp_path / "score")
    args.wandb_mode = "online"
    args.wandb_fail_open = True
    wandb_sdk.outage = outage
    result = campaign.score_checkpoint(args)
    assert result == handoff.read_json(args.output_dir / "code_metrics.json")
    assert (
        handoff.read_json(args.output_dir / "scoring_receipt.json")["status"]
        == "completed"
    )
    tracking_dir = args.output_dir.with_name("score.tracking")
    receipt = handoff.read_json(tracking_dir / "wandb_run.json")
    assert receipt["status"] == "completed"
    assert receipt["upload_status"] == "failed"
    assert (
        handoff.read_json(tracking_dir / "tracking_failure.json")["exception_type"]
        == "TimeoutError"
    )


def test_strict_tracking_failure_also_preserves_already_completed_score(
    generated, wandb_sdk, docker, tmp_path
):
    args = score_args(generated, tmp_path / "score")
    args.wandb_mode = "online"
    wandb_sdk.outage = "log"
    with pytest.raises(TimeoutError):
        campaign.score_checkpoint(args)
    assert (
        handoff.read_json(args.output_dir / "scoring_receipt.json")["status"]
        == "completed"
    )
    assert (
        handoff.read_json(args.output_dir / "code_metrics.json")["humaneval"][
            "task_count"
        ]
        == 2
    )
    assert handoff.validate_generation(generated[0]) == generated[1]


def test_receipt_save_never_follows_a_preexisting_temporary_symlink(
    tmp_path, checkpoint
):
    summary = checkpoint[1] / "summary.json"
    before = summary.read_bytes()
    (tmp_path / "scoring_receipt.tmp").symlink_to(summary)
    with pytest.raises(FileExistsError):
        campaign.save(tmp_path / "scoring_receipt.json", {"status": "failed"})
    assert summary.read_bytes() == before


def test_base_generation_has_no_training_dependency(generation_args, mock_generation):
    generation_args.arm = "base"
    generation_args.checkpoint_dir = None
    manifest = campaign.generate_checkpoint(generation_args)
    assert manifest["checkpoint"]["backend"] == "pretrained"
    assert manifest["checkpoint"]["files"] == {}
    assert mock_generation.restore == []
    assert handoff.validate_generation(generation_args.output_dir) == manifest
