"""Serial, deadline-bounded confirmation with visible incomplete/failed stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from examples.moe_privacy import handoff
from examples.moe_privacy.campaign_report import write_report

ARMS = ("reference", "reference_aux", "dp", "dp_aux")
STAGE_GRACE_SECONDS = 60


def save(path, value):
    """Atomically replace only campaign-owned JSON receipts."""
    temporary = path.with_suffix(".tmp")
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x") as stream:
        stream.write(raw)
    temporary.replace(path)


def normalize_code_result(result):
    """Keep protocol identity independent of model outputs and checkpoint names."""
    protocol = {
        key: result[key]
        for key in ("benchmark", "generation", "evalplus_version", "image_id", "limits")
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return {
        "base_pass1": result["base"]["pass@1"],
        "plus_pass1": result["plus"]["pass@1"],
        "plus_wilson_95": result["plus"]["wilson_95"],
        "task_count": result["task_count"],
        "smoke": result["mode"] != "full",
        "protocol_sha256": digest,
        "tasks_with_extended_tests": result["benchmark"].get(
            "tasks_with_extended_tests"
        ),
        "tasks_without_extended_tests": result["benchmark"].get(
            "tasks_without_extended_tests"
        ),
        **{
            key: result[key]
            for key in (
                "syntax_parse_success_rate",
                "timeout_rate",
                "truncation_rate",
                "generation_seconds",
                "generated_tokens",
            )
        },
    }


def _stage_tracker(args, directory, config, checkpoint, stage):
    from examples.moe_privacy.tracking import ExperimentTracker, TrackingOptions

    mode = getattr(args, "wandb_mode", "off")
    options = TrackingOptions(
        mode="disabled" if mode == "off" else mode,
        group=getattr(args, "wandb_group", None) or getattr(args, "group", None),
        fail_open=getattr(args, "wandb_fail_open", False),
        **{
            name: getattr(args, f"wandb_{name}")
            for name in ("base_url", "entity", "project", "name")
            if getattr(args, f"wandb_{name}", None) is not None
        },
    )
    # SDK logs are not portable payloads. A fresh sibling also gives this stage
    # its own run identity, never the completed training run's ID or directory.
    tracking_dir = directory.with_name(directory.name + ".tracking")
    if options.mode != "disabled":
        tracking_dir.mkdir(parents=True, exist_ok=False)
    return ExperimentTracker(
        tracking_dir,
        config=config,
        arm=checkpoint["arm"],
        seed=checkpoint["seed"],
        experiment=f"moe_code_{stage}",
        resume=False,
        options=options,
    )


def _publish_code_metrics(tracker, manifest, results=None):
    if tracker.run is not None and not tracker._tracking_failed:
        with tracker._sdk_operation("code_metrics"):
            scalars = {
                f"code/{name}/{key}": value
                for name, result in (results or {}).items()
                for key, value in result.items()
                if isinstance(value, (float, int)) and not isinstance(value, bool)
            }
            if scalars:
                tracker.run.log({"trainer/step": 0, **scalars})
            tracker.run.summary.update(
                {
                    **scalars,
                    "code/generation_manifest_sha256": manifest["manifest_sha256"],
                    "code/training_executed": False,
                    "code/evaluation_completed": results is not None,
                    "code/smoke": any(
                        item["mode"] != "full"
                        for item in manifest["benchmarks"].values()
                    ),
                }
            )
    tracker.completed = True


def generate_checkpoint(args):
    """Generate a fresh flat handoff, without Docker or any training operation.

    Uses the same ``sft_run.load_model`` / export restore / merge / tokenizer
    path for *all* arms, including native TRL exports. No trainer is constructed
    or resumed. Optional SDK logs live in the sibling ``OUTPUT.tracking``.
    """
    import torch
    from examples.moe_privacy.code_eval import (
        generate_samples,
        generation_settings,
        load_benchmark,
        merge_for_evaluation,
    )
    from examples.moe_privacy.sft_adapters import load_trainable
    from examples.moe_privacy.sft_run import SFTExperimentConfig, load_model
    from transformers import AutoTokenizer

    config = SFTExperimentConfig(**handoff.read_json(Path(args.config)))
    directory = Path(args.output_dir)
    if directory.exists() or directory.is_symlink():
        message = f"Refusing to overwrite generation: {directory}"
        raise FileExistsError(message)
    execution_metadata = getattr(args, "execution_metadata", None)
    if execution_metadata is not None and (
        Path(execution_metadata).exists() or Path(execution_metadata).is_symlink()
    ):
        message = "Refusing to overwrite generation execution metadata"
        raise FileExistsError(message)
    names = list(args.benchmarks)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in ("humaneval", "mbpp") for name in names)
    ):
        message = "Require distinct supported benchmarks"
        raise ValueError(message)
    max_new_tokens = getattr(args, "max_new_tokens", 512)
    smoke_limit = getattr(args, "smoke_limit", None)
    generation_settings(max_new_tokens)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    checkpoint, payload = handoff.checkpoint_payload(
        asdict(config), checkpoint_dir, args.arm, args.seed
    )
    for name in names:
        path = Path(args.benchmark_dir) / f"{name}.jsonl"
        handoff.file_record(path)
        tasks, _ = load_benchmark(name, path)
        if smoke_limit is not None and (
            type(smoke_limit) is not int or not 1 <= smoke_limit <= len(tasks)
        ):
            message = "smoke_limit must be between 1 and the full task count"
            raise ValueError(message)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        for filename, raw in payload.items():
            with (directory / filename).open("xb") as stream:
                stream.write(raw)
        for name in names:
            handoff.copy_file(
                Path(args.benchmark_dir) / f"{name}.jsonl", directory / f"{name}.jsonl"
            )
        with _stage_tracker(
            args, directory, asdict(config), checkpoint, "generation"
        ) as tracker:
            tracker.start()
            torch.set_num_threads(4)
            started = time.monotonic()
            if execution_metadata is not None:
                if not torch.cuda.is_available():
                    message = "CUDA execution metadata requires a real GPU; no fallback"
                    raise ValueError(message)
                torch.cuda.reset_peak_memory_stats()
            model, _ = load_model(config, args.seed)
            if (
                execution_metadata is not None
                and next(model.parameters()).device.type != "cuda"
            ):
                message = "Generation model is not on CUDA"
                raise ValueError(message)
            if args.arm != "base":
                load_trainable(model, Path(checkpoint_dir) / "trainable")
            merge_for_evaluation(model)
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id, revision=config.model_revision, local_files_only=True
            )
            for name in names:
                work = directory / f".{name}-pending"
                generate_samples(
                    model,
                    tokenizer,
                    dataset=name,
                    dataset_path=directory / f"{name}.jsonl",
                    output_dir=work,
                    model_id=config.model_id,
                    model_revision=config.model_revision,
                    arm=args.arm,
                    max_new_tokens=max_new_tokens,
                    smoke_limit=smoke_limit,
                )
                for filename in (
                    "generation.json",
                    "samples.jsonl",
                    "raw_samples.jsonl",
                ):
                    handoff.file_record(work / filename)
                    (work / filename).rename(directory / f"{name}-{filename}")
                work.rmdir()
            if execution_metadata is not None:
                torch.cuda.synchronize()
                handoff.write_json(
                    Path(execution_metadata),
                    {
                        "status": "completed",
                        "device": str(next(model.parameters()).device),
                        "device_name": torch.cuda.get_device_name(),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                )
            current, _ = handoff.checkpoint_payload(
                asdict(config), checkpoint_dir, args.arm, args.seed
            )
            if current != checkpoint:
                message = "Checkpoint changed during generation"
                raise ValueError(message)
            manifest = handoff.seal_generation(
                directory,
                checkpoint=checkpoint,
                benchmarks=names,
                max_new_tokens=max_new_tokens,
                smoke_limit=smoke_limit,
            )
            _publish_code_metrics(tracker, manifest)
    except BaseException as error:
        if not (directory / handoff.MANIFEST).exists():
            save(
                directory / handoff.RECEIPT,
                {
                    "schema_version": 1,
                    "stage": "generation",
                    "status": "failed",
                    "checkpoint": checkpoint,
                    "exception_type": type(error).__name__,
                },
            )
        raise
    return manifest


def _scoring_identity(image, limits, docker_sudo):
    from examples.moe_privacy.code_eval import inspect_image

    image_id = inspect_image(image, docker_sudo=docker_sudo)
    prefix = ["sudo", "-n", "docker"] if docker_sudo else ["docker"]

    def inspect(arguments):
        result = subprocess.run(
            [*prefix, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    info = inspect(["image", "inspect", "--format", "{{json .}}", image_id])
    engine = inspect(["version", "--format", "{{json .Server}}"])
    if (
        not isinstance(info, dict)
        or info.get("Id") != image_id
        or not isinstance(engine, dict)
        or any(
            not isinstance(info.get(key), str) or not info[key]
            for key in ("Os", "Architecture")
        )
        or any(
            not isinstance(engine.get(key), str) or not engine[key]
            for key in ("Os", "Arch", "Version")
        )
    ):
        message = "Missing or inconsistent Docker image/engine platform identity"
        raise ValueError(message)
    return {
        "requested_image": image,
        "image_id": image_id,
        "image_platform": {"os": info["Os"], "architecture": info["Architecture"]},
        "docker_engine": {
            "os": engine["Os"],
            "architecture": engine["Arch"],
            "version": engine["Version"],
            "platform": engine.get("Platform", {}).get("Name"),
        },
        "host": {"os": platform.system(), "architecture": platform.machine()},
        "limits": asdict(limits),
        "timing_parity": (
            "Not established across hosts, architectures, Docker virtualization or "
            "emulation; timeout-sensitive scores and timings may differ."
        ),
    }


def score_checkpoint(args):
    """Score retained samples into a fresh attempt; never load a model or train.

    ``expected_manifest_sha256`` and an immutable ``eval_image`` are required.
    Returns/writes the existing benchmark-keyed ``code_metrics.json`` format.
    """
    return _score_checkpoint(args, require_pinned_image=True)


def _score_checkpoint(args, *, require_pinned_image):
    from examples.moe_privacy.code_eval import SandboxLimits, evaluate_samples

    expected = getattr(args, "expected_manifest_sha256", None)
    if not expected:
        message = "Scoring requires --expected-manifest-sha256 from generation"
        raise ValueError(message)
    image = args.eval_image
    if require_pinned_image and not re.fullmatch(
        r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", image
    ):
        message = "Scoring requires a pinned image ID or repository digest"
        raise ValueError(message)
    directory, source = Path(args.output_dir), Path(args.samples_dir)
    if directory.exists() or directory.is_symlink():
        message = f"Refusing to overwrite a scoring attempt: {directory}"
        raise FileExistsError(message)
    if directory.resolve().is_relative_to(source.resolve()):
        message = "A scoring attempt must not modify the generation directory"
        raise ValueError(message)
    manifest = handoff.validate_generation(source, expected)
    limits = SandboxLimits(
        cpus=getattr(args, "cpus", 2),
        memory_gb=getattr(args, "memory_gb", 8),
        timeout_seconds=getattr(args, "timeout_seconds", 3600),
    )
    docker_sudo = getattr(args, "docker_sudo", False)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "stage": "scoring",
        "status": "running",
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "requested_image": image,
        "results": {},
    }
    results = {}
    save(directory / "scoring_receipt.json", receipt)
    try:
        snapshot = directory / "generation"
        snapshot.mkdir()
        for filename in (*manifest["files"], handoff.MANIFEST):
            handoff.copy_file(source / filename, snapshot / filename)
        handoff.validate_generation(snapshot, expected)
        config = handoff.read_json(snapshot / "config.json")
        with _stage_tracker(
            args, directory, config, manifest["checkpoint"], "scoring"
        ) as tracker:
            tracker.start()
            identity = _scoring_identity(image, limits, docker_sudo)
            receipt["evaluator"] = identity
            save(directory / "scoring_receipt.json", receipt)
            for name, generation in manifest["benchmarks"].items():
                samples = handoff.materialize_samples(
                    snapshot, name, directory / f"{name}-input"
                )
                execution = directory / f"{name}-execution"
                result = evaluate_samples(
                    dataset=name,
                    dataset_path=snapshot / f"{name}.jsonl",
                    samples_dir=samples,
                    output_dir=execution,
                    image=identity["image_id"],
                    limits=limits,
                    docker_sudo=docker_sudo,
                )
                if (
                    any(
                        result[key] != generation[key]
                        for key in ("model", "benchmark", "generation", "mode")
                    )
                    or result["image_id"] != identity["image_id"]
                    or result["task_count"] != len(generation["task_ids"])
                ):
                    message = "Scoring result differs from the retained generation"
                    raise ValueError(message)
                per_task = handoff.read_json(execution / "per_task.json")
                if [row["task_id"] for row in per_task] != generation["task_ids"]:
                    message = "Scoring outcomes do not align with generation tasks"
                    raise ValueError(message)
                receipt["results"][name] = {
                    "task_ids": generation["task_ids"],
                    "mode": generation["mode"],
                    "files": {
                        f"{name}-execution/{filename}": handoff.file_record(
                            execution / filename
                        )
                        for filename in (
                            "summary.json",
                            "per_task.json",
                            "execution.json",
                            "samples_eval_results.json",
                        )
                    },
                }
                results[name] = normalize_code_result(result)
                save(directory / "code_metrics.json", results)
                receipt["code_metrics"] = handoff.file_record(
                    directory / "code_metrics.json"
                )
                save(directory / "scoring_receipt.json", receipt)
            receipt["status"] = "completed"
            save(directory / "scoring_receipt.json", receipt)
            _publish_code_metrics(tracker, manifest, results)
    except BaseException as error:
        if receipt["status"] != "completed":
            receipt.update(status="failed", exception_type=type(error).__name__)
            save(directory / "scoring_receipt.json", receipt)
        raise
    return results


def evaluate_checkpoint(args):
    """Compatibility entry point: generate, score in Docker, publish legacy paths.

    New stage directories and their receipts are retained on failure. Retry the
    retained generation with ``score`` and a fresh output, not by regenerating.
    Completed training summaries/receipts and earlier benchmark outputs are never
    rewritten. ``append_benchmark`` adds only new completed benchmark results.
    """
    from examples.moe_privacy.sft_run import SFTExperimentConfig

    config = SFTExperimentConfig(**handoff.read_json(Path(args.config)))
    root = Path(args.output_dir)
    append = getattr(args, "append_benchmark", False)
    results_path = root / "code_metrics.json"
    if append:
        results = json.loads(results_path.read_text())
        if not results or any(name in results for name in args.benchmarks):
            message = "append only new benchmarks to an existing completed evaluation"
            raise ValueError(message)
        for name in results:
            if name not in ("humaneval", "mbpp"):
                message = "unrecognized existing benchmark"
                raise ValueError(message)
            old = json.loads((root / f"{name}-execution" / "summary.json").read_text())
            if (
                old["model"]["id"] != config.model_id
                or old["model"]["revision"] != config.model_revision
            ):
                message = "appended evaluation must use the same base checkpoint"
                raise ValueError(message)
    else:
        if results_path.exists():
            message = "Completed code metrics already exist; append only new benchmarks"
            raise FileExistsError(message)
        results = {}
    checkpoint, _ = handoff.checkpoint_payload(
        asdict(config), root, args.arm, args.seed
    )
    for name in results:
        pointer = root / f"{name}-handoff.json"
        if pointer.exists() and handoff.read_json(pointer)["checkpoint"] != checkpoint:
            message = "Appended evaluation must use the same completed checkpoint"
            raise ValueError(message)
    if not args.benchmarks or any(
        name not in ("humaneval", "mbpp") for name in args.benchmarks
    ):
        message = "Require supported benchmarks"
        raise ValueError(message)
    label = "-".join(args.benchmarks)
    generated, scored = root / f"generation-{label}", root / f"score-{label}"
    for path in (
        generated,
        scored,
        *(
            root / f"{name}-{kind}"
            for name in args.benchmarks
            for kind in ("samples", "execution", "handoff.json")
        ),
    ):
        if path.exists() or path.is_symlink():
            message = "Evaluation artifacts already exist; score retained samples into a fresh attempt"
            raise FileExistsError(message)
    if args.arm == "base" and not append:
        root.mkdir(parents=True, exist_ok=False)
    manifest = generate_checkpoint(
        argparse.Namespace(
            **{**vars(args), "checkpoint_dir": root, "output_dir": generated}
        )
    )
    try:
        _score_checkpoint(
            argparse.Namespace(
                **{
                    **vars(args),
                    "samples_dir": generated,
                    "output_dir": scored,
                    "eval_image": args.image,
                    "expected_manifest_sha256": manifest["manifest_sha256"],
                }
            ),
            require_pinned_image=False,
        )
    finally:
        score_metrics = scored / "code_metrics.json"
        if score_metrics.exists():
            for name, metrics in handoff.read_json(score_metrics).items():
                execution = root / f"{name}-execution"
                execution.mkdir()
                for filename in (
                    "summary.json",
                    "per_task.json",
                    "execution.json",
                    "samples_eval_results.json",
                    "container.log",
                ):
                    source = scored / f"{name}-execution" / filename
                    if source.exists():
                        handoff.copy_file(source, execution / filename)
                handoff.materialize_samples(generated, name, root / f"{name}-samples")
                handoff.write_json(
                    root / f"{name}-handoff.json",
                    {
                        "generation": generated.name,
                        "generation_manifest_sha256": manifest["manifest_sha256"],
                        "score_attempt": scored.name,
                        "checkpoint": checkpoint,
                    },
                )
                results[name] = metrics
                save(results_path, results)
    return results


def run_stage(command, *, root, name, deadline):
    """Wait for one owned process, stopping on an error or absolute deadline."""
    remaining = deadline - time.time()
    if remaining <= STAGE_GRACE_SECONDS:
        message = "campaign deadline reached before the next stage"
        raise TimeoutError(message)
    save(
        root / "campaign_state.json",
        {
            "status": "running",
            "stage": name,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    with (
        (root / "logs" / f"{name}.log").open("x") as stream,
        subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as child,
    ):
        try:
            returncode = child.wait(timeout=remaining - STAGE_GRACE_SECONDS / 2)
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    save(
        root / "campaign_state.json",
        {"status": "running", "stage": name, "stage_completed": True},
    )


def run_campaign(args):
    """Run each planned arm once, then replicas; never choose the best seed."""
    from examples.moe_privacy.sft_run import SFTExperimentConfig

    config = SFTExperimentConfig(**json.loads(args.config.read_text()))
    deadline_time = datetime.fromisoformat(args.deadline)
    if deadline_time.utcoffset() is None:
        message = "deadline must include an explicit UTC offset"
        raise ValueError(message)
    deadline = deadline_time.timestamp()
    if not 0 < deadline - time.time() <= 72 * 3600:
        message = "deadline must be in the future and no more than 72 hours away"
        raise ValueError(message)
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or any(seed < 0 for seed in args.seeds)
    ):
        message = "provide distinct nonnegative public initialization seeds"
        raise ValueError(message)
    secondary = getattr(args, "secondary_benchmarks", [])
    requested = [*args.benchmarks, *secondary]
    if len(requested) != len(set(requested)):
        message = "primary and secondary benchmarks must be distinct"
        raise ValueError(message)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(
        root / "campaign.json",
        {
            "protocol": "matched-clipped-lagged-v2",
            "config": asdict(config),
            "seeds": args.seeds,
            "arms": list(ARMS),
            "deadline_utc": args.deadline,
            "group": args.group,
            "image": args.image,
            "primary_benchmarks": args.benchmarks,
            "secondary_benchmarks": secondary,
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.iterdir()
                if path.suffix in (".py", ".md")
            },
        },
    )
    write_report(root)
    evaluation = [
        sys.executable,
        "-B",
        "-m",
        "examples.moe_privacy.campaign",
        "evaluate",
        "--config",
        str(args.config),
        "--image",
        args.image,
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--benchmarks",
        *args.benchmarks,
        "--group",
        args.group,
        "--wandb-mode",
        args.wandb_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.docker_sudo:
        evaluation.append("--docker-sudo")
    try:
        durations = []
        for index, seed in enumerate(args.seeds):
            offset = (2 * index) % len(ARMS)
            for arm in (*ARMS[offset:], *ARMS[:offset]):
                if durations and deadline - time.time() < max(durations) * 1.25 + 600:
                    message = "not enough remaining time for another complete arm; completed results retained"
                    raise TimeoutError(message)
                started = time.time()
                output = root / "main" / f"seed-{seed}" / arm
                run_stage(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "examples.moe_privacy.sft_run",
                        "--config",
                        str(args.config),
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output),
                        "--wandb-mode",
                        args.wandb_mode,
                        "--wandb-group",
                        args.group,
                    ],
                    root=root,
                    name=f"s{seed}-{arm}-train",
                    deadline=deadline,
                )
                if not (output / "summary.json").is_file():
                    message = "training exited without a completed summary"
                    raise RuntimeError(message)
                write_report(root)
                training_elapsed = time.time() - started
                if not (root / "base" / "code_metrics.json").exists():
                    run_stage(
                        [
                            *evaluation,
                            "--arm",
                            "base",
                            "--output-dir",
                            str(root / "base"),
                        ],
                        root=root,
                        name="base-code",
                        deadline=deadline,
                    )
                    write_report(root)
                code_started = time.time()
                run_stage(
                    [
                        *evaluation,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output),
                    ],
                    root=root,
                    name=f"s{seed}-{arm}-code",
                    deadline=deadline,
                )
                write_report(root)
                durations.append(training_elapsed + time.time() - code_started)
        if secondary:
            command = evaluation.copy()
            command[command.index("--benchmarks") + 1 : command.index("--group")] = (
                secondary
            )
            command.append("--append-benchmark")
            checkpoints = [
                ("base", 0, root / "base"),
                *(
                    (arm, seed, root / "main" / f"seed-{seed}" / arm)
                    for seed in args.seeds
                    for arm in ARMS
                ),
            ]
            secondary_durations = []
            for arm, seed, output in checkpoints:
                estimate = max(secondary_durations, default=1800) * 1.25 + 600
                if deadline - time.time() < estimate:
                    message = "primary comparison complete; insufficient budget for another secondary evaluation"
                    raise TimeoutError(message)
                started = time.time()
                run_stage(
                    [
                        *command,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output),
                    ],
                    root=root,
                    name=f"s{seed}-{arm}-secondary",
                    deadline=deadline,
                )
                secondary_durations.append(time.time() - started)
                write_report(root)
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
        state = (
            json.loads((root / "campaign_state.json").read_text())
            if (root / "campaign_state.json").exists()
            else {}
        )
        state.update(
            status="budget_exhausted"
            if isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
            else "failed",
            error=str(error),
        )
        save(root / "campaign_state.json", state)
        write_report(root)
        raise
    save(root / "campaign_state.json", {"status": "completed"})
    return write_report(root)


def main(argv=None):
    """Select the controller, combined evaluation, generation, or isolated scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--image", required=True)
        sub.add_argument("--benchmark-dir", type=Path, required=True)
        sub.add_argument(
            "--benchmarks",
            nargs="+",
            choices=("humaneval", "mbpp"),
            default=["humaneval"],
        )
        sub.add_argument("--docker-sudo", action="store_true")
        sub.add_argument("--group", required=True)
        sub.add_argument(
            "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
        )
        sub.add_argument("--max-new-tokens", type=int, default=512)
        if name == "run":
            sub.add_argument("--deadline", required=True)
            sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            sub.add_argument(
                "--secondary-benchmarks",
                nargs="*",
                choices=("humaneval", "mbpp"),
                default=["mbpp"],
            )
        else:
            sub.add_argument(
                "--arm",
                choices=("base", *ARMS, "trl_reference", "trl_reference_aux"),
                required=True,
            )
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--smoke-limit", type=int)
            sub.add_argument("--append-benchmark", action="store_true")
    for name in ("generate", "score"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--output-dir", type=Path, required=True, help="Must not exist"
        )
        sub.add_argument(
            "--wandb-mode",
            choices=("off", "online", "offline", "disabled"),
            default="off",
        )
        sub.add_argument("--group", "--wandb-group", dest="group")
        sub.add_argument("--wandb-base-url")
        sub.add_argument("--wandb-entity")
        sub.add_argument("--wandb-project")
        sub.add_argument("--wandb-name")
        sub.add_argument("--wandb-fail-open", action="store_true")
        if name == "generate":
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument(
                "--checkpoint-dir",
                type=Path,
                help="Completed training export; required except for base",
            )
            sub.add_argument("--arm", choices=handoff.ARMS, required=True)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--benchmark-dir", type=Path, required=True)
            sub.add_argument(
                "--benchmarks",
                nargs="+",
                choices=("humaneval", "mbpp"),
                default=["humaneval"],
            )
            sub.add_argument("--max-new-tokens", type=int, default=512)
            sub.add_argument(
                "--execution-metadata",
                type=Path,
                help="Separate CUDA-only hardware/cost receipt; must not exist.",
            )
            sub.add_argument(
                "--smoke-limit", type=int, help="Test only; never a full score"
            )
        else:
            sub.add_argument("--samples-dir", type=Path, required=True)
            sub.add_argument(
                "--eval-image",
                required=True,
                help="Local immutable image ID or repository digest",
            )
            sub.add_argument("--expected-manifest-sha256", required=True)
            sub.add_argument("--docker-sudo", action="store_true")
            sub.add_argument("--cpus", type=int, default=2)
            sub.add_argument("--memory-gb", type=int, default=8)
            sub.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_campaign(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    elif args.command == "generate":
        manifest = generate_checkpoint(args)
        print(
            json.dumps({"manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True)
        )
    else:
        print(json.dumps(score_checkpoint(args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
