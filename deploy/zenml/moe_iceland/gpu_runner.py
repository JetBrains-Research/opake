"""Fresh-process CUDA execution of the existing, immutable MoE experiments."""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .errors import refuse
from .gpu_policy import CACHE_BYTES, MIN_GPU_MEMORY, utc_deadline, validate_config
from .settings import validate_execution_plan
from .source import sha256

PARTITIONS = {
    "train": "c1211fe46f56249b47e9a5a5394e3ca8b8bd5c3eda4eb2fcd7028f03e96ff1de",
    "validation": "a609ad356206992fbdfa1857ffb99eae030b3e63a304d2e8a228439087287046",
    "test": "a685873689e5a2d85a09e769b5ba52a5378c29ad0fe446a19f60937b823b9b0d",
    "diagnostic": "d6cf608cfa2ef61321dd754ffa6d6e8260c19cf29e1651145789494803d1df5c",
}
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model*.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.model",
)
DATA_FILES = ("README.md", ".gitattributes", "*.json", "*.jsonl", "*.parquet")
RESERVE_BYTES = 8 * 1024**3


def write_receipt(path: Path, value: dict) -> None:
    """Write bounded public metadata without overwriting a previous stage receipt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cuda_details() -> dict:
    """Require one suitably sized BF16 GPU and execute a synchronized computation."""
    import torch

    if (
        not torch.version.cuda
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        refuse("Exactly one real CUDA device is required; there is no CPU fallback.")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < MIN_GPU_MEMORY or not torch.cuda.is_bf16_supported():
        refuse("The CUDA profile requires at least 80 GB and native BF16 support.")
    left = torch.full((16, 16), 2.0, dtype=torch.bfloat16, device="cuda:0")
    product = left @ left
    torch.cuda.synchronize()
    if product.device.type != "cuda" or not torch.equal(
        product, torch.full_like(product, 64.0)
    ):
        refuse("The real CUDA computation failed; an import is not a GPU gate.")
    return {
        "verified": True,
        "device": "cuda:0",
        "device_name": properties.name,
        "memory_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "capability": list(torch.cuda.get_device_capability(0)),
        "computation": "bf16_matmul_16x16_verified",
        "visible_devices": 1,
    }


def child_environment(
    environ: dict, *, pythonpath: str, plan: dict, run_id: str
) -> dict:
    """Forward only required runtime references; never forward cluster credentials."""
    allowed = {
        "PATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "WANDB_API_KEY",
    }
    env = {name: value for name, value in environ.items() if name in allowed}
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": "/scratch/hf",
            "HF_DATASETS_CACHE": "/scratch/hf/datasets",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TMPDIR": "/scratch",
            "WANDB_DIR": "/scratch/wandb",
            "WANDB_MODE": "online",
            "WANDB_BASE_URL": "https://jetbrains.wandb.io",
            "WANDB_ENTITY": "federated-compute",
            "WANDB_PROJECT": "opaque",
            "OPAQUE_ZENML_RUN_ID": run_id,
            "OPAQUE_ZENML_PROJECT_ID": plan["project_id"],
            "OPAQUE_ZENML_STACK_ID": plan["stack_id"],
            "OPAQUE_DEPLOYMENT_SOURCE_SHA256": plan["source_sha256"],
            "OPAQUE_DEPLOYMENT_IMAGE": plan["image"],
            "OPAQUE_DEPLOYMENT_RESOURCE_PROFILE": plan["resource_profile"],
        }
    )
    return env


def offline_environment(environ: dict) -> dict:
    """Ensure the fresh trainer never resolves a mutable/network model or dataset."""
    return {
        **environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def selected_files(info, patterns: tuple[str, ...]) -> tuple[list[str], int]:
    """Accept only bounded regular snapshot paths with known download sizes."""
    files = []
    size = 0
    for item in info.siblings:
        name = item.rfilename
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            refuse("Unsafe file path in the pinned public snapshot.")
        if type(item.size) is not int or item.size < 0:
            refuse("Snapshot has unknown file sizes; refuse an unbounded download.")
        files.append(name)
        size += item.size
    if not files or size > CACHE_BYTES:
        refuse(
            "Snapshot exceeds the bounded cache allowance or contains no usable files."
        )
    return sorted(files), size


def cache_bytes(directory: Path) -> int:
    """Count actual cache files, not duplicate snapshot symlink references."""
    total = 0
    for parent, _, names in os.walk(directory, followlinks=False):
        for name in names:
            path = Path(parent) / name
            if not path.is_symlink():
                total += path.stat().st_size
    if total > CACHE_BYTES:
        refuse(
            "Cache use exceeds the fixed allowance; do not enlarge the pod silently."
        )
    return total


def prefetch(config: dict, cache: Path) -> dict:
    """Stage pinned public snapshots and prime the exact offline datasets loader."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    cache.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=False)
    snapshots = []
    total = cache_bytes(cache)
    for kind, identifier, revision, patterns in (
        ("model", config["model_id"], config["model_revision"], MODEL_FILES),
        ("dataset", config["dataset_id"], config["dataset_revision"], DATA_FILES),
    ):
        info = api.repo_info(
            identifier, repo_type=kind, revision=revision, files_metadata=True
        )
        if info.sha != revision:
            refuse("Hub response did not resolve the pinned commit.")
        names, size = selected_files(info, patterns)
        total += size
        snapshots.append(
            {
                "repo_type": kind,
                "repo_id": identifier,
                "revision": revision,
                "files": names,
                "bytes": size,
            }
        )
    if total > CACHE_BYTES or shutil.disk_usage(cache).free < total + RESERVE_BYTES:
        refuse(
            "Pinned snapshots do not fit the bounded writable cache/scratch allocation."
        )
    for snapshot in snapshots:
        path = Path(
            snapshot_download(
                snapshot["repo_id"],
                repo_type=snapshot["repo_type"],
                revision=snapshot["revision"],
                allow_patterns=snapshot["files"],
                cache_dir=cache / "hub",
                token=False,
                max_workers=2,
            )
        )
        if path.name != snapshot["revision"] or not path.resolve().is_relative_to(
            cache.resolve()
        ):
            refuse("Downloaded snapshot escaped its revision-pinned cache.")
    load_dataset(
        config["dataset_id"],
        revision=config["dataset_revision"],
        token=False,
        cache_dir=str(cache / "datasets"),
    )
    return {
        "status": "completed",
        **{
            key: config[key]
            for key in ("model_id", "model_revision", "dataset_id", "dataset_revision")
        },
        "snapshots": snapshots,
        "cache_bytes": cache_bytes(cache),
        "cache_limit_bytes": CACHE_BYTES,
    }


def execution_config(config: dict, profile: str) -> dict:
    """Only the two-update gate changes length; data partitions remain identical."""
    if profile == "sft-smoke-v1":
        return {**config, "steps": 2, "eval_every": 1}
    return dict(config)


def tracking_name(plan: dict, config: dict) -> str:
    """Name each stage by backend, effective coefficient and privacy allocation."""
    coefficient = config["router_aux_loss_coef"] if plan["arm"].endswith("aux") else 0
    ratio = config["load_noise_ratio"] if plan["arm"] == "dp_aux" else "na"
    return f"iceland-{plan['profile']}-{plan['backend']}-aux{coefficient:g}-ratio{ratio}-{plan['run_name'].rsplit('-', 1)[-1]}"


def trainer_command(plan: dict, config_path: Path, output: Path) -> list[str]:
    """Dispatch directly to the unchanged private/native entry point, never a shell."""
    module = {"opaque": "sft_run", "trl": "trl_run"}.get(plan["backend"])
    if module is None:
        refuse("Unknown trainer backend.")
    config = json.loads(config_path.read_text())
    return [
        sys.executable,
        "-m",
        f"examples.moe_privacy.{module}",
        "--config",
        str(config_path),
        "--arm",
        plan["arm"],
        "--seed",
        str(plan["seed"]),
        "--device",
        "cuda",
        "--output-dir",
        str(output),
        "--wandb-mode",
        "online",
        "--wandb-group",
        "moe-aux-coefficients-iceland",
        "--wandb-name",
        tracking_name(plan, config),
        "--wandb-fail-open",
    ]


def validate_training(summary: dict, config: dict, plan: dict) -> None:
    """Keep partition, actual CUDA execution and unchanged training semantics fixed."""
    if summary.get("status") != "completed" or summary.get("config") != config:
        refuse("The trainer did not complete exactly its approved configuration.")
    if summary.get("arm") != plan["arm"] or summary.get("model_seed") != plan["seed"]:
        refuse("Trainer arm or public initialization changed.")
    execution = summary.get("execution", {})
    if not str(execution.get("device", "")).startswith("cuda"):
        refuse("Training did not execute on CUDA.")
    if plan["backend"] == "trl" and execution.get("opaque_patches") is not False:
        refuse("Native training must reject Opaque patches.")
    for name, expected in PARTITIONS.items():
        if summary.get("data", {}).get(name, {}).get("sha256") != expected:
            refuse(f"The audited {name} partition changed; do not compare this run.")
    privacy = summary.get("privacy", {})
    if privacy.get("steps") != config["steps"]:
        refuse("Incomplete optimizer updates.")
    if plan["arm"] == "dp_aux" and (
        privacy.get("private") is not True
        or privacy.get("load_release") is not True
        or privacy.get("load_noise_ratio") != config["load_noise_ratio"]
        or not 0 < privacy.get("epsilon", float("inf")) <= config["target_epsilon"]
        or privacy.get("delta") != config["delta"]
    ):
        refuse("Private training did not satisfy the joint accounting contract.")


def reload_export(plan: dict, output: Path) -> dict:
    """Reconstruct the backend in a fresh interpreter and load the exported tensors."""
    import torch
    from examples.moe_privacy.sft_adapters import load_trainable
    from examples.moe_privacy.sft_run import SFTExperimentConfig

    if plan["backend"] == "trl":
        from examples.moe_privacy.trl_run import load_model
    else:
        from examples.moe_privacy.sft_run import load_model
    summary = json.loads((output / "summary.json").read_text())
    model, layout = load_model(
        SFTExperimentConfig(**summary["config"]), plan["seed"], device="cuda"
    )
    if layout != summary["trainability"]:
        refuse("Reloaded model differs from the trained expert/router scope.")
    load_trainable(model, output / "trainable")
    model.eval()
    with torch.inference_mode():
        values = torch.arange(8, device="cuda").unsqueeze(0)
        logits = model(input_ids=values, attention_mask=torch.ones_like(values)).logits
    if logits.device.type != "cuda" or not torch.isfinite(logits).all().item():
        refuse("Reloaded adapters failed the real CUDA forward computation.")
    return {
        "status": "completed",
        "verified": True,
        "trainable_sha256": sha256(output / "trainable/trainable.safetensors"),
        "device": "cuda:0",
        "tensor_count": len(
            json.loads((output / "trainable/adapter_spec.json").read_text())["tensors"]
        ),
    }


def _run_child(command: list[str], *, root: Path, env: dict, expires: float) -> None:
    remaining = int(expires - time.monotonic())
    if remaining <= 0:
        refuse("Execution deadline expired; no next stage is allowed.")
    subprocess.run(command, cwd=root, env=env, check=True, timeout=remaining)


def _probe(
    plan: dict,
    root: Path,
    output: Path,
    plan_path: Path,
    env: dict,
    expires: float,
    cuda: dict,
) -> None:
    from examples.moe_privacy.tracking import ExperimentTracker, TrackingOptions

    output.mkdir()
    for backend in ("opaque", "trl"):
        _run_child(
            [
                sys.executable,
                "-m",
                "deploy.zenml.moe_iceland.gpu_runner",
                "--plan",
                str(plan_path),
                "--check-backend",
                backend,
            ],
            root=root,
            env=offline_environment(env),
            expires=expires,
        )
    with ExperimentTracker(
        output,
        config={"name": plan["run_name"]},
        arm="infrastructure_probe",
        seed=0,
        experiment="iceland_gpu_artifact_probe",
        options=TrackingOptions(
            mode="online", name=f"gpu-artifact-probe-{plan['run_name']}"
        ),
    ) as tracker:
        tracker.start()
        tracker.complete({"status": "completed", "execution": cuda})
    write_receipt(
        output / "probe.json",
        {
            "status": "completed",
            "cuda": cuda,
            "wandb_verified": True,
            "verified_backends": ["opaque", "trl"],
        },
    )


def generation_command(plan: dict, config: Path, directory: Path) -> list[str]:
    """Generate inert answers with the same common inference path and decoding."""
    return [
        sys.executable,
        "-m",
        "examples.moe_privacy.campaign",
        "generate",
        "--config",
        str(config),
        "--checkpoint-dir",
        str(directory / "checkpoint"),
        "--output-dir",
        str(directory / "runner-output/answers"),
        "--arm",
        plan["arm"],
        "--seed",
        str(plan["seed"]),
        "--benchmarks",
        plan["benchmark"],
        "--benchmark-dir",
        str(directory / "benchmarks"),
        "--max-new-tokens",
        "512",
        "--wandb-mode",
        "off",
        "--execution-metadata",
        str(directory / "runner-output/generation_execution.json"),
    ]


def _generate(
    plan: dict,
    root: Path,
    raw: Path,
    config: dict,
    env: dict,
    expires: float,
    cuda: dict,
) -> None:
    from examples.moe_privacy.handoff import validate_generation
    from examples.moe_privacy.tracking import ExperimentTracker, TrackingOptions

    from .checkpoint import validate_export

    validate_export(
        raw.parent / "checkpoint", config=config, arm=plan["arm"], seed=plan["seed"]
    )
    if (
        sha256(raw.parent / "benchmarks" / f"{plan['benchmark']}.jsonl")
        != plan["benchmark_input"]["sha256"]
    ):
        refuse("Benchmark bytes changed before generation.")
    staged = prefetch(config, Path(env["HF_HOME"]))
    config_path = raw.parent / "runner-config.json"
    write_receipt(config_path, config)
    raw.mkdir(exist_ok=False)
    with ExperimentTracker(
        raw,
        config=config,
        arm=plan["arm"],
        seed=plan["seed"],
        experiment="iceland_answer_generation",
        options=TrackingOptions(
            mode="online",
            fail_open=True,
            name=tracking_name(plan, config),
            group="moe-aux-coefficients-iceland",
        ),
    ) as tracker:
        tracker.start()
        _run_child(
            generation_command(plan, config_path, raw.parent),
            root=root,
            env=offline_environment(env),
            expires=expires,
        )
        validate_generation(raw / "answers")
        tracker.complete({"status": "completed", "execution": cuda})
    write_receipt(raw / "prefetch.json", staged)


def run_gpu(plan: dict, plan_path: Path) -> Path:
    """Run a bounded profile; return only locally validated, public artifact inputs."""
    validate_execution_plan(plan)
    root = Path(__file__).resolve().parents[3]
    config = validate_config(root, plan)
    output = Path(plan["output_dir"])
    raw = output.parent / "runner-output"
    expires = time.monotonic() + min(
        plan["timeout_seconds"] - 120,
        (
            utc_deadline(
                plan.get("run_deadline_utc", plan["authorization"]["deadline_utc"])
            )
            - datetime.now(UTC)
        ).total_seconds()
        - 120,
    )
    cuda = cuda_details()
    started = time.monotonic()
    if not os.environ.get("WANDB_API_KEY"):
        refuse("The required W&B runtime secret is absent; do not start training.")
    env = dict(os.environ)
    if plan["profile"] == "gpu-probe-v1":
        _probe(plan, root, raw, plan_path, env, expires, cuda)
    elif plan["profile"] in {"sft-smoke-v1", "sft-train-v1"}:
        staged = prefetch(config, Path(env["HF_HOME"]))
        actual = execution_config(config, plan["profile"])
        config_path = output.parent / "runner-config.json"
        write_receipt(config_path, actual)
        _run_child(
            trainer_command(plan, config_path, raw),
            root=root,
            env=offline_environment(env),
            expires=expires,
        )
        summary = json.loads((raw / "summary.json").read_text())
        validate_training(summary, actual, plan)
        execution_path = raw / "execution.json"
        if not execution_path.exists():
            write_receipt(execution_path, summary["execution"])
        write_receipt(raw / "prefetch.json", staged)
        _run_child(
            [
                sys.executable,
                "-m",
                "deploy.zenml.moe_iceland.gpu_runner",
                "--plan",
                str(plan_path),
                "--reload-output",
                str(raw),
            ],
            root=root,
            env=offline_environment(env),
            expires=expires,
        )
    elif plan["profile"] == "generate-v1":
        _generate(plan, root, raw, config, env, expires, cuda)
    else:
        refuse("Unsupported GPU execution profile.")
    runtime = {
        "cuda": cuda,
        "backend": plan["backend"],
        "elapsed_seconds": time.monotonic() - started,
    }
    if plan["profile"] == "generate-v1":
        generation_execution = json.loads(
            (raw / "generation_execution.json").read_text()
        )
        if generation_execution.get(
            "status"
        ) != "completed" or not generation_execution.get("device", "").startswith(
            "cuda"
        ):
            refuse("Generation did not record completed CUDA execution.")
        runtime["generation_execution"] = generation_execution
    write_receipt(raw / "runtime.json", runtime)
    tracking = json.loads((raw / "wandb_run.json").read_text())
    wandb_verified = (
        tracking.get("upload_status") == "completed"
        and not (raw / "tracking_failure.json").exists()
    )
    write_receipt(
        raw / "stage_receipt.json",
        {
            "stage": "probe"
            if plan["profile"] == "gpu-probe-v1"
            else "generation"
            if plan["profile"] == "generate-v1"
            else "training",
            "profile": plan["profile"],
            "status": "completed",
            "backend": plan["backend"],
            "arm": plan["arm"],
            "model_seed": plan["seed"],
            "steps": execution_config(config, plan["profile"])["steps"]
            if plan["profile"] in {"sft-smoke-v1", "sft-train-v1"}
            else 0,
            "cuda_verified": True,
            "wandb_verified": wandb_verified,
            "config_sha256": plan["config_sha256"],
            "source_sha256": plan["source_sha256"],
            "image": plan["image"],
            "zenml_run_id": os.environ["OPAQUE_ZENML_RUN_ID"],
            **(
                {
                    "generation_manifest_sha256": sha256(
                        raw / "answers/generation-manifest.json"
                    ),
                    "benchmark": plan["benchmark"],
                }
                if plan["profile"] == "generate-v1"
                else {}
            ),
        },
    )
    return raw


def main() -> None:
    """Execute only dispatcher-owned modes and paths, never arbitrary commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reload-output", type=Path)
    modes.add_argument("--check-backend", choices=("opaque", "trl"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_execution_plan(plan)
    if args.check_backend:
        if args.check_backend == "trl":
            from examples.moe_privacy.trl_run import assert_native_model

            assert_native_model()
        else:
            from opaque.transformers import DPTrainer

            if DPTrainer.__name__ != "DPTrainer":
                refuse("The private trainer import differs from the approved backend.")
    elif args.reload_output:
        if args.reload_output != Path(plan["output_dir"]).parent / "runner-output":
            refuse("Reload must use this job's own exported adapter directory.")
        write_receipt(
            args.reload_output / "adapter_reload.json",
            reload_export(plan, args.reload_output),
        )
    else:
        run_gpu(plan, args.plan)


if __name__ == "__main__":
    main()
