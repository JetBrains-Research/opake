"""One arm/seed, one native Kubernetes step, one directory artifact."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from zenml import pipeline, step
from zenml.materializers.path_materializer import PathMaterializer

from .bundle import validate_bundle
from .errors import refuse
from .gpu_policy import GPU_PROFILES, utc_deadline
from .provenance import overlay_workspace, validate_contract
from .settings import DEADLINES, POD_URL, validate_execution_plan
from .source import verify_source


def run_job(plan: dict) -> Path:
    """Execute the real checked runner using verified archived source and image code."""
    validate_execution_plan(plan)
    root = Path(__file__).resolve().parents[3]
    source = verify_source(root)
    if source["source_sha256"] != plan["source_sha256"]:
        refuse("The runtime archive differs from the submitted source.")
    if os.environ.get("ZENML_STORE_URL") != POD_URL:
        refuse("The step pod does not have the required external ZenML URL.")
    gpu = plan["profile"] in GPU_PROFILES
    versions = validate_contract(
        root, plan["image_contract"], runtime=True, accelerator="cuda" if gpu else "cpu"
    )
    if gpu:
        return run_gpu_job(root, plan, source, versions)
    env = os.environ.copy()
    env["PYTHONPATH"] = overlay_workspace(root)
    for name in list(env):
        if name.startswith("ZENML_") or name in {
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "WANDB_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
        }:
            env.pop(name)
    output = Path(plan["output_dir"])
    if (
        output.parent.parent != Path("/tmp")
        or not output.parent.name.startswith("moe-iceland-")
        or output.name != "output"
    ):
        refuse("Unexpected runtime output path.")
    output.parent.mkdir(exist_ok=False)
    command = [sys.executable, *plan["runner_command"][1:]]
    subprocess.run(
        command,
        cwd=root,
        env=env,
        check=True,
        timeout=DEADLINES[plan["profile"]] - 180,
    )
    artifact = validate_bundle(output, plan["profile"])
    receipt = {
        "schema": 1,
        "plan": plan,
        "runner_returncode": 0,
        "checks_requested": True,
        "source": source,
        "dependencies": versions,
        "bundle": artifact,
    }
    with (output / "deployment.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    validate_bundle(output, plan["profile"])
    # Keep the directory alive until PathMaterializer has uploaded data.tar.gz.
    return output


def public_outputs(raw: Path, output: Path) -> None:
    """Select public receipts and weights, never optimizer, RNG, or W&B caches."""
    allowed = {
        "summary.json",
        "metrics.jsonl",
        "trainability.json",
        "execution.json",
        "test-records.jsonl",
        "wandb_run.json",
        "tracking_failure.json",
        "probe.json",
        "runtime.json",
        "prefetch.json",
        "adapter_reload.json",
        "stage_receipt.json",
    }
    if (raw / "probe.json").exists():
        allowed = {"probe.json", "runtime.json", "stage_receipt.json"}
    answers = raw / "answers"
    if answers.exists():
        from examples.moe_privacy.handoff import validate_generation

        validate_generation(answers)
        allowed = {
            "runtime.json",
            "prefetch.json",
            "stage_receipt.json",
            "wandb_run.json",
            "tracking_failure.json",
        }
    selected = []
    for path in raw.iterdir():
        if path.name in allowed:
            if path.is_symlink() or not path.is_file():
                refuse("A public output is not a regular file.")
            selected.append(path)
    if answers.exists():
        selected.extend(answers.iterdir())
    trainable = raw / "trainable"
    if trainable.exists():
        if trainable.is_symlink() or not trainable.is_dir():
            refuse("Expected a regular trainable export directory.")
        for name in ("adapter_spec.json", "trainable.safetensors"):
            path = trainable / name
            if path.is_symlink() or not path.is_file():
                refuse("The weights-only export is missing or unsafe.")
            selected.append(path)
    if sum(path.stat().st_size for path in selected) > 1024**3:
        refuse("Public outputs exceed the 1Gi artifact allowance before copying.")
    output.mkdir(exist_ok=False)
    for path in selected:
        destination = output / path.relative_to(raw)
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)


def run_gpu_job(root: Path, plan: dict, source: dict, versions: dict) -> Path:
    """Run the CUDA dispatcher without inheriting cluster credentials or patches."""
    from datetime import UTC, datetime

    from zenml import get_step_context

    from .bundle import seal_bundle
    from .gpu_runner import child_environment, write_receipt

    output = Path(plan["output_dir"])
    output.parent.mkdir(exist_ok=False)
    plan_path = output.parent / "execution-plan.json"
    write_receipt(plan_path, plan)
    if plan["profile"] == "generate-v1":
        from .checkpoint import download_benchmark, download_checkpoint, validate_export
        from .gpu_policy import validate_config

        checkpoint = download_checkpoint(plan, output.parent)
        validate_export(
            checkpoint,
            config=validate_config(root, plan),
            arm=plan["arm"],
            seed=plan["seed"],
        )
        download_benchmark(plan, output.parent / "benchmarks")
    run_id = str(get_step_context().pipeline_run.id)
    env = child_environment(
        os.environ, pythonpath=overlay_workspace(root), plan=plan, run_id=run_id
    )
    deadline = utc_deadline(
        plan.get("run_deadline_utc", plan["authorization"]["deadline_utc"])
    )
    remaining = (
        min(
            plan["timeout_seconds"], int((deadline - datetime.now(UTC)).total_seconds())
        )
        - 60
    )
    if remaining <= 0:
        refuse("No authorized time remains for the GPU step.")
    subprocess.run(
        [sys.executable, *plan["runner_command"][1:]],
        cwd=root,
        env=env,
        check=True,
        timeout=remaining,
    )
    raw = output.parent / "runner-output"
    public_outputs(raw, output)
    runtime = json.loads((output / "runtime.json").read_text())
    probe = plan["profile"] == "gpu-probe-v1"
    wandb_verified = json.loads((output / "stage_receipt.json").read_text())[
        "wandb_verified"
    ]
    reload_path = output / "adapter_reload.json"
    reloaded = (
        reload_path.exists()
        and json.loads(reload_path.read_text()).get("verified") is True
    )
    write_receipt(
        output / "deployment.json",
        {
            "schema": 2,
            "plan": plan,
            "runner_returncode": 0,
            "source": source,
            "dependencies": versions,
            "cuda_verified": runtime["cuda"]["verified"],
            "wandb_verified": wandb_verified,
            "adapter_reload_verified": reloaded,
            "zenml_run_id": run_id,
            "checks_requested": probe or plan["profile"] == "sft-smoke-v1",
        },
    )
    seal_bundle(output, plan["profile"])
    validate_bundle(output, plan["profile"])
    return output


@step(
    enable_cache=False,
    enable_artifact_metadata=False,
    enable_artifact_visualization=False,
    experiment_tracker=False,
    step_operator=False,
    output_materializers=PathMaterializer,
)
def execute_moe(plan: dict) -> Annotated[Path, "bundle"]:
    """Materialize the actual output directory, not its ephemeral path string."""
    return run_job(plan)


@pipeline(
    name="moe_iceland",
    enable_cache=False,
    enable_artifact_metadata=False,
)
def moe_iceland(plan: dict) -> None:
    """Run exactly one checked arm/seed job without dynamic mapping or retries."""
    execute_moe(plan)
