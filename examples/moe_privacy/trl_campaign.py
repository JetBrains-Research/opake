"""Queue two native TRL controls without interrupting an existing private run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from examples.moe_privacy.campaign import run_stage, save

ARMS = ("trl_reference", "trl_reference_aux")
PARTITIONS = ("train", "validation", "test", "diagnostic")
NATIVE_DIFFERENCES = frozenset(
    (
        "name",
        "microbatch_size",
        "clipping_norm",
        "gradient_checkpointing",
        "router_aux_loss_coef",
    )
)
GATE_STEPS = 2
HUMANEVAL_TASKS = 164
POLL_SECONDS = 30


def read_json(path):
    """Read an execution receipt, not model or dataset contents."""
    return json.loads(path.read_text())


def validate_shared_config(config, reference):
    """Permit documented native-training differences, not a different task."""
    if config.keys() != reference.keys():
        message = "native and private configurations must declare the same fields"
        raise ValueError(message)
    differences = {
        key for key in config if config[key] != reference[key]
    } - NATIVE_DIFFERENCES
    if differences:
        message = f"unmatched comparison settings: {sorted(differences)}"
        raise ValueError(message)


def validate_result(summary, reference, config, arm):
    """Check real completion, trainable scope and all reserved data partitions."""
    if summary["status"] != "completed" or summary["arm"] != arm:
        message = "native training did not complete the requested arm"
        raise ValueError(message)
    if summary["config"] != config or summary["model_seed"] != reference["model_seed"]:
        message = "training configuration or initialization does not match"
        raise ValueError(message)
    privacy = summary["privacy"]
    if (
        privacy["private"] is not False
        or privacy["epsilon"] is not None
        or privacy["noise_multiplier"] != 0
        or privacy["load_release"] is not False
        or privacy["steps"] != config["steps"]
    ):
        message = "TRL controls must be complete, non-private and without releases"
        raise ValueError(message)
    for partition in PARTITIONS:
        digest = reference["data"][partition]["sha256"]
        if not digest or summary["data"][partition]["sha256"] != digest:
            message = f"unmatched {partition} data"
            raise ValueError(message)
    for key in (
        "trainable_parameters",
        "target_parameters",
        "geometry",
        "rank",
        "alpha",
    ):
        if summary["trainability"][key] != reference["trainability"][key]:
            message = f"unmatched trainability: {key}"
            raise ValueError(message)


def wait_for_previous(unit, previous_root, *, deadline):
    """Wait only; never stop, restart, or modify the predecessor's service."""
    while time.time() < deadline:
        result = subprocess.run(
            ["systemctl", "show", unit, "--property=ActiveState,SubState,Result"],
            check=True,
            capture_output=True,
            text=True,
        )
        state = dict(line.split("=", 1) for line in result.stdout.splitlines())
        if state.get("ActiveState") == "failed":
            message = "private follow-up failed; native controls were not started"
            raise RuntimeError(message)
        if state.get("ActiveState") == "inactive":
            receipt = previous_root / "campaign_state.json"
            if not receipt.is_file() or read_json(receipt).get("status") != "completed":
                message = "predecessor exited without a completed campaign receipt"
                raise RuntimeError(message)
            return
        if state.get("ActiveState") not in ("active", "activating", "deactivating"):
            message = f"unexpected predecessor state: {state}"
            raise RuntimeError(message)
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.time())))
    message = "deadline reached while waiting; no native training started"
    raise TimeoutError(message)


def write_comparison(root, reference_root, previous_root):
    """Keep task-quality comparisons descriptive, including unscored runs."""
    baseline = read_json(reference_root / "main/seed-0/dp/summary.json")
    baseline_code = read_json(reference_root / "main/seed-0/dp/code_metrics.json")[
        "humaneval"
    ]
    locations = (
        [
            (arm, reference_root / "main/seed-0" / arm)
            for arm in ("reference", "reference_aux", "dp", "dp_aux")
        ]
        + [("dp_aux_coef0.2", previous_root / "dp_aux")]
        + [(arm, root / "results" / arm) for arm in ARMS]
    )
    rows = []
    for name, directory in locations:
        path = directory / "summary.json"
        row = {"variant": name, "status": "pending", "code_status": "pending"}
        if path.is_file():
            summary = read_json(path)
            if summary["status"] != "completed":
                message = f"incomplete training summary: {name}"
                raise ValueError(message)
            for partition in PARTITIONS:
                if (
                    summary["data"][partition]["sha256"]
                    != baseline["data"][partition]["sha256"]
                ):
                    message = f"comparison data mismatch: {name}/{partition}"
                    raise ValueError(message)
            measurements = summary["measurements"]
            row.update(
                status="completed",
                test_nll=measurements["test_nll"],
                test_token_accuracy=measurements["test_token_accuracy"],
                training_hours=summary["training_seconds"] / 3600,
                epsilon=summary["privacy"]["epsilon"],
                balancing=summary["privacy"].get("balancing", "off"),
                coefficient=summary["config"]["router_aux_loss_coef"]
                if "aux" in name
                else 0,
            )
            code_path = directory / "code_metrics.json"
            if code_path.is_file():
                code = read_json(code_path)["humaneval"]
                if (
                    code["smoke"]
                    or code["task_count"] != HUMANEVAL_TASKS
                    or code["protocol_sha256"] != baseline_code["protocol_sha256"]
                ):
                    message = f"unmatched executable-code protocol: {name}"
                    raise ValueError(message)
                row.update(
                    code_status="completed", humaneval_plus_pass1=code["plus_pass1"]
                )
            receipt = directory / "wandb_run.json"
            if receipt.is_file():
                row["wandb_url"] = read_json(receipt)["url"]
        rows.append(row)
    report = {
        "rows": rows,
        "scope": "Single initialization; reused public benchmark; descriptive, not a significance claim.",
        "comparison_limits": (
            "Native TRL uses shuffled fixed-size batches, averaged physical-batch token means and batch-gradient "
            "clipping. Opaque uses Poisson sampling, per-record task loss and clipping. The native "
            "controls are standard-SFT quality references, not an isolation of privacy noise."
        ),
    }
    save(root / "comparison.json", report)
    lines = [
        "### Task-quality comparison",
        "",
        "| Run | Training | HumanEval+ pass@1 | Test NLL | Token accuracy | Training hours |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        score = row.get("humaneval_plus_pass1")
        values = (
            f"{100 * score:.2f}%" if score is not None else "pending",
            f"{row['test_nll']:.4f}" if "test_nll" in row else "pending",
            f"{100 * row['test_token_accuracy']:.2f}%"
            if "test_token_accuracy" in row
            else "pending",
            f"{row['training_hours']:.2f}" if "training_hours" in row else "pending",
        )
        lines.append(
            f"| {row['variant']} | {row['status']} | " + " | ".join(values) + " |"
        )
    lines.extend(("", report["scope"], "", report["comparison_limits"], ""))
    (root / "comparison.md").write_text("\n".join(lines))
    return report


def run_campaign(args):
    """Train both controls before scoring, inside the already-authorized cutoff."""
    root = args.root
    manifest = read_json(args.reference_root / "campaign.json")
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    config = read_json(args.config)
    reference = read_json(args.reference_root / "main/seed-0/dp/summary.json")
    validate_shared_config(config, reference["config"])
    if time.time() >= deadline:
        message = "campaign authorization deadline has already passed"
        raise TimeoutError(message)
    root.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "queued",
        "arms": list(ARMS),
        "runs_per_arm": 1,
        "config": config,
        "predecessor": args.wait_for_unit,
        "deadline_utc": manifest["deadline_utc"],
        "queued_utc": datetime.now(UTC).isoformat(),
        "group": args.group,
    }
    with (root / "queue.json").open("x") as stream:
        json.dump(queue, stream, indent=2)
    (root / "logs").mkdir()
    save(
        root / "campaign_state.json",
        {"status": "queued", "stage": "waiting_for_private_followup"},
    )
    try:
        wait_for_previous(args.wait_for_unit, args.previous_root, deadline=deadline)
        gate_config = dict(config, steps=GATE_STEPS, eval_every=GATE_STEPS)
        gate_path = root / "gate-config.json"
        save(gate_path, gate_config)

        def train(arm, config_path, output, *, gate=False):
            command = [
                sys.executable,
                "-B",
                "-m",
                "examples.moe_privacy.trl_run",
                "--config",
                str(config_path),
                "--arm",
                arm,
                "--seed",
                "0",
                "--output-dir",
                str(output),
                "--wandb-mode",
                "disabled" if gate else "online",
                "--wandb-group",
                args.group,
                "--wandb-name",
                f"magicoder-{arm.replace('_', '-')}-s0-20260922",
            ]
            if gate:
                command.append("--development-only")
            run_stage(
                command,
                root=root,
                name="native-gpu-gate" if gate else f"{arm}-train",
                deadline=deadline,
            )
            result = read_json(output / "summary.json")
            validate_result(result, reference, gate_config if gate else config, arm)

        train("trl_reference_aux", gate_path, root / "gate", gate=True)
        if read_json(root / "gate/summary.json")["test_evaluated"]:
            message = "engineering gate must not evaluate the final test set"
            raise ValueError(message)
        for arm in ARMS:
            train(arm, args.config, root / "results" / arm)
            write_comparison(root, args.reference_root, args.previous_root)
        for arm in ARMS:
            run_stage(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "examples.moe_privacy.campaign",
                    "evaluate",
                    "--config",
                    str(args.config),
                    "--arm",
                    arm,
                    "--seed",
                    "0",
                    "--output-dir",
                    str(root / "results" / arm),
                    "--image",
                    manifest["image"],
                    "--benchmark-dir",
                    str(args.reference_root.parent / "evaluation/benchmarks"),
                    "--benchmarks",
                    "humaneval",
                    "--max-new-tokens",
                    "512",
                    "--group",
                    args.group,
                    "--wandb-mode",
                    "online",
                    "--docker-sudo",
                ],
                root=root,
                name=f"{arm}-code",
                deadline=deadline,
            )
            write_comparison(root, args.reference_root, args.previous_root)
        save(
            root / "campaign_state.json",
            {"status": "completed", "training_runs": len(ARMS)},
        )
    except Exception as error:
        state = read_json(root / "campaign_state.json")
        state.update(status="failed", error=str(error))
        save(root / "campaign_state.json", state)
        raise


def main():
    """Queue controls behind a known service; no shutdown policy is changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference-root", "previous-root", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--wait-for-unit", required=True)
    parser.add_argument("--group", required=True)
    run_campaign(parser.parse_args())


if __name__ == "__main__":
    main()
