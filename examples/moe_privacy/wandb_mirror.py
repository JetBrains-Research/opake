"""Backfill/follow public MoE JSON logs without importing or restarting training."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from examples.moe_privacy import ARMS
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    _write_json,
    add_tracking_arguments,
    options_from_args,
    public_metrics,
)


def metric_rows(path: Path) -> list[dict]:
    """Ignore an unfinished final write; complete malformed rows are errors."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


def campaign_progress(path: Path, arms: tuple[str, ...]) -> dict[str, list[dict]]:
    """The serial runner emits completed/arm before the next arm's progress."""
    progress = {arm: [] for arm in arms}
    current = 0
    if not path.exists():
        return progress
    for line in path.read_text().splitlines(keepends=True):
        if not line.startswith("{") or not line.endswith("\n"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "completed":
            arm = event.get("arm")
            if current >= len(arms) or arm != arms[current]:
                message = "campaign completion order differs from configured arms"
                raise ValueError(message)
            current += 1
        elif event.get("event") == "optimizer_step" and current < len(arms):
            progress[arms[current]].append(event)
    return progress


class ArmMirror:
    """Persist cursors locally and reuse a stable W&B run on a mirror restart."""

    def __init__(self, output_dir: Path, tracker: ExperimentTracker):
        self.output_dir = output_dir
        self.tracker = tracker
        self.state_path = output_dir / "wandb_mirror_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"progress": 0, "metrics": {}, "completed": False}
        )
        self.started = False

    @property
    def completed(self):
        return self.state["completed"]

    def poll(self, progress: list[dict]) -> None:
        if self.completed or not self.output_dir.is_dir():
            return
        if not self.started:
            self.tracker.start()
            self.started = True
        rows = metric_rows(self.output_dir / "metrics.jsonl")
        events = [(row["step"], 1, row) for row in rows]
        events += [(row["step"], 0, row) for row in progress]
        for step, kind, row in sorted(events, key=lambda item: item[:2]):
            if kind == 0:
                if step > self.state["progress"]:
                    self.tracker.log_progress(step, row["steps"], row["seconds"])
                    self.state["progress"] = step
                continue
            values = public_metrics(row)
            seen = self.state["metrics"].setdefault(str(step), {})
            changed = {
                key: value for key, value in values.items() if seen.get(key) != value
            }
            if changed:
                delta = {"step": step}
                for key, value in changed.items():
                    original = (
                        "elapsed_seconds"
                        if key == "runtime/evaluation_elapsed_seconds"
                        else "eval_" + key.removeprefix("eval/")
                    )
                    delta[original] = value
                self.tracker.log_metrics(delta)
                seen.update(changed)
        summary_path = self.output_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = None
            if summary is not None:
                if summary.get("arm") != self.tracker.config["arm"]:
                    message = "summary arm does not match the mirrored run"
                    raise ValueError(message)
                self.tracker.complete(summary)
                self.tracker.__exit__(None, None, None)
                self.started = False
                self.state["completed"] = True
        _write_json(self.state_path, self.state)

    def close(self, error: BaseException | None = None):
        if self.started:
            self.tracker.__exit__(type(error) if error else None, error, None)
            self.started = False


def training_active(unit: str) -> bool:
    """Inspect the campaign without changing its service or processes."""
    result = subprocess.run(
        ["systemctl", "show", unit, "--property=ActiveState", "--value"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() in {"active", "activating", "reloading"}


def main():
    """Backfill existing logs and optionally follow the rest of a serial campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--campaign-dir", type=Path)
    location.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--experiment",
        choices=("synthetic_moe", "pretrained_moe_sft"),
        default="pretrained_moe_sft",
    )
    parser.add_argument("--campaign-log", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-seconds", type=float, default=86400)
    parser.add_argument(
        "--training-unit",
        help="Stop if this systemd campaign exits without all summaries",
    )
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.run_dir and not args.arm:
        parser.error("--run-dir requires --arm")
    if args.campaign_dir and args.arm:
        parser.error("a campaign mirrors all three arms; --arm is only for --run-dir")
    if not 0 < args.poll_seconds <= args.max_seconds < float("inf"):
        parser.error("require finite 0 < poll-seconds <= max-seconds")
    options = options_from_args(args)
    if options.mode not in {"online", "offline"}:
        parser.error(
            "an external mirror requires explicit --wandb-mode online or offline"
        )
    root = args.campaign_dir or args.run_dir
    if not root.is_dir():
        parser.error("result directory must already exist")
    config = json.loads(args.config.read_text())
    arms = ARMS if args.campaign_dir else (args.arm,)
    log = args.campaign_log or root / "campaign.log"
    with (root / ".wandb-mirror.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mirrors = []
        for arm in arms:
            directory = root / arm if args.campaign_dir else root
            tracker = ExperimentTracker(
                directory,
                config=config,
                arm=arm,
                seed=args.seed,
                experiment=args.experiment,
                options=options,
                resume=True,
            )
            mirrors.append(ArmMirror(directory, tracker))
        deadline = time.monotonic() + args.max_seconds
        error = None
        try:
            while True:
                progress = campaign_progress(log, arms)
                for arm, mirror in zip(arms, mirrors, strict=True):
                    mirror.poll(progress[arm])
                if all(mirror.completed for mirror in mirrors) or not args.follow:
                    break
                if args.training_unit and not training_active(args.training_unit):
                    message = "training stopped before all experiment summaries were completed"
                    raise RuntimeError(message)
                if time.monotonic() >= deadline:
                    message = "mirror deadline reached; training was not modified"
                    raise TimeoutError(message)
                time.sleep(args.poll_seconds)
        except BaseException as caught:
            error = caught
            raise
        finally:
            for mirror in mirrors:
                mirror.close(error)


if __name__ == "__main__":
    main()
