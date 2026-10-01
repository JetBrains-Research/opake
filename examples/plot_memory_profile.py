"""Plot per-phase memory inside a training step from a --memory-profile run.

Reads the files ``train_dpsgd.py --memory-profile PREFIX`` writes and produces
a figure with two views of the same data:

    top     per-phase peak memory per step (bars), from PREFIX.csv
    bottom  sampled live-bytes-over-time inside each recorded step (lines),
            from PREFIX.timeline.csv, with phase transitions marked

The bottom view is the one that answers "where inside the step": ``grad_fn``
is a single call, so peak-per-phase alone cannot separate the forward, the
per-example backward, and the clipping that follows them. The timeline can,
because it samples continuously while the step runs.

Usage:
    python examples/plot_memory_profile.py PREFIX
    python examples/plot_memory_profile.py runs/q7b_mb2 --steps 4 5 6

Needs matplotlib only - it reads files, it does not need a GPU.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GIB = 1024**3
# Whole-step columns that end in _peak_gb but are not a phase name.
NOT_A_PHASE = ("memory_peak_gb",)


def _read_timeline(path: Path) -> dict[int, list[tuple[float, str, float]]]:
    """Group sampled rows as ``{step: [(t_ms, phase, live_GiB), ...]}``."""
    steps: dict[int, list[tuple[float, str, float]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            steps.setdefault(int(row["step"]), []).append(
                (float(row["t_ms"]), row["phase"], int(row["alloc_bytes"]) / GIB)
            )
    return steps


def _phase_breaks(samples: list[tuple[float, str, float]]) -> list[tuple[str, float]]:
    """``(entering_phase, t_ms)`` where the sampled phase label changes."""
    breaks: list[tuple[str, float]] = []
    previous = samples[0][1] if samples else None
    for t, phase, _gb in samples:
        if phase != previous:
            breaks.append((phase, t))
            previous = phase
    return breaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", help="PREFIX used for --memory-profile")
    ap.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="Restrict the timeline panel to these step numbers",
    )
    args = ap.parse_args()

    prefix = Path(args.prefix)
    csv_path = prefix.with_suffix(".csv")
    timeline_path = Path(f"{prefix}.timeline.csv")
    if not csv_path.exists():
        raise SystemExit(f"no profile CSV at {csv_path}; run with --memory-profile")

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    phases = sorted(
        {
            key[: -len("_peak_gb")]
            for key in rows[0]
            if key.endswith("_peak_gb") and key not in NOT_A_PHASE
        }
    )

    steps: dict[int, list[tuple[float, str, float]]] = {}
    if timeline_path.exists():
        steps = _read_timeline(timeline_path)
        if args.steps:
            steps = {
                step: samples
                for step, samples in steps.items()
                if step in args.steps
            }

    ncols = max(1, len(steps))
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.25])
    ax_peak = fig.add_subplot(grid[0, :])

    # --- per-phase peak memory, grouped bars per step ----------------------
    step_ids = [int(row["step"]) for row in rows]
    width = 0.8 / max(1, len(phases))
    for idx, phase in enumerate(phases):
        ax_peak.bar(
            [step + idx * width for step in step_ids],
            [float(row[f"{phase}_peak_gb"]) for row in rows],
            width=width,
            label=phase,
        )
    ax_peak.set_xlabel("optimizer step")
    ax_peak.set_ylabel("peak allocated (GiB)")
    ax_peak.set_title("Per-phase peak memory")
    ax_peak.legend(fontsize=8)

    # --- sampled live bytes inside each recorded step ----------------------
    if not steps:
        ax = fig.add_subplot(grid[1, :])
        ax.set_axis_off()
        ax.text(
            0.02,
            0.5,
            f"No timeline samples in {timeline_path.name}",
            fontsize=9,
        )
    else:
        # One panel per step: overlaying them hides the shape, and steps differ
        # in length, so a shared x axis would compare unrelated moments.
        ymax = max(point[2] for samples in steps.values() for point in samples)
        for col, (step, samples) in enumerate(sorted(steps.items())):
            ax = fig.add_subplot(grid[1, col])
            ax.plot(
                [point[0] for point in samples],
                [point[2] for point in samples],
                lw=0.8,
                color="tab:blue",
            )
            for phase, t_ms in _phase_breaks(samples):
                ax.axvline(t_ms, lw=0.7, ls=":", color="grey")
                ax.text(
                    t_ms,
                    ymax * 0.98,
                    f" {phase}",
                    fontsize=6,
                    color="grey",
                    rotation=90,
                    va="top",
                )
            ax.set_title(f"step {step}", fontsize=9)
            ax.set_xlabel("ms since step start", fontsize=8)
            ax.set_ylabel("live GiB", fontsize=8)
            ax.set_ylim(0, ymax * 1.05)
            ax.set_xlim(left=0)
            ax.tick_params(labelsize=7)
        fig.suptitle(
            "Bottom: live bytes within the step, 1 ms sampler. The floor is "
            "weights + optimizer state; the bumps are per-step work.",
            fontsize=9,
        )

    out = prefix.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
