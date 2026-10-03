"""Compare per-phase memory across runs, to see WHY a variant did or did not save.

Two views over the ``--memory-profile`` output of ``train_dpsgd.py``:

    top     step peak per variant, split into the static floor (weights +
            optimizer state, present before any per-step work) and the
            per-step working set above it. Only the second part is something a
            memory optimization can remove.
    bottom  one live-bytes timeline per variant on a shared y scale, so a
            variant that changes nothing looks like what it is: identical.

Usage:
    python examples/plot_memory_compare.py notes/profiles \\
        "kernels, no ckpt:q7bE" "attention-only ckpt:q7b_attnckpt" \\
        "full-layer ckpt:q7bF" "eager, no kernels:q7b_eager" --step 3

Run it wherever you want to look at the picture; it reads files and needs no
GPU.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GIB = 1024**3
FLOOR_COLUMN = "noise_peak_gb"
PEAK_COLUMN = "clip_peak_gb"


def _load(prefix: str, roots: list[Path]) -> tuple[Path, Path]:
    for root in roots:
        timeline = root / f"{prefix}.timeline.csv"
        if timeline.exists():
            return root / f"{prefix}.csv", timeline
    raise SystemExit(f"no {prefix}.timeline.csv found in {[str(r) for r in roots]}")


def _read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _read_timeline(path: Path, step: int | None) -> list[tuple[float, float]]:
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if step is None or int(r["step"]) == step]
    return [(float(r["t_ms"]), int(r["alloc_bytes"]) / GIB) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", help="Directory holding the profile files")
    ap.add_argument("runs", nargs="+", help="LABEL:PREFIX pairs")
    ap.add_argument("--step", type=int, default=None, help="Step to plot per run")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()

    roots = [Path("."), Path(args.outdir)]
    entries = []
    for item in args.runs:
        label, sep, prefix = item.partition(":")
        if not sep:
            label = prefix = item
        csv_path, timeline_path = _load(prefix, roots)
        rows = _read_rows(csv_path)
        samples = _read_timeline(timeline_path, args.step)
        if not samples:
            raise SystemExit(f"{prefix}.timeline.csv has no samples for the request")
        floor = max(float(r[FLOOR_COLUMN]) for r in rows)
        peak = max(float(r[PEAK_COLUMN]) for r in rows)
        entries.append((label, floor, peak, samples))

    ncols = min(2, len(entries))
    nrows = -(-len(entries) // ncols)
    fig = plt.figure(figsize=(13, 4 + 3.4 * nrows))
    grid = fig.add_gridspec(1 + nrows, ncols, height_ratios=[1.0] + [1.5] * nrows)
    ax_peak = fig.add_subplot(grid[0, :])

    # --- peak per variant, split into floor and working set ----------------
    labels = [label for label, *_ in entries]
    ax_peak.barh(labels, [floor for _, floor, _, _ in entries], color="0.75")
    ax_peak.barh(
        labels,
        [peak - floor for _, floor, peak, _ in entries],
        left=[floor for _, floor, _, _ in entries],
        color="tab:red",
    )
    top = max(peak for _, _, peak, _ in entries)
    for index, (_, floor, peak, _) in enumerate(entries):
        ax_peak.annotate(
            f"peak {peak:.2f} = floor {floor:.2f} + working {peak - floor:.2f} GiB",
            xy=(peak, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax_peak.invert_yaxis()
    ax_peak.set_xlim(0, top * 1.62)
    ax_peak.set_xlabel("GiB")
    ax_peak.set_title(
        "Step peak per variant. Grey = static floor, red = per-step working "
        "set (only the red part is removable)",
        fontsize=10,
    )

    # --- live-bytes timelines, shared y so shapes are comparable ------------
    ymax = max(max(value for _, value in samples) for _, _, _, samples in entries)
    for index, (label, floor, _peak, samples) in enumerate(entries):
        ax = fig.add_subplot(grid[1 + index // ncols, index % ncols])
        ax.plot(*zip(*samples, strict=True), lw=0.8, color="tab:blue")
        ax.axhline(floor, color="0.5", lw=0.8, ls="--")
        ax.annotate(
            f"floor {floor:.1f}",
            xy=(samples[0][0], floor),
            xytext=(2, 3),
            textcoords="offset points",
            fontsize=7,
            color="0.4",
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("ms since step start", fontsize=8)
        ax.set_ylim(0, ymax * 1.06)
        ax.set_xlim(left=0)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Live bytes within one step (1 ms sampler), shared scale",
        y=0.995,
        fontsize=9,
    )

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.outdir) / "compare.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
