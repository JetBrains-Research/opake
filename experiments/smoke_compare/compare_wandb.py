"""Compare the opake smoke variants with the 2026-09-25 ckpt-sweep runs on W&B.

Reads grazie-ml/next-edit-prediction-trl-fed-dp on jetbrains.wandb.io. The
baseline is the sweep's ``baseline`` run (opaque 0.15.6rc1); other sweep
variants are listed for context. New runs are found by model-ID prefix
(``nes-opake-dp-7B-cmp-<variant>``); the newest run per variant is used.

Per run: step times (paired with the baseline by step; batches must match),
phase times, peak memory, train/eval loss, epsilon, clipping norm and rate.
Writes ``notes/smoke-compare/comparison.json`` and prints a markdown table.

Usage (W&B key from the macOS keychain item ``wandb.api_key``):
  WANDB_API_KEY=$(security find-generic-password -s wandb.api_key -w) \
  WANDB_BASE_URL=https://jetbrains.wandb.io uv run --with wandb \
  python experiments/smoke_compare/compare_wandb.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import wandb

PROJECT = "grazie-ml/next-edit-prediction-trl-fed-dp"
BASELINE = "ada8c57faf8d44cea8991ed4022aa385"
SWEEP = {
    "5be328ba41d34626a2aa59d156ef9b88": "sweep: checkpointing off",
    "e6f58a8064e74c93ba7ebd9136b476a0": "sweep: off + kernels",
    "067ee19e9fae46b59bff1f6bc2e5d1fe": "sweep: off + kernels + mlp-save",
}
VARIANTS = {
    "a": "opake A: baseline replica",
    "b": "opake B: + fused clip",
    "c": "opake C: + CUDA graphs",
}
def load(run):
    full = [r for r in run.scan_history() if r.get("train/step_time_sec") is not None]
    rows = sorted(full, key=lambda r: r.get("_step", 0))
    s = run.summary
    return {
        "id": run.id, "name": run.name, "state": run.state, "created": run.created_at,
        "steps": [r.get("train/step_time_sec") for r in rows],
        "batches": [r.get("train/batch_size") for r in rows],
        "clip": [r.get("train/clip_sec") for r in rows],
        "noise": [r.get("train/noise_sec") for r in rows],
        "optimizer": [r.get("train/optimizer_sec") for r in rows],
        "peak_gb": max((r.get("train/memory_peak_gb") or 0) for r in rows) if rows else None,
        "loss": [r.get("train/loss") for r in rows],
        "epsilon": s.get("train/privacy_epsilon"),
        "clipping_norm": [r.get("train/privacy_clipping_norm") for r in rows],
        "clip_rate": [r.get("train/privacy_clip_rate") for r in rows],
        "eval_loss": s.get("eval/loss"),
        "train_runtime": s.get("train_runtime"),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


api = wandb.Api(timeout=90)
base = load(api.run(f"{PROJECT}/{BASELINE}"))
runs = {"baseline (opaque 0.15.6rc1)": base}
for rid, label in SWEEP.items():
    runs[label] = load(api.run(f"{PROJECT}/{rid}"))
for v, label in VARIANTS.items():
    found = api.runs(PROJECT, filters={"display_name": {"$regex": f"^nes-opake-dp-7B-cmp-{v}-"}}, order="-created_at", per_page=5)
    found = list(found)
    if found:
        runs[label] = load(found[0])

out = Path("notes/smoke-compare/comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(runs, indent=1, default=str))

def per_sample(r, skip_first=True):
    """Seconds per sample over the steps (optionally excluding step 1: warmup/capture)."""
    pairs = [(t, b) for t, b in zip(r["steps"], r["batches"], strict=False) if t and b]
    if skip_first:
        pairs = pairs[1:]
    return sum(t for t, _ in pairs) / sum(b for _, b in pairs) if pairs else float("nan")


print(f"baseline batches: {base['batches']}")
print("| run | state | batches | steps (s) | s/sample (steps 2+) | vs baseline (s/sample) | samples/s | clip / noise / optimizer (mean s) | peak | eval loss | epsilon |")
print("|---|---|---|---|---:|---:|---:|---|---:|---:|---:|")
b_ps = per_sample(base)
for label, r in runs.items():
    ps = per_sample(r)
    print(f"| {label} | {r['state']} | {r['batches']} | {' / '.join(f'{x:.1f}' for x in r['steps'])} | {ps:.3f} | "
          f"x{b_ps / ps:.2f} | {1 / ps:.2f} | {mean(r['clip']):.1f} / {mean(r['noise']):.2f} / {mean(r['optimizer']):.2f} | "
          f"{r['peak_gb'] or float('nan'):.1f} GB | {r['eval_loss']} | {r['epsilon']} |")
print()
print("Paired by step (only meaningful if batches match):")
print("| run | state | steps (s) | mean step | vs baseline (per step) | clip / noise / optimizer (mean s) | peak | final train loss | eval loss | epsilon | train_runtime |")
print("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
for label, r in runs.items():
    same = r["batches"] == base["batches"]
    n = min(len(r["steps"]), len(base["steps"]))
    ratios = [base["steps"][i] / r["steps"][i] for i in range(n) if r["steps"][i]]
    vs = (f"x{mean(base['steps'][:n]) / mean(r['steps'][:n]):.2f} (x{min(ratios):.2f}-{max(ratios):.2f})" if ratios else "") + ("" if same else " [batches differ]")
    print(f"| {label} | {r['state']} | {' / '.join(f'{x:.1f}' for x in r['steps'] if x)} | {mean(r['steps']):.1f} | {vs} | "
          f"{mean(r['clip']):.1f} / {mean(r['noise']):.2f} / {mean(r['optimizer']):.2f} | {r['peak_gb'] or float('nan'):.1f} GB | "
          f"{(r['loss'] or [None])[-1]} | {r['eval_loss']} | {r['epsilon']} | {r['train_runtime']} |")
print(f"\nwritten {out}")
