"""Summarize the host-overhead lever runs (kernel-targets report).

Reads runs/lv/<name>.{csv,log,smi} copied to a local directory. Mean step
time and throughput come from the per-step memory-profile CSV (all six steps;
batches must match the baseline), peak memory from the same CSV, GPU
utilization from nvidia-smi samples while memory was within 10% of its peak.

Usage: python experiments/kernel_targets/summarize_levers.py notes/profiles/lv [baseline]
"""

from __future__ import annotations

import csv
import re
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "notes/profiles/lv")
BASE = sys.argv[2] if len(sys.argv) > 2 else "lv_base2"
LABELS = {
    "lv_base": "baseline (eager, mb=2), first run",
    "lv_base2": "baseline (eager, mb=2), clean rerun",
    "lv_mb4": "microbatch 4 [overlapped a CPU microbenchmark]",
    "lv_mb4b": "microbatch 4, clean rerun",
    "lv_tf32": "TF32 for fp32 matmuls",
    "lv_nopeft": "stock PEFT LoRA (Opake PEFT kernels off)",
    "lv_comp": "torch.compile (default), clip auto",
    "lv_compro": "torch.compile reduce-overhead, clip auto",
    "lv_kcomp": "torch.compile + model kernels, clip auto",
    "lv_gc_mb4": "checkpointing, microbatch 4",
    "lv_gc_mb8": "checkpointing, microbatch 8",
    "lv_gc_mb16": "checkpointing, microbatch 16",
    "lv_np_mb4": "stock PEFT LoRA + microbatch 4",
    "lv_np_gc_mb8": "stock PEFT LoRA + checkpointing, microbatch 8",
    "lv_np_comp": "stock PEFT LoRA + torch.compile (default)",
    "lv_np_compro": "stock PEFT LoRA + torch.compile reduce-overhead",
    "lv_tf32_mb4": "TF32 + microbatch 4",
    "lv_np_tf32_mb4": "stock PEFT LoRA + TF32 + microbatch 4",
    "lv_kern_mb4": "Opake model kernels on + microbatch 4",
}


def load(name):
    p = D / f"{name}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    f = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    return {"batches": [int(b) for b in f("batch")], "steps": f("step_time_sec"),
            "step": st.mean(f("step_time_sec")), "clip": st.mean(f("clip_sec")),
            "noise": st.mean(f("noise_sec")), "opt": st.mean(f("optimizer_sec")),
            "peak": max(f("memory_peak_gb")), "sps": sum(f("batch")) / sum(f("step_time_sec"))}


def util(name):
    p = D / f"{name}.smi"
    if not p.exists():
        return float("nan")
    rows = [tuple(map(int, r.split(","))) for r in p.read_text().splitlines() if r.strip()]
    peak = max(m for _, m in rows)
    return st.mean(u for u, m in rows if m >= 0.9 * peak)


def status(name):
    p = D / f"{name}.log"
    if not p.exists():
        return "missing"
    t = p.read_text(errors="replace")
    for pat in ("OutOfMemory", "out of memory", "Traceback"):
        if pat in t:
            m = re.findall(r"^\w*Error[^\n]*", t, re.M)
            return f"failed ({m[-1][:100] if m else pat})"
    return "ok"


base = load(BASE)
print(f"baseline {BASE}: batches {base['batches']}")
print("| variant | status | mean step | clip phase | noise | peak | throughput | GPU util | vs baseline | per-step |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, label in LABELS.items():
    r = load(name)
    if r is None:
        st_ = status(name)
        if st_ == "ok":
            st_ = "no step data (stopped early)"
        print(f"| {label} | {st_} | | | | | | | | |")
        continue
    if "comp" in name:  # compiled runs: steps 0-2 include compilation
        tail = r["steps"][3:]
        print(f"| {label} | {status(name)} | steps 4-6: {' / '.join(f'{x:.1f}' for x in tail)} s "
              f"(steps 1-3 incl. compile: {' / '.join(f'{x:.0f}' for x in r['steps'][:3])} s) | | | {r['peak']:.2f} GiB | | "
              f"{util(name):.0f}% | steady state x{base['step'] / (sum(tail) / len(tail)):.2f} | |")
        continue
    same = r["batches"] == base["batches"]
    ratios = [a / b for a, b in zip(base["steps"], r["steps"], strict=True)] if same else []
    rng = f"x{min(ratios):.2f}-{max(ratios):.2f}, {sum(x > 1 for x in ratios)}/{len(ratios)}" if same else "batches differ"
    print(f"| {label} | {status(name)} | {r['step']:.2f} s | {r['clip']:.2f} s | {r['noise']:.2f} s | {r['peak']:.2f} GiB | "
          f"{r['sps']:.2f} smp/s | {util(name):.0f}% | x{base['step'] / r['step']:.2f} ({(r['step'] / base['step'] - 1) * 100:+.1f}%) | {rng} |")
