"""Summarize the CUDA-graph chunk runs (cg_driver.sh).

Mean step time and per-step ratios come from the memory-profile CSVs
(identical batches checked against the baseline). Steps that contain a
capture (their log prints "[cudagraph] captured") are reported separately:
"steady" means the capture-free steps. Peak memory is nvidia-smi's maximum
memory.used, because allocator statistics miss CUDA-graph pool memory.

Usage: python experiments/cuda_graph/summarize_cg.py notes/profiles/cg
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "notes/profiles/cg")
RUNS = [
    ("base_mb2", "eager, fused triton clip, mb 2", None),
    ("cg_mb2", "CUDA graph, torch clip, mb 2", "base_mb2"),
    ("cg_kern_mb2", "CUDA graph + Opake model kernels, mb 2", "base_mb2"),
    ("cg_mb4", "CUDA graph, mb 4", "base_mb2"),
    ("base_best", "eager best: stock PEFT + TF32 + triton clip, mb 4", "base_mb2"),
    ("cg_best", "CUDA graph + stock PEFT + TF32, mb 4", "base_best"),
    ("cg_kern_tf32_mb4", "CUDA graph + model kernels + TF32, mb 4", "base_best"),
]


def load(name):
    p = D / f"{name}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    steps = [float(r["step_time_sec"]) for r in rows]
    log = (D / f"{name}.log").read_text(errors="replace").replace("\r", "\n")
    smi = [tuple(map(int, r.split(","))) for r in (D / f"{name}.smi").read_text().splitlines() if r.strip()]
    return {"batches": [int(r["batch"]) for r in rows], "steps": steps,
            "noise": st.mean(float(r["noise_sec"]) for r in rows),
            "smi_peak_gib": max(m for _, m in smi) / 1024, "log": log}


R = {n: load(n) for n, _, _ in RUNS}
mbs = {"base_mb2": 2, "cg_mb2": 2, "cg_kern_mb2": 2, "cg_mb4": 4, "base_best": 4, "cg_best": 4, "cg_kern_tf32_mb4": 4}
print("| variant | status | mean step (all 6) | steady steps (no capture) | vs ref (steady, per step) | noise | nvidia-smi peak |")
print("|---|---|---:|---:|---:|---:|---:|")
for n, label, ref in RUNS:
    r = R[n]
    if r is None:
        print(f"| {label} | missing | | | | | |")
        continue
    caps = r["log"].count("[cudagraph] captured")
    cap_rows = set()
    if caps:
        cap_rows.add(0)
        partial = [i for i, b in enumerate(r["batches"]) if b % mbs[n]]
        if caps >= 2 and partial:
            cap_rows.add(partial[0])
    steady = [t for i, t in enumerate(r["steps"]) if i not in cap_rows]
    cmp = ""
    if ref and R.get(ref):
        b = R[ref]
        assert b["batches"] == r["batches"], f"batches differ: {n} vs {ref}"
        idx = [i for i in range(len(r["steps"])) if i not in cap_rows]
        ratios = [b["steps"][i] / r["steps"][i] for i in idx]
        base_steady = st.mean(b["steps"][i] for i in idx)
        cmp = (f"x{base_steady / st.mean(steady):.2f} ({(st.mean(steady) / base_steady - 1) * 100:+.1f}%), "
               f"per step x{min(ratios):.2f}-{max(ratios):.2f}")
    status = "ok" if "Traceback" not in r["log"] else "failed"
    print(f"| {label} | {status}, captures {caps} (rows {sorted(cap_rows)}) | {st.mean(r['steps']):.2f} s | "
          f"{st.mean(steady):.2f} s | {cmp} | {r['noise']:.2f} s | {r['smi_peak_gib']:.1f} GiB |")
print("batches:", R["base_mb2"]["batches"] if R["base_mb2"] else None)
