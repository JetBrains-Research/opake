"""Summarize the multi-tensor / vmap(chunk_size) runs (report §12).

Input directory holds mt_*.{csv,log,smi} copied from the workspace's runs/.
Naming: mt_<kernels>_<engine>_<batching>, engine pl = packaged per-leaf
triton, mt = multi-tensor + shared markers; batching mb = Python
microbatching (2), ch = vmap(chunk_size=2) with one clip call per step.

Usage: python experiments/clip_fused/summarize_mt_runs.py notes/profiles/mt
"""

from __future__ import annotations

import ast
import csv
import re
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "notes/profiles/mt")


def prof(name):
    rows = list(csv.DictReader(open(D / f"{name}.csv")))
    f = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    return {"batches": [int(b) for b in f("batch")], "steps": f("step_time_sec"),
            "step": st.mean(f("step_time_sec")), "clip": st.mean(f("clip_sec")),
            "noise": st.mean(f("noise_sec")), "opt": st.mean(f("optimizer_sec")),
            "peak": max(f("memory_peak_gb")), "sps": sum(f("batch")) / sum(f("step_time_sec"))}


def util(name):
    rows = [tuple(map(int, r.split(","))) for r in (D / f"{name}.smi").read_text().splitlines() if r.strip()]
    peak = max(m for _, m in rows)
    act = [u for u, m in rows if m >= 0.9 * peak]
    return st.mean(act)


def log_info(name):
    text = (D / f"{name}.log").read_text().replace("\r", "\n")
    losses = [float(x) for x in re.findall(r"^Step\s+\d+ .*?Loss: ([\d.]+)", text, re.M)]
    thr = re.findall(r"Throughput: ([\d.]+) samples/s", text)
    times = [float(x) for x in re.findall(r"^Step\s+\d+ .*?Time: ([\d.]+)s", text, re.M)]
    return losses, (float(thr[-1]) if thr else float("nan")), times


def dt_steps(name):
    rows = []
    for line in (D / f"{name}.log").read_text().replace("\r", "\n").splitlines():
        if line.startswith("{'loss'"):
            d = ast.literal_eval(line)
            rows.append({k: float(d[k]) for k in ("loss", "batch_size", "step_time_sec", "memory_peak_gb")})
    return rows


LABEL = {"pl_mb": "per-leaf, microbatch loop", "mt_mb": "multi-tensor, microbatch loop",
         "pl_ch": "per-leaf, vmap(chunk_size)", "mt_ch": "multi-tensor, vmap(chunk_size)"}
for K, kname in (("eager", "eager (kernels off)"), ("kern", "Triton model kernels")):
    runs = {v: prof(f"mt_{K}_{v}") for v in LABEL}
    base = runs["pl_mb"]
    assert all(r["batches"] == base["batches"] for r in runs.values()), "batches differ"
    print(f"\n## train_dpsgd.py, {kname}, profiler on; batches {base['batches']}")
    print("| variant | mean step | clip phase | noise | optimizer | peak | throughput | GPU util | vs per-leaf+loop | per-step speedup |")
    for v, r in runs.items():
        ratios = [a / b for a, b in zip(base["steps"], r["steps"])]
        print(f"| {LABEL[v]} | {r['step']:.2f} s | {r['clip']:.2f} s | {r['noise']:.2f} s | {r['opt']:.2f} s | "
              f"{r['peak']:.2f} GiB | {r['sps']:.2f} smp/s | {util(f'mt_{K}_{v}'):.0f}% | "
              f"x{base['step'] / r['step']:.2f} ({(r['step'] / base['step'] - 1) * 100:+.1f}%) | "
              f"x{min(ratios):.2f}-{max(ratios):.2f}, faster {sum(x > 1 for x in ratios)}/{len(ratios)} |")
    for v in LABEL:
        print(f"   {v} losses (steps 2,4,6): {log_info(f'mt_{K}_{v}')[0]}")

print("\n## clean (eager, 8 steps, no profiler)")
for v in ("pl_mb", "mt_mb", "mt_ch"):
    _, thr, times = log_info(f"mt_clean_{v}")
    print(f"| {LABEL[v]} | {thr:.2f} smp/s | logged step times {times} | GPU util {util(f'mt_clean_{v}'):.0f}% |")

print("\n## DPTrainer (steps 1-6)")
dts = {v: dt_steps(f"mt_dt_{v}") for v in ("pl_mb", "mt_mb", "mt_ch")}
assert all([r["batch_size"] for r in x] == [r["batch_size"] for r in dts["pl_mb"]] for x in dts.values())
print("batches:", [int(r["batch_size"]) for r in dts["pl_mb"]])
b = st.mean(r["step_time_sec"] for r in dts["pl_mb"])
for v, rows in dts.items():
    m = st.mean(r["step_time_sec"] for r in rows)
    sps = sum(r["batch_size"] for r in rows) / sum(r["step_time_sec"] for r in rows)
    ratios = [x["step_time_sec"] / y["step_time_sec"] for x, y in zip(dts["pl_mb"], rows)]
    print(f"| {LABEL[v]} | {m:.2f} s | x{b / m:.2f} | per-step x{min(ratios):.2f}-{max(ratios):.2f} | "
          f"peak {max(r['memory_peak_gb'] for r in rows):.2f} GB | {sps:.2f} smp/s | GPU util {util(f'mt_dt_{v}'):.0f}% |")
    print("   losses:", " ".join(f"{r['loss']:.4f}" for r in rows))
