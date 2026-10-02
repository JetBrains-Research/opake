"""Summarize the clip-backend full-step runs (report §11).

Inputs (copied from the workspace's ``~/opaque/runs/``) in a directory:
  cb_dt_*.log           DPTrainer runs: per-step dicts with step_time_sec,
                        memory_peak_gb, batch_size, loss
  cb_{kern,eager}_*.csv train_dpsgd.py --memory-profile per-phase CSVs
  cb_*.log              train_dpsgd.py stdout (Step lines, Throughput)
  cb_*.smi              nvidia-smi utilization.gpu,memory.used at 1 Hz

Usage: python experiments/clip_fused/summarize_cb_runs.py notes/profiles/cb
"""

from __future__ import annotations

import ast
import csv
import re
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "notes/profiles/cb")


def dt_steps(name):
    rows = []
    for line in (D / f"{name}.log").read_text().replace("\r", "\n").splitlines():
        if line.startswith("{'loss'"):
            d = ast.literal_eval(line)
            rows.append({k: float(d[k]) for k in ("loss", "batch_size", "step_time_sec", "memory_peak_gb")})
    return rows


def smi_util(name):
    path = D / f"{name}.smi"
    if not path.exists():
        return float("nan")
    rows = [tuple(map(int, r.split(","))) for r in path.read_text().splitlines() if r.strip()]
    peak = max(m for _, m in rows)
    active = [u for u, m in rows if m >= 0.9 * peak]  # same rule for every run
    return st.mean(active) if active else float("nan")


def td_profile(name):
    rows = list(csv.DictReader(open(D / f"{name}.csv")))
    f = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    steps, batch = f("step_time_sec"), f("batch")
    return {
        "batches": [int(b) for b in batch],
        "step": st.mean(steps),
        "clip": st.mean(f("clip_sec")),
        "noise": st.mean(f("noise_sec")),
        "opt": st.mean(f("optimizer_sec")),
        "peak": max(f("memory_peak_gb")),
        "sps": sum(batch) / sum(steps),
    }


def td_log(name):
    text = (D / f"{name}.log").read_text().replace("\r", "\n")
    steps = re.findall(r"^Step\s+(\d+) .*?BS: (\d+) \| Loss: ([\d.]+).*?Time: ([\d.]+)s", text, re.M)
    thr = re.findall(r"Throughput: ([\d.]+) samples/s", text)
    return steps, (float(thr[-1]) if thr else float("nan"))


def pct(new, old):
    return f"{(new / old - 1) * 100:+.1f}%"


print("## DPTrainer (train_dpsgd_trainer.py), steps 1-6")
dt = {n: dt_steps(n) for n in ("cb_dt_torch", "cb_dt_triton", "cb_dt_save", "cb_dt_triton_save")}
base = dt["cb_dt_torch"]
assert all([r["batch_size"] for r in v] == [r["batch_size"] for r in base] for v in dt.values()), "batches differ"
print("batches:", [int(r["batch_size"]) for r in base], "(identical in all four runs)")
print("| run | mean step | step peak | throughput | GPU util | vs torch |")
for n, v in dt.items():
    mean = st.mean(r["step_time_sec"] for r in v)
    sps = sum(r["batch_size"] for r in v) / sum(r["step_time_sec"] for r in v)
    peak = max(r["memory_peak_gb"] for r in v)
    bmean = st.mean(r["step_time_sec"] for r in base)
    print(f"| {n} | {mean:.2f} s | {peak:.2f} GB | {sps:.2f} smp/s | {smi_util(n):.0f}% | "
          f"{pct(mean, bmean)} step, x{bmean / mean:.2f} |")
    print("   per-step s:", " ".join(f"{r['step_time_sec']:.2f}" for r in v),
          "| loss:", " ".join(f"{r['loss']:.4f}" for r in v))

print("\n## train_dpsgd.py --memory-profile (6 steps, profiler on)")
td = {n: td_profile(n) for n in ("cb_kern_torch", "cb_kern_triton", "cb_eager_torch", "cb_eager_triton")}
assert all(v["batches"] == td["cb_kern_torch"]["batches"] for v in td.values()), "batches differ"
print("batches:", td["cb_kern_torch"]["batches"], "(identical in all four runs)")
print("| run | mean step | clip | noise | optimizer | step peak | throughput | GPU util |")
for n, v in td.items():
    print(f"| {n} | {v['step']:.2f} s | {v['clip']:.2f} s | {v['noise']:.2f} s | {v['opt']:.2f} s | "
          f"{v['peak']:.2f} GiB | {v['sps']:.2f} smp/s | {smi_util(n):.0f}% |")
for a, b in (("cb_kern_torch", "cb_kern_triton"), ("cb_eager_torch", "cb_eager_triton"),
             ("cb_kern_torch", "cb_eager_triton")):
    print(f"  {b} vs {a}: step {pct(td[b]['step'], td[a]['step'])} (x{td[a]['step'] / td[b]['step']:.2f}), "
          f"clip phase {td[b]['clip'] - td[a]['clip']:+.2f} s, throughput {pct(td[b]['sps'], td[a]['sps'])}")
for n in td:
    steps, _ = td_log(n)
    print(f"  {n} log steps:", " ".join(f"{s}:{l}" for s, _, l, _ in steps))

print("\n## train_dpsgd.py clean (8 steps, no profiler)")
for n in ("cb_clean_torch", "cb_clean_triton"):
    steps, thr = td_log(n)
    print(f"| {n} | throughput {thr:.2f} smp/s | logged step times: "
          + " ".join(f"{t}s" for _, _, _, t in steps) + f" | GPU util {smi_util(n):.0f}% |")
