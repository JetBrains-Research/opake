"""Summarize the packaged multi-tensor confirmation runs (report §12.3).

Input directory holds pk_*.{csv,log,smi} from pk_driver.sh. ``old`` = the
previous packaged behaviour (per-tensor kernels, per-leaf markers, leafwise
accumulator), ``new`` = as packaged now, ``nofe`` = packaged kernel with the
leafwise accumulator.

Usage: python experiments/clip_fused/summarize_pk_runs.py notes/profiles/pk
"""

from __future__ import annotations

import ast
import csv
import re
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "notes/profiles/pk")


def prof(name):
    rows = list(csv.DictReader(open(D / f"{name}.csv")))
    f = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    return {"batches": [int(b) for b in f("batch")], "steps": f("step_time_sec"),
            "step": st.mean(f("step_time_sec")), "clip": st.mean(f("clip_sec")),
            "noise": st.mean(f("noise_sec")), "peak": max(f("memory_peak_gb")),
            "sps": sum(f("batch")) / sum(f("step_time_sec"))}


def util(name):
    rows = [tuple(map(int, r.split(","))) for r in (D / f"{name}.smi").read_text().splitlines() if r.strip()]
    peak = max(m for _, m in rows)
    return st.mean(u for u, m in rows if m >= 0.9 * peak)


def log_text(name):
    return (D / f"{name}.log").read_text().replace("\r", "\n")


def losses(name):
    return [float(x) for x in re.findall(r"^Step\s+\d+ .*?Loss: ([\d.]+)", log_text(name), re.M)]


for K, title in (("eager", "eager (kernels off)"), ("kern", "Triton model kernels")):
    arms = ["old", "new", "nofe"] if K == "eager" else ["old", "new"]
    runs = {a: prof(f"pk_{K}_{a}") for a in arms}
    base = runs["old"]
    assert all(r["batches"] == base["batches"] for r in runs.values()), "batches differ"
    print(f"\n## train_dpsgd.py, {title}, profiler on; batches {base['batches']}")
    print("| arm | mean step | clip phase | noise | peak | throughput | GPU util | vs old | per-step |")
    for a, r in runs.items():
        ratios = [x / y for x, y in zip(base["steps"], r["steps"], strict=True)]
        print(f"| {a} | {r['step']:.2f} s | {r['clip']:.2f} s | {r['noise']:.2f} s | {r['peak']:.2f} GiB | "
              f"{r['sps']:.2f} smp/s | {util(f'pk_{K}_{a}'):.0f}% | x{base['step'] / r['step']:.2f} "
              f"({(r['step'] / base['step'] - 1) * 100:+.1f}%) | x{min(ratios):.2f}-{max(ratios):.2f}, "
              f"faster {sum(x > 1 for x in ratios)}/{len(ratios)} |")
        print(f"   losses: {losses(f'pk_{K}_{a}')}")

print("\n## clean (eager, 8 steps, no profiler)")
for a in ("old", "new"):
    t = log_text(f"pk_clean_{a}")
    thr = re.findall(r"Throughput: ([\d.]+) samples/s", t)
    times = [float(x) for x in re.findall(r"^Step\s+\d+ .*?Time: ([\d.]+)s", t, re.M)]
    print(f"| {a} | {float(thr[-1]):.2f} smp/s | logged step times {times} | GPU util {util(f'pk_clean_{a}'):.0f}% |")

print("\n## DPTrainer (steps 1-6)")
dts = {}
for a in ("old", "new"):
    dts[a] = [ast.literal_eval(line) for line in log_text(f"pk_dt_{a}").splitlines() if line.startswith("{'loss'")]
assert [r["batch_size"] for r in dts["old"]] == [r["batch_size"] for r in dts["new"]], "batches differ"
print("batches:", [int(r["batch_size"]) for r in dts["old"]])
b = st.mean(float(r["step_time_sec"]) for r in dts["old"])
for a, rows in dts.items():
    m = st.mean(float(r["step_time_sec"]) for r in rows)
    ratios = [float(x["step_time_sec"]) / float(y["step_time_sec"]) for x, y in zip(dts["old"], rows, strict=True)]
    sps = sum(float(r["batch_size"]) for r in rows) / sum(float(r["step_time_sec"]) for r in rows)
    print(f"| {a} | {m:.2f} s | x{b / m:.2f} | per-step x{min(ratios):.2f}-{max(ratios):.2f} | "
          f"peak {max(float(r['memory_peak_gb']) for r in rows):.2f} GB | {sps:.2f} smp/s | GPU util {util(f'pk_dt_{a}'):.0f}% |")
    print("   losses:", " ".join(f"{float(r['loss']):.4f}" for r in rows))
