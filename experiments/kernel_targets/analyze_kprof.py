"""Kernel-level breakdown of one profiled training step.

Input: <prefix>.kernels.csv.gz written by run_train_variant.py (CB_KPROF), with
columns name, start_us, dur_us for every CUDA kernel/memcpy/memset event.

Reports GPU busy time (union of event intervals) vs the profiled window, idle
gaps, and time/count per kernel category and per top kernel name.

Usage: python experiments/kernel_targets/analyze_kprof.py notes/profiles/kp/kp_eager.step4 [...]
"""

from __future__ import annotations

import collections
import csv
import gzip
import re
import sys

CATEGORIES = [  # first match wins
    ("clip (Opake fused)", r"_partial_sq_kernel|_tensor_sum_kernel|_apply_sum_kernel"),
    ("attention fwd", r"flash_fwd|fmha.*fwd|attention_kernel|efficient_attention_forward|cudnn.*fprop|sdpa.*fwd"),
    ("attention bwd", r"flash_bwd|fmha.*bwd|attention_backward|efficient_attention_backward|cudnn.*dgrad|sdpa.*bwd"),
    ("GEMM", r"gemm|cutlass|xmma|cublas|Kernel2|sm80_|ampere_|magma|splitK|_mm_|matmul"),
    ("Opake Triton model kernels", r"swiglu|geglu|rope|rms|lora|_lce|cross_entropy|linear_ce|_ce_"),
    ("reduce", r"reduce_kernel|Reduce"),
    ("softmax / log-softmax", r"softmax"),
    ("copy / cat / index", r"copy|cat|CatArray|index|gather|scatter|Memcpy|memcpy"),
    ("fill / memset", r"[Ff]ill|Memset|memset|zero"),
    ("elementwise", r"elementwise|vectorized|unrolled|Functor|pointwise|triton_poi"),
    ("RNG", r"philox|normal|randn|curand|distribution"),
]


def category(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return "other"


def analyze(prefix):
    rows = []
    with gzip.open(f"{prefix}.kernels.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["name"], float(r["start_us"]), float(r["dur_us"])))
    rows.sort(key=lambda r: r[1])
    t0 = rows[0][1]
    t1 = max(s + d for _, s, d in rows)
    busy, cur_s, cur_e, gaps = 0.0, None, None, []
    for _, s, d in rows:
        e = s + d
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
                gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = t1 - t0
    by_cat = collections.defaultdict(lambda: [0.0, 0])
    by_name = collections.defaultdict(lambda: [0.0, 0])
    for n, _, d in rows:
        c = by_cat[category(n)]
        c[0] += d
        c[1] += 1
        k = by_name[n]
        k[0] += d
        k[1] += 1
    total = sum(v[0] for v in by_cat.values())
    print(f"\n## {prefix}")
    print(f"events {len(rows):,}; GPU window {span / 1e6:.3f} s; busy (union) {busy / 1e6:.3f} s "
          f"({busy / span * 100:.1f}%); idle {(span - busy) / 1e6:.3f} s; summed kernel time {total / 1e6:.3f} s")
    gaps.sort()
    small = sum(g for g in gaps if g < 10)
    mid = sum(g for g in gaps if 10 <= g < 100)
    big = sum(g for g in gaps if g >= 100)
    print(f"idle gaps: {len(gaps):,}; <10us {small / 1e6:.3f} s, 10-100us {mid / 1e6:.3f} s, "
          f">=100us {big / 1e6:.3f} s (largest {gaps[-1] / 1e3:.1f} ms)")
    print("| category | kernel time | share of kernel time | events | mean us |")
    print("|---|---:|---:|---:|---:|")
    for cat, (t, n) in sorted(by_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"| {cat} | {t / 1e6:.3f} s | {t / total * 100:.1f}% | {n:,} | {t / n:.1f} |")
    print("\ntop kernels by time:")
    for name, (t, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {t / 1e6:7.3f} s  {n:7,}x  {t / n:8.1f} us  [{category(name)}]  {name[:120]}")
    return {"busy": busy, "span": span, "by_cat": dict(by_cat), "events": len(rows)}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
