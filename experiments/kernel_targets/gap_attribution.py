"""Attribute GPU idle gaps to the CUDA events around them.

For every idle gap >= --min-us between consecutive (merged) GPU events, record
the event that ends before the gap and the event that starts after it, and sum
gap time per (after) name and per (before -> after) pair. The event after a
stall is what the host enqueued once it caught up; pageable copies and syncs
show up here.

Usage: python experiments/kernel_targets/gap_attribution.py notes/profiles/kp/kp_kern.step4 [--min-us 100]
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip


def short(name, n=90):
    return name if len(name) <= n else name[: n - 3] + "..."


ap = argparse.ArgumentParser()
ap.add_argument("prefix", nargs="+")
ap.add_argument("--min-us", type=float, default=100.0)
args = ap.parse_args()
for prefix in args.prefix:
    rows = []
    with gzip.open(f"{prefix}.kernels.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["name"], float(r["start_us"]), float(r["dur_us"])))
    rows.sort(key=lambda r: r[1])
    after = collections.defaultdict(lambda: [0.0, 0])
    pair = collections.defaultdict(lambda: [0.0, 0])
    end, prev = rows[0][1] + rows[0][2], rows[0][0]
    total = 0.0
    memcpy = collections.Counter()
    for name, s, d in rows:
        if "Memcpy" in name or "Memset" in name:
            memcpy[name] += 1
        gap = s - end
        if gap >= args.min_us:
            after[short(name)][0] += gap
            after[short(name)][1] += 1
            pair[(short(prev, 50), short(name, 50))][0] += gap
            pair[(short(prev, 50), short(name, 50))][1] += 1
            total += gap
        if s + d > end:
            end, prev = s + d, name
    print(f"\n## {prefix}: gaps >= {args.min_us:.0f} us total {total / 1e6:.3f} s")
    print("memcpy/memset events:", dict(memcpy))
    print("by event AFTER the gap:")
    for n, (t, c) in sorted(after.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {t / 1e6:6.3f} s {c:5d}x  {n}")
    print("by (before -> after):")
    for (b, a), (t, c) in sorted(pair.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  {t / 1e6:6.3f} s {c:5d}x  {b}  ->  {a}")
