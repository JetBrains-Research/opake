"""Adversarial + subnormal-regime stored-value bound test (zero tolerance).

check_bound_strict.py used random Gaussian trees at C=1, whose clipped values
sit in each dtype's NORMAL range. This test targets the regimes where the
guard actually has to work:

  A1 normal-range round-back: every element is v = 2^-4 (1 + 2^-m) (m = mantissa
     bits), the value with the largest relative half-ulp below it; the example
     norm is (1 + 0.9*half_ulp) * C. A scale shrink smaller than the storage
     half-ulp rounds x*s back to x, leaving the example unclipped.
  A2 subnormal round-up: every element is 3 * (smallest subnormal); C is set so
     the ratio is 0.84, so x*s = 2.52 grid units, which rounds UP to 3 (= x).
  A3 realistic small-C fp16 regime: 4M-element Gaussian leaf, C = 0.01, so
     clipped elements are ~5e-6, i.e. deep in fp16's subnormal range
     (fp16 smallest normal is 6.1e-5). Regimes: just above C, and 4C.

B=1 trick as in check_bound_strict.py (engine output == stored example);
norms in fp64 on CPU; C passed as an fp64 tensor so it is not itself rounded.

Usage (CUDA host): .venv/bin/python experiments/clip_fused/check_bound_adversarial.py old_engine.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping._streaming import _FIXED_SCALE
from opake.api.engine.pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402

MANT = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23}
SUB = {torch.float16: 2.0**-24, torch.bfloat16: 2.0**-133, torch.float32: 2.0**-149}


def reduce_leaf(x):
    return CF._sum_clipped_tensor(x, dim=0, output_dtype=None, compute_dtype=None)


def stored_ratio(fn, x, C):
    reduced, *_ = fn({"w": x}, torch.tensor(C, dtype=torch.float64, device="cuda"),
                     scale=_FIXED_SCALE, batch_size=1, reduce_leaf=reduce_leaf,
                     compute_dtype=None, second_moment=False, return_aux=False,
                     return_stats=False)
    leaves, _ = tree_flatten(reduced)
    n = math.sqrt(sum(float((leaf.detach().double().cpu() ** 2).sum()) for leaf in leaves))
    return n / C - 1.0


def signs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double()


def case_a1(dt, n=1024, seed=0):
    m = MANT[dt]
    v = 2.0**-4 * (1 + 2.0**-m)
    half = 2.0 ** -(m + 1) / (1 + 2.0**-m)
    x = (signs(n, seed) * v).to(dt)
    C = v * math.sqrt(n) / (1 + 0.9 * half)
    return x.reshape(1, n).cuda(), C, f"A1 round-back  ({1 + 0.9 * half:.7f}C)"


def case_a2(dt, n=1024, seed=0):
    v = 3 * SUB[dt]
    x = (signs(n, seed) * v).to(dt)
    C = v * math.sqrt(n) * 0.84
    return x.reshape(1, n).cuda(), C, "A2 subnormal round-up (1.19C)"


def case_a3(dt, mult, n=4 * 1024 * 1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = 0.01
    raw = torch.randn(n, generator=g, dtype=torch.float64)
    x = (raw * (mult * C / raw.norm())).to(dt)
    return x.reshape(1, n).cuda(), C, f"A3 C=0.01, 4M elems, {mult}C"


def main():
    engines = {"production": fused_engine._ORIGINAL_STREAM,
               "fused(current)": fused_engine.fused_stream_clip_and_sum}
    if len(sys.argv) > 1:
        spec = importlib.util.spec_from_file_location("old_engine", sys.argv[1])
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        engines["old(d6be8dca)"] = old.fused_stream_clip_and_sum
    builders = [case_a1, case_a2,
                lambda dt: case_a3(dt, 1.001), lambda dt: case_a3(dt, 4.0)]
    print(f"{'dtype':5s} {'case':34s} " + " ".join(f"{e:>18s}" for e in engines))
    for dname, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        for b in builders:
            x, C, label = b(dt)
            cells = []
            for fn in engines.values():
                r = stored_ratio(fn, x, C)
                cells.append(f"{r:+.3e}{' VIOL' if r > 0 else '     '}")
            print(f"{dname:5s} {label:34s} " + " ".join(f"{c:>18s}" for c in cells))
    print("\ncell = ||stored||/C - 1 (must be <= 0). fused dispatch:",
          dict(fused_engine.STATS["fallback"]), "fused calls:", fused_engine.STATS["fused"])


if __name__ == "__main__":
    main()
