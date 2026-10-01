"""Strict stored-value bound test: ‖stored clipped example‖₂ <= C, zero tolerance.

B=1 trick: the dim-0 sum of a single stored value, accumulated at fp32 and
cast back to storage, is bit-identical to the stored value. So the engine's
output IS the per-example clipped vector it would contribute. Its norm is
evaluated in fp64 on CPU (relative error ~1e-16, far below every guard margin),
and compared to C with NO tolerance — the formal clip_pytree contract.

Earlier rigs used ‖·‖ <= C + 1e-5, which cannot see violations at the fp32
guard scale (~1e-7). This closes that gap.

Engines compared:
  production  original _stream_clip_and_sum                (control)
  fused       experiments/clip_fused/fused_engine.py      (current)
  old         fused_engine.py at commit d6be8dca, if given (regression demo)

Usage (CUDA host):
  .venv/bin/python experiments/clip_fused/check_bound_strict.py [old_engine.py]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping._streaming import _FIXED_SCALE
from opake.api.engine.pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402


def reduce_leaf(x):
    return CF._sum_clipped_tensor(x, dim=0, output_dtype=None, compute_dtype=None)


def stored_norm(fn, tree, C):
    reduced, *_ = fn(
        tree, torch.tensor(C, device="cuda"), scale=_FIXED_SCALE, batch_size=1,
        reduce_leaf=reduce_leaf, compute_dtype=None, second_moment=False,
        return_aux=False, return_stats=False,
    )
    leaves, _ = tree_flatten(reduced)
    sq = sum((leaf.detach().double().cpu() ** 2).sum() for leaf in leaves)
    return float(sq.sqrt())


def make_tree(spec, target_norm, seed):
    """spec: list of (shape, dtype). Scale the whole tree to target_norm in fp64,
    then round to each leaf's dtype (rounding moves the norm slightly)."""
    g = torch.Generator().manual_seed(seed)
    raw = [torch.randn(1, *shape, generator=g, dtype=torch.float64)
           for shape, _ in spec]
    tot = torch.sqrt(sum((r ** 2).sum() for r in raw))
    return {f"p{i}": (r * (target_norm / tot)).to(dt).cuda()
            for i, ((_, dt), r) in enumerate(zip(spec, raw))}


TREES = {
    "single-16k": lambda dt: [((16384,), dt)],
    "multi-3leaf": lambda dt: [((128, 128), dt), ((4096,), dt), ((3584, 16), dt)],
}
REGIMES = {"heavy(4C)": 4.0, "just-above(1.0001C)": 1.0001, "near-below(0.99995C)": 0.99995}


def main():
    assert torch.cuda.is_available()
    engines = {"production": fused_engine._ORIGINAL_STREAM,
               "fused": fused_engine.fused_stream_clip_and_sum}
    if len(sys.argv) > 1:
        spec = importlib.util.spec_from_file_location("old_engine", sys.argv[1])
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        engines["old(d6be8dca)"] = old.fused_stream_clip_and_sum

    C = 1.0
    trials = 100
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    total_viol = {k: 0 for k in engines}
    print(f"{'engine':15s} {'dtype':5s} {'tree':12s} {'regime':22s} "
          f"{'violations':>10s}  max(‖s‖/C - 1)")
    for ename, fn in engines.items():
        for dname, dt in dtypes.items():
            for tname, tspec in TREES.items():
                for rname, mult in REGIMES.items():
                    viol, worst = 0, -1.0
                    for s in range(trials):
                        tree = make_tree(tspec(dt), mult * C, seed=1000 * s + 7)
                        n = stored_norm(fn, tree, C)
                        worst = max(worst, n / C - 1.0)
                        viol += n > C
                    total_viol[ename] += viol
                    flag = "  <-- VIOLATION" if viol else ""
                    print(f"{ename:15s} {dname:5s} {tname:12s} {rname:22s} "
                          f"{viol:>4d}/{trials:<5d}  {worst:+.3e}{flag}")
    # mixed-dtype tree (fp32 + bf16 leaves sharing one per-example norm)
    for ename, fn in engines.items():
        viol, worst = 0, -1.0
        for s in range(trials):
            tree = make_tree([((128, 128), torch.float32), ((4096,), torch.bfloat16),
                              ((3584, 16), torch.float32)], 4.0, seed=1000 * s + 9)
            n = stored_norm(fn, tree, C)
            worst = max(worst, n / C - 1.0)
            viol += n > C
        total_viol[ename] += viol
        print(f"{ename:15s} mixed multi-3leaf  heavy(4C)              "
              f"{viol:>4d}/{trials:<5d}  {worst:+.3e}{'  <-- VIOLATION' if viol else ''}")
    print("\nfused-engine dispatch:", dict(fused_engine.STATS["fallback"]),
          "fused calls:", fused_engine.STATS["fused"])
    print("TOTAL violations:", total_viol)


if __name__ == "__main__":
    main()
