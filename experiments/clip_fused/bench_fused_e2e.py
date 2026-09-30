"""End-to-end rig: real clipped_grad pipeline with the fused engine swapped
at the _stream_clip_and_sum seam (test-harness scope, CUDA-only).

Per config, runs:
  r_default  unpatched stream path, fresh params/state
  r_drift    unpatched again (must be bitwise equal to r_default; if not,
             there is pre-existing nondeterminism and parity must be judged
             against drift, not zero)
  r_fused    patched with fused_stream_clip_and_sum

Checks: max|fused - default| per leaf (tolerance + reported), aggregate
parity on the summed gradient (the DP-visible quantity), peak memory, and
median step time per path.

Usage (CUDA host):
  .venv/bin/python experiments/clip_fused/bench_fused_e2e.py --quick
  .venv/bin/python experiments/clip_fused/bench_fused_e2e.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call

import opake.api.engine.clipping._clipped_fun as CF
from opake.dpsgd.clipping import clipped_grad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402


def sync():
    torch.cuda.synchronize()


def build_model(kind, device, dtype):
    if kind == "few-large":
        mods = []
        for _ in range(2):
            mods += [torch.nn.Linear(4096, 4096), torch.nn.GELU(),
                     torch.nn.LayerNorm(4096)]
        width = 4096
    elif kind == "many-small":
        mods = []
        for _ in range(12):
            mods += [torch.nn.Linear(128, 128), torch.nn.GELU(),
                     torch.nn.LayerNorm(128)]
        width = 128
    else:
        raise ValueError(kind)
    return torch.nn.Sequential(*mods).to(device=device, dtype=dtype), width


def flatten_leaves(x):
    try:
        from opake.api.engine.pytree import tree_flatten
        inner = getattr(x, "pytree", x)
        leaves, _ = tree_flatten(inner)
        return list(leaves)
    except Exception:
        import torch.utils._pytree as tpt
        return list(tpt.tree_leaves(getattr(x, "pytree", x)))


def run_case(kind, batch, dtype, iters=8):
    dev = "cuda"
    torch.manual_seed(0)
    net, width = build_model(kind, dev, dtype)
    seq = 64 if kind == "few-large" else 32
    X = torch.randn(batch, seq, width, device=dev, dtype=dtype)
    Y = torch.randn(batch, seq, width, device=dev, dtype=dtype) * 3.0  # force
    # active clipping: large residuals => many per-example grad norms above C

    params0 = {n: p.detach().clone() for n, p in net.named_parameters()}

    def loss_fn(pd, x1, y1):
        out = functional_call(net, pd, (x1,))
        return F.mse_loss(out, y1)

    def make_cg():
        return clipped_grad(
            loss_fn, argnums=0, batch_argnums=(1, 2),
            clipping_norm=1.0, normalize_by=batch,
        )

    def call(cg_state):
        cg, st = cg_state
        res = cg(params0, X, Y, state=st)
        out = res[0] if isinstance(res, tuple) else res
        return out

    def timed(make, it=iters, warm=3):
        for _ in range(warm):
            call(make())
        sync()
        ts = []
        for _ in range(it):
            sync()
            t0 = time.perf_counter()
            call(make())
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    # --- unpatched pair (parity baseline + drift) ---
    o_default = call(make_cg())
    o_drift = call(make_cg())
    drift = max((a - b).abs().max().item()
                for a, b in zip(flatten_leaves(o_default),
                                flatten_leaves(o_drift)))

    # --- patched ---
    CF._stream_clip_and_sum = fused_engine.fused_stream_clip_and_sum
    try:
        o_fused = call(make_cg())
        mem_fused = torch.cuda.max_memory_allocated() / 2**20
        sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = call(make_cg())
        sync()
        mem_fused = torch.cuda.max_memory_allocated() / 2**20
        t_fused = timed(make_cg)
    finally:
        CF._stream_clip_and_sum = fused_engine._ORIGINAL_STREAM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = call(make_cg())
    sync()
    mem_stream = torch.cuda.max_memory_allocated() / 2**20
    t_stream = timed(make_cg)

    dmax, rmax = 0.0, 0.0
    bad = []
    for name, (a, b) in zip(_leaf_names(net),
                            zip(flatten_leaves(o_default),
                                flatten_leaves(o_fused))):
        d = (a.float() - b.float()).abs().max().item()
        r = d / max(a.float().abs().max().item(), 1e-12)
        if d > dmax:
            dmax, rmax, worst = d, r, name
        tol = 5e-5 if dtype == torch.float32 else 5e-3
        if r > tol:
            bad.append((name, d, r))
    ok = not bad and drift == 0.0
    tag = f"[{kind},B={batch},{str(dtype).split('.')[-1]}]"
    print(f"{'PASS' if ok else 'FAIL'} {tag:26s} step: stream {t_stream:7.2f}ms "
          f"fused {t_fused:7.2f}ms (x{t_stream / t_fused:5.2f})  "
          f"peak {mem_stream:6.1f} -> {mem_fused:6.1f} MB  "
          f"maxrel {rmax:.2e} at {worst if bad or rmax else '-'}  drift={drift:.1e}")
    if bad:
        for name, d, r in bad[:5]:
            print(f"   MISMATCH {name}: abs={d:.3e} rel={r:.3e}")
    if drift != 0.0:
        print(f"   NOTE: unpatched reruns differ (drift={drift:.3e}) — "
              f"pre-existing nondeterminism; fused parity judged vs drift")
    return ok


def _leaf_names(net):
    return [n for n, _ in net.named_parameters()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA host required"
    cases = [
        ("few-large", 16, torch.float32),
        ("few-large", 64, torch.float32),
        ("many-small", 128, torch.float32),
        ("few-large", 64, torch.bfloat16),
        ("many-small", 128, torch.bfloat16),
    ]
    if args.quick:
        cases = cases[:1]
    results = [run_case(*c) for c in cases]
    print(f"\nfailures: {sum(1 for r in results if not r)}")
    raise SystemExit(1 if not all(results) else 0)


if __name__ == "__main__":
    main()
