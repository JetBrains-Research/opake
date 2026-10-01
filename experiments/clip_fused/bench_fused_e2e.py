"""End-to-end rig: real clipped_grad / adaptive_clipped_grad with the fused
engine swapped at the _stream_clip_and_sum seam (harness scope, CUDA-only).

Modes (the last one is what examples/train_dpsgd.py runs by default):
  fixed            clipped_grad, return_aux=False, no microbatching
  fixed+aux        clipped_grad, return_aux=True,  no microbatching
  fixed+aux+mb8    clipped_grad, return_aux=True,  microbatch_size=8
  adaptive+aux+mb8 adaptive_clipped_grad, return_aux=True, microbatch_size=8

Per (config, mode):
  default  unpatched, run twice -> drift must be exactly 0
  fused    patched; every tensor in the result (grads, aux norms, clipped
           norms, adaptive state) compared to default; engine dispatch
           counters must show fused calls and zero fallbacks
Timing: clipped_grad CALL (per-example grads + clip + sum [+ adaptive
update]), constructed once outside the timed region, CUDA-synced, median.
This is NOT a full training step (no noise, no optimizer).

Usage (CUDA host):
  .venv/bin/python experiments/clip_fused/bench_fused_e2e.py [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from opake.dpsgd.clipping import adaptive_clipped_grad, clipped_grad
from opake.random import key

sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402


def build_model(kind, dtype):
    if kind == "few-large":
        mods, width = [], 4096
        for _ in range(2):
            mods += [torch.nn.Linear(4096, 4096), torch.nn.GELU(), torch.nn.LayerNorm(4096)]
    elif kind == "many-small":
        mods, width = [], 128
        for _ in range(12):
            mods += [torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.LayerNorm(128)]
    else:
        raise ValueError(kind)
    return torch.nn.Sequential(*mods).to(device="cuda", dtype=dtype), width


def collect(obj, prefix="out"):
    """Yield (path, tensor) for every tensor reachable in a result object."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from collect(getattr(obj, f.name), f"{prefix}.{f.name}")
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from collect(obj[k], f"{prefix}[{k}]")
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            yield from collect(v, f"{prefix}[{i}]")
    elif hasattr(obj, "pytree"):
        yield from collect(obj.pytree, f"{prefix}.pytree")


def make_fn(mode, loss_fn, batch):
    common = dict(argnums=0, batch_argnums=(1, 2), normalize_by=batch)
    if mode == "fixed":
        return clipped_grad(loss_fn, clipping_norm=1.0, **common)
    if mode == "fixed+aux":
        return clipped_grad(loss_fn, clipping_norm=1.0, return_aux=True, **common)
    if mode == "fixed+aux+mb8":
        return clipped_grad(loss_fn, clipping_norm=1.0, return_aux=True,
                            microbatch_size=8, **common)
    if mode == "adaptive+aux+mb8":
        return adaptive_clipped_grad(
            loss_fn, initial_clipping_norm=1.0, target_quantile=0.5,
            clipping_norm_max=100.0, microbatch_size=8, return_aux=True,
            key=key(0), **common)
    raise ValueError(mode)


def timed(fn, iters=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def peak_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def run_case(kind, batch, dtype, mode):
    torch.manual_seed(0)
    net, width = build_model(kind, dtype)
    seq = 64 if kind == "few-large" else 32
    X = torch.randn(batch, seq, width, device="cuda", dtype=dtype)
    Y = torch.randn(batch, seq, width, device="cuda", dtype=dtype) * 3.0
    params0 = {n: p.detach().clone() for n, p in net.named_parameters()}

    def loss_fn(pd, x1, y1):
        return F.mse_loss(functional_call(net, pd, (x1,)), y1)

    fused_engine.uninstall()
    fn, st = make_fn(mode, loss_fn, batch)
    call = lambda: fn(params0, X, Y, state=st)  # noqa: E731
    r_default = list(collect(call()))
    r_drift = list(collect(call()))
    drift = max(((a - b).abs().max().item() if a.is_floating_point()
                 else float((a != b).any())) for (_, a), (_, b) in zip(r_default, r_drift))
    t_default = timed(call)
    m_default = peak_mib(call)

    fused_engine.install()
    fused_engine.STATS["fused"] = 0
    fused_engine.STATS["fallback"].clear()
    try:
        fn_f, st_f = make_fn(mode, loss_fn, batch)
        call_f = lambda: fn_f(params0, X, Y, state=st_f)  # noqa: E731
        r_fused = list(collect(call_f()))
        dispatch = (fused_engine.STATS["fused"], dict(fused_engine.STATS["fallback"]))
        t_fused = timed(call_f)
        m_fused = peak_mib(call_f)
    finally:
        fused_engine.uninstall()

    assert [p for p, _ in r_default] == [p for p, _ in r_fused], "result structure differs"
    worst, worst_path, bad = 0.0, "-", []
    for (path, a), (_, b) in zip(r_default, r_fused):
        if not a.is_floating_point():
            if (a != b).any():
                bad.append((path, "int-mismatch"))
            continue
        d = (a.float() - b.float()).abs().max().item()
        r = d / max(a.float().abs().max().item(), 1e-30)
        is_norm = "norm" in path
        tol = 1e-5 if is_norm else (5e-5 if dtype == torch.float32 else 5e-3)
        if r > worst:
            worst, worst_path = r, path
        if r > tol:
            bad.append((path, f"rel={r:.2e}>{tol:.0e}"))
    ok = not bad and drift == 0.0 and dispatch[0] > 0 and not dispatch[1]
    tag = f"[{kind},B={batch},{str(dtype).split('.')[-1]},{mode}]"
    print(f"{'PASS' if ok else 'FAIL'} {tag:44s} call: {t_default:8.2f} -> {t_fused:8.2f} ms "
          f"(x{t_default / t_fused:5.2f})  peak {m_default:8.1f} -> {m_fused:8.1f} MiB  "
          f"worst rel {worst:.1e} @ {worst_path}  drift={drift:.0e}  "
          f"dispatch fused={dispatch[0]} fallback={dispatch[1]}")
    for path, why in bad[:6]:
        print(f"     MISMATCH {path}: {why}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    modes = ["fixed", "fixed+aux", "fixed+aux+mb8", "adaptive+aux+mb8"]
    configs = [("few-large", 64, torch.float32), ("many-small", 128, torch.float32),
               ("few-large", 64, torch.bfloat16)]
    if args.quick:
        configs, modes = configs[:1], ["adaptive+aux+mb8"]
    results = [run_case(k, b, d, m) for (k, b, d) in configs for m in modes]
    print(f"\nfailures: {sum(not r for r in results)} / {len(results)}")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
