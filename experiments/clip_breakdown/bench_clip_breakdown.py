"""DP-SGD clip-path breakdown benchmark (report OQ1/OQ5; plan §2.3 gate).

Splits DP step time into stages to answer WHERE the step cost lives:

  t_nodp   - batched fwd+bwd via autograd (no per-example gradients)
  t_grad   - torch.func vmap(grad(loss)) + batch sum, NO clipping
             (isolates per-example gradient materialization cost)
  t_clip_* - opake dpsgd clipped_grad (clip+aggregate inside), variants:
             stream   : default leaf-streamed path (post #1108)
             nostream : _scale_fn=clip_pytree forces the pre-#1108 vmapped path
             mb8      : microbatched chunk path, microbatch_size=8

Derived:
  clip_overhead        = t_clip - t_grad        (what a fused kernel could remove)
  materialization_cost = t_grad - t_nodp        (what only Ghost/FlashDP-style
                                                  no-materialize designs remove)

Synthetic data on purpose: this measures the plumbing, not learning. MSE loss on
random targets is fine — per-example gradient NORM magnitudes (what triggers
clipping) are controlled explicitly, not learned ones.

Usage:
  .venv/bin/python experiments/clip_breakdown/bench_clip_breakdown.py --smoke      # CPU quick check
  .venv/bin/python experiments/clip_breakdown/bench_clip_breakdown.py --device mps # full matrix
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from opake.dpsgd.clipping import clipped_grad
from opake.api.engine.clipping._pytree import clip_pytree

RESULTS_PATH = Path(__file__).parent / "results.json"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_model(kind: str, device: str):
    if kind == "few-large":
        mods = []
        for _ in range(2):
            mods += [torch.nn.Linear(4096, 4096), torch.nn.GELU(), torch.nn.LayerNorm(4096)]
        width = 4096
    elif kind == "many-small":
        mods = []
        for _ in range(12):
            mods += [torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.LayerNorm(128)]
        width = 128
    else:
        raise ValueError(kind)
    return torch.nn.Sequential(*mods).to(device=device, dtype=torch.float32), width


def timed(fn, device: str, warmup: int = 3, iters: int = 8) -> float:
    for _ in range(warmup):
        fn()
    sync(device)  # drain warmup queue before measuring
    ts = []
    for _ in range(iters):
        sync(device)  # each measurement starts with an empty queue
        t0 = time.perf_counter()
        fn()
        sync(device)  # and ends only when the device is idle
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]  # median ms


def peak_mb(device: str) -> float | None:  # unused helper; peak stats are inline
    return None


def run_case(device: str, kind: str, batch: int, iters: int):
    net, width = build_model(kind, device)
    seq = 64 if kind == "few-large" else 32
    gen = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device)

    X = rnd(batch, seq, width)
    Y = rnd(batch, seq, width)

    params0 = {n: p.detach().clone() for n, p in net.named_parameters()}

    def loss_fn(pd, x1, y1):  # per-example; batched via vmap inside
        out = functional_call(net, pd, (x1,))
        return F.mse_loss(out, y1)

    out = {"model": kind, "batch": batch, "n_leaves": len(params0)}

    # --- t_nodp: batched autograd fwd+bwd (non-DP reference) ---
    def t_nodp():
        pd = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
        o = functional_call(net, pd, (X,))
        loss = F.mse_loss(o, Y)
        loss.backward()
        del pd
        sync(device)

    out["t_nodp_ms"] = timed(t_nodp, device, iters=iters)

    # --- t_grad: vmap(grad) without clipping ---
    per_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    def t_grad():
        g = per_grad(params0, X, Y)
        return {k: v.sum(0) for k, v in g.items()}

    out["t_grad_ms"] = timed(t_grad, device, iters=iters)

    # --- clipping variants ---
    def make_cg(**kw):
        cg, st = clipped_grad(
            loss_fn, argnums=0, batch_argnums=(1, 2),
            clipping_norm=1.0, normalize_by=batch, **kw,
        )
        return cg, st

    variants = {
        "t_clip_stream_ms": make_cg(),
        "t_clip_nostream_ms": make_cg(_scale_fn=lambda v: clip_pytree(v, 1.0)),
        "t_clip_mb8_ms": make_cg(microbatch_size=8),
    }
    for name, (cg, st0) in variants.items():
        def call(cg=cg, st0=st0):
            return cg(params0, X, Y, state=st0)

        out[name] = timed(call, device, iters=iters)
        # memory (approximate, single run, peak-reset where supported)
        try:
            if device in ("mps", "cuda"):
                dev_mod = torch.cuda if device == "cuda" else torch.mps
                if device == "cuda":
                    torch.cuda.empty_cache()
                dev_mod.reset_peak_memory_stats()
                call()
                sync(device)
                out[name + "_peak_mb"] = round(dev_mod.max_memory_allocated() / 2**20, 1)
        except Exception as exc:  # memory stats are best-effort
            out[name + "_peak_mb"] = f"unavailable: {type(exc).__name__}"

    # ratios (informational; noise-prone on tiny configs)
    if out["t_nodp_ms"] > 0:
        out["materialization_over_t_nodp"] = round(out["t_grad_ms"] / out["t_nodp_ms"], 2)
        out["clip_overhead_ms"] = round(out["t_clip_stream_ms"] - out["t_grad_ms"], 3)
        out["clip_overhead_share_of_dp"] = round(
            (out["t_clip_stream_ms"] - out["t_nodp_ms"]) / out["t_clip_stream_ms"], 3
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--smoke", action="store_true", help="one tiny CPU config")
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    if args.smoke:
        cases = [("cpu", "many-small", 4)]
        args.iters = 2
    else:
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        cases = [
            (dev, "many-small", 32),
            (dev, "many-small", 128),
            (dev, "few-large", 16),
            (dev, "few-large", 64),
        ]

    results = {
        "meta": {
            "when": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "note": "SYNTHETIC data; single host; ratios for triage only, not paper numbers.",
        },
        "cases": [],
    }
    for device, kind, batch in cases:
        print(f"== {kind} B={batch} on {device} ==", flush=True)
        case = run_case(device, kind, batch, args.iters)
        for k, v in case.items():
            if isinstance(v, float):
                case[k] = round(v, 3)
        print(json.dumps(case, indent=2), flush=True)
        results["cases"].append(case)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
