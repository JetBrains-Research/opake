"""Clip-path measurement rig: fp64 oracle parity + stored-value bound invariants
+ noise-convention probe (plan §2.5 / §2.6.1 / §2.6.4 gates; report §4.1).

Runs against the CURRENT Opake clip pipeline and will run unchanged against any
fused replacement kernel, so the rig is the gate, not just a test:
  (A) parity   — clip+sum vs fp64 oracle, per path (streaming / legacy / mb8)
  (B) invariant— per-example stored ||clipped|| <= C (+ margin for diagnostic
                 rounding), on clipping-active regimes and near-C norms
  (C) noise    — gaussian_noise realized std == sigma * max_norm on the AGGREGATE
                 (documents the convention any fused kernel must reproduce)

Synthetic data. Usage:
  .venv/bin/python experiments/clip_oracle/run_oracle.py            # cpu
  .venv/bin/python experiments/clip_oracle/run_oracle.py --mps      # + mps checks
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.api.engine.clipping._pytree import clip_pytree
from opake.dpsgd.clipping import clipped_grad  # noqa: F401  (path used in real stack)
from opake.dpsgd.noise import gaussian_noise
from opake.random import key

FAILURES: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ---------------------------------------------------------------- helpers


def oracle_clip_sum(G: torch.Tensor, C: float) -> torch.Tensor:
    """fp64 reference: per-example clip to C, then sum over dim0."""
    g = G.double()
    norms = g.norm(dim=(-1, -2), keepdim=True)
    scale = (C / (norms + 1e-10)).clamp(max=1.0)
    return (g * scale).sum(0)


def _raw(out):
    return getattr(out, "pytree", out)


def _raw_cpu(out):
    t = getattr(out, "pytree", out)
    return t.detach().cpu()


def run_path(G: torch.Tensor, C: float, path: str, device: str, dtype: torch.Tensor.dtype):
    """Return (summed_result, aux_dict) for the given clip path on a [B,...] stack."""
    common = dict(batch_argnums=0, clipping_norm=C, normalize_by=1.0, return_aux=True)
    if path == "stream":
        cf, st = clipped_fun(lambda x: x, **common)
        (out, aux), _ = cf(G, state=st)
    elif path == "nostream":
        cf, st = clipped_fun(
            lambda x: x, _scale_fn=lambda v: clip_pytree(v, C), **common
        )
        (out, aux), _ = cf(G, state=st)
    elif path.startswith("mb"):
        mb = int(path[2:])
        cf, st = clipped_fun(lambda x: x, microbatch_size=mb, **common)
        (out, aux), _ = cf(G, state=st)
    else:
        raise ValueError(path)
    return _raw(out), aux


# ------------------------------------------------------------------ test A/B


def test_parity_and_invariant(device: str, dtype: torch.dtype, paths: list[str]) -> None:
    C = 1.0
    gen = torch.Generator(device="cpu").manual_seed(0)

    def stack(B: int, D: int, P: int, mode: str):
        """Build a [B,D,P] stack whose per-example norms straddle C, sit below it,
        or tower above it (norm of an i.i.d. block ≈ scale·sqrt(D·P), so scales
        are set from that)."""
        base = torch.randn(B, D, P, generator=gen, dtype=torch.float64)
        unit = 1.0 / math.sqrt(D * P)
        if mode == "low":      # norms ≈ 0.4·C → no clipping exercised
            per = torch.full((B,), 0.4)
        elif mode == "straddle":  # per-sample norms uniformly in [0.3C, 2C]
            per = torch.empty(B).uniform_(0.3, 2.0, generator=gen)
        elif mode == "clipped":  # norms ≈ 4·C → heavy clipping
            per = torch.full((B,), 4.0)
        else:
            raise ValueError(mode)
        G64 = base * (per * unit).reshape(-1, 1, 1)
        return G64

    configs = [
        ("low-norm (no clipping)", 8, 64, 512, "low"),
        ("near-C boundary", 64, 256, 512, "straddle"),
        ("near-C small", 32, 32, 128, "straddle"),
        ("heavy clip", 16, 64, 256, "clipped"),
    ]
    tol = {torch.float32: (1e-4, 1e-5), torch.bfloat16: (2e-2, 2e-3)}[dtype]

    for name, B, D, P, mode in configs:
        G64 = stack(B, D, P, mode)                       # cpu fp64 truth
        Gr = G64.to(dtype)                               # dtype-rounded (what the path sees)
        ref = oracle_clip_sum(Gr.double(), C)             # fp64 oracle on ROUNDED input
        G = Gr.to(device)
        norms = G64.norm(dim=(-1, -2))
        clip_share = (norms > C).float().mean().item()
        for path in paths:
            try:
                out, aux = run_path(G, C, path, device, dtype)
                out64 = _raw_cpu(out).double()
                max_abs = (out64 - ref).abs().max().item()
                rel = max_abs / max(ref.abs().max().item(), 1e-12)
                ok = torch.allclose(out64, ref, rtol=tol[0], atol=tol[1])
                check(
                    f"parity [{device},{str(dtype).split('.')[-1]},{name},{path}]",
                    ok,
                    f"clip_share={clip_share:.2f} max_abs={max_abs:.2e} rel={rel:.2e}",
                )
                cn = aux.clipped_norms
                if cn is not None:
                    cn = cn.detach().to("cpu", dtype=torch.float32)
                    over = (cn > C + 1e-5).sum().item()  # +margin for diagnostic-side fp error
                    check(
                        f"bound  [{device},{str(dtype).split('.')[-1]},{name},{path}]",
                        over == 0,
                        f"max_clipped_norm={cn.max().item():.6f} (C={C}, tol 1e-5)",
                    )
            except Exception as exc:
                check(f"run [{device},{dtype},{name},{path}]", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ test C


def test_noise_convention(device: str) -> None:
    """Opake convention probe: gaussian_noise adds std = noise_multiplier * max_norm
    to the ALREADY-AGGREGATED pytree (mean aggregation => C/B scaling via
    normalize_by). A fused path that adds sigma/sqrt(B) on the mean (FlashDP
    reference) will NOT match this — this probe pins the convention."""
    C, sigma, B = 1.0, 1.0, 100000  # one fat "aggregate" vector, B used as normalize_by
    x = torch.zeros(2**20, device=device)
    noise_fn, st = gaussian_noise(noise_multiplier=sigma, key=key(0))

    # (1) sum-aggregation convention: std on aggregate should be sigma * C = 1.0
    from opake.api.engine.types import clipped as clipped_wrap

    out, _ = noise_fn(clipped_wrap(x.clone(), max_norm=C), st)
    t = getattr(out, "pytree", out)
    realized = t.std().item()
    check(
        "noise  [sum-convention: std ≈ sigma*C on aggregate]",
        abs(realized - sigma * C) < 0.05 * sigma * C,
        f"realized std={realized:.4f} vs sigma*C={sigma * C}",
    )

    # (2) mean convention via clipped_fun(normalize_by=B): max_norm becomes C/B,
    #     so realized noise std on the mean should be sigma*C/B — NOT sigma*sqrt(C/B)-ish
    cf, c_st = clipped_fun(lambda v: v / B, clipping_norm=C, normalize_by=B)
    # small end-to-end proxy: just confirm the max_norm metadata equals C/B
    G = 0.1 * torch.randn(16, 4, 64, device=device)
    res, _ = cf(G, state=c_st)
    expected = C / B
    check(
        "noise  [mean-convention: max_norm == C/B]",
        abs(res.max_norm - expected) < 1e-12,
        f"max_norm={res.max_norm} expected={expected}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    paths = ["stream", "nostream", "mb4"]

    print("== CPU / float32 ==")
    test_parity_and_invariant("cpu", torch.float32, paths)
    print("== CPU / bfloat16 ==")
    test_parity_and_invariant("cpu", torch.bfloat16, paths)
    if args.mps:
        if torch.backends.mps.is_available():
            print("== MPS / float32 ==")
            test_parity_and_invariant("mps", torch.float32, paths)
            print("== MPS / bfloat16 ==")
            test_parity_and_invariant("mps", torch.bfloat16, paths)
        else:
            NOTES.append("MPS unavailable; skipped")
    if args.cuda and torch.cuda.is_available():
        print("== CUDA / float32 ==")
        test_parity_and_invariant("cuda", torch.float32, paths)
        print("== CUDA / bfloat16 ==")
        test_parity_and_invariant("cuda", torch.bfloat16, paths)
    print("== noise convention ==")
    test_noise_convention("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print("\n==== SUMMARY ====")
    print(f"failures: {len(FAILURES)}")
    for f in FAILURES:
        print("  FAIL:", f)


if __name__ == "__main__":
    main()
