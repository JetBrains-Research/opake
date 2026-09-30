"""Prototype: fused in-vmap clip kernel (plan v1, Phase 1.1 alternative path).

Target op: clip+sum over a STACKED per-example leaf x of shape [B, *N] whose
dim-0 is the per-example axis and whose norm is taken over ALL trailing dims
(the per-example flat-L2 contract of the vmap clip pipeline).

Replaces, per leaf, the current stream chain
  pass1: nan_to_num(copy) -> vmap(_leaf_sq_sum) [several kernels]
  scalar: sqrt / div / guard-shrink / clamp / finite-check [several kernels]
  pass2: nan_to_num(copy) -> vmap(_scale_tensor): to(f32) mul, cast storage,
         subnormal nudge -> torch.sum(dim=0, dtype=f32) -> cast back
with three launches:
  K1  per-(example, column-tile) partial sum-of-squares (fp32 squares, fp64
      accumulator), NaN/Inf sanitized on read, deterministic per program
  C1  tiny combine: partial[B,T] -> norm -> scale (torch ops on [B] fp64
      vectors; revisit as one fused Triton prologue if launches dominate)
  K2  per-column-block program loops examples b=0..B-1 in register, fusing
      sanitize + scale-multiply-cast-storage + subnormal nudge + fp32
      accumulation. No atomics, no clipped-copy materialization, no
      sanitized-copy materialization; deterministic b-ascending order.

Semantics mirrored from opake.api.engine.clipping._pytree / _clipped_fun:
  * sanitize nan/+-inf -> 0 before norm AND before scaling
  * ratio = C / norm, NO epsilon (non-finite check replaces inf -> scale 0)
  * scale = ratio*relative - absolute, then min(1,·), then !finite -> 0,
    then clamp(min=0)   [guard BEFORE min(1), per _finalize_scale]
  * multiply at promote(storage, f32) == f32; single cast to storage dtype
  * subnormal round-up nudge only when multiply_dtype != storage (fp32:
    stored == product, never fires; bf16: real, so implemented)
  * dim-0 sum accumulates fp32 then casts back to input dtype

Gates before any package integration: fp64-oracle parity (sanitized oracle),
stored-value bound ‖clipped‖ <= C + 1e-5 recomputed on audited stored values,
run-to-run determinism, peak memory, and speed vs the reconstructed chain.
CUDA-only. Prototype under experiments/, not the shipped package.

Run on the CUDA host:
  .venv/bin/python experiments/clip_fused/fused_clip.py --quick
  .venv/bin/python experiments/clip_fused/fused_clip.py
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl

import opake.api.engine.clipping._pytree as cp


# ------------------------------------------------------------------ kernels


@triton.jit
def _partial_sq_kernel(X, PARTIAL, N, BLOCK: tl.constexpr):
    """K1: PARTIAL[b, t] = sum of sanitized(x*x) over one (example, slab)."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
    # sanitize without libdevice: finite iff (x - x) == 0
    x = tl.where((x - x) == 0.0, x, 0.0)
    xf = x.to(tl.float32)
    sq = xf * xf
    partial = tl.sum(sq.to(tl.float64), axis=0)
    T = tl.num_programs(1)
    tl.store(PARTIAL + b * T + t, partial)


@triton.jit
def _clip_sum_kernel(X, SCALE, OUT, N, B, BLOCK: tl.constexpr,
                     SM_NORMAL: tl.constexpr):
    """K2: column-slab program, deterministic example loop, fused ops.

    Per element: read x, sanitize, multiply by scale[b] at f32, single cast
    to storage dtype, bf16 subnormal nudge, accumulate at f32. Writes only
    OUT (one output vector). Reduction order: b ascending, fixed.
    """
    t = tl.program_id(0)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in range(B):
        x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
        x = tl.where((x - x) == 0.0, x, 0.0)
        s = tl.load(SCALE + b)
        prod = x.to(tl.float32) * s
        stored = prod.to(X.dtype.element_ty)  # single cast to storage
        if X.dtype.element_ty == tl.bfloat16:
            sa = tl.abs(stored.to(tl.float32))
            bad = (sa <= SM_NORMAL) & (sa > tl.abs(prod))
            stored = tl.where(bad, 0.0, stored)
        acc += stored.to(tl.float32)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


# ------------------------------------------------------------------ wrappers

_BLOCK = 2048


def _tiles(n: int, block: int) -> int:
    return (n + block - 1) // block


def scale_from_partial(partial: torch.Tensor, C: float, rel: float,
                       abs_: float) -> tuple[torch.Tensor, torch.Tensor]:
    """C1: combine partials, norm, scale. [B] fp64 math, exact formula."""
    sq = partial.sum(1)                      # fp64, tree-order per row
    norm = torch.sqrt(sq)
    r = C / norm                              # inf allowed here
    s = r * rel - abs_
    s = torch.minimum(torch.ones_like(s), s)  # min(1) AFTER guard
    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    s = torch.clamp(s, min=0.0)
    return s.to(torch.float32), norm


def fused_clip_sum(x: torch.Tensor, C: float, rel: float, abs_: float,
                   block: int = _BLOCK):
    """Clip+sum one stacked leaf [B, *N] -> (sum in input dtype, scale[B])."""
    x = x.contiguous()
    B = x.shape[0]
    N = x.numel() // B
    T = _tiles(N, block)
    partial = torch.empty((B, T), dtype=torch.float64, device=x.device)
    _partial_sq_kernel[(B, T)](x, partial, N, BLOCK=block)
    scale, _norm = scale_from_partial(partial, C, rel, abs_)
    out = torch.empty(x.shape[1:], dtype=x.dtype, device=x.device)
    sm = (torch.finfo(torch.bfloat16).smallest_normal
          if x.dtype == torch.bfloat16 else 0.0)
    _clip_sum_kernel[(T,)](x, scale, out, N, B, BLOCK=block,
                           SM_NORMAL=float(sm), num_warps=4)
    return out, scale


# ------------------------------------------- reference chain + oracle


def production_chain(x: torch.Tensor, C: float, rel: float, abs_: float):
    """Reconstruction of the current stream chain for ONE leaf, using the
    same private primitives (what the fused kernel must match and beat)."""
    from torch.func import vmap

    acc = torch.float32
    sqdt = torch.float64
    x1 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sq = vmap(lambda t: cp._leaf_sq_sum(t, acc, sqdt))(x1)
    norm = torch.sqrt(sq)
    scale = C / norm
    scale = scale * rel - abs_
    scale = torch.minimum(torch.ones_like(scale), scale)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    scale = torch.clamp(scale, min=0.0)
    x2 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def scale_one(v, s):
        md = torch.promote_types(v.dtype, torch.float32)
        scaled = v.to(dtype=md) * s.to(dtype=md)
        stored = scaled.to(dtype=v.dtype)
        if v.dtype == torch.bfloat16:
            sn = torch.finfo(torch.bfloat16).smallest_normal
            bad = (stored.abs() <= sn) & (stored.abs() > scaled.abs())
            stored = torch.where(bad, torch.zeros_like(stored), stored)
        return stored

    stored = vmap(scale_one)(x2, scale)
    out = stored.sum(0, dtype=torch.float32).to(x.dtype)
    return out, stored


def fp64_oracle(x: torch.Tensor, C: float) -> torch.Tensor:
    g = torch.nan_to_num(x.double(), nan=0.0, posinf=0.0, neginf=0.0)
    norms = g.norm(dim=tuple(range(1, g.ndim)), keepdim=True)
    scale = (C / norms).clamp(max=1.0)
    return (g * scale).sum(0)


# ------------------------------------------------------------------- harness


def make_data(B, shape, mode, dtype, device, seed):
    """Per-example norms straddle/tower/sit-below C; salt NaN/Inf entries."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 1
    for s in shape:
        n *= s
    unit = 1.0 / (n**0.5)
    base = torch.randn(B, *shape, generator=gen, dtype=torch.float64) * unit
    if mode == "straddle":
        per = torch.empty(B).uniform_(0.3, 2.0, generator=gen)
    elif mode == "clipped":
        per = torch.full((B,), 4.0)
    elif mode == "below":
        per = torch.full((B,), 0.4)
    else:
        raise ValueError(mode)
    x64 = base * per.reshape(-1, 1, 1)
    x64.view(-1)[::500003] = float("nan")     # ~0.03% salt
    x64.view(-1)[1::500027] = float("inf")
    return x64.to(dtype).to(device)


def guard_constants(dtype, widest_n):
    """Same formulas as _guard_scale/_norm_roundoff, CUDA fp64 accumulator."""
    if dtype == torch.float32:
        rounding = torch.finfo(torch.float32).eps
        storage = torch.finfo(torch.float32)
    else:
        rounding = (torch.finfo(torch.bfloat16).eps / 2
                    + torch.finfo(torch.float32).eps)
        storage = torch.finfo(torch.bfloat16)
    if widest_n > cp._BLOCKED_REDUCTION_MIN:
        terms = cp._blocked_reduction_terms(widest_n)
    else:
        terms = widest_n
    roundoff = cp._norm_roundoff(torch.float32, torch.float64, 1, terms)
    rel = 1.0 - float(rounding) - 2.0 * roundoff
    abs_ = float(storage.smallest_normal * storage.eps)
    return rel, abs_


def run_case(B, shape, mode, dtype, C=1.0, iters=10):
    dev = "cuda"
    n = 1
    for s in shape:
        n *= s
    x = make_data(B, shape, mode, dtype, dev,
                  seed=hash((B, n, mode, str(dtype))) % 2**31)

    rel, abs_ = guard_constants(dtype, n)

    # --- correctness ---
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fused, scale = fused_clip_sum(x, C, rel, abs_)
    torch.cuda.synchronize()
    peak_fused = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    ref, stored_ref = production_chain(x, C, rel, abs_)
    torch.cuda.synchronize()
    peak_chain = torch.cuda.max_memory_allocated()

    oracle = fp64_oracle(x.cpu(), C)
    f64 = fused.float().double().cpu()
    d_of = (f64 - oracle).abs().max().item()
    r_of = d_of / max(oracle.abs().max().item(), 1e-12)
    d_chain = (ref.float().double().cpu() - oracle).abs().max().item()
    # fused vs chain: both deterministic here, compare loosely at dtype tol
    ok_chain = torch.allclose(fused.float(), ref.float(),
                              rtol=3e-4 if dtype == torch.float32 else 4e-2,
                              atol=1e-4 if dtype == torch.float32 else 2e-2)
    # tolerance: generous on abs because fp64 oracle has different
    # accumulation order; the SHARED reduction errors must stay ~1e-4 rel.
    ok_oracle = r_of < (3e-4 if dtype == torch.float32 else 6e-2)

    # stored-value bound on AUDITED stored values from the fused scales
    x1 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = x1.float() * scale.view(-1, *([1] * (x.ndim - 1)))
    stored_audit = scaled.to(dtype)
    if dtype == torch.bfloat16:
        sn = torch.finfo(torch.bfloat16).smallest_normal
        bad = (stored_audit.abs() <= sn) & (stored_audit.abs() > scaled.abs())
        stored_audit = torch.where(bad, torch.zeros_like(stored_audit),
                                   stored_audit)
    cn = stored_audit.double().norm(dim=tuple(range(1, x.ndim)))
    ok_bound = bool((cn <= C + 1e-5).all())
    max_cn = cn.max().item()

    fused2, _ = fused_clip_sum(x, C, rel, abs_)
    ok_det = bool((fused == fused2).all())

    # --- perf ---
    def bench(fn, w=3):
        for _ in range(w):
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

    t_f = bench(lambda: fused_clip_sum(x, C, rel, abs_))
    t_c = bench(lambda: production_chain(x, C, rel, abs_))

    tag = f"[{B}x{'x'.join(map(str, shape))},{str(dtype).split('.')[-1]},{mode}]"
    status = "PASS" if (ok_oracle and ok_chain and ok_bound and ok_det) else "FAIL"
    print(f"{status} {tag:40s} fused {t_f:8.2f}ms  chain {t_c:8.2f}ms  "
          f"speedup x{t_c / t_f:6.2f}  |d|orc={d_of:.2e} chain={d_chain:.2e}  "
          f"bound‖·‖={max_cn:.6f} det={ok_det}  "
          f"peak f={peak_fused / 2**20:.0f}MB c={peak_chain / 2**20:.0f}MB")
    ok = status == "PASS"
    if not ok:
        reasons = []
        if not ok_oracle:
            reasons.append("oracle-parity")
        if not ok_chain:
            reasons.append("chain-mismatch")
        if not ok_bound:
            reasons.append("BOUND-VIOLATION")
        if not ok_det:
            reasons.append("nondeterminism")
        print("      reasons:", ", ".join(reasons))
    return ok, t_f, t_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA host required"
    print("bf16 smallest_normal:", float(torch.finfo(torch.bfloat16).smallest_normal))

    cases = []
    for dt in (torch.float32, torch.bfloat16):
        cases += [
            (64, (4096, 4096), "straddle", dt),
            (64, (4096, 4096), "clipped", dt),
            (64, (4096,), "straddle", dt),
            (128, (128, 128), "straddle", dt),
        ]
    if args.quick:
        cases = cases[:1]
    results = [run_case(*c) for c in cases]
    n_fail = sum(1 for r in results if not r[0])
    tf = sum(r[1] for r in results)
    tc = sum(r[2] for r in results)
    print(f"\nfailures: {n_fail}   total fused {tf:.1f}ms vs chain {tc:.1f}ms "
          f"(x{tc / tf:.2f})")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
