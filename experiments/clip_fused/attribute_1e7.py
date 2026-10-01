"""Attribute fused-vs-production differences to (1) norm order, (2) batch-sum order, (3) bf16 cast."""
import math, sys, torch
from torch.func import vmap
import opake.api.engine.clipping._pytree as cp
from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping._streaming import _FIXED_SCALE
sys.path.insert(0, "experiments/clip_fused"); import fused_engine as fe
dev = "cuda"
def rel(a, b, ref):
    return ((a.double() - b.double()).abs().max() / ref.double().abs().max()).item()
def run(B, shape, dt, seed=0):
    g = torch.Generator().manual_seed(seed); n = math.prod(shape)
    x64 = torch.randn(B, *shape, generator=g, dtype=torch.float64) / math.sqrt(n)
    x = (x64 * torch.empty(B).uniform_(0.3, 2.0, generator=g).double().view(-1, *[1] * len(shape))).to(dt).to(dev)
    C = torch.tensor(1.0, device=dev); bshape = (-1, *[1] * len(shape))
    acc, sq = torch.float32, torch.float64
    ro = cp._norm_roundoff(acc, sq, 1, cp._reduction_terms([x[0]]))
    # production norm and scale
    norm_p = torch.sqrt(vmap(lambda t: cp._leaf_sq_sum(t, acc, sq))(x))
    scale_p = cp._finalize_scale(C.double() / norm_p, dt, acc, ro, clamp_to_one=True).to(acc)
    # fused norm and scale (kernel K1 + same production guard)
    _, norm_f, _, _ = fe._fused_impl([x], C, with_postsq=False)
    scale_f = cp._finalize_scale(C.double() / norm_f, dt, acc, ro, clamp_to_one=True).to(acc)
    def stored(s):
        p = x.float() * s.view(bshape); st = p.to(dt)
        if dt == torch.bfloat16:
            bad = (st.abs() <= torch.finfo(dt).smallest_normal) & (st.float().abs() > p.abs())
            st = torch.where(bad, torch.zeros_like(st), st)
        return st
    sp, sf = stored(scale_p), stored(scale_f)
    tsum = lambda s: torch.sum(s, dim=0, dtype=torch.float32)
    def seqsum(s):
        a = torch.zeros(s.shape[1:], dtype=torch.float32, device=dev)
        for b in range(s.shape[0]): a += s[b].float()
        return a
    red32 = lambda t: torch.sum(t, dim=0, dtype=torch.float32)  # fp32 output: no cast-back
    P = fe._ORIGINAL_STREAM({"w": x}, C, scale=_FIXED_SCALE, batch_size=B, reduce_leaf=red32, compute_dtype=None,
                            second_moment=False, return_aux=False, return_stats=False)[0]["w"]
    F, _, _, _ = fe._fused_impl([x], C, with_postsq=False, out_dtypes={dt: torch.float32}); F = F[0]
    # decomposition sanity: production == torch-order sum of production-scaled values; fused == sequential sum of fused-scaled
    assert torch.equal(P, tsum(sp)), "production decomposition mismatch"
    def fmasum(st, sc):  # FMA model: exact product x*s (fits in fp64), one rounding per accumulate
        a = torch.zeros(st.shape[1:], dtype=torch.float32, device=dev)
        for b in range(st.shape[0]): a = (a.double() + x[b].double() * sc[b].double()).float()
        return a
    eq_seq = float((F == seqsum(sf)).float().mean())
    eq_fma = float((F == fmasum(sf, scale_f)).float().mean()) if dt == torch.float32 else float("nan")
    ndiff = int((scale_p != scale_f).sum())
    print(f"[B={B:3d} {str(shape):12s} {str(dt).split('.')[-1]:8s}] "
          f"(1) norm fp64 rel {((norm_f - norm_p).abs() / norm_p).max().item():.1e}, fp32 scales differing {ndiff}/{B} | "
          f"(2) sum-order only {rel(tsum(sp), seqsum(sp), P):.1e} | scale only {rel(tsum(sp), tsum(sf), P):.1e} | "
          f"total fp32 {rel(P, F, P):.1e} | bitwise-equal elems {float((P == F).float().mean()):.1%}"
          + f" | kernel==round-then-add {eq_seq:.2%} kernel==FMA-model {eq_fma:.2%}"
          + (f" | contraction only {rel(seqsum(sf), fmasum(sf, scale_f), P):.1e}" if dt == torch.float32 else "")
          + (f" | (3) after bf16 cast {rel(P.to(dt), F.to(dt), P):.1e}" if dt == torch.bfloat16 else ""))
for dt in (torch.float32, torch.bfloat16):
    for B, shape in [(64, (4096, 1024)), (128, (128, 128)), (2, (3584, 16))]:
        run(B, shape, dt)
