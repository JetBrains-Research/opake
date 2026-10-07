"""Gate + microbenchmark for the multi-tensor clip kernel and vmap(chunk_size).

A. Correctness (through clipped_fun / adaptive_clipped_grad, clip_backend="triton"):
   parity multi-tensor vs per-leaf vs torch; zero-tolerance stored-value bound
   (B=1 trick, random + adversarial); determinism; spy that it really ran;
   vmap(chunk_size) vs Python microbatching parity.
B. Per-call seam cost on a LoRA-7B-shaped tree (392 leaves, 40.4M elems):
   torch / per-leaf fused / multi-tensor fused, B=2 and B=16, with CUDA
   kernel-launch counts from torch.profiler.

Usage (CUDA host, repo root): .venv/bin/python experiments/clip_fused/bench_mt.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.patches import apply_runtime_patches
from opake.random import key

sys.path.insert(0, str(Path(__file__).parent))
import mt_engine  # noqa: E402

apply_runtime_patches()

DEV = "cuda"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAIL.append(name)


LORA_LAYER = [(16, 3584), (3584, 16), (16, 3584), (512, 16), (16, 3584), (512, 16), (16, 3584),
              (3584, 16), (16, 3584), (18944, 16), (16, 3584), (18944, 16), (16, 18944), (3584, 16)]


def tree(batch, layers, dtype_of, seed, norms=(0.3, 2.0)):
    g = torch.Generator().manual_seed(seed)
    shapes = [s for _ in range(layers) for s in LORA_LAYER]
    raw = [torch.randn(batch, *s, generator=g, dtype=torch.float64) for s in shapes]
    total = torch.sqrt(sum((r.flatten(1) ** 2).sum(1) for r in raw))
    target = torch.empty(batch, dtype=torch.float64).uniform_(*norms, generator=g)
    return {f"p{i}": (r * (target / total).view(-1, 1, 1)).to(dtype_of(i)).to(DEV)
            for i, r in enumerate(raw)}


def clip(tree_, **kw):
    fn, st = clipped_fun(lambda v: v, batch_argnums=0, clip_backend="triton", **kw)
    return fn(tree_, state=st)[0]


def run_with(mt, fn):
    (mt_engine.install if mt else mt_engine.uninstall)()
    try:
        return fn()
    finally:
        mt_engine.uninstall()


FP32 = lambda i: torch.float32  # noqa: E731
BF16 = lambda i: torch.bfloat16  # noqa: E731
MIXED = lambda i: torch.bfloat16 if i % 3 == 1 else torch.float32  # noqa: E731

# ---------------- A. correctness ----------------
calls = []
real_mt = mt_engine.mt_fused_clip_sum
mt_engine.mt_fused_clip_sum = lambda *a, **k: (calls.append(1), real_mt(*a, **k))[1]  # spy
for dname, dt in (("fp32", FP32), ("bf16", BF16), ("mixed", MIXED)):
    for batch in (1, 2, 16):
        for mb in (None, 4):
            if mb and mb >= batch:
                continue
            t = tree(batch, 4, dt, seed=batch)
            kw = dict(clipping_norm=1.0, normalize_by=batch, return_aux=True, microbatch_size=mb)
            calls.clear()
            (o_mt, a_mt) = run_with(True, lambda: clip(t, **kw))
            ran = len(calls) > 0
            (o_pl, a_pl) = run_with(False, lambda: clip(t, **kw))
            fn, st = clipped_fun(lambda v: v, batch_argnums=0, clip_backend="torch", **kw)
            (o_t, a_t) = fn(t, state=st)[0]
            bit = all(torch.equal(o_mt.pytree[k], o_pl.pytree[k]) for k in t)
            worst = max(((o_mt.pytree[k].double() - o_t.pytree[k].double()).abs().max()
                         / o_t.pytree[k].double().abs().max()).item() for k in t)
            tol = 1e-5 if dname == "fp32" else 2e-2
            ok = ran and worst < tol and torch.allclose(a_mt.norms, a_pl.norms, rtol=1e-6) \
                and torch.allclose(a_mt.clipped_norms, a_t.clipped_norms, rtol=1e-5, atol=1e-7)
            check(f"parity {dname} B={batch} mb={mb}", ok,
                  f"mt-ran={ran} bitwise-vs-per-leaf={bit} max rel vs torch={worst:.1e}")

def stored_norm(o):
    return math.sqrt(sum(float((v.double().cpu() ** 2).sum()) for v in o.pytree.values()))

for dname, dt in (("fp32", FP32), ("bf16", BF16), ("mixed", MIXED)):
    for ratio in (4.0, 1.0001, 0.99995):
        worst = max(stored_norm(run_with(True, lambda: clip(tree(1, 2, dt, seed=s, norms=(ratio, ratio)),
                                                         clipping_norm=1.0))) for s in range(8))
        check(f"bound {dname} ratio={ratio}", worst <= 1.0, f"max ||stored|| = {worst:.9f} (<= 1 exactly)")
for dtype, m in ((torch.float32, 23), (torch.bfloat16, 7)):
    n = 1024
    v = 2.0**-4 * (1 + 2.0**-m)
    half = 2.0 ** -(m + 1) / (1 + 2.0**-m)
    sg = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(0)) < 0.5, -1.0, 1.0)
    x = {"a": (sg[:512] * v).to(dtype).reshape(1, 512).to(DEV), "b": (sg[512:] * v).to(dtype).reshape(1, 512).to(DEV)}
    c = v * math.sqrt(n) / (1 + 0.9 * half)
    s = stored_norm(run_with(True, lambda: clip(x, clipping_norm=c)))
    check(f"bound adversarial round-back {dtype}", s <= c, f"{s / c - 1:+.2e}")

t = tree(16, 4, MIXED, seed=5)
o1 = run_with(True, lambda: clip(t, clipping_norm=1.0))
o2 = run_with(True, lambda: clip(t, clipping_norm=1.0))
check("determinism", all(torch.equal(o1.pytree[k], o2.pytree[k]) for k in t))
mt_engine.mt_fused_clip_sum = real_mt

# vmap(chunk_size) vs Python microbatching, through adaptive_clipped_grad
torch.manual_seed(0)
net = torch.nn.Sequential(*[m for _ in range(6) for m in (torch.nn.Linear(96, 96), torch.nn.GELU())]).to(DEV)
params = {n_: p.detach() for n_, p in net.named_parameters()}
X = torch.randn(16, 7, 96, device=DEV)
Y = torch.randn(16, 7, 96, device=DEV) * 3
loss_fn = lambda p, x, y: F.mse_loss(functional_call(net, p, (x,)), y)  # noqa: E731
def adaptive(mb, backend):
    fn, st = adaptive_clipped_grad(loss_fn, batch_argnums=(1, 2), initial_clipping_norm=0.5,
                                   microbatch_size=mb, return_aux=True, key=key(0),
                                   normalize_by=16, clip_backend=backend)
    return fn(params, X, Y, state=st)
((g_ref, a_ref), s_ref) = adaptive(2, "torch")
mt_engine.install(); mt_engine.install_vmap_chunk(2)
try:
    ((g_ch, a_ch), s_ch) = adaptive(None, "triton")
finally:
    mt_engine.uninstall(); mt_engine.uninstall_vmap_chunk()
worst = max(((g_ch.pytree[k] - g_ref.pytree[k]).abs().max() / g_ref.pytree[k].abs().max()).item() for k in params)
check("vmap(chunk_size)+mt vs microbatch torch (adaptive+aux)",
      worst < 1e-5 and torch.allclose(a_ch.grad_norms, a_ref.grad_norms, rtol=1e-5)
      and abs(float(s_ch._next_clipping_norm) - float(s_ref._next_clipping_norm)) < 1e-6,
      f"max rel grad diff {worst:.1e}")

# ---------------- B. per-call seam cost ----------------
def timed(fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

def launches(fn):
    fn(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)

print("\nper-call seam cost, LoRA-7B-shaped tree (392 leaves), fp32, return_aux=True:")
for batch in (2, 16):
    t = tree(batch, 28, FP32, seed=9)
    print(f"  tree: {len(t)} leaves, {sum(v[0].numel() for v in t.values()) / 1e6:.1f}M elems/example, B={batch}")
    for name, backend, mt in (("torch", "torch", False), ("per-leaf triton", "triton", False),
                              ("multi-tensor triton", "triton", True)):
        fn, st = clipped_fun(lambda v: v, batch_argnums=0, clip_backend=backend,
                             clipping_norm=1.0, return_aux=True)
        call = lambda: fn(t, state=st)  # noqa: E731
        (mt_engine.install if mt else mt_engine.uninstall)()
        try:
            ms, nk = timed(call), launches(call)
        finally:
            mt_engine.uninstall()
        print(f"    {name:20s} {ms:8.2f} ms/call   {nk:6d} CUDA kernels/call")
print(f"\nfailures: {len(FAIL)} {FAIL}")
