"""Exact check: CUDA-graphed chunk kernel vs eager on a small model (CUDA).

Two independent runs from the same initial state and data: eager, and with
``_chunk_compiler=CudaGraphChunk``. Each runs several SGD steps with batch
sizes that leave partial microbatches (several capture keys). For fixed and
adaptive clipping (the threshold changes every step), gradients, the clip
state's threshold and aux norms must be bitwise equal at every step.

Usage (CUDA host, repo root):
  .venv/bin/python experiments/cuda_graph/check_cudagraph_small.py [torch|triton]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import cudagraph_chunk  # noqa: E402
from cudagraph_chunk import CudaGraphChunk, report  # noqa: E402

from opake.dpsgd.clipping import adaptive_clipped_grad, clipped_grad  # noqa: E402
from opake.random import key  # noqa: E402

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "torch"
if BACKEND != "torch":
    from opake.patches import apply_runtime_patches

    apply_runtime_patches()
    cudagraph_chunk.allow_fused_clip()
torch.manual_seed(0)
DEV = "cuda"
D, H, O = 32, 64, 8
P0 = {
    "w1": torch.randn(D, H, device=DEV) * 0.2, "b1": torch.zeros(H, device=DEV),
    "w2": torch.randn(H, O, device=DEV) * 0.2, "b2": torch.zeros(O, device=DEV),
}
BATCHES = [12, 10, 13, 12, 9, 12]
DATA = [(torch.randn(b, D, device=DEV), torch.randint(0, O, (b,), device=DEV)) for b in BATCHES]


def loss(p, x, y):
    h = torch.tanh(x @ p["w1"] + p["b1"])
    return F.cross_entropy(h @ p["w2"] + p["b2"], y)


def run(mode, compiler):
    kw = {"argnums": 0, "batch_argnums": (1, 2), "microbatch_size": 4, "return_aux": True,
          "normalize_by": 12, "clip_backend": BACKEND}
    if compiler is not None:
        kw["_chunk_compiler"] = compiler
    if mode == "adaptive":
        fn, state = adaptive_clipped_grad(loss, initial_clipping_norm=0.5, target_quantile=0.5,
                                          key=key(7), **kw)
    else:
        fn, state = clipped_grad(loss, clipping_norm=0.5, **kw)
    p = {k: v.clone() for k, v in P0.items()}
    trace = []
    for x, y in DATA:
        (grads, aux), state = fn(p, x, y, state=state)
        g = grads.pytree if hasattr(grads, "pytree") else grads
        thr = getattr(state, "_current_clipping_norm", None)
        trace.append(({k: v.clone() for k, v in g.items()},
                      aux.grad_norms.clone(), aux.clipped_grad_norms.clone(),
                      None if thr is None else (thr.clone() if isinstance(thr, torch.Tensor) else thr)))
        p = {k: v - 0.5 * g[k] for k, v in p.items()}  # parameters change every step
    return trace


ok_all = True
for mode in ("fixed", "adaptive"):
    eager = run(mode, None)
    graphed = run(mode, CudaGraphChunk)
    for step, (e, g) in enumerate(zip(eager, graphed, strict=True)):
        same = all(torch.equal(e[0][k], g[0][k]) for k in e[0])
        same &= torch.equal(e[1], g[1]) and torch.equal(e[2], g[2])
        same &= (e[3] == g[3]) if not isinstance(e[3], torch.Tensor) else torch.equal(e[3], g[3])
        ok_all &= same
        print(f"{mode:8s} step {step} batch {BATCHES[step]:2d}: bitwise={same} threshold={e[3]}")
    if mode == "adaptive":
        thresholds = {t[3] for t in eager}
        assert len(thresholds) > 1, f"adaptive threshold never changed: {thresholds}"
        print(f"adaptive threshold took {len(thresholds)} distinct values across {len(eager)} steps")
print("report:", {k: v for k, v in report().items() if k != "keys"}, "capture keys (chunk sizes):", report()["keys"])
print("ALL BITWISE EQUAL" if ok_all else "MISMATCH")
sys.exit(0 if ok_all else 1)
