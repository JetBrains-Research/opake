"""Bitwise check: packaged multi-tensor clip vs the previous per-tensor kernels.

Runs ``clipped_fun(clip_backend="triton")`` twice per case: as packaged
(multi-tensor kernels, shared markers, foreach accumulator), then with the
c13e3007 per-tensor kernels, per-leaf markers and leafwise accumulator
swapped in.

Blocking: clipped sums and per-example norms bitwise equal (they set the scale
and the privacy bound). Diagnostic: the ``clipped_norms`` aux sums tile
partials in a different fp32 order, so it may differ in the last bits; it must
stay within 1e-6 relative. Nothing in clipping, noise or accounting reads it.

Usage (CUDA host, repo root): .venv/bin/python experiments/clip_fused/check_packaged_vs_per_tensor.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import _clip_sum_per_tensor as old  # noqa: E402

from opake.api.engine.clipping._clipped_fun import clipped_fun  # noqa: E402
from opake.api.engine.pytree import tree_map  # noqa: E402

CS = importlib.import_module("opake.api.engine.kernels._clip_sum")
CF = importlib.import_module("opake.api.engine.clipping._clipped_fun")
NEW = (CS.fused_clip_sum, CS._dtype_marker, CF._add_trees)
OLD = (old.fused_clip_sum, lambda leaf: leaf.new_zeros(()),
       lambda t, n: tree_map(lambda a, b: a + b, t, n))

LORA = [(16, 3584), (3584, 16), (16, 3584), (512, 16), (16, 3584), (512, 16),
        (16, 3584), (3584, 16), (16, 3584), (18944, 16), (16, 3584), (18944, 16),
        (16, 18944), (3584, 16)]


def tree(kind, batch, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "lora":
        shapes = [s for _ in range(28) for s in LORA]
        dts = [torch.float32] * len(shapes)
    else:
        shapes = [(64, 48), (48,), (3, 5, 7), (2049,), (7, 300)]
        dts = {"fp32": [torch.float32] * 5, "bf16": [torch.bfloat16] * 5,
               "mixed": [torch.float32, torch.bfloat16] * 2 + [torch.float32]}[kind]
    out = {}
    for i, (s, d) in enumerate(zip(shapes, dts, strict=True)):
        v = torch.randn(batch, *s, generator=g) * (0.02 if kind == "lora" else 0.2)
        out[f"p{i}"] = v.to(device="cuda", dtype=d)
    out["p0"][0] *= 50.0  # one example far above C, the rest mixed
    return out


def run(impl, t, mb):
    CS.fused_clip_sum, CS._dtype_marker, CF._add_trees = impl
    fn, st = clipped_fun(lambda v: v, batch_argnums=0, clip_backend="triton",
                         clipping_norm=1.0, return_aux=True, microbatch_size=mb)
    (out, aux), _ = fn(t, state=st)
    torch.cuda.synchronize()
    return out, aux


fails, n = [], 0
for kind in ("fp32", "bf16", "mixed", "lora"):
    for batch in (1, 2, 16):
        for mb in (None, 4):
            if mb and mb >= batch:
                continue
            t = tree(kind, batch, seed=batch)
            (a, aa), (b, ba) = run(NEW, t, mb), run(OLD, t, mb)
            vals = all(torch.equal(a.pytree[k], b.pytree[k]) and a.pytree[k].dtype == b.pytree[k].dtype
                       for k in t)
            rel = lambda x, y: float(((x.double() - y.double()).abs() / y.double().abs().clamp_min(1e-300)).max())  # noqa: E731
            vdiff = max(float((a.pytree[k].double() - b.pytree[k].double()).abs().max()
                              / b.pytree[k].double().abs().max()) for k in t)
            norms = torch.equal(aa.norms, ba.norms)
            cnorms = torch.equal(aa.clipped_norms, ba.clipped_norms)
            same = vals and norms and a.max_norm == b.max_norm and rel(aa.clipped_norms, ba.clipped_norms) <= 1e-6
            n += 1
            tag = f"{kind} B={batch} mb={mb}"
            print(f"{'PASS' if same else 'FAIL'} {tag}: values={vals} (max rel {vdiff:.1e}) "
                  f"norms={norms} (max rel {rel(aa.norms, ba.norms):.1e}) "
                  f"clipped_norms={cnorms} (max rel {rel(aa.clipped_norms, ba.clipped_norms):.1e})")
            if not same:
                fails.append(tag)
CS.fused_clip_sum, CS._dtype_marker, CF._add_trees = NEW
print(f"cases: {n} (values+norms bitwise, clipped_norms <= 1e-6 rel), failures: {len(fails)} {fails}")
