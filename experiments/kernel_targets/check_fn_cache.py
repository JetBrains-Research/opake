"""Correctness + overhead check for functorch_fn_cache (CPU is enough).

Correctness: per-example grads through custom autograd.Functions with saved
tensors and a vmap staticmethod (including a Function whose backward calls
another Function, Opake's two-level pattern) must be bitwise equal with and
without the cache, under vmap(grad) and plain grad, and in torch.autograd.
Overhead: bench_function_overhead.per_call_us with and without the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.func import grad, vmap

sys.path.insert(0, str(Path(__file__).parent))
import bench_function_overhead as bfo  # noqa: E402
import functorch_fn_cache as cache  # noqa: E402


class _SquareBackward(torch.autograd.Function):
    @staticmethod
    def forward(g, x):
        return 2 * x * g

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])

    @staticmethod
    def backward(ctx, gg):
        (x,) = ctx.saved_tensors
        return 2 * x * gg, None

    @staticmethod
    def vmap(info, in_dims, g, x):
        gd, xd = in_dims
        g = g.movedim(gd, 0) if gd is not None else g.expand(info.batch_size, *g.shape)
        x = x.movedim(xd, 0) if xd is not None else x.expand(info.batch_size, *x.shape)
        return 2 * x * g, 0


class Square(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x * x

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return _SquareBackward.apply(g, x)

    @staticmethod
    def vmap(info, in_dims, x):
        return x * x, in_dims[0]


def loss(w, x):
    h = Square.apply(x @ w)
    h = Square.apply(h * 0.1) + h
    return h.sum()


torch.manual_seed(0)
w = torch.randn(5, 3, dtype=torch.float64)
xs = torch.randn(4, 7, 5, dtype=torch.float64)


def results():
    per_ex = vmap(grad(loss), in_dims=(None, 0))(w, xs)
    single = grad(loss)(w, xs[0])
    wr = w.clone().requires_grad_()
    loss(wr, xs[1]).backward()
    return per_ex, single, wr.grad


ref = results()
cache.install()
try:
    new = results()
    new2 = results()  # second call reuses cached classes
finally:
    cache.uninstall()
ok = all(torch.equal(a, b) for a, b in zip(ref, new)) and all(torch.equal(a, b) for a, b in zip(ref, new2))
exact = torch.stack([grad(loss)(w, x) for x in xs])
ok &= torch.allclose(ref[0], exact, rtol=0, atol=0)
print(f"bitwise equal with/without cache (vmap(grad), grad, autograd), and vs per-example loop: {ok}")
print(f"cached classes: {len(cache._CACHE)}")

base = bfo.per_call_us(bfo.FnVmap.apply, 200, 20)
cache.install()
try:
    cached = bfo.per_call_us(bfo.FnVmap.apply, 200, 20)
finally:
    cache.uninstall()
plain = bfo.per_call_us(bfo.OPS["plain"], 200, 20)
print(f"per call inside vmap(grad): plain {plain:.1f} us | autograd.Function {base:.1f} us | with cache {cached:.1f} us")
sys.exit(0 if ok else 1)
