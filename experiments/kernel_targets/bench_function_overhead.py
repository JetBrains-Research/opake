"""Host overhead per call of custom-kernel wrappers under vmap(grad(...)).

Compares, inside torch.func.vmap(torch.func.grad(loss)) with tiny tensors so
compute is negligible:

  plain       one ATen op (x * 2)
  fn_vmap     autograd.Function with setup_context + backward + vmap staticmethod
              (the pattern Opake's Triton kernels use)
  fn_genrule  autograd.Function with generate_vmap_rule = True
  custom_op   torch.library.custom_op + register_autograd + register_vmap
              (reported as FAILS if it does not compose with torch.func.grad)

Each variant applies the op N times in a chain; per-call time = (t - t_empty)/N.
Host-only effect: results transfer across devices in ratio, not in absolute us.

Usage: python experiments/kernel_targets/bench_function_overhead.py [--n 200] [--reps 20]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.func import grad, vmap


class FnVmap(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return x * 2

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, g):
        return g * 2

    @staticmethod
    def vmap(info, in_dims, x):
        return x * 2, in_dims[0]


class FnGenRule(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return x * 2

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, g):
        return g * 2


@torch.library.custom_op("opake_bench::scale2", mutates_args=())
def scale2(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@scale2.register_fake
def _(x):
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    pass


def _backward(ctx, g):
    return scale2(g)


scale2.register_autograd(_backward, setup_context=_setup)


@scale2.register_vmap
def _(info, in_dims, x):
    return x * 2, in_dims[0]


OPS = {
    "plain": lambda x: x * 2,
    "fn_vmap": FnVmap.apply,
    "fn_genrule": FnGenRule.apply,
    "custom_op": scale2,
}


def per_call_us(op, n, reps, batch=2):
    def loss(x):
        for _ in range(n):
            x = op(x) * 0.5
        return x.sum()

    def empty_loss(x):
        for _ in range(n):
            x = x * 0.5
        return x.sum()

    xs = torch.randn(batch, 8)
    f, f0 = vmap(grad(loss)), vmap(grad(empty_loss))
    for _ in range(3):
        f(xs)
        f0(xs)
    t, t0 = [], []
    for _ in range(reps):
        s = time.perf_counter()
        out = f(xs)
        t.append(time.perf_counter() - s)
        s = time.perf_counter()
        f0(xs)
        t0.append(time.perf_counter() - s)
    torch.testing.assert_close(out, torch.ones_like(xs))  # d/dx of prod(2*0.5)=1
    return (statistics.median(t) - statistics.median(t0)) / n * 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    print(f"torch {torch.__version__}; per-call host overhead inside vmap(grad), N={args.n}")
    for name, op in OPS.items():
        try:
            print(f"  {name:11s} {per_call_us(op, args.n, args.reps):8.1f} us/call (fwd+bwd, minus x*0.5 chain)")
        except RuntimeError as exc:
            print(f"  {name:11s} FAILS under vmap(grad): {str(exc).splitlines()[0][:150]}")
