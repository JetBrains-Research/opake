#!/usr/bin/env python3
"""Benchmark streaming matrix and noise hot paths.

Usage:
    uv run benchmarks/bench.py              # MPS
    uv run benchmarks/bench.py --cpu        # CPU

Compare baseline vs optimized:
    git stash
    uv run benchmarks/bench.py --json > baseline.json
    git stash pop
    uv run benchmarks/bench.py --json > optimized.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

DEVICE_NAME = "mps"
WARMUP = 15
ITERATIONS = 50
SEED = 42


def sync():
    if DEVICE_NAME == "cuda":
        torch.cuda.synchronize()
    elif DEVICE_NAME == "mps":
        torch.mps.synchronize()


def median_ms(func):
    """Warm up then time *func*, return median ms."""
    for _ in range(WARMUP):
        func()
        sync()
    times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        func()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# --- Fixtures ---


def _blt_sm():
    from opake.api.dpftrl.noise._blt_math import (
        BufferedToeplitz,
        _streaming_matrix_builder,
    )

    blt = BufferedToeplitz.build(
        buf_decay=[0.95, 0.90, 0.85, 0.80, 0.70],
        output_scale=[0.1, 0.2, 0.25, 0.2, 0.15],
    )
    C_inv = _streaming_matrix_builder(blt).build_inverse()
    template = torch.zeros(768, device=DEVICE_NAME)
    state = C_inv.init_multiply(template)
    return C_inv, state


def _toeplitz_sm():
    from opake.api.dpftrl.noise._toeplitz import inverse_as_streaming_matrix

    coef = torch.tensor([1.0, 0.3, 0.2, 0.1, 0.05], dtype=torch.float64)
    C_inv = inverse_as_streaming_matrix(coef)
    template = torch.zeros(768, device=DEVICE_NAME)
    state = C_inv.init_multiply(template)
    return C_inv, state


def _blt_tree_sm():
    """Lifted BLT inverse state for a 200-leaf pytree (Llama-like)."""
    import optree

    from opake.api.dpftrl.noise._blt_math import (
        BufferedToeplitz,
        _streaming_matrix_builder,
    )
    from opake.api.dpftrl.noise._streaming_matrix import StreamingMatrix

    blt = BufferedToeplitz.build(
        buf_decay=[0.95, 0.90, 0.85, 0.80, 0.70],
        output_scale=[0.1, 0.2, 0.25, 0.2, 0.15],
    )
    C_inv = _streaming_matrix_builder(blt).build_inverse()

    # Build lifted version manually (same as from_array_implementation does)
    def lifted_init(abstract_value):
        if isinstance(abstract_value, torch.Tensor):
            return C_inv.init_multiply(abstract_value)
        flat, spec = optree.tree_flatten(abstract_value)
        flat_states = [C_inv.init_multiply(v) for v in flat]
        return (spec, flat_states)

    def lifted_next(value, state):
        if isinstance(value, torch.Tensor):
            return C_inv.multiply_next(value, state)
        spec, flat_states = state
        flat_values = optree.tree_leaves(value)
        flat_outputs = []
        new_flat_states = []
        for v, s in zip(flat_values, flat_states, strict=True):
            out, ns = C_inv.multiply_next(v, s)
            flat_outputs.append(out)
            new_flat_states.append(ns)
        return spec.unflatten(flat_outputs), (spec, new_flat_states)

    lifted = StreamingMatrix(lifted_init, lifted_next)

    leaves = [torch.zeros(4096, device=DEVICE_NAME) for _ in range(200)]
    tree = {"layers": leaves}
    state = lifted.init_multiply(tree)
    return lifted, state


def _mf_noise_setup():
    """MF noise_fn + state + clipped_grads for a small pytree of 1D leaves.

    BLT inverse _read uses output_scale.unsqueeze(-1) * state which only
    broadcasts correctly for 1D (per-parameter) leaves.  Multi-D leaves
    hit a pre-existing broadcast bug (size 5 vs 4096) — tracked but not
    yet fixed in this pass.
    """
    from opake.api.dpftrl.noise._blt_math import (
        BufferedToeplitz,
        inverse_as_streaming_matrix,
    )
    from opake.api.dpftrl.noise._engine import _matrix_factorization_noise
    from opake.random import key

    blt = BufferedToeplitz.build(
        buf_decay=[0.95, 0.90, 0.85, 0.80, 0.70],
        output_scale=[0.1, 0.2, 0.25, 0.2, 0.15],
    )
    noising = inverse_as_streaming_matrix(blt)
    # 1D leaf sizes matching a small model (3 parameters)
    dims = [4096, 768, 30]
    grad_template = [torch.zeros(d, device=DEVICE_NAME) for d in dims]
    noise_fn, state = _matrix_factorization_noise(grad_template, noising, key=key(SEED))
    clipped_grads = [torch.randn(d, device=DEVICE_NAME) for d in dims]
    return noise_fn, state, clipped_grads


def _dpsgd_noise_setup():
    from opake.dpsgd.noise import gaussian_noise
    from opake.random import key
    from opake.types import clipped

    dims = [(4096, 768), (768, 30), (768, 4096)]
    grads = [torch.randn(*d, device=DEVICE_NAME) for d in dims]
    clipped_tree = clipped(grads, max_norm=1.0)
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(SEED))
    return noise_fn, state, clipped_tree


def _iid_normal_noise_setup():
    """Setup for _iid_normal_noise on a 200-leaf tree (same layout as BLT tree)."""
    from opake.api.dpftrl.noise._engine import _iid_normal_noise
    from opake.random import generator_from_key, key

    gen = generator_from_key(key(SEED))
    leaves = [torch.zeros(4096, device=DEVICE_NAME) for _ in range(200)]
    tree = {"layers": leaves}
    return _iid_normal_noise, tree, gen


# --- Benchmarks ---


def bench_blt_single_tensor():
    """BLT inverse: single tensor, per-step multiply."""
    C_inv, state = _blt_sm()
    xi = torch.randn(768, device=DEVICE_NAME)

    def run():
        nonlocal state
        _out, state = C_inv.multiply_next(xi, state)

    return median_ms(run)


def bench_toeplitz_single_tensor():
    """Toeplitz inverse: single tensor, per-step multiply."""
    C_inv, state = _toeplitz_sm()
    yi = torch.randn(768, device=DEVICE_NAME)

    def run():
        nonlocal state
        _out, state = C_inv.multiply_next(yi, state)

    return median_ms(run)


def bench_blt_large_tree():
    """BLT inverse: 200-leaf tree via lifted streaming matrix."""
    lifted, state = _blt_tree_sm()
    input_leaves = [torch.randn(4096, device=DEVICE_NAME) for _ in range(200)]
    input_tree = {"layers": input_leaves}

    def run():
        nonlocal state
        _out, state = lifted.multiply_next(input_tree, state)

    return median_ms(run)


def bench_mf_noise():
    """Full MF noise step (1D leaves): IID noise generation + streaming matrix multiply."""
    noise_fn, state, clipped_grads = _mf_noise_setup()

    def run():
        nonlocal state
        _noisy, state = noise_fn(clipped_grads, state, stddev=1.0)

    return median_ms(run)


def bench_iid_normal_noise():
    """_iid_normal_noise on a 200-leaf tree (isolates the IID draw path)."""
    fn, tree, gen = _iid_normal_noise_setup()

    def run():
        _noise = fn(tree, 1.0, generator=gen)

    return median_ms(run)


def bench_dpsgd_noise():
    """DP-SGD gaussian noise step on a small pytree."""
    noise_fn, state, clipped_tree = _dpsgd_noise_setup()

    def run():
        nonlocal state
        _noisy, state = noise_fn(clipped_tree, state)

    return median_ms(run)


BENCHMARKS = [
    ("BLT single tensor (768)", bench_blt_single_tensor),
    ("Toeplitz single tensor (768)", bench_toeplitz_single_tensor),
    ("BLT 200-leaf tree (4096 each)", bench_blt_large_tree),
    ("_iid_normal_noise 200-leaf (4096 each)", bench_iid_normal_noise),
    ("MF noise small tree (1D)", bench_mf_noise),
    ("DP-SGD noise small tree", bench_dpsgd_noise),
]


def main():
    global DEVICE_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    DEVICE_NAME = args.device
    ITERATIONS = args.iterations
    torch.manual_seed(SEED)

    results = {}
    print(f"\nDevice: {DEVICE_NAME} | Warmup: {WARMUP} | Iterations: {ITERATIONS}")
    print("-" * 60)

    for name, fn in BENCHMARKS:
        try:
            ms = fn()
            results[name] = round(ms, 3)
            print(f"{name:40s}  {ms:8.2f} ms")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            print(f"{name:40s}  ERROR: {e}")

    print("-" * 60)

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
