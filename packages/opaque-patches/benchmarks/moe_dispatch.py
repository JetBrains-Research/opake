#!/usr/bin/env python
# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure MoE implementations and print dispatch diagnostics as JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext

import torch

from opaque.api.patches.kernels import _moe_memory
from opaque.api.patches.kernels._grouped_moe import (
    Opaque_GroupedMoE,
    grouped_mm_available,
)
from opaque.api.patches.kernels._moe_dispatch import moe_dispatch_decision
from opaque.api.patches.kernels._moe_memory import estimate_moe_workspace
from opaque.api.patches.kernels.moe import Opaque_MoE


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _routes(tokens: int, top_k: int, experts: int, distribution: str, device: str):
    active = experts if distribution == "balanced" else max(1, experts // 8)
    return (torch.arange(tokens * top_k, device=device) % active).reshape(tokens, top_k)


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def main() -> None:
    """Run one configured benchmark geometry and emit JSON evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument(
        "--autocast", choices=("none", "bfloat16", "float16"), default="none"
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--routes", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--trainable-experts", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    autocast_dtype = None if args.autocast == "none" else getattr(torch, args.autocast)
    _moe_memory._MAX_WORKSPACE_BYTES = args.workspace_mib * 1024**2

    def autocast():
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(args.device, dtype=autocast_dtype)

    torch.manual_seed(966)
    x = torch.randn(args.tokens, args.hidden, device=args.device, dtype=dtype)
    gate = args.weight_scale * torch.randn(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        device=args.device,
        dtype=dtype,
    )
    down = args.weight_scale * torch.randn(
        args.experts,
        args.hidden,
        args.intermediate,
        device=args.device,
        dtype=dtype,
    )
    index = _routes(args.tokens, args.top_k, args.experts, args.routes, args.device)
    weights = torch.rand(args.tokens, args.top_k, device=args.device, dtype=dtype)
    if args.trainable_experts:
        gate.requires_grad_()
        down.requires_grad_()
    x.requires_grad_()
    weights.requires_grad_()

    estimate = estimate_moe_workspace(x, gate, down, index)
    triton_available = False
    try:
        import triton  # noqa: F401

        triton_available = True
    except ImportError:
        pass
    with autocast():
        decision = moe_dispatch_decision(
            x,
            gate,
            down,
            index,
            weights,
            estimate=estimate,
            budget_bytes=args.workspace_mib * 1024**2,
            grouped_enabled=True,
            grouped_available=grouped_mm_available(),
            triton_available=triton_available,
            inspect_routes=True,
        )

    implementations = {"dense": Opaque_MoE.apply}
    if grouped_mm_available() and args.device in ("cpu", "mps"):
        implementations["grouped"] = Opaque_GroupedMoE.apply
    if (
        triton_available
        and args.device == "cuda"
        and decision.features.dtype in (torch.float16, torch.bfloat16)
    ):
        from opaque.api.patches.kernels.fused_moe import Opaque_FusedMoE

        implementations["triton"] = Opaque_FusedMoE.apply

    with autocast():
        reference = Opaque_MoE.apply(x, gate, down, index, weights).detach()
    rows = []
    for name, implementation in implementations.items():

        def operation(implementation=implementation):
            with autocast():
                output = implementation(x, gate, down, index, weights)
                inputs = [x, weights]
                if args.trainable_experts:
                    inputs[1:1] = [gate, down]
                torch.autograd.grad(output.float().square().mean(), inputs)

        for _ in range(args.warmup):
            operation()
        _sync(args.device)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            operation()
            _sync(args.device)
            samples.append((time.perf_counter() - start) * 1_000)
        with autocast():
            output = implementation(x, gate, down, index, weights).detach()
        median_ms = statistics.median(samples)
        rows.append(
            {
                "implementation": name,
                "selected": name == decision.backend,
                "median_ms": median_ms,
                "p95_ms": _percentile(samples, 0.95),
                "tokens_per_second": args.tokens * 1_000 / median_ms,
                "max_abs_error": (output - reference).abs().max().item(),
            }
        )

    diagnostics = {
        "selected_backend": decision.backend,
        "reason": decision.reason,
        "dense_cost": decision.dense_cost,
        "sparse_cost": decision.sparse_cost,
        **decision.features.__dict__,
        "master_dtype": str(dtype),
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
    }
    diagnostics["dtype"] = str(diagnostics["dtype"])
    print(json.dumps({"dispatch": diagnostics, "measurements": rows}, indent=2))


if __name__ == "__main__":
    main()
