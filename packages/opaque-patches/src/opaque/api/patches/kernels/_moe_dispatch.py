# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic cost model for MoE backend dispatch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isqrt
from typing import Literal

import torch

from ._moe_memory import (
    MoEWorkspaceEstimate,
    chunk_size,
    dense_backward_bytes_per_row,
    dense_routing_bytes_per_row,
    grouped_backward_bytes_per_route,
    grouped_forward_bytes_per_route,
)

MoEBackend = Literal["dense", "grouped", "triton"]

_SPARSE_MARGIN_PERCENT = 10


@dataclass(frozen=True)
class MoEDispatchFeatures:
    """Scalar inputs to the backend cost model.

    Costs are deterministic integer work units. They are comparable only within
    one decision; they are not latency predictions.
    """

    backend: str
    dtype: torch.dtype
    tokens: int
    experts: int
    top_k: int
    hidden: int
    intermediate: int
    active_experts: int
    max_routes_per_expert: int
    trainable_expert_matrices: int
    backward_likely: bool
    workspace_budget_bytes: int
    dense_workspace_bytes: int
    sparse_workspace_bytes: int
    dense_chunks: int
    sparse_chunks: int
    grouped_available: bool
    triton_available: bool
    sparse_routes: bool = True
    grouped_enabled: bool = True
    route_evidence: Literal["bounds", "histogram"] = "histogram"

    @property
    def routes(self) -> int:
        return self.tokens * self.top_k


@dataclass(frozen=True)
class MoEDispatchDecision:
    """Selected backend plus stable diagnostics for evidence collection."""

    backend: MoEBackend
    reason: str
    dense_cost: int
    sparse_cost: int | None
    features: MoEDispatchFeatures


def _dtype_key(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def _active_dtype(x: torch.Tensor) -> torch.dtype:
    # Non-CUDA ``torch._grouped_mm`` does not follow autocast, so its cost model
    # must use the input dtype. CUDA dispatch receives explicitly autocast inputs
    # from ``opaque_moe``; the fallback also covers direct diagnostic calls.
    if x.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


def _sparse_backend(features: MoEDispatchFeatures) -> MoEBackend | None:
    if (
        features.backend == "cuda"
        and features.triton_available
        and features.dtype in (torch.float16, torch.bfloat16)
    ):
        return "triton"
    if (
        features.backend in ("cpu", "mps")
        and features.grouped_available
        and features.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return "grouped"
    return None


def _costs(features: MoEDispatchFeatures) -> tuple[int, int]:
    # H*I captures GEMM geometry while the floor preserves launch costs for tiny
    # test/decode shapes. Forward is three GEMMs; backward approximately doubles
    # activation work, and each trainable expert bank adds one gradient phase.
    geometry = max(1, features.hidden * features.intermediate // 256)
    effective_trainability = (
        features.trainable_expert_matrices if features.backward_likely else 0
    )
    phases = 3 + (3 if features.backward_likely else 0)
    phases += 2 * effective_trainability
    dtype_key = _dtype_key(features.dtype)

    dense_dtype_factor = 100
    sparse_compute_factor = 130
    dense_launch = 400
    sparse_fixed = 12_000
    sparse_group = 17_000
    imbalance_factor = 8

    if features.backend == "mps":
        sparse_compute_factor = 140
        dense_launch = 2_000
        sparse_fixed = 50_000
        sparse_group = 42_000
        if dtype_key == "bfloat16":
            sparse_group = 46_000
    elif features.backend == "cuda":
        sparse_compute_factor = 90
        dense_launch = 3_000
        sparse_fixed = 120_000
        sparse_group = 500
        imbalance_factor = 20
    elif features.backend == "cpu" and dtype_key in ("bfloat16", "float16"):
        # Measured CPU builds without low-precision dense GEMM acceleration pay
        # heavily for the E-way dense loop. Keep this conservative: grouped must
        # still clear the common 10% selection margin.
        dense_dtype_factor = 160
        sparse_compute_factor = 90
        sparse_fixed = 6_000
        sparse_group = 6_000

    dense = (
        features.tokens
        * features.experts
        * geometry
        * phases
        * dense_dtype_factor
        // 100
    )
    dense += features.experts * features.dense_chunks * dense_launch

    sparse = features.routes * geometry * phases * sparse_compute_factor // 100
    active = max(1, features.active_experts)
    group_scale = isqrt(active * 1024)
    trainability_scale = 4 + 3 * effective_trainability
    sparse += features.sparse_chunks * (
        sparse_fixed + sparse_group * group_scale * trainability_scale // (32 * 4)
    )
    # A high maximum/mean route count models under-utilized grouped GEMMs. The
    # integer form keeps boundary decisions stable across Python/platforms.
    mean_routes_ceil = max(1, (features.routes + active - 1) // active)
    imbalance_percent = max(
        100, features.max_routes_per_expert * 100 // mean_routes_ceil
    )
    sparse += sparse * (imbalance_percent - 100) * imbalance_factor // 10_000
    sparse += features.routes * max(1, features.routes.bit_length() - 1)
    return dense, sparse


def decide_moe_backend(features: MoEDispatchFeatures) -> MoEDispatchDecision:
    """Choose dense, grouped, or Triton execution from scalar features."""
    if not features.grouped_enabled:
        return MoEDispatchDecision("dense", "grouped-disabled", 0, None, features)
    if not features.sparse_routes:
        return MoEDispatchDecision("dense", "dense-route-weights", 0, None, features)

    sparse_backend = _sparse_backend(features)
    if sparse_backend is None:
        return MoEDispatchDecision(
            "dense", "sparse-backend-unavailable", 0, None, features
        )

    dense_cost, sparse_cost = _costs(features)
    if sparse_cost * 100 <= dense_cost * (100 - _SPARSE_MARGIN_PERCENT):
        reason = "modeled-speedup"
        backend: MoEBackend = sparse_backend
    else:
        reason = "dense-cost-or-margin"
        backend = "dense"
    return MoEDispatchDecision(backend, reason, dense_cost, sparse_cost, features)


def _route_histogram(top_k_index: torch.Tensor, experts: int) -> tuple[int, int]:
    if top_k_index.numel() == 0:
        return 0, 0
    counts = torch.bincount(top_k_index.reshape(-1), minlength=experts)
    counts_cpu = counts.detach().to(device="cpu")
    return int(torch.count_nonzero(counts_cpu)), int(counts_cpu.max())


def _can_inspect_routes(top_k_index: torch.Tensor) -> bool:
    if torch.compiler.is_compiling():
        return False
    functorch = getattr(torch._C, "_functorch", None)
    return functorch is None or not functorch.is_functorch_wrapped_tensor(top_k_index)


def _features_for_tensor(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    estimate: MoEWorkspaceEstimate,
    budget_bytes: int,
    grouped_enabled: bool,
    grouped_available: bool,
    triton_available: bool,
    active_experts: int,
    max_routes_per_expert: int,
    route_evidence: Literal["bounds", "histogram"],
) -> MoEDispatchFeatures:
    tokens = x.numel() // x.shape[-1]
    experts = gate_up_proj.shape[0]
    top_k = top_k_index.shape[-1]
    hidden = x.shape[-1]
    intermediate = gate_up_proj.shape[1] // 2
    routes = tokens * top_k
    output_bytes = tokens * hidden * 4
    dense_per_row = dense_backward_bytes_per_row(
        hidden, intermediate, x.element_size()
    ) + dense_routing_bytes_per_row(top_k_index, top_k_weights, experts)
    backward_likely = torch.is_grad_enabled() and (
        x.requires_grad
        or gate_up_proj.requires_grad
        or down_proj.requires_grad
        or top_k_weights.requires_grad
    )
    sparse_per_route = grouped_forward_bytes_per_route(
        hidden, intermediate, x.element_size()
    )
    if backward_likely:
        sparse_per_route = max(
            sparse_per_route,
            grouped_backward_bytes_per_route(
                hidden,
                intermediate,
                x.element_size(),
                backend_multiplier=1 if x.device.type == "cuda" else 3,
            ),
        )
    dense_rows = chunk_size(
        tokens,
        dense_per_row,
        x.device,
        fixed_bytes=output_bytes,
        budget_bytes=budget_bytes,
    )
    sparse_rows = chunk_size(
        routes,
        sparse_per_route,
        x.device,
        fixed_bytes=output_bytes,
        budget_bytes=budget_bytes,
    )
    return MoEDispatchFeatures(
        backend=x.device.type,
        dtype=_active_dtype(x),
        tokens=tokens,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        active_experts=active_experts,
        max_routes_per_expert=max_routes_per_expert,
        trainable_expert_matrices=(
            int(gate_up_proj.requires_grad) + int(down_proj.requires_grad)
        ),
        backward_likely=backward_likely,
        workspace_budget_bytes=budget_bytes,
        dense_workspace_bytes=estimate.dense_bytes,
        sparse_workspace_bytes=estimate.grouped_bytes,
        dense_chunks=max(1, (tokens + dense_rows - 1) // dense_rows),
        sparse_chunks=max(1, (routes + sparse_rows - 1) // sparse_rows),
        grouped_available=grouped_available,
        triton_available=triton_available,
        sparse_routes=top_k_weights.shape[-1] != experts,
        grouped_enabled=grouped_enabled,
        route_evidence=route_evidence,
    )


def moe_dispatch_decision(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    estimate: MoEWorkspaceEstimate,
    budget_bytes: int,
    grouped_enabled: bool,
    grouped_available: bool,
    triton_available: bool,
    inspect_routes: bool = False,
) -> MoEDispatchDecision:
    """Build features and select a backend, inspecting routes only near a boundary."""
    experts = gate_up_proj.shape[0]
    routes = top_k_index.numel()
    max_active = min(experts, routes)
    active_bounds = (0, 0) if routes == 0 else (1, max_active)

    decisions = []
    for active in active_bounds:
        max_routes = routes if active <= 1 else (routes + active - 1) // active
        features = _features_for_tensor(
            x,
            gate_up_proj,
            down_proj,
            top_k_index,
            top_k_weights,
            estimate=estimate,
            budget_bytes=budget_bytes,
            grouped_enabled=grouped_enabled,
            grouped_available=grouped_available,
            triton_available=triton_available,
            active_experts=active,
            max_routes_per_expert=max_routes,
            route_evidence="bounds",
        )
        decisions.append(decide_moe_backend(features))

    if not inspect_routes and decisions[0].backend == decisions[-1].backend:
        conservative = max(
            decisions,
            key=lambda decision: (decision.sparse_cost or 0) - decision.dense_cost,
        )
        return replace(conservative, reason=f"{conservative.reason}-route-bounds")

    if not _can_inspect_routes(top_k_index):
        dense = next(
            (decision for decision in decisions if decision.backend == "dense"),
            decisions[-1],
        )
        return replace(dense, reason="dense-route-inspection-unavailable")

    active, max_routes = _route_histogram(top_k_index, experts)
    features = _features_for_tensor(
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        estimate=estimate,
        budget_bytes=budget_bytes,
        grouped_enabled=grouped_enabled,
        grouped_available=grouped_available,
        triton_available=triton_available,
        active_experts=active,
        max_routes_per_expert=max_routes,
        route_evidence="histogram",
    )
    return decide_moe_backend(features)


__all__ = [
    "MoEDispatchDecision",
    "MoEDispatchFeatures",
    "decide_moe_backend",
    "moe_dispatch_decision",
]
