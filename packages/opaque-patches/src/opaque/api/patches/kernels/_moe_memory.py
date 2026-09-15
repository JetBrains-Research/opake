# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Private workspace planning for the MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

_MIB = 1024**2
_GIB = 1024**3
_FALLBACK_WORKSPACE_BYTES = 256 * _MIB
_MAX_WORKSPACE_BYTES = 8 * _GIB
_MAX_FP32_ACCUMULATOR_BYTES = 32 * _MIB
_FREE_MEMORY_FRACTION = 0.25
_ROUTING_INDEX_BYTES = 8


@dataclass(frozen=True)
class MoEWorkspaceEstimate:
    """Conservative route estimates, excluding inputs and returned outputs."""

    dense_bytes: int
    grouped_bytes: int
    required_weight_grad_bytes: int


def _workspace_budget_bytes(device: torch.device) -> int:
    """Return a conservative temporary-workspace budget for ``device``."""
    free_bytes: int | None = None
    try:
        if device.type == "cuda":
            driver_free = int(torch.cuda.mem_get_info(device)[0])
            cached_free = max(
                0,
                int(torch.cuda.memory_reserved(device))
                - int(torch.cuda.memory_allocated(device)),
            )
            free_bytes = driver_free + cached_free
        elif device.type == "mps":
            total = int(torch.mps.recommended_max_memory())
            used = int(torch.mps.driver_allocated_memory())
            free_bytes = max(total - used, 0)
    except (AttributeError, RuntimeError, TypeError):
        free_bytes = None

    if free_bytes is None:
        return min(_FALLBACK_WORKSPACE_BYTES, _MAX_WORKSPACE_BYTES)
    return max(1, min(_MAX_WORKSPACE_BYTES, int(free_bytes * _FREE_MEMORY_FRACTION)))


def _shape_rows(x: torch.Tensor) -> int:
    return prod(x.shape[:-1])


def estimate_moe_workspace(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k_index: torch.Tensor,
    *,
    batch_size: int = 1,
    compute_gate_wgrad: bool | None = None,
    compute_down_wgrad: bool | None = None,
) -> MoEWorkspaceEstimate:
    """Estimate peak avoidable workspace for dense and grouped execution.

    ``batch_size`` is the physical vmap batch represented by ``x``. Returned
    expert gradients are reported separately because trainable experts require
    that storage regardless of the selected route.
    """
    rows = _shape_rows(x)
    experts = gate_up_proj.shape[0]
    intermediate = gate_up_proj.shape[1] // 2
    hidden = x.shape[-1]
    top_k = top_k_index.shape[-1]
    itemsize = max(
        x.element_size(), gate_up_proj.element_size(), down_proj.element_size()
    )
    compute_gate_wgrad = (
        gate_up_proj.requires_grad if compute_gate_wgrad is None else compute_gate_wgrad
    )
    compute_down_wgrad = (
        down_proj.requires_grad if compute_down_wgrad is None else compute_down_wgrad
    )

    dense_fixed = rows * hidden * x.element_size()
    dense_per_row = (
        itemsize * (4 * intermediate + 3 * hidden)
        + 4 * (3 * intermediate + 2 * hidden)
        + top_k * (itemsize + 1)
    )

    routed_rows = rows * top_k
    grouped_fixed = rows * hidden * x.element_size()
    grouped_per_route = (
        3 * _ROUTING_INDEX_BYTES
        + itemsize * (5 * intermediate + 5 * hidden)
        + 4 * (3 * intermediate + 3 * hidden)
    )

    required_wgrad = (
        batch_size
        * experts
        * intermediate
        * hidden
        * (
            (2 * gate_up_proj.element_size() if compute_gate_wgrad else 0)
            + (down_proj.element_size() if compute_down_wgrad else 0)
        )
    )
    return MoEWorkspaceEstimate(
        dense_bytes=dense_fixed + rows * dense_per_row,
        grouped_bytes=grouped_fixed + routed_rows * grouped_per_route,
        required_weight_grad_bytes=required_wgrad,
    )


def use_grouped_route(
    estimate: MoEWorkspaceEstimate,
    *,
    experts: int,
    min_experts: int,
    budget_bytes: int,
) -> bool:
    """Select grouped execution using both its speed gate and memory estimate."""
    dense_fits = estimate.dense_bytes <= budget_bytes
    grouped_fits = estimate.grouped_bytes <= budget_bytes
    if dense_fits and not grouped_fits:
        return False
    if not dense_fits:
        return estimate.grouped_bytes < estimate.dense_bytes
    return experts >= min_experts and grouped_fits


def grouped_forward_bytes_per_route(
    hidden: int, intermediate: int, itemsize: int
) -> int:
    """Conservative live bytes for one routed row in grouped forward."""
    return (
        3 * _ROUTING_INDEX_BYTES
        + itemsize * (2 * hidden + 3 * intermediate)
        + 4 * (hidden + intermediate)
    )


def grouped_backward_bytes_per_route(
    hidden: int,
    intermediate: int,
    itemsize: int,
    *,
    backend_multiplier: int = 3,
) -> int:
    """Conservative live bytes for one routed row in grouped backward."""
    tensor_bytes = (
        4 * _ROUTING_INDEX_BYTES
        + itemsize * (5 * hidden + 7 * intermediate)
        + 4 * (4 * hidden + 5 * intermediate)
    )
    # ``torch._grouped_mm`` on MPS retains implementation workspaces beyond
    # visible tensors and needs the default slack. The CUDA fused path passes
    # one: its Triton weight-gradient kernel has no proportional hidden
    # workspace, while the free-memory budget still supplies global headroom.
    return backend_multiplier * tensor_bytes


def dense_routing_bytes_per_row(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
) -> int:
    """Live chunk-local bytes needed to select one sparse expert route."""
    if top_k_weights.shape[-1] == num_experts:
        return 0
    return (
        top_k_index.shape[-1] * (top_k_weights.element_size() + 1)
        + top_k_weights.element_size()
    )


def dense_backward_bytes_per_row(hidden: int, intermediate: int, itemsize: int) -> int:
    """Conservative live bytes for one token row in dense backward."""
    return itemsize * (4 * hidden + 7 * intermediate) + 4 * (
        3 * hidden + 5 * intermediate
    )


def chunk_size(
    total: int,
    bytes_per_row: int,
    device: torch.device,
    *,
    fixed_bytes: int = 0,
    budget_bytes: int | None = None,
    row_multiple: int = 1,
) -> int:
    """Largest bounded chunk not exceeding ``total`` (always at least one row)."""
    if total <= 0:
        return 1
    budget = _workspace_budget_bytes(device) if budget_bytes is None else budget_bytes
    available = max(budget - fixed_bytes, bytes_per_row)
    rows = max(1, available // max(bytes_per_row, 1))
    if row_multiple > 1:
        rows = max(row_multiple, rows - rows % row_multiple)
    return min(total, rows)


def fp32_accumulator_rows(total: int, width: int) -> int:
    """Bound a token-local FP32 accumulator independently of free memory."""
    if total <= 0:
        return 1
    return min(total, max(1, _MAX_FP32_ACCUMULATOR_BYTES // max(width * 4, 1)))
