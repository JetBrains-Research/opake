# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Stable boundary tests for the deterministic MoE dispatch model."""

from __future__ import annotations

import pytest
import torch

from opaque.api.patches.kernels._moe_dispatch import (
    MoEDispatchFeatures,
    _active_dtype,
    decide_moe_backend,
)


def _features(**overrides) -> MoEDispatchFeatures:
    values = {
        "backend": "mps",
        "dtype": torch.float32,
        "tokens": 64,
        "experts": 16,
        "top_k": 2,
        "hidden": 64,
        "intermediate": 128,
        "active_experts": 16,
        "max_routes_per_expert": 8,
        "trainable_expert_matrices": 2,
        "backward_likely": True,
        "workspace_budget_bytes": 256 * 1024**2,
        "dense_workspace_bytes": 8 * 1024**2,
        "sparse_workspace_bytes": 4 * 1024**2,
        "dense_chunks": 1,
        "sparse_chunks": 1,
        "grouped_available": True,
        "triton_available": True,
    }
    values.update(overrides)
    return MoEDispatchFeatures(**values)


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_features(backend="cpu", experts=8, active_experts=8), "dense"),
        (
            _features(
                backend="cpu",
                experts=16,
                active_experts=16,
                max_routes_per_expert=8,
            ),
            "grouped",
        ),
        (_features(experts=32, active_experts=32, max_routes_per_expert=4), "dense"),
        (_features(experts=64, active_experts=64, max_routes_per_expert=2), "grouped"),
        (
            _features(
                experts=16,
                active_experts=2,
                max_routes_per_expert=64,
            ),
            "grouped",
        ),
    ],
)
def test_measured_cpu_mps_boundary_geometries(features, expected):
    assert decide_moe_backend(features).backend == expected


def test_dtype_changes_cpu_boundary():
    fp32 = decide_moe_backend(_features(backend="cpu", experts=8, active_experts=8))
    bf16 = decide_moe_backend(
        _features(
            backend="cpu",
            dtype=torch.bfloat16,
            experts=8,
            active_experts=8,
        )
    )
    assert fp32.backend == "dense"
    assert bf16.backend == "grouped"


def test_cpu_grouped_dtype_does_not_follow_autocast():
    x = torch.ones(1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _active_dtype(x) == torch.float32


def test_no_grad_ignores_expert_trainability_cost():
    frozen = decide_moe_backend(
        _features(backward_likely=False, trainable_expert_matrices=0)
    )
    trainable = decide_moe_backend(
        _features(backward_likely=False, trainable_expert_matrices=2)
    )
    assert (trainable.backend, trainable.dense_cost, trainable.sparse_cost) == (
        frozen.backend,
        frozen.dense_cost,
        frozen.sparse_cost,
    )


def test_chunked_sparse_workspace_can_reverse_decision():
    unchunked = decide_moe_backend(
        _features(experts=64, active_experts=64, max_routes_per_expert=2)
    )
    chunked = decide_moe_backend(
        _features(
            experts=64,
            active_experts=64,
            max_routes_per_expert=2,
            sparse_chunks=8,
            sparse_workspace_bytes=512 * 1024**2,
            workspace_budget_bytes=8 * 1024**2,
        )
    )
    assert unchunked.backend == "grouped"
    assert chunked.backend == "dense"


def test_cuda_triton_uses_geometry_not_availability_alone():
    tiny = decide_moe_backend(
        _features(
            backend="cuda",
            dtype=torch.bfloat16,
            tokens=1,
            experts=8,
            top_k=2,
            hidden=16,
            intermediate=16,
            active_experts=2,
            max_routes_per_expert=1,
            trainable_expert_matrices=0,
            backward_likely=False,
        )
    )
    model_scale = decide_moe_backend(
        _features(
            backend="cuda",
            dtype=torch.bfloat16,
            tokens=16,
            experts=8,
            top_k=2,
            hidden=4096,
            intermediate=14336,
            active_experts=8,
            max_routes_per_expert=4,
            trainable_expert_matrices=0,
            backward_likely=False,
        )
    )
    assert tiny.backend == "dense"
    assert model_scale.backend == "triton"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"grouped_enabled": False}, "grouped-disabled"),
        ({"sparse_routes": False}, "dense-route-weights"),
        (
            {
                "backend": "cuda",
                "dtype": torch.float32,
                "triton_available": True,
                "grouped_available": False,
            },
            "sparse-backend-unavailable",
        ),
        ({"backend": "cpu", "dtype": torch.float64}, "sparse-backend-unavailable"),
    ],
)
def test_deterministic_dense_fallbacks(updates, reason):
    decision = decide_moe_backend(_features(**updates))
    assert decision.backend == "dense"
    assert decision.reason == reason
