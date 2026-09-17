# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Workspace-planning and forced-chunk regressions for MoE kernels."""

# ``I`` is the per-expert intermediate dimension, matching the kernel modules.
# ruff: noqa: E741

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opaque.api.patches.kernels import _grouped_moe, _moe_memory
from opaque.api.patches.kernels import moe as moe_kernel
from opaque.api.patches.kernels._grouped_moe import Opaque_GroupedMoE
from opaque.api.patches.kernels.moe import (
    Opaque_MoE,
    _expert_route,
    torch_reference_moe,
)


def _inputs(E=16, I=32, H=16, K=2, B=3, T=6):
    torch.manual_seed(924)
    gate_up = torch.randn(E, 2 * I, H)
    down = torch.randn(E, H, I)
    x = torch.randn(B, T, H)
    index = torch.randint(E, (B, T, K))
    weights = torch.rand(B, T, K)
    return x, gate_up, down, index, weights


def test_workspace_estimate_separates_required_weight_grad_output():
    x, gate_up, down, index, _ = _inputs(B=1)
    batch = 7
    estimate = _moe_memory.estimate_moe_workspace(
        x[0],
        gate_up,
        down,
        index[0],
        batch_size=batch,
        compute_gate_wgrad=True,
        compute_down_wgrad=True,
    )
    E, I, H = 16, 32, 16
    assert estimate.required_weight_grad_bytes == batch * E * 3 * I * H * 4

    gate_only = _moe_memory.estimate_moe_workspace(
        x[0],
        gate_up,
        down,
        index[0],
        batch_size=batch,
        compute_gate_wgrad=True,
        compute_down_wgrad=False,
    )
    assert gate_only.required_weight_grad_bytes == batch * E * 2 * I * H * 4

    frozen = _moe_memory.estimate_moe_workspace(
        x[0],
        gate_up,
        down,
        index[0],
        batch_size=batch,
        compute_gate_wgrad=False,
        compute_down_wgrad=False,
    )
    assert frozen.required_weight_grad_bytes == 0
    assert frozen.dense_bytes == estimate.dense_bytes
    assert frozen.grouped_bytes == estimate.grouped_bytes


def test_workspace_estimates_scale_with_shape_and_top_k():
    x, gate_up, down, index, _ = _inputs(B=1, K=2)
    small = _moe_memory.estimate_moe_workspace(x[0], gate_up, down, index[0])
    long = _moe_memory.estimate_moe_workspace(
        x[0].repeat(2, 1), gate_up, down, index[0].repeat(2, 1)
    )
    wide_routing = _moe_memory.estimate_moe_workspace(
        x[0], gate_up, down, index[0].repeat(1, 2)
    )
    assert long.dense_bytes > small.dense_bytes
    assert long.grouped_bytes > small.grouped_bytes
    assert wide_routing.dense_bytes > small.dense_bytes
    assert wide_routing.grouped_bytes > small.grouped_bytes


def test_workspace_budget_caps_accelerator_memory(monkeypatch):
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 8_000)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100_000, 100_000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 40_000)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 10_000)
    assert _moe_memory._workspace_budget_bytes(torch.device("cuda")) == 8_000

    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 100_000)
    assert _moe_memory._workspace_budget_bytes(torch.device("cuda")) == 32_500


def test_workspace_budget_uses_bounded_fallback(monkeypatch):
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 1_000_000_000)
    assert _moe_memory._workspace_budget_bytes(torch.device("cpu")) == 256 * 1024**2


def test_grouped_backward_backend_multiplier():
    base = _moe_memory.grouped_backward_bytes_per_route(
        hidden=16, intermediate=32, itemsize=2, backend_multiplier=1
    )
    assert _moe_memory.grouped_backward_bytes_per_route(16, 32, 2) == 3 * base


def test_opaque_moe_dispatch_respects_workspace_budget(monkeypatch):
    x, gate_up, down, index, weights = _inputs(B=1)
    x, index, weights = x[0], index[0], weights[0]
    sentinel = torch.tensor(924.0)
    monkeypatch.setattr(_grouped_moe, "grouped_mm_available", lambda: True)
    monkeypatch.setattr(
        _grouped_moe.Opaque_GroupedMoE,
        "apply",
        staticmethod(lambda *args: sentinel),
    )
    from opaque.api.patches.kernels import _moe_dispatch

    original_decision = _moe_dispatch.moe_dispatch_decision
    monkeypatch.setattr(
        _moe_dispatch,
        "moe_dispatch_decision",
        lambda *args, **kwargs: _moe_dispatch.MoEDispatchDecision(
            "dense",
            "test",
            1,
            2,
            original_decision(*args, **kwargs).features,
        ),
    )
    dense = moe_kernel.opaque_moe(x, gate_up, down, index, weights)
    assert dense.shape == x.shape

    monkeypatch.setattr(
        _moe_dispatch,
        "moe_dispatch_decision",
        lambda *args, **kwargs: _moe_dispatch.MoEDispatchDecision(
            "grouped",
            "test",
            2,
            1,
            original_decision(*args, **kwargs).features,
        ),
    )
    assert moe_kernel.opaque_moe(x, gate_up, down, index, weights) is sentinel

    monkeypatch.setattr(_moe_dispatch, "moe_dispatch_decision", original_decision)
    dense_weights = weights.new_zeros(weights.shape[0], gate_up.shape[0])
    dense_weights.scatter_add_(-1, index, weights)
    dense_result = moe_kernel.opaque_moe(
        x, gate_up, down, index, dense_weights, grouped=True
    )
    assert dense_result.shape == x.shape


def test_dense_and_grouped_compute_only_requested_expert_gradient():
    x, gate_up, down, index, weights = _inputs(E=4)

    def reference(xx, g, d, ii, ww):
        return moe_kernel.torch_reference_moe(xx, g, d, ii, ww).square().mean()

    def make_loss(op):
        def loss(xx, g, d, ii, ww):
            return op.apply(xx, g, d, ii, ww).square().mean()

        return loss

    in_dims = (0, None, None, 0, 0)
    for op in (Opaque_MoE, Opaque_GroupedMoE):
        loss = make_loss(op)
        for argnum in (1, 2):
            actual = vmap(grad(loss, argnums=argnum), in_dims=in_dims)(
                x, gate_up, down, index, weights
            )
            expected = vmap(grad(reference, argnums=argnum), in_dims=in_dims)(
                x, gate_up, down, index, weights
            )
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_empty_routes_produce_zero_expert_gradients(monkeypatch):
    E, I, H, K = 4, 32, 16, 2
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 3_000)
    index = torch.empty(0, K, dtype=torch.long)
    for op in (Opaque_MoE, Opaque_GroupedMoE):
        x = torch.empty(0, H, requires_grad=True)
        gate_up = torch.randn(E, 2 * I, H, requires_grad=True)
        down = torch.randn(E, H, I, requires_grad=True)
        weights = torch.empty(0, K, requires_grad=True)
        op.apply(x, gate_up, down, index, weights).sum().backward()
        assert torch.count_nonzero(gate_up.grad) == 0
        assert torch.count_nonzero(down.grad) == 0


def test_dense_output_allows_inplace_consumer():
    x, gate_up, down, index, weights = _inputs(E=4)
    x.requires_grad_()
    gate_up.requires_grad_()
    down.requires_grad_()
    out = Opaque_MoE.apply(x, gate_up, down, index, weights)
    out.add_(torch.zeros_like(out))
    out.sum().backward()
    assert x.grad is not None


def test_dense_backward_flattens_multiple_token_dimensions():
    x, gate_up, down, index, weights = _inputs(E=4)
    x.requires_grad_()
    gate_up.requires_grad_()
    down.requires_grad_()
    weights.requires_grad_()
    Opaque_MoE.apply(x, gate_up, down, index, weights).square().mean().backward()
    assert x.grad.shape == x.shape
    assert gate_up.grad.shape == gate_up.shape
    assert down.grad.shape == down.shape
    assert weights.grad.shape == weights.shape


def test_expert_only_backward_skips_token_gradient_outputs():
    x, gate_up, down, index, weights = _inputs(B=1)
    grad_out = torch.randn_like(x[0])
    dense = moe_kernel._moe_backward(
        grad_out,
        x[0],
        gate_up,
        down,
        index[0],
        weights[0],
        batch_dims=0,
        compute_x_grad=False,
        compute_route_grad=False,
        compute_gate_wgrad=True,
        compute_down_wgrad=False,
    )
    grouped = _grouped_moe._fused_moe_backward(
        grad_out,
        x[0],
        gate_up,
        down,
        index[0].reshape(-1),
        weights[0].reshape(-1),
        index.shape[-1],
        n_groups=gate_up.shape[0],
        compute_x_grad=False,
        compute_route_grad=False,
        compute_gate_wgrad=True,
        compute_down_wgrad=False,
    )
    for result in (dense, grouped):
        assert result[0] is None
        assert result[1] is not None
        assert result[2] is None
        assert result[3] is None


def test_sparse_expert_route_accumulates_duplicate_top_k_entries():
    index = torch.tensor([[[1, 1], [0, 2]]])
    weights = torch.tensor([[[0.25, 0.75], [0.4, 0.6]]])
    route = _expert_route(index, weights, expert=1, num_experts=4)
    torch.testing.assert_close(route, torch.tensor([[[1.0], [0.0]]]))


def test_forced_forward_chunks_bound_dense_linear_rows(monkeypatch):
    x, gate_up, down, index, weights = _inputs()
    x2 = x.reshape(-1, x.shape[-1])
    index2 = index.reshape(-1, index.shape[-1])
    weights2 = weights.reshape(-1, weights.shape[-1])
    expected = Opaque_MoE.apply(x2, gate_up, down, index2, weights2)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 4_096)

    rows = []
    linear = moe_kernel.F.linear

    def recording_linear(input, weight, bias=None):
        rows.append(input.numel() // input.shape[-1])
        return linear(input, weight, bias)

    monkeypatch.setattr(moe_kernel.F, "linear", recording_linear)
    actual = Opaque_MoE.apply(x2, gate_up, down, index2, weights2)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert len(rows) > 2 * gate_up.shape[0]
    assert max(rows) < x2.shape[0]


def test_forced_dense_backward_matches_autograd_reference(monkeypatch):
    x, gate_up, down, index, weights = _inputs(B=2, T=8)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 4_096)

    def loss(kernel, xx, g, d, ii, ww):
        return kernel(xx, g, d, ii, ww).square().mean()

    in_dims = (0, None, None, 0, 0)
    actual = vmap(
        grad(
            lambda xx, g, d, ii, ww: loss(Opaque_MoE.apply, xx, g, d, ii, ww),
            argnums=(0, 1, 2, 4),
        ),
        in_dims=in_dims,
    )(x, gate_up, down, index, weights)
    expected = vmap(
        grad(
            lambda xx, g, d, ii, ww: loss(torch_reference_moe, xx, g, d, ii, ww),
            argnums=(0, 1, 2, 4),
        ),
        in_dims=in_dims,
    )(x, gate_up, down, index, weights)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-4, atol=2e-4)


def test_forced_forward_chunks_bound_grouped_mm_rows(monkeypatch):
    x, gate_up, down, index, weights = _inputs()
    x2 = x.reshape(-1, x.shape[-1])
    index2 = index.reshape(-1, index.shape[-1])
    weights2 = weights.reshape(-1, weights.shape[-1])
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 4_096)

    rows = []
    sort_rows = []
    grouped_mm = _grouped_moe._grouped_mm
    route_sort = _grouped_moe._route_sort

    def recording_grouped_mm(A, Bw, ends):
        rows.append(A.shape[0])
        return grouped_mm(A, Bw, ends)

    def recording_route_sort(expert_of_row, n_groups):
        sort_rows.append(expert_of_row.numel())
        return route_sort(expert_of_row, n_groups)

    monkeypatch.setattr(_grouped_moe, "_grouped_mm", recording_grouped_mm)
    monkeypatch.setattr(_grouped_moe, "_route_sort", recording_route_sort)
    actual = Opaque_GroupedMoE.apply(x2, gate_up, down, index2, weights2)
    expected = Opaque_MoE.apply(x2, gate_up, down, index2, weights2)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert len(rows) > 2
    assert max(rows) < index2.numel()
    assert max(sort_rows) < index2.numel()


def test_forced_weight_grad_tiles_when_routed_activations_fit(monkeypatch):
    x, gate_up, down, index, weights = _inputs(B=1, T=1, K=1)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 3_000)
    streamed = []
    stream_weight_grads = _grouped_moe._stream_grouped_weight_grads

    def recording_stream(*args, **kwargs):
        streamed.append(True)
        return stream_weight_grads(*args, **kwargs)

    monkeypatch.setattr(_grouped_moe, "_stream_grouped_weight_grads", recording_stream)

    def loss(xx, g, d, ii, ww):
        return Opaque_GroupedMoE.apply(xx, g, d, ii, ww).square().mean()

    actual = vmap(grad(loss, argnums=(1, 2)), in_dims=(0, None, None, 0, 0))(
        x, gate_up, down, index, weights
    )
    assert streamed == [True]
    assert actual[0].shape[:2] == (1, gate_up.shape[0])
    assert actual[1].shape[:2] == (1, down.shape[0])


def test_forced_trainable_backward_chunks_single_long_example(monkeypatch):
    x, gate_up, down, index, weights = _inputs(B=1, T=32)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 4_096)

    rows = []
    grouped_mm = _grouped_moe._grouped_mm

    def recording_grouped_mm(A, Bw, ends):
        rows.append(A.shape[0])
        return grouped_mm(A, Bw, ends)

    streamed = []
    stream_weight_grads = _grouped_moe._stream_grouped_weight_grads

    def recording_stream(*args, **kwargs):
        streamed.append(True)
        return stream_weight_grads(*args, **kwargs)

    monkeypatch.setattr(_grouped_moe, "_grouped_mm", recording_grouped_mm)
    monkeypatch.setattr(_grouped_moe, "_stream_grouped_weight_grads", recording_stream)

    def grouped_loss(xx, g, d, ii, ww):
        return Opaque_GroupedMoE.apply(xx, g, d, ii, ww).square().mean()

    def dense_loss(xx, g, d, ii, ww):
        return Opaque_MoE.apply(xx, g, d, ii, ww).square().mean()

    in_dims = (0, None, None, 0, 0)
    actual = vmap(grad(grouped_loss, argnums=(0, 1, 2, 4)), in_dims=in_dims)(
        x, gate_up, down, index, weights
    )
    expected = vmap(grad(dense_loss, argnums=(0, 1, 2, 4)), in_dims=in_dims)(
        x, gate_up, down, index, weights
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-3)
    assert streamed == [True]
    assert max(rows) <= 2


def test_forced_trainable_vmap_chunks_examples_and_preserves_gradients(monkeypatch):
    x, gate_up, down, index, weights = _inputs()
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 30_000)

    batch_rows = []
    grouped_backward = _grouped_moe._fused_moe_backward

    def recording_backward(grad_flat, x_flat, *args, **kwargs):
        batch_rows.append(x_flat.shape[0])
        return grouped_backward(grad_flat, x_flat, *args, **kwargs)

    monkeypatch.setattr(_grouped_moe, "_fused_moe_backward", recording_backward)

    def grouped_loss(xx, g, d, ii, ww):
        return Opaque_GroupedMoE.apply(xx, g, d, ii, ww).square().mean()

    def dense_loss(xx, g, d, ii, ww):
        return Opaque_MoE.apply(xx, g, d, ii, ww).square().mean()

    in_dims = (0, None, None, 0, 0)
    actual = vmap(grad(grouped_loss, argnums=(0, 1, 2, 4)), in_dims=in_dims)(
        x, gate_up, down, index, weights
    )
    expected = vmap(grad(dense_loss, argnums=(0, 1, 2, 4)), in_dims=in_dims)(
        x, gate_up, down, index, weights
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
    assert batch_rows == [x.shape[1]] * x.shape[0]
