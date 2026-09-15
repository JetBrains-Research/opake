# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Non-Triton sparse MoE (``torch._grouped_mm``) parity tests (MPS/CPU).

The sparse grouped path must match the dense ``Opaque_MoE`` oracle — forward and,
crucially, the DP-SGD ``vmap(grad)`` **per-sample** expert-weight gradients
(``dW1`` ``(B,E,2I,H)`` / ``dW2`` ``(B,E,H,I)``, never summed across the batch).
"""

# ``I`` is the per-expert intermediate dim (paired with ``2I``), as in the kernel.
# ruff: noqa: E741

from __future__ import annotations

import gc

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.patches.kernels import _moe_memory
from opaque.api.patches.kernels._grouped_moe import (
    Opaque_GroupedMoE,
    _grouped_AtB,
    grouped_mm_available,
)
from opaque.api.patches.kernels.moe import Opaque_MoE, opaque_moe

_TOL = 1e-3  # fp32 grouped_mm vs the dense expert loop: accumulation-order roundoff

pytestmark = pytest.mark.skipif(
    not grouped_mm_available(), reason="torch._grouped_mm unavailable"
)


def _inputs(device, E, I=32, H=16, K=2, B=3, T=6):
    torch.manual_seed(0)
    gu = torch.randn(E, 2 * I, H, device=device)
    dn = torch.randn(E, H, I, device=device)
    x3 = torch.randn(B, T, H, device=device)
    idx3 = torch.randint(0, E, (B, T, K), device=device)
    w3 = torch.rand(B, T, K, device=device)
    return gu, dn, x3, idx3, w3, B, T, H


def _check_parity(device: str) -> None:
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=4)
    N = B * T
    x2, idx2, w2 = (
        x3.reshape(N, H),
        idx3.reshape(N, -1),
        w3.reshape(N, -1),
    )

    # forward: sparse Function vs dense Function
    o_s = Opaque_GroupedMoE.apply(x2, gu, dn, idx2, w2)
    o_d = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
    assert (o_s - o_d).abs().max().item() < _TOL, "forward"

    # DP vmap(grad): per-example grads wrt hidden + both expert weights + router.
    def f_s(xx, gg, dd, ii, ww):
        return Opaque_GroupedMoE.apply(xx, gg, dd, ii, ww).sum()

    def f_d(xx, gg, dd, ii, ww):
        return Opaque_MoE.apply(xx, gg, dd, ii, ww).sum()

    in_dims = (0, None, None, 0, 0)
    gs = vmap(grad(f_s, argnums=(0, 1, 2, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    gd = vmap(grad(f_d, argnums=(0, 1, 2, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    for name, a, b in zip(
        ("dx", "d_gate_up", "d_down", "d_router"), gs, gd, strict=False
    ):
        assert (a - b).abs().max().item() < _TOL, f"vmap(grad) {name}"


def test_grouped_moe_parity_cpu():
    _check_parity("cpu")


@pytest.mark.mps
def test_grouped_moe_parity_mps():
    _check_parity("mps")


def _frob(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / b.norm().clamp(min=1e-6).item()


def _check_bf16_within_floor(device: str) -> None:
    """bf16 sparse vs dense within the bf16 floor (relative Frobenius — the
    DP-relevant metric; per-element max-relative is meaningless on the near-zero
    grad entries). The dense oracle is itself bf16 (matmuls) + fp32 expert-sum, so
    the bar is bf16 precision, not fp32."""
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=16)
    gu, dn, x3, w3 = (t.to(torch.bfloat16) for t in (gu, dn, x3, w3))
    N = B * T
    x2, idx2, w2 = x3.reshape(N, H), idx3.reshape(N, -1), w3.reshape(N, -1)
    assert (
        _frob(
            Opaque_GroupedMoE.apply(x2, gu, dn, idx2, w2),
            Opaque_MoE.apply(x2, gu, dn, idx2, w2),
        )
        < 1e-2
    ), "forward"

    def f_s(xx, gg, dd, ii, ww):
        return Opaque_GroupedMoE.apply(xx, gg, dd, ii, ww).float().sum()

    def f_d(xx, gg, dd, ii, ww):
        return Opaque_MoE.apply(xx, gg, dd, ii, ww).float().sum()

    in_dims = (0, None, None, 0, 0)
    gs = vmap(grad(f_s, argnums=(0, 1, 2)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    gd = vmap(grad(f_d, argnums=(0, 1, 2)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    for name, a, b in zip(("dx", "d_gate_up", "d_down"), gs, gd, strict=False):
        assert _frob(a, b) < 1e-2, f"vmap(grad) {name}"


def test_grouped_moe_bf16_cpu():
    _check_bf16_within_floor("cpu")


@pytest.mark.mps
def test_grouped_moe_bf16_mps():
    _check_bf16_within_floor("mps")


def _check_frozen_experts(device: str) -> None:
    """DP-SGD LoRA-on-attention: experts frozen, only hidden + router differentiated.

    Exercises the ``compute_wgrad=False`` skip (mirrors the Triton fused path): the
    backward must NOT build the per-sample ``(B, E, ...)`` weight grads, yet ``dx``
    and ``d_router`` must still match the dense oracle bit-for-bit-within-_TOL."""
    gu, dn, x3, idx3, w3, _B, _T, _H = _inputs(device, E=16)

    def f_s(xx, gg, dd, ii, ww):
        return Opaque_GroupedMoE.apply(xx, gg, dd, ii, ww).sum()

    def f_d(xx, gg, dd, ii, ww):
        return Opaque_MoE.apply(xx, gg, dd, ii, ww).sum()

    # argnums=(0, 4): only hidden + router weights need grad — experts (1, 2) stay
    # frozen, so ctx.needs_input_grad makes compute_wgrad False.
    in_dims = (0, None, None, 0, 0)
    gs = vmap(grad(f_s, argnums=(0, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    gd = vmap(grad(f_d, argnums=(0, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    for name, a, b in zip(("dx", "d_router"), gs, gd, strict=False):
        assert (a - b).abs().max().item() < _TOL, f"frozen vmap(grad) {name}"
    # Per-sample dx still genuinely distinct across the batch.
    assert (gs[0][0] - gs[0][1]).abs().max().item() > 0


def test_grouped_moe_frozen_experts_cpu():
    _check_frozen_experts("cpu")


@pytest.mark.mps
def test_grouped_moe_frozen_experts_mps():
    _check_frozen_experts("mps")


def test_grouped_atb_handles_empty_groups_on_device(monkeypatch):
    torch.manual_seed(964)
    A = torch.randn(5, 16)
    B = torch.randn(5, 8)
    ends = torch.tensor([2, 2, 5, 5], dtype=torch.int32)
    monkeypatch.setattr(
        torch.Tensor,
        "tolist",
        lambda self: pytest.fail("grouped offsets must stay on device"),
    )
    actual = _grouped_AtB(A, B, ends, 4)
    expected = torch.stack(
        [
            A[:2].mT @ B[:2],
            torch.zeros(16, 8),
            A[2:].mT @ B[2:],
            torch.zeros(16, 8),
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "routes",
    [
        pytest.param(
            [[[0, 1], [2, 3], [0, 2], [1, 3]]],
            id="balanced",
        ),
        pytest.param(
            [[[0, 0], [0, 0], [0, 0], [0, 0]]],
            id="skewed-duplicate-empty",
        ),
    ],
)
@pytest.mark.parametrize("expert_argnums", [(), (1,), (2,), (1, 2)])
def test_route_plans_preserve_ordinary_and_dp_gradients(
    monkeypatch, routes, expert_argnums
):
    torch.manual_seed(964)
    B, T, K, E, H, I = 2, 4, 2, 4, 16, 16
    x = torch.randn(B, T, H)
    gate_up = torch.randn(E, 2 * I, H)
    down = torch.randn(E, H, I)
    index = torch.tensor(routes).expand(B, -1, -1).clone()
    weights = torch.rand(B, T, K)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 4_096)

    def grouped_loss(xx, g, d, ii, ww):
        return Opaque_GroupedMoE.apply(xx, g, d, ii, ww).square().mean()

    def dense_loss(xx, g, d, ii, ww):
        return Opaque_MoE.apply(xx, g, d, ii, ww).square().mean()

    argnums = (0, *expert_argnums, 4)
    ordinary = grad(grouped_loss, argnums=argnums)(
        x[0], gate_up, down, index[0], weights[0]
    )
    ordinary_ref = grad(dense_loss, argnums=argnums)(
        x[0], gate_up, down, index[0], weights[0]
    )
    per_example = vmap(
        grad(grouped_loss, argnums=argnums), in_dims=(0, None, None, 0, 0)
    )(x, gate_up, down, index, weights)
    per_example_ref = vmap(
        grad(dense_loss, argnums=argnums), in_dims=(0, None, None, 0, 0)
    )(x, gate_up, down, index, weights)
    for actual, expected in zip(ordinary, ordinary_ref, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(per_example, per_example_ref, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def _check_dispatch(device: str) -> None:
    # opaque_moe routes large-E to the sparse path and small-E to dense; both must
    # match the dense oracle. (Only correctness is asserted; the E threshold is a
    # perf heuristic.)
    for E in (4, 16):
        gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=E)
        x2, idx2, w2 = (
            x3.reshape(B * T, H),
            idx3.reshape(B * T, -1),
            w3.reshape(B * T, -1),
        )
        out = opaque_moe(x2, gu, dn, idx2, w2)
        ref = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
        assert (out - ref).abs().max().item() < _TOL, f"dispatch E={E}"


def test_grouped_moe_dispatch_cpu():
    _check_dispatch("cpu")


@pytest.mark.mps
def test_grouped_moe_dispatch_mps():
    _check_dispatch("mps")


def _check_grouped_gate(device: str) -> None:
    # ``grouped=False`` forces the dense ``Opaque_MoE`` compat path even for large
    # E where the grouped-GEMM path is eligible: the output is bit-identical to the
    # dense oracle (a clean proof of the route), while ``grouped=True`` takes the
    # grouped path (equal to the oracle only within the accumulation-roundoff
    # floor). This is the gate the ``grouped_moe`` patch flag rides on.
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=16)
    x2, idx2, w2 = (
        x3.reshape(B * T, H),
        idx3.reshape(B * T, -1),
        w3.reshape(B * T, -1),
    )
    ref = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
    dense = opaque_moe(x2, gu, dn, idx2, w2, grouped=False)
    grouped = opaque_moe(x2, gu, dn, idx2, w2, grouped=True)
    assert (dense - ref).abs().max().item() == 0.0, "grouped=False must take dense path"
    assert (grouped - ref).abs().max().item() < _TOL, "grouped=True grouped parity"


def test_grouped_moe_grouped_gate_cpu():
    _check_grouped_gate("cpu")


@pytest.mark.mps
def test_grouped_moe_grouped_gate_mps():
    _check_grouped_gate("mps")


@pytest.mark.mps
def test_grouped_moe_trainable_backward_stays_near_budget(monkeypatch):
    from opaque.device import device_capabilities
    from opaque.profiling import get_memory_stats, reset_peak_memory

    if not device_capabilities("mps").peak_memory_trackable:
        pytest.skip("MPS allocator peak statistics require PyTorch 2.13+")
    budget = 4 * 1024**2
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", budget)
    torch.manual_seed(924)
    B, T, K, E, H, I = 1, 1024, 4, 16, 64, 128
    x = torch.randn(B, T, H, device="mps", dtype=torch.bfloat16)
    gate_up = torch.randn(E, 2 * I, H, device="mps", dtype=torch.bfloat16)
    down = torch.randn(E, H, I, device="mps", dtype=torch.bfloat16)
    index = torch.randint(E, (B, T, K), device="mps")
    weights = torch.rand(B, T, K, device="mps", dtype=torch.bfloat16)

    def loss(xx, g, d, ii, ww):
        return Opaque_GroupedMoE.apply(xx, g, d, ii, ww).float().square().mean()

    def run():
        return vmap(grad(loss, argnums=(0, 1, 2, 4)), in_dims=(0, None, None, 0, 0))(
            x, gate_up, down, index, weights
        )

    result = run()
    torch.mps.synchronize()
    del result
    gc.collect()
    torch.mps.empty_cache()
    reset_peak_memory("mps")
    baseline = get_memory_stats("mps").allocated_gb * 1024**3
    result = run()
    torch.mps.synchronize()
    peak = get_memory_stats("mps").peak_gb * 1024**3 - baseline
    required = sum(t.numel() * t.element_size() for t in result)
    # grouped_mm has a backend-owned fixed workspace, so the estimator bounds
    # visible route tensors to the budget and allows one budget of backend slack.
    assert peak - required <= 2 * budget


@pytest.mark.mps
def test_grouped_moe_chunking_reduces_mps_forward_peak(monkeypatch):
    from opaque.device import device_capabilities
    from opaque.profiling import get_memory_stats, reset_peak_memory

    if not device_capabilities("mps").peak_memory_trackable:
        pytest.skip("MPS allocator peak statistics require PyTorch 2.13+")
    torch.manual_seed(924)
    N, K, E, H, I = 4096, 4, 16, 64, 128
    x = torch.randn(N, H, device="mps", dtype=torch.bfloat16)
    gate_up = torch.randn(E, 2 * I, H, device="mps", dtype=torch.bfloat16)
    down = torch.randn(E, H, I, device="mps", dtype=torch.bfloat16)
    index = torch.randint(E, (N, K), device="mps")
    weights = torch.rand(N, K, device="mps", dtype=torch.bfloat16)

    def run():
        return Opaque_GroupedMoE.apply(x, gate_up, down, index, weights)

    def peak_bytes():
        out = run()
        torch.mps.synchronize()
        del out
        gc.collect()
        torch.mps.empty_cache()
        reset_peak_memory("mps")
        baseline = get_memory_stats("mps").allocated_gb * 1024**3
        out = run()
        torch.mps.synchronize()
        peak = get_memory_stats("mps").peak_gb * 1024**3 - baseline
        del out
        return peak

    unchunked = peak_bytes()
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 8 * 1024**2)
    chunked = peak_bytes()
    assert chunked < 0.75 * unchunked, (
        f"expected chunking to reduce MPS peak: {chunked / 1024**2:.1f} MiB vs "
        f"{unchunked / 1024**2:.1f} MiB"
    )
