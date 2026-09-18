# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused (sparse grouped-GEMM) MoE kernel tests.

The fused path replaces the dense ``Opake_MoE`` internals with ``O(T*K)`` sparse
compute (``torch._grouped_mm`` + a custom Triton weight-grad kernel). It must
match independent references within bf16 noise across:

1. Forward vs sparse-gather oracle AND the dense autograd-composed reference.
2. Backward (dx + expert weight grads + routing-weight grad) vs the dense ref.
3. vmap forward.
4. vmap(grad): per-example dx AND per-sample expert-weight grads — the DP-SGD
   path — match a loop reference and are distinct across the batch.
5. opake_moe dispatches CUDA bf16 → the fused path.
"""

# ``I`` is the intermediate dim in the (E, K, H, I, T) shape tuple — intentional.
# ruff: noqa: E741

import gc

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels import _moe_memory
from opake.api.patches.kernels.fused_moe import opake_fused_moe
from opake.api.patches.kernels.moe import opake_moe, torch_reference_moe

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

E, K, H, I, T = 8, 2, 256, 128, 64


def _inputs(device="cuda", dtype=torch.bfloat16, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    x = torch.randn(T, H, device=device, dtype=dtype, requires_grad=requires_grad)
    gate_up = (
        torch.randn(E, 2 * I, H, device=device, dtype=dtype) * 0.05
    ).requires_grad_(requires_grad)
    down = (torch.randn(E, H, I, device=device, dtype=dtype) * 0.05).requires_grad_(
        requires_grad
    )
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    tw, ti = torch.topk(F.softmax(logits, dim=-1), K, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(dtype).requires_grad_(requires_grad)
    return x, gate_up, down, ti, tw


def _sparse_gather_moe(x, gate_up, down, ti, tw):
    """Independent ground-truth sparse MoE via per-expert nonzero gather."""
    out = torch.zeros_like(x)
    for e in range(gate_up.shape[0]):
        sel = ti == e
        if not sel.any():
            continue
        tok, kp = sel.nonzero(as_tuple=True)
        gu = F.linear(x[tok], gate_up[e])
        g, u = gu[:, :I], gu[:, I:]
        ye = F.linear(F.silu(g) * u, down[e])
        out.index_add_(0, tok, (ye * tw[tok, kp].unsqueeze(-1)).to(out.dtype))
    return out


def test_forward_matches_references(assert_precision):
    x, gate_up, down, ti, tw = _inputs()
    out = opake_fused_moe(x, gate_up, down, ti, tw)
    assert_precision(
        out,
        _sparse_gather_moe(x, gate_up, down, ti, tw),
        rtol=1e-2,
        atol=1e-2,
        label="vs sparse",
    )
    assert_precision(
        out,
        torch_reference_moe(x, gate_up, down, ti, tw),
        rtol=1e-2,
        atol=1e-2,
        label="vs dense",
    )


def test_dispatch_cuda_bf16_uses_fused():
    """opake_moe routes CUDA bf16 to the fused path (bit-identical to it)."""
    x, gate_up, down, ti, tw = _inputs()
    torch.testing.assert_close(
        opake_moe(x, gate_up, down, ti, tw),
        opake_fused_moe(x, gate_up, down, ti, tw),
        rtol=0,
        atol=0,
    )


def test_backward_matches_reference(assert_precision):
    x, gate_up, down, ti, tw = _inputs(requires_grad=True)
    xr, gr, dr, twr = (
        x.clone().detach().requires_grad_(True),
        gate_up.clone().detach().requires_grad_(True),
        down.clone().detach().requires_grad_(True),
        tw.clone().detach().requires_grad_(True),
    )
    # Independent autograd-composed reference (derives its own backward).
    torch_reference_moe(xr, gr, dr, ti, twr).square().mean().backward()
    opake_fused_moe(x, gate_up, down, ti, tw).square().mean().backward()
    assert_precision(x.grad, xr.grad, rtol=2e-2, atol=2e-2, label="dx")
    assert_precision(gate_up.grad, gr.grad, rtol=2e-2, atol=2e-2, label="dgate_up")
    assert_precision(down.grad, dr.grad, rtol=2e-2, atol=2e-2, label="ddown")
    assert_precision(tw.grad, twr.grad, rtol=2e-2, atol=2e-2, label="dtop_k_weights")


def test_vmap_forward(assert_precision):
    B = 4
    _, gate_up, down, ti, tw = _inputs()
    xb = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    tib = ti.unsqueeze(0).expand(B, -1, -1).contiguous()
    twb = tw.unsqueeze(0).expand(B, -1, -1).contiguous()
    out = vmap(lambda xs, t, w: opake_fused_moe(xs, gate_up, down, t, w))(xb, tib, twb)
    ref = vmap(lambda xs, t, w: torch_reference_moe(xs, gate_up, down, t, w))(
        xb, tib, twb
    )
    assert_precision(out, ref, rtol=1e-2, atol=1e-2, label="vmap forward")


def test_vmap_grad_per_sample(assert_precision):
    """Per-sample gradients (DP-SGD path): dx AND expert weight grads match a loop
    reference and are genuinely distinct across the batch."""
    B = 4
    _, gate_up, down, ti, tw = _inputs()
    xb = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    tib = ti.unsqueeze(0).expand(B, -1, -1).contiguous()
    twb = tw.unsqueeze(0).expand(B, -1, -1).contiguous()

    def loss(xs, t, w, g1, g2):
        return opake_fused_moe(xs, g1, g2, t, w).square().mean()

    def rloss(xs, t, w, g1, g2):
        return torch_reference_moe(xs, g1, g2, t, w).square().mean()

    # in_dims: map x/ti/tw over the batch; weights are shared (None) — the DP case.
    g_op = vmap(grad(loss, argnums=(0, 3, 4)), in_dims=(0, 0, 0, None, None))(
        xb, tib, twb, gate_up, down
    )
    refs = [
        grad(rloss, argnums=(0, 3, 4))(xb[i], tib[i], twb[i], gate_up, down)
        for i in range(B)
    ]
    dx_r = torch.stack([r[0] for r in refs])
    dgu_r = torch.stack([r[1] for r in refs])
    ddn_r = torch.stack([r[2] for r in refs])

    assert g_op[1].shape == (B, E, 2 * I, H)
    assert g_op[2].shape == (B, E, H, I)
    assert_precision(g_op[0], dx_r, rtol=2e-2, atol=2e-2, label="vmap(grad) dx")
    assert_precision(g_op[1], dgu_r, rtol=2e-2, atol=2e-2, label="vmap(grad) dgate_up")
    assert_precision(g_op[2], ddn_r, rtol=2e-2, atol=2e-2, label="vmap(grad) ddown")

    # Per-sample grads must differ across the batch (the point of DP per-sample grads).
    assert (g_op[0][0] - g_op[0][1]).abs().max().item() > 0
    assert (g_op[1][0] - g_op[1][1]).abs().max().item() > 0
    assert (g_op[2][0] - g_op[2][1]).abs().max().item() > 0


def test_trainable_vmap_grad_forced_route_chunks(assert_precision, monkeypatch):
    x, gate_up, down, ti, tw = _inputs()
    xb, tib, twb = x.unsqueeze(0), ti.unsqueeze(0), tw.unsqueeze(0)
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 480 * 1024)

    def loss(xs, t, w, g1, g2):
        return opake_fused_moe(xs, g1, g2, t, w).square().mean()

    def reference(xs, t, w, g1, g2):
        return torch_reference_moe(xs, g1, g2, t, w).square().mean()

    actual = vmap(grad(loss, argnums=(0, 2, 3, 4)), in_dims=(0, 0, 0, None, None))(
        xb, tib, twb, gate_up, down
    )
    expected = grad(reference, argnums=(0, 2, 3, 4))(x, ti, tw, gate_up, down)
    for name, result, target in zip(
        ("dx", "dtop_k_weights", "dgate_up", "ddown"),
        actual,
        expected,
        strict=True,
    ):
        assert_precision(result[0], target, rtol=2e-2, atol=2e-2, label=name)


@pytest.mark.parametrize(("argnum", "label"), [(3, "dgate_up"), (4, "ddown")])
def test_vmap_grad_selective_expert_weight(assert_precision, argnum, label):
    x, gate_up, down, ti, tw = _inputs()

    def loss(xs, t, w, g1, g2):
        return opake_fused_moe(xs, g1, g2, t, w).square().mean()

    def reference(xs, t, w, g1, g2):
        return torch_reference_moe(xs, g1, g2, t, w).square().mean()

    actual = vmap(grad(loss, argnums=argnum), in_dims=(0, 0, 0, None, None))(
        x.unsqueeze(0), ti.unsqueeze(0), tw.unsqueeze(0), gate_up, down
    )
    expected = grad(reference, argnums=argnum)(x, ti, tw, gate_up, down)
    assert_precision(actual[0], expected, rtol=2e-2, atol=2e-2, label=label)


def test_vmap_grad_frozen_experts(assert_precision):
    """Frozen experts (attention-only LoRA): only ``x``/``tw`` need grad, so the
    backward takes the ``compute_wgrad=False`` skip. ``dx``/``dtw`` must still
    match the loop reference."""
    B = 4
    _, gate_up, down, ti, tw = _inputs()
    xb = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    tib = ti.unsqueeze(0).expand(B, -1, -1).contiguous()
    twb = tw.unsqueeze(0).expand(B, -1, -1).contiguous()

    def loss(xs, t, w, g1, g2):
        return opake_fused_moe(xs, g1, g2, t, w).square().mean()

    def rloss(xs, t, w, g1, g2):
        return torch_reference_moe(xs, g1, g2, t, w).square().mean()

    # Only x (0) and top_k_weights (2, the ``w`` arg) are differentiated;
    # experts g1/g2 (args 3, 4) stay shared and frozen — exercises the
    # compute_wgrad=False skip.
    g_op = vmap(grad(loss, argnums=(0, 2)), in_dims=(0, 0, 0, None, None))(
        xb, tib, twb, gate_up, down
    )
    refs = [
        grad(rloss, argnums=(0, 2))(xb[i], tib[i], twb[i], gate_up, down)
        for i in range(B)
    ]
    dx_r = torch.stack([r[0] for r in refs])
    dtw_r = torch.stack([r[1] for r in refs])

    assert_precision(g_op[0], dx_r, rtol=2e-2, atol=2e-2, label="frozen dx")
    assert_precision(
        g_op[1], dtw_r, rtol=2e-2, atol=2e-2, label="frozen dtop_k_weights"
    )
    # Per-sample dx still genuinely distinct across the batch.
    assert (g_op[0][0] - g_op[0][1]).abs().max().item() > 0


def test_empty_expert_routing(assert_precision):
    """Routing that leaves some experts with zero tokens still matches the ref."""
    x, gate_up, down, _, _ = _inputs()
    # Force all tokens to experts {0, 1} only — experts 2..E-1 are empty.
    ti = torch.randint(0, 2, (T, K), device="cuda")
    tw = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
    tw = tw / tw.sum(-1, keepdim=True)
    out = opake_fused_moe(x, gate_up, down, ti, tw)
    ref = torch_reference_moe(x, gate_up, down, ti, tw)
    assert_precision(out, ref, rtol=1e-2, atol=1e-2, label="empty experts")


def test_chunking_reduces_cuda_forward_peak(monkeypatch):
    torch.manual_seed(924)
    n = 4096
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
    ti = torch.randint(E, (n, K), device="cuda")
    tw = torch.rand(n, K, device="cuda", dtype=torch.bfloat16)

    def run():
        return opake_fused_moe(x, gate_up, down, ti, tw)

    def peak_bytes():
        out = run()
        torch.cuda.synchronize()
        del out
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = run()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del out
        return peak

    unchunked = peak_bytes()
    monkeypatch.setattr(_moe_memory, "_MAX_WORKSPACE_BYTES", 8 * 1024**2)
    chunked = peak_bytes()
    assert chunked < 0.8 * unchunked, (
        f"expected chunking to reduce CUDA peak: {chunked / 1024**2:.1f} MiB vs "
        f"{unchunked / 1024**2:.1f} MiB"
    )
