# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""MoE expert-FFN kernel tests.

1. Forward vs sparse-gather reference (the ground-truth MoE math)
2. Backward vs reference
3. vmap forward
4. vmap(grad): per-example gradients — the DP-SGD path
5. CPU fallback path
"""

# ``I`` is the intermediate dim in the (E, K, H, I, T) shape tuple — intentional.
# ruff: noqa: E741

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.moe import (
    opake_moe,
    torch_reference_moe,
)

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
    tw = (tw / tw.sum(-1, keepdim=True)).to(dtype)
    return x, gate_up, down, ti, tw


def _sparse_gather_moe(x, gate_up, down, ti, tw):
    """Ground-truth sparse MoE via per-expert nonzero gather."""
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


def test_forward_matches_sparse_reference(assert_precision):
    x, gate_up, down, ti, tw = _inputs()
    ref = _sparse_gather_moe(x, gate_up, down, ti, tw)
    out = opake_moe(x, gate_up, down, ti, tw)
    assert_precision(out, ref, rtol=1e-2, atol=1e-2, label="forward")


def test_dense_masked_equals_sparse():
    """Dense-masked == sparse gather (within bf16 tolerance)."""
    x, gate_up, down, ti, tw = _inputs()
    ref = _sparse_gather_moe(x, gate_up, down, ti, tw)
    out = torch_reference_moe(x, gate_up, down, ti, tw)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_backward_matches_reference(assert_precision):
    x, gate_up, down, ti, tw = _inputs(requires_grad=True)
    xr, gr, dr = (
        x.clone().detach().requires_grad_(True),
        gate_up.clone().detach().requires_grad_(True),
        down.clone().detach().requires_grad_(True),
    )
    _sparse_gather_moe(xr, gr, dr, ti, tw).square().mean().backward()
    opake_moe(x, gate_up, down, ti, tw).square().mean().backward()
    assert_precision(x.grad, xr.grad, rtol=2e-2, atol=2e-2, label="dx")
    assert_precision(gate_up.grad, gr.grad, rtol=2e-2, atol=2e-2, label="dgate_up")
    assert_precision(down.grad, dr.grad, rtol=2e-2, atol=2e-2, label="ddown")


def test_vmap_forward(assert_precision):
    B = 4
    _x, gate_up, down, ti, tw = _inputs()
    xb = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    tib = ti.unsqueeze(0).expand(B, -1, -1).contiguous()
    twb = tw.unsqueeze(0).expand(B, -1, -1).contiguous()
    out = vmap(lambda xs, t, w: opake_moe(xs, gate_up, down, t, w))(xb, tib, twb)
    ref = vmap(lambda xs, t, w: torch_reference_moe(xs, gate_up, down, t, w))(
        xb, tib, twb
    )
    assert_precision(out, ref, rtol=1e-2, atol=1e-2, label="vmap forward")


def test_vmap_grad_per_example(assert_precision):
    """Per-example gradients (DP-SGD path) match a loop reference and are distinct."""
    B = 4
    _x, gate_up, down, ti, tw = _inputs()
    xb = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    tib = ti.unsqueeze(0).expand(B, -1, -1).contiguous()
    twb = tw.unsqueeze(0).expand(B, -1, -1).contiguous()

    def loss(xs, t, w):
        return opake_moe(xs, gate_up, down, t, w).square().mean()

    g_op = vmap(grad(loss))(xb, tib, twb)
    g_ref = torch.stack(
        [
            grad(
                lambda xs, t, w: (
                    torch_reference_moe(xs, gate_up, down, t, w).square().mean()
                )
            )(xb[i], tib[i], twb[i])
            for i in range(B)
        ]
    )
    assert_precision(g_op, g_ref, rtol=2e-2, atol=2e-2, label="vmap(grad) dx")
    # per-example gradients are genuinely different across the batch
    assert (g_op[0] - g_op[1]).abs().max().item() > 0


def test_dense_cuda_autocast_normalizes_manual_backward_dtypes():
    x, gate_up, down, ti, tw = _inputs(dtype=torch.float32, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = opake_moe(x, gate_up, down, ti, tw, grouped=False)
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert x.grad is not None
    assert gate_up.grad is not None
    assert down.grad is not None


def test_cpu_fallback():
    """CPU inputs take the torch fallback (even on a CUDA+Triton host)."""
    x, gate_up, down, ti, tw = (
        t.cpu() if torch.is_tensor(t) else t
        for t in _inputs(device="cpu", dtype=torch.float32)
    )
    out = opake_moe(x, gate_up, down, ti, tw)
    ref = _sparse_gather_moe(x, gate_up, down, ti, tw)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
