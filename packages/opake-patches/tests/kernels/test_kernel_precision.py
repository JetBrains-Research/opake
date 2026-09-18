# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""bf16 precision regression guard for the Triton kernels.

For each kernel we measure its bf16-vs-fp32 error against the *eager bf16*-vs-fp32
error (the inherent bf16 "noise floor" for that op). The Opake kernels upcast
the error-prone reductions (RMSNorm variance, CE logsumexp, MoE expert sum) to
fp32 internally, so each kernel should land at or below its eager floor.

This catches a silent regression where a kernel starts doing a reduction in bf16
(which would push it well above the floor) — the kind of bug a CPU-only CI can't
see.
"""

# ``I`` is the intermediate dim in the (E, K, H, I, T) shape tuple — intentional.
# ruff: noqa: E741

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from opake.api.patches.kernels.moe import opake_moe
from opake.patches.kernels import (
    opake_cross_entropy_loss,
    opake_geglu_approx,
    opake_geglu_exact,
    opake_rms_norm,
    opake_swiglu,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# A kernel may be at most this multiple of the eager bf16 error vs fp32. Eager
# already upcasts some reductions (e.g. RMSNorm variance), so the bar is the
# eager floor with headroom for benign rounding differences; a kernel doing a
# reduction in bf16 would be ~5-10x worse and trip this.
SLACK = 3.0
EPS = 1e-7  # additive floor so near-zero eager errors don't make the ratio explode


def _err(actual_bf16, truth_fp32):
    return (actual_bf16.float() - truth_fp32).pow(2).mean().item()


def _assert_at_or_below_floor(name, opake_bf16, eager_bf16, truth_fp32):
    e_op = _err(opake_bf16, truth_fp32)
    e_floor = _err(eager_bf16, truth_fp32)
    print(
        f"\n{name}: opake_err={e_op:.3e}  eager_floor={e_floor:.3e}  ratio={e_op / (e_floor + EPS):.2f}"
    )
    assert e_op <= e_floor * SLACK + EPS, (
        f"{name}: bf16 error {e_op:.3e} exceeds {SLACK}x eager floor {e_floor:.3e} "
        "— a reduction may have dropped to bf16."
    )


def _rand(*shape, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32) * scale


def test_rms_norm_precision():
    H = 2048
    x32 = _rand(64, H, seed=1)
    w32 = _rand(H, seed=2)
    eps = 1e-6

    def eager(x, w):  # HF llama RMSNorm: variance in fp32, weight mult in x dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * w

    truth = eager(x32, w32)
    eager_bf16 = eager(x32.bfloat16(), w32.bfloat16())
    opake_bf16 = opake_rms_norm(x32.bfloat16(), w32.bfloat16(), eps)
    _assert_at_or_below_floor("rms_norm", opake_bf16, eager_bf16, truth)


@pytest.mark.parametrize(
    ("name", "eager_fn", "opake_fn"),
    [
        ("swiglu", lambda g, u: F.silu(g) * u, opake_swiglu),
        ("geglu_exact", lambda g, u: F.gelu(g) * u, opake_geglu_exact),
        (
            "geglu_approx",
            lambda g, u: F.gelu(g, approximate="tanh") * u,
            opake_geglu_approx,
        ),
    ],
)
def test_activation_precision(name, eager_fn, opake_fn):
    g32, u32 = _rand(64, 1024, seed=3), _rand(64, 1024, seed=4)
    truth = eager_fn(g32, u32)
    eager_bf16 = eager_fn(g32.bfloat16(), u32.bfloat16())
    opake_bf16 = opake_fn(g32.bfloat16(), u32.bfloat16())
    _assert_at_or_below_floor(name, opake_bf16, eager_bf16, truth)


def test_cross_entropy_precision():
    B, V = 256, 4096
    logits32 = _rand(B, V, seed=5)
    labels = torch.randint(0, V, (B,), device="cuda")

    def eager(logits):  # per-token loss (no reduction), matching the kernel output
        return F.cross_entropy(logits.float(), labels, reduction="none")

    truth = eager(logits32)
    eager_bf16 = eager(logits32.bfloat16())
    opake_bf16 = opake_cross_entropy_loss(logits32.bfloat16(), labels).reshape(B)
    _assert_at_or_below_floor("cross_entropy", opake_bf16, eager_bf16, truth)


def test_fused_moe_precision():
    E, K, H, I, T = 16, 4, 512, 256, 128
    x32 = _rand(T, H, seed=6)
    gate_up32 = _rand(E, 2 * I, H, seed=7, scale=0.05)
    down32 = _rand(E, H, I, seed=8, scale=0.05)
    probs = F.softmax(_rand(T, E, seed=9), dim=-1)
    tw, ti = torch.topk(probs, K, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)

    def eager(x, gate_up, down):  # sparse gather reference
        out = torch.zeros_like(x)
        for e in range(E):
            sel = ti == e
            tok, kp = sel.nonzero(as_tuple=True)
            gu = F.linear(x[tok], gate_up[e])
            g, u = gu[:, :I], gu[:, I:]
            ye = F.linear(F.silu(g) * u, down[e])
            out.index_add_(0, tok, (ye * tw[tok, kp].unsqueeze(-1).to(x.dtype)))
        return out

    truth = eager(x32, gate_up32, down32)
    eager_bf16 = eager(x32.bfloat16(), gate_up32.bfloat16(), down32.bfloat16())
    opake_bf16 = opake_moe(
        x32.bfloat16(),
        gate_up32.bfloat16(),
        down32.bfloat16(),
        ti,
        tw.bfloat16(),
    )
    _assert_at_or_below_floor("fused_moe", opake_bf16, eager_bf16, truth)
