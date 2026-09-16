"""``torch.compile`` over the opaque-patches Triton kernels.

Validates that the in-house Liger-style kernels survive ``torch.compile``
of the loss closure when used inside a functional DP step.

Triton kernels typically appear as opaque ops to Dynamo and may produce
graph breaks; we don't require ``fullgraph=True``. What we enforce:

  1. Compiled grads ≈ eager grads within tolerance.
  2. The compiler does not raise.

CUDA-only — Triton kernels in opaque-patches don't have a CPU path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.func import grad, vmap
from torch.utils._pytree import tree_leaves

from opaque.api.engine.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches.kernels import opaque_rms_norm, opaque_swiglu

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="opaque-patches kernels require CUDA"
    ),
]


class _RmsNormBlock(nn.Module):
    """Tiny module exercising ``opaque_rms_norm`` + a SwiGLU MLP."""

    def __init__(self, hidden: int = 16, intermediate: int = 32):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = opaque_rms_norm(x, self.norm_weight)
        gate = self.gate(x)
        up = self.up(x)
        hidden = opaque_swiglu(gate, up)
        return self.down(hidden)


def _build(seed: int = 0):
    torch.manual_seed(seed)
    model = _RmsNormBlock().cuda()
    x = torch.randn(4, 7, 16, device="cuda")
    target = torch.randn(4, 7, 16, device="cuda")
    return model, x, target


def _run(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, compile_backend: str | None
):
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    if compile_backend is not None:
        loss_fn = torch.compile(loss_fn, backend=compile_backend, fullgraph=False)

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    return grads


def _assert_close(a, b, *, rtol: float, atol: float):
    # clipped_grad returns a ClippedPytree; compare the leaf tensors.
    la, lb = tree_leaves(a.pytree), tree_leaves(b.pytree)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol)


def test_kernels_eager_baseline_no_nan_cuda():
    """Sanity: eager kernel path produces finite gradients before we ask about compile."""
    model, x, y = _build()
    eager = _run(model, x, y, compile_backend=None)
    for g in tree_leaves(eager.pytree):
        assert torch.isfinite(g).all(), "kernel produced non-finite eager gradient"


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_kernels_under_compile_cuda(backend: str):
    """opaque-patches Triton kernels survive torch.compile on CUDA."""
    model, x, y = _build()
    eager = _run(model, x, y, compile_backend=None)
    compiled = _run(model, x, y, compile_backend=backend)
    _assert_close(eager, compiled, rtol=1e-3, atol=1e-4)


def test_rmsnorm_vmap_grad_compiles_fullgraph_cuda():
    """Static backward metadata keeps the transformed RMSNorm graph capturable."""
    torch.manual_seed(1)
    x = torch.randn(4, 8, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, device="cuda", dtype=torch.bfloat16)

    def loss_fn(inp, weight):
        return opaque_rms_norm(inp, weight).float().square().mean()

    grad_fn = vmap(grad(loss_fn, argnums=(0, 1)), in_dims=(0, None))
    eager = grad_fn(x, w)
    compiled_fn = torch.compile(grad_fn, backend="inductor", fullgraph=True)
    compiled = compiled_fn(x, w)

    for actual, expected in zip(compiled, eager, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-3)
