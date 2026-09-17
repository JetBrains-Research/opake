"""``torch.autocast`` over the opaque-patches Triton kernels.

Validates that the public ``opaque_*`` wrappers honor an active
``torch.autocast`` context: when autocast is enabled, each kernel runs in the
autocast dtype end-to-end. Boundary tensors are cast before dispatch; fused
weight kernels cast full weights transiently inside their custom Functions.
Backward dtype follows forward, so upstream gradient propagation stays
consistent.

Without this wrapper-side cast, the kernels are dtype-passthrough — they
preserve their input dtype regardless of autocast — which produces a
hybrid graph (``nn.Linear`` runs in autocast dtype, ``opaque_rms_norm``
runs in fp32) that defeats the user's autocast intent.

CUDA-only.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils._pytree import tree_leaves

from opaque.api.engine.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches.kernels import (
    opaque_fused_add_rms_norm,
    opaque_geglu_approx,
    opaque_geglu_exact,
    opaque_rms_norm,
    opaque_swiglu,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="autocast + opaque kernels require CUDA"
    ),
]


# ----------------------------------------------------------------------------
# Per-kernel: under autocast the public wrapper returns the autocast dtype.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_opaque_rms_norm_returns_autocast_dtype(amp_dtype: torch.dtype):
    x = torch.randn(2, 3, 16, device="cuda")
    w = torch.ones(16, device="cuda")
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        y = opaque_rms_norm(x, w)
    assert y.dtype == amp_dtype, f"expected {amp_dtype} under autocast, got {y.dtype}"


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_opaque_swiglu_returns_autocast_dtype(amp_dtype: torch.dtype):
    g = torch.randn(2, 3, 32, device="cuda")
    u = torch.randn(2, 3, 32, device="cuda")
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        y = opaque_swiglu(g, u)
    assert y.dtype == amp_dtype


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_opaque_geglu_returns_autocast_dtype(amp_dtype: torch.dtype):
    g = torch.randn(2, 3, 32, device="cuda")
    u = torch.randn(2, 3, 32, device="cuda")
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        y_exact = opaque_geglu_exact(g, u)
        y_approx = opaque_geglu_approx(g, u)
    assert y_exact.dtype == amp_dtype
    assert y_approx.dtype == amp_dtype


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_opaque_fused_add_rms_norm_returns_autocast_dtype(amp_dtype: torch.dtype):
    x = torch.randn(2, 3, 16, device="cuda")
    res = torch.randn(2, 3, 16, device="cuda")
    w = torch.ones(16, device="cuda")
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out, residual = opaque_fused_add_rms_norm(x, res, w)
    assert out.dtype == amp_dtype
    assert residual.dtype == amp_dtype


# ----------------------------------------------------------------------------
# Block-level: every intermediate respects the autocast dtype.
# ----------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, h: int = 16, i: int = 32):
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(h))
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = opaque_rms_norm(x, self.norm_w)
        gate = self.gate(x)
        up = self.up(x)
        hidden = opaque_swiglu(gate, up)
        return self.down(hidden)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_block_under_autocast_uniform_intermediate_dtype(amp_dtype: torch.dtype):
    """RMSNorm + Linear + SwiGLU + Linear: every intermediate matches autocast dtype."""
    torch.manual_seed(0)
    model = _Block().cuda()
    x = torch.randn(2, 3, 16, device="cuda")
    seen = []

    orig_forward = model.forward

    def tap_forward(inp):
        seen.append(("input", inp.dtype))
        x = opaque_rms_norm(inp, model.norm_w)
        seen.append(("rms", x.dtype))
        g = model.gate(x)
        seen.append(("gate", g.dtype))
        u = model.up(x)
        seen.append(("up", u.dtype))
        h = opaque_swiglu(g, u)
        seen.append(("swiglu", h.dtype))
        out = model.down(h)
        seen.append(("down", out.dtype))
        return out

    model.forward = tap_forward  # type: ignore[assignment]
    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _ = model(x)
    finally:
        model.forward = orig_forward  # type: ignore[assignment]

    # Input is fp32 (typical user setup); every internal step should be amp_dtype.
    assert seen[0] == ("input", torch.float32), seen
    for stage, dt in seen[1:]:
        assert dt == amp_dtype, (
            f"intermediate '{stage}' is {dt}, expected {amp_dtype}; full trace: {seen}"
        )


# ----------------------------------------------------------------------------
# Backward / DP pipeline under autocast.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_kernels_under_autocast_dp_step_finite(amp_dtype: torch.dtype):
    """Full DP step (vmap(grad) → clip) under autocast returns finite grads."""
    torch.manual_seed(0)
    model = _Block().cuda()
    x = torch.randn(4, 7, 16, device="cuda")
    y = torch.randn(4, 7, 16, device="cuda")

    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    for g in tree_leaves(grads.pytree):
        assert torch.isfinite(g).all(), f"non-finite grad under autocast({amp_dtype})"


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_kernels_autocast_proximity_to_full_cast(amp_dtype: torch.dtype):
    """Autocast(dtype) ≈ full-cast(dtype) at the kernel level within tolerance.

    Once wrapper-side autocast-follow lands, autocast-driven execution should
    closely match an explicitly cast model — same kernel paths, same intermediate
    dtypes.
    """
    torch.manual_seed(0)
    model_fp32 = _Block().cuda()
    x = torch.randn(2, 3, 16, device="cuda")

    # full-cast reference
    model_full = _Block().cuda().to(dtype=amp_dtype)
    model_full.load_state_dict(
        {k: v.to(dtype=amp_dtype) for k, v in model_fp32.state_dict().items()}
    )
    with torch.no_grad():
        out_full = model_full(x.to(dtype=amp_dtype))

    # autocast on the fp32 model
    with torch.autocast(device_type="cuda", dtype=amp_dtype), torch.no_grad():
        out_amp = model_fp32(x)

    assert out_amp.dtype == amp_dtype
    assert out_full.dtype == amp_dtype
    rtol = 5e-3 if amp_dtype is torch.float16 else 1e-2
    atol = 5e-3 if amp_dtype is torch.float16 else 1e-2
    torch.testing.assert_close(out_amp.float(), out_full.float(), rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Numerical backward parity: autocast vs fp32 reference (mse-clipped)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_block_backward_grad_proximity_to_fp32(amp_dtype: torch.dtype):
    """Per-parameter grad mse(autocast, fp32) within calibrated max_norm.

    Goes beyond ``finite_grads`` — a lower-max_norm on numerical agreement
    between an autocast-driven backward and the fp32 reference. Bound is
    generous (autocast is intentionally lower-precision) but tight enough
    to fail if the kernel backward silently drops precision.
    """
    torch.manual_seed(0)
    model = _Block().cuda()
    x = torch.randn(2, 3, 16, device="cuda", requires_grad=False)
    y = torch.randn(2, 3, 16, device="cuda")

    # fp32 reference
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out = model(x)
    ((out - y) ** 2).mean().backward()
    fp32_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    # autocast pass
    for p in model.parameters():
        p.grad = None
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_amp = model(x)
        loss_amp = ((out_amp - y) ** 2).mean()
    loss_amp.backward()
    amp_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    # Grads must come back at the parameter's dtype (fp32 here, since params are fp32);
    # autograd applies the inverse cast across the .to(amp_dtype) edge.
    for n in fp32_grads:
        a = fp32_grads[n].float()
        b = amp_grads[n].float()
        mse = (a - b).pow(2).mean().item()
        # bf16 has ~3 decimal digits, fp16 ~3-4; threshold calibrated against the block size.
        mse_bound = 5e-2 if amp_dtype is torch.float16 else 1e-1
        assert mse < mse_bound, (
            f"param {n}: grad mse {mse:.4f} exceeds mse_bound {mse_bound} for autocast({amp_dtype})"
        )


# ----------------------------------------------------------------------------
# Backward dtype edge case: autocast Linear → opaque kernel → autocast Linear
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_autocast_linear_then_kernel_then_linear_backward(amp_dtype: torch.dtype):
    """Mixed graph: autograd flows through autocast Linear → kernel → Linear.

    Pins down the dtype edge: an autocast-registered op (Linear) followed by
    our manually-cast kernel (RMSNorm) followed by another Linear. Backward
    must connect cleanly without dtype mismatches and produce finite grads.
    """
    torch.manual_seed(0)
    pre = nn.Linear(16, 16, bias=False).cuda()
    norm_w = nn.Parameter(torch.ones(16, device="cuda"))
    post = nn.Linear(16, 4, bias=False).cuda()
    x = torch.randn(2, 3, 16, device="cuda")
    y = torch.randn(2, 3, 4, device="cuda")

    for p in (*pre.parameters(), norm_w, *post.parameters()):
        if p.grad is not None:
            p.grad = None

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        h = pre(x)
        h = opaque_rms_norm(h, norm_w)
        out = post(h)
        loss = ((out - y) ** 2).mean()

    loss.backward()
    for p, name in [
        (pre.weight, "pre.weight"),
        (norm_w, "norm_w"),
        (post.weight, "post.weight"),
    ]:
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), (
            f"{name} grad has non-finite under autocast"
        )


# ----------------------------------------------------------------------------
# Cross-product: autocast × torch.compile
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_kernels_autocast_under_compile(backend: str, amp_dtype: torch.dtype):
    """The cross-product: autocast inside a ``torch.compile``'d loss closure
    over ``vmap(grad)`` + opaque kernels. Compiled grads ≈ eager grads.
    """
    torch.manual_seed(0)
    model = _Block().cuda()
    x = torch.randn(4, 7, 16, device="cuda")
    y = torch.randn(4, 7, 16, device="cuda")

    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            return ((pred - yi) ** 2).mean()

    grad_fn_eager, st_e = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    eager_grads, _ = grad_fn_eager(params, x, y, state=st_e)

    compiled_loss = torch.compile(loss_fn, backend=backend, fullgraph=False)
    grad_fn_c, st_c = clipped_grad(
        compiled_loss, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    compiled_grads, _ = grad_fn_c(params, x, y, state=st_c)

    for a, b in zip(
        tree_leaves(eager_grads.pytree),
        tree_leaves(compiled_grads.pytree),
        strict=True,
    ):
        diff = (a.float() - b.float()).pow(2).mean().item()
        assert diff < 1e-3, f"compile×autocast mse {diff:.5f} too large"


# ----------------------------------------------------------------------------
# Real-architecture HF model under apply_model_patches + autocast + vmap(grad)
# ----------------------------------------------------------------------------


def _build_tiny_llama():
    """Build a tiny offline Llama model (no HF download)."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )
    return transformers.LlamaModel(cfg).cuda()


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_tiny_llama_under_autocast_and_patches(amp_dtype: torch.dtype):
    """Tiny offline LlamaModel + apply_model_patches + autocast → finite forward+backward."""
    pytest.importorskip("transformers")
    from opaque.patches import apply_model_patches

    torch.manual_seed(0)
    model = _build_tiny_llama()
    apply_model_patches(model, performance=True, compat=True, peft=False)

    input_ids = torch.randint(0, 128, (2, 8), device="cuda")
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model(input_ids=input_ids)
        last = out.last_hidden_state
        loss = last.float().pow(2).mean()

    loss.backward()
    grads_seen = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        grads_seen += 1
        assert torch.isfinite(p.grad).all(), (
            f"non-finite grad on {n} under autocast({amp_dtype})"
        )
    assert grads_seen > 0, "no parameter received a gradient — backward path is severed"


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_tiny_llama_under_autocast_dp_step(amp_dtype: torch.dtype):
    """Same tiny model under apply_model_patches + autocast + vmap(grad) + clipped_grad."""
    pytest.importorskip("transformers")
    from opaque.patches import apply_model_patches

    torch.manual_seed(0)
    model = _build_tiny_llama()
    apply_model_patches(model, performance=True, compat=True, peft=False)

    fmodel, params = make_functional(model)
    input_ids = torch.randint(0, 128, (3, 8), device="cuda")

    def loss_fn(p, ids):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = fmodel(p, input_ids=ids.unsqueeze(0))
            return out.last_hidden_state.float().pow(2).mean()

    grad_fn, st = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1,), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, input_ids, state=st)
    for g in tree_leaves(grads.pytree):
        assert torch.isfinite(g).all(), (
            f"non-finite grad in DP step under autocast({amp_dtype}) on tiny Llama"
        )
