"""Precision support over the functional DP step.

Three axes:

1. **Full-cast bf16** — ``model.to(bfloat16)`` flows cleanly through
   ``vmap(grad)`` → ``clipped_grad`` → ``gaussian_noise`` → ``adamw``.
   Reduction-stability promotion to fp32 already lives in
   ``opake/clipping/clipped_fun.py:_sum_clipped_tensor``; we exercise it
   end-to-end and check dtype is preserved on the public boundary.

2. **TF32** — global toggle (``torch.backends.cuda.matmul.allow_tf32`` /
   ``torch.set_float32_matmul_precision``) is orthogonal to the
   functional DP step. CUDA-only.

3. **fp16 autocast** — ``torch.autocast(fp16)`` inside the loss closure
   works with ``vmap(grad)`` and the standard clipped/noised update path.
   CUDA-only (CPU autocast does not support fp16).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torchopt

from opake.api.engine.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.optimizers import adamw
from opake.random import key


def _build_model(seed: int = 0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )


def _make_batch(dtype: torch.dtype, device: torch.device, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(5, 8, generator=g).to(device=device, dtype=dtype)
    y = torch.randn(5, 4, generator=g).to(device=device, dtype=dtype)
    return x, y


def _full_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, noise_stddev: float
):
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=noise_stddev, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer = adamw(lr=1e-2)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(noised, opt_state, params=params)
    new_params = torchopt.apply_updates(params, updates, inplace=False)
    return grads.pytree, noised.pytree, new_params


# ----------------------------------------------------------------------------
# Full-cast bf16
# ----------------------------------------------------------------------------


def test_bf16_grad_finite_cpu():
    """Full-cast bf16 produces finite per-example clipped gradients."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    grads, noised, _ = _full_step(model, x, y, noise_stddev=0.0)
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite bf16 gradient"
    for n in noised:
        assert torch.isfinite(n).all(), "non-finite noised gradient"


def test_bf16_grad_dtype_preserved_cpu():
    """Reduction promotes to fp32 internally, but caller sees bf16 back."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    grads, noised, _ = _full_step(model, x, y, noise_stddev=0.0)
    for g in grads:
        assert g.dtype == torch.bfloat16, f"grad dtype leaked to {g.dtype}"
    for n in noised:
        assert n.dtype == torch.bfloat16, f"noised dtype leaked to {n.dtype}"


def test_bf16_full_pipeline_proximity_to_fp32_cpu():
    """bf16 grads are close to fp32 reference within calibrated tolerance."""
    model_fp32 = _build_model()
    x_fp32, y_fp32 = _make_batch(torch.float32, torch.device("cpu"))
    grads_fp32, _, _ = _full_step(model_fp32, x_fp32, y_fp32, noise_stddev=0.0)

    model_bf16 = _build_model().to(dtype=torch.bfloat16)
    x_bf16 = x_fp32.to(dtype=torch.bfloat16)
    y_bf16 = y_fp32.to(dtype=torch.bfloat16)
    grads_bf16, _, _ = _full_step(model_bf16, x_bf16, y_bf16, noise_stddev=0.0)

    for g32, g16 in zip(grads_fp32, grads_bf16, strict=True):
        diff = (g32.float() - g16.float()).pow(2).mean().item()
        assert diff < 5e-2, f"bf16 grad too far from fp32 reference: mse={diff:.4f}"


def test_bf16_optimizer_step_finite_cpu():
    """adamw.update on bf16 grads produces finite updated params."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    _, _, new_params = _full_step(model, x, y, noise_stddev=0.5)
    for p in new_params:
        assert torch.isfinite(p).all(), "adamw produced non-finite param update"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_bf16_full_pipeline_cuda():
    """Same end-to-end claim on CUDA (bf16 hardware path)."""
    model = _build_model().to(device="cuda", dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cuda"))
    grads, noised, new_params = _full_step(model, x, y, noise_stddev=0.5)
    for g in grads:
        assert torch.isfinite(g).all()
        assert g.dtype == torch.bfloat16
    for n in noised:
        assert torch.isfinite(n).all()
        assert n.dtype == torch.bfloat16
    for p in new_params:
        assert torch.isfinite(p).all()
        assert p.dtype == torch.bfloat16


# ----------------------------------------------------------------------------
# TF32 (CUDA-only)
# ----------------------------------------------------------------------------


@pytest.fixture
def _saved_tf32():
    """Snapshot + restore TF32 flags so tests don't bleed into one another."""
    saved_matmul = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn = torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved_matmul
    torch.backends.cudnn.allow_tf32 = saved_cudnn


def _step_simple(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Trimmed DP step for TF32 parity — torchopt.adamw, no AdamW-BC."""
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=0.0, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer = torchopt.adamw(lr=1e-2)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(noised.pytree, opt_state, params=params)
    new_params = torchopt.apply_updates(params, updates, inplace=False)
    return grads.pytree, new_params


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is CUDA-only")
@pytest.mark.usefixtures("_saved_tf32")
def test_tf32_on_off_parity_within_tolerance():
    """TF32 on vs off: matmul precision differs but gradient flow stays close."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grads_off, params_off = _step_simple(model, x, y)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    grads_on, params_on = _step_simple(model, x, y)

    for g_off, g_on in zip(grads_off, grads_on, strict=True):
        torch.testing.assert_close(g_off, g_on, rtol=2e-3, atol=1e-4)
    for p_off, p_on in zip(params_off, params_on, strict=True):
        torch.testing.assert_close(p_off, p_on, rtol=2e-3, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is CUDA-only")
@pytest.mark.usefixtures("_saved_tf32")
def test_tf32_via_set_float32_matmul_precision():
    """``torch.set_float32_matmul_precision('high')`` is also non-disruptive."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4)).cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    torch.set_float32_matmul_precision("highest")
    grads_high, _ = _step_simple(model, x, y)

    torch.set_float32_matmul_precision("medium")
    grads_med, _ = _step_simple(model, x, y)

    for ghigh, gmed in zip(grads_high, grads_med, strict=True):
        torch.testing.assert_close(ghigh, gmed, rtol=5e-3, atol=5e-4)

    torch.set_float32_matmul_precision("highest")


# ----------------------------------------------------------------------------
# fp16 autocast (CUDA-only)
# ----------------------------------------------------------------------------


_AUTOCAST_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fp16 autocast requires CUDA"
)


def _step_eager_no_autocast(model, x, y):
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    return grads.pytree


def _step_with_autocast(model, x, y):
    """Run the DP step with autocast(fp16)."""
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    return grads.pytree


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_bare_probe_runs():
    """Sanity: vmap(grad(autocast(loss))) executes without exception."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")
    grads = _step_with_autocast(model, x, y)
    assert grads is not None
    assert len(grads) > 0


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_bare_probe_parity_to_fp32():
    """Bare autocast grads should be near fp32 reference (no underflow yet)."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    g_fp32 = _step_eager_no_autocast(model, x, y)
    g_fp16 = _step_with_autocast(model, x, y)

    for a, b in zip(g_fp32, g_fp16, strict=True):
        diff = (a.float() - b.float()).pow(2).mean().item()
        assert diff < 1e-2, f"fp16 autocast mse vs fp32 too large: {diff:.5f}"


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_full_pipeline_with_optimizer():
    """End-to-end: autocast -> clipped -> noised -> AdamW step."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=0.5, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer = torchopt.adamw(lr=1e-2)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(noised.pytree, opt_state, params=params)
    new_params = torchopt.apply_updates(params, updates, inplace=False)

    for p in new_params:
        assert torch.isfinite(p).all(), (
            "fp16 autocast pipeline produced non-finite params"
        )
