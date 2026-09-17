"""Lightweight MPS compatibility checks.

These tests are intentionally tiny and safe to run on MPS in default test runs.
"""

import pytest
import torch

from opake.api.engine.clipping import clipped_grad


@pytest.mark.mps
def test_mps_tiny_clipped_grad_smoke():
    """Run clipped_grad on a tiny tensor workload on MPS."""

    params = torch.tensor([0.2, -0.3], device="mps")
    batch_x = torch.randn(4, 2, device="mps")
    batch_y = torch.randn(4, device="mps")

    def loss_fn(weights, x_single, y_single):
        pred = torch.sum(weights * x_single)
        return (pred - y_single).pow(2).sum()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=(1, 2),
    )

    grads, _ = grad_fn(params, batch_x, batch_y, state=clip_state)

    # `clipped_grad` returns a ``ClippedPytree`` wrapper; the tensor is
    # under ``.pytree``.
    assert grads.pytree.device.type == "mps"
    assert torch.isfinite(grads.pytree).all()


@pytest.mark.mps
def test_mps_tiny_matmul_smoke():
    """Run a tiny native torch op on MPS to exercise basic backend path."""

    x = torch.randn(8, 8, device="mps")
    y = x @ x.T

    assert y.device.type == "mps"
    assert torch.isfinite(y).all()
