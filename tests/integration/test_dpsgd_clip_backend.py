"""Automatic fused clipping dispatch through ``adaptive_clipped_grad``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from opake.api.engine.clipping import _clipped_fun as cf
from opake.device import fused_kernels_available
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.patches import apply_runtime_patches
from opake.random import key


@pytest.mark.cuda
@pytest.mark.skipif(
    not fused_kernels_available(), reason="needs a CUDA device and Triton"
)
@pytest.mark.parametrize("microbatch_size", [None, 4])
def test_adaptive_clipping_auto_matches_torch(microbatch_size, monkeypatch):
    apply_runtime_patches()
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).cuda()
    params = {n: p.detach() for n, p in net.named_parameters()}
    x = torch.randn(12, 5, 32, device="cuda")
    y = torch.randn(12, 5, 8, device="cuda") * 3.0

    def loss_fn(p, xb, yb):
        return F.mse_loss(functional_call(net, p, (xb,)), yb)

    with monkeypatch.context() as context:
        context.setattr(cf, "get_fused_clip_backend", lambda: None)
        torch_fn, torch_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.5,
            target_quantile=0.5,
            microbatch_size=microbatch_size,
            return_aux=True,
            key=key(0),
            normalize_by=12,
        )
        expected, expected_state = torch_fn(params, x, y, state=torch_state)
    fn, state = adaptive_clipped_grad(
        loss_fn,
        batch_argnums=(1, 2),
        initial_clipping_norm=0.5,
        target_quantile=0.5,
        microbatch_size=microbatch_size,
        return_aux=True,
        key=key(0),
        normalize_by=12,
    )
    actual, actual_state = fn(params, x, y, state=state)
    ((expected, expected_aux), (actual, actual_aux)) = expected, actual
    for name in params:
        torch.testing.assert_close(actual.pytree[name], expected.pytree[name])
    torch.testing.assert_close(
        actual_aux.grad_norms, expected_aux.grad_norms, rtol=1e-6, atol=0
    )
    torch.testing.assert_close(
        torch.as_tensor(actual_state._next_clipping_norm),
        torch.as_tensor(expected_state._next_clipping_norm),
    )
