"""DP-FTRL clipped grad + MF Gaussian noise under NCCL."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from dpftrl_ddp_helpers import (
    SimpleModel,
    _fixed_sd_mf,
    _spawn,
    _worker_mf_clip_three_steps,
    _worker_mf_parity,
)

from opake.distributed import sum_gradients
from opake.dpftrl.clipping import clipped_grad
from opake.dpftrl.noise import identity_strategy, mf_gaussian_noise
from opake.functional import make_functional
from opake.pytree import tree_map
from opake.random import key

pytestmark = pytest.mark.cuda


def _baseline_identity_mf_fc1() -> torch.Tensor:
    device = torch.device("cuda:0")
    m = SimpleModel().to(device)
    m.load_state_dict(_fixed_sd_mf())
    func_model, params, _frozen = make_functional(m, partition_trainable=True)

    def loss_fn(params, x, y):
        pred = func_model(params, x)
        return ((pred - y) ** 2).mean()

    grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
    tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
    noise_fn, noise_state = mf_gaussian_noise(
        tmpl,
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.1,
        key=key(0),
    )
    x = torch.arange(80, dtype=torch.float32, device=device).reshape(8, 10)
    y = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.1
    grads, _ = grad_fn(params, x, y, state=clip_state)
    summed = sum_gradients(grads)
    noised, _ = noise_fn(summed, noise_state)
    return noised.pytree["fc1.weight"].detach().cpu()


class TestClippedGradMFDistributed:
    def test_clipped_mf_three_steps(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_mf_clip_three_steps)

    def test_parity_identity_mf_single_vs_two_ranks(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        ref = _baseline_identity_mf_fc1()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "fc1.pt")
            _spawn(2, _worker_mf_parity, out)
            got = torch.load(out, map_location="cpu")
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
