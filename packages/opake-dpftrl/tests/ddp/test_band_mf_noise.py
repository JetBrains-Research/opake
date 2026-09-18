"""Band MF noise parity under NCCL vs single-process."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from dpftrl_ddp_helpers import (
    TinyModel,
    _fixed_sd_tiny,
    _spawn,
    _worker_band_parity,
)

from opake.distributed import sum_gradients
from opake.dpftrl.clipping import clipped_grad
from opake.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opake.functional import make_functional
from opake.pytree import tree_map
from opake.random import key

pytestmark = pytest.mark.cuda


def _baseline_band_fc1() -> torch.Tensor:
    device = torch.device("cuda:0")
    m = TinyModel().to(device)
    m.load_state_dict(_fixed_sd_tiny())
    func_model, params, _frozen = make_functional(m, partition_trainable=True)

    def loss_fn(params, x, y):
        pred = func_model(params, x)
        return ((pred - y) ** 2).mean()

    grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
    tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
    noise_fn, noise_state = mf_gaussian_noise(
        tmpl,
        band_mf_strategy(bands=2, momentum=0.9),
        n_steps=4,
        noise_multiplier=0.85,
        key=key(5),
    )
    x = torch.arange(48, dtype=torch.float32, device=device).reshape(8, 6)
    y = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.05
    grads, _ = grad_fn(params, x, y, state=clip_state)
    summed = sum_gradients(grads)
    noised, _ = noise_fn(summed, noise_state)
    return noised.pytree["fc1.weight"].detach().cpu()


class TestBandMfNoiseDistributed:
    def test_parity_band_mf_single_vs_two_ranks(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        ref = _baseline_band_fc1()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "fc1.pt")
            _spawn(2, _worker_band_parity, out)
            got = torch.load(out, map_location="cpu")
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
