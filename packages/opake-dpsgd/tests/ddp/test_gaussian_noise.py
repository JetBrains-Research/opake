"""DP-SGD Gaussian noise (single-process + NCCL spawn)."""

from __future__ import annotations

import pytest
import torch
from dpsgd_ddp_helpers import _spawn, _worker_shared_noise_is_deterministic

from opake.dpsgd.noise import gaussian_noise
from opake.random import key
from opake.types import clipped


def _noise_raw(noise_fn, grads, state):
    noised, state = noise_fn(clipped(grads, max_norm=1.0), state)
    return noised.pytree, state


class TestDistributedNoise:
    """Tests for gaussian_noise() behavior."""

    def test_distributed_false_matches_single_device(self):
        stddev = 1.0
        seed = 42

        noise_fn1, state1 = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=stddev, key=key(seed))

        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        noisy1, state1 = _noise_raw(noise_fn1, grads, state1)
        noisy2, state2 = _noise_raw(noise_fn2, grads, state2)

        assert torch.allclose(noisy1["weight"], noisy2["weight"])
        assert torch.allclose(noisy1["bias"], noisy2["bias"])

    def test_rank_fold_in_produces_distinct_streams(self):
        from opake.random import fold_in

        noise_fn1, state1 = gaussian_noise(
            noise_multiplier=1.0, key=fold_in(key(42), 0)
        )
        noise_fn2, state2 = gaussian_noise(
            noise_multiplier=1.0, key=fold_in(key(42), 1)
        )

        grads = {"weight": torch.zeros(4)}

        noisy1, state1 = _noise_raw(noise_fn1, grads, state1)
        noisy2, state2 = _noise_raw(noise_fn2, grads, state2)

        assert not torch.allclose(noisy1["weight"], noisy2["weight"])

    def test_distributed_noise_is_deterministic(self):
        stddev = 1.1
        seed = 123

        noise_fn1, state1 = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=stddev, key=key(seed))

        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        noisy1, state1 = _noise_raw(noise_fn1, grads, state1)
        noisy2, state2 = _noise_raw(noise_fn2, grads, state2)

        assert torch.allclose(noisy1["weight"], noisy2["weight"])
        assert torch.allclose(noisy1["bias"], noisy2["bias"])

    def test_distributed_false_different_seeds_produce_different_noise(self):
        stddev = 1.0
        grads = {"weight": torch.randn(10, 5)}

        noise_fn1, state1 = gaussian_noise(noise_multiplier=stddev, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=stddev, key=key(43))

        noisy1, state1 = _noise_raw(noise_fn1, grads, state1)
        noisy2, state2 = _noise_raw(noise_fn2, grads, state2)

        assert not torch.allclose(noisy1["weight"], noisy2["weight"], atol=1e-3)

    def test_distributed_preserves_stddev(self):
        seed = 42
        grads = {"weight": torch.randn(10, 5)}

        noise_fn_small, state_small = gaussian_noise(
            noise_multiplier=0.1, key=key(seed)
        )
        noisy_small, state_small = _noise_raw(noise_fn_small, grads, state_small)

        noise_fn_large, state_large = gaussian_noise(
            noise_multiplier=10.0, key=key(seed)
        )
        noisy_large, state_large = _noise_raw(noise_fn_large, grads, state_large)

        diff_small = (noisy_small["weight"] - grads["weight"]).abs().mean()
        diff_large = (noisy_large["weight"] - grads["weight"]).abs().mean()

        assert diff_large > diff_small * 10

    def test_zero_stddev_no_noise(self):
        seed = 42
        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(seed))
        noised, state = _noise_raw(noise_fn, grads, state)

        assert torch.allclose(noised["weight"], grads["weight"])
        assert torch.allclose(noised["bias"], grads["bias"])

    def test_negative_stddev_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            gaussian_noise(noise_multiplier=-1.0, key=key(42))


class TestDistributedNoiseWithPyTree:
    """Tests for noise with different PyTree structures."""

    def test_nested_pytree(self):
        stddev = 1.0
        seed = 42

        grads = {
            "layer1": {"weight": torch.randn(10, 5), "bias": torch.randn(5)},
            "layer2": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
        }

        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grads, state)

        assert "layer1" in noised
        assert "layer2" in noised
        assert "weight" in noised["layer1"]
        assert "bias" in noised["layer1"]

        assert not torch.allclose(noised["layer1"]["weight"], grads["layer1"]["weight"])

    def test_list_of_tensors(self):
        stddev = 1.0
        seed = 42

        grads = [torch.randn(10, 5), torch.randn(5)]

        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grads, state)

        assert len(noised) == 2
        assert noised[0].shape == grads[0].shape
        assert noised[1].shape == grads[1].shape

    def test_single_tensor(self):
        stddev = 1.0
        seed = 42

        grad = torch.randn(10, 5)

        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grad, state)

        assert not torch.allclose(noised, grad)
        assert noised.shape == grad.shape

    def test_preserves_dtype(self):
        stddev = 1.0
        seed = 42

        grads = {
            "float32": torch.randn(5, dtype=torch.float32),
            "float64": torch.randn(5, dtype=torch.float64),
        }

        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grads, state)

        assert noised["float32"].dtype == torch.float32
        assert noised["float64"].dtype == torch.float64

    def test_preserves_device(self, device):
        stddev = 1.0
        seed = 42

        grads = {
            "weight": torch.randn(10, 5, device=device),
            "bias": torch.randn(5, device=device),
        }

        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grads, state)

        assert noised["weight"].device.type == device.type
        assert noised["bias"].device.type == device.type


class TestNoiseCalibration:
    """Noise scale sanity checks."""

    def test_noise_std_matches_stddev_roughly(self):
        stddev = 2.0
        seed = 42
        n_samples = 10000

        grad = torch.zeros(n_samples)
        noise_fn, state = gaussian_noise(noise_multiplier=stddev, key=key(seed))
        noised, state = _noise_raw(noise_fn, grad, state)

        noise = noised - grad
        assert abs(noise.mean().item()) < 0.1
        assert abs(noise.std().item() - stddev) < 0.1


@pytest.mark.cuda
class TestDistributedGaussianNoiseSpawn:
    def test_shared_noise_is_deterministic(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_shared_noise_is_deterministic)
