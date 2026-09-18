"""Tests for clipped gaussian_noise with PerGroup bounds."""

import math

import pytest
import torch

from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.noise.types import GaussianNoiseState
from opake.random import key
from opake.types import NoisedPytree, PerGroup, clipped


class TestGaussianNoisePerGroup:
    """Tests for gaussian_noise() with PerGroup bounds."""

    def _make_pg(self, param_keys, group_values):
        groups = {k: k for k in param_keys}
        values = dict(zip(param_keys, group_values, strict=False))
        return PerGroup(groups=groups, values=values)

    def test_returns_tuple(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_per_group_noise(self):
        max_norm = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 1.0, "mlp": 5.0},
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
        grads = {
            "weight": torch.zeros(100),
            "bias": torch.zeros(100),
        }
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)

        assert isinstance(output, NoisedPytree)
        assert isinstance(output.noise_stddev, PerGroup)
        assert output.noise_stddev.values == {
            "attn": pytest.approx(math.sqrt(6.0)),
            "mlp": pytest.approx(math.sqrt(30.0)),
        }
        assert not torch.allclose(output.pytree["weight"], grads["weight"])
        assert not torch.allclose(output.pytree["bias"], grads["bias"])

        attn_var = output.pytree["weight"].var().item()
        mlp_var = output.pytree["bias"].var().item()
        assert mlp_var > attn_var * 2.5

    def test_zero_bound_group_returns_original(self):
        max_norm = PerGroup(
            groups={"noised": "g1", "clean": "g2"},
            values={"g1": 1.0, "g2": 0.0},
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grads = {
            "noised": torch.ones(5),
            "clean": torch.ones(5),
        }
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)
        torch.testing.assert_close(output.pytree["clean"], grads["clean"])
        assert not torch.allclose(output.pytree["noised"], grads["noised"])

    def test_all_zero_bound_returns_original(self):
        max_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.0, "g2": 0.0},
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grads = {"a": torch.randn(3), "b": torch.randn(3)}
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)
        torch.testing.assert_close(output.pytree["a"], grads["a"])
        torch.testing.assert_close(output.pytree["b"], grads["b"])

    def test_dtype_preservation(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        grads_f32 = {"w": torch.randn(5, dtype=torch.float32)}
        out_f32, state = noise_fn(clipped(grads_f32, max_norm=max_norm), state)
        assert out_f32.pytree["w"].dtype == torch.float32

        grads_f64 = {"w": torch.randn(5, dtype=torch.float64)}
        out_f64, state = noise_fn(clipped(grads_f64, max_norm=max_norm), state)
        assert out_f64.pytree["w"].dtype == torch.float64

    def test_step_counter_advances(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        assert state._step_counter == 0

        grads = clipped({"w": torch.zeros(3)}, max_norm=max_norm)
        _, state = noise_fn(grads, state)
        assert state._step_counter == 1
        _, state = noise_fn(grads, state)
        assert state._step_counter == 2

    def test_deterministic_noise(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        grads = clipped({"w": torch.zeros(10)}, max_norm=max_norm)

        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noisy1, _ = noise_fn1(grads, state1)

        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noisy2, _ = noise_fn2(grads, state2)

        torch.testing.assert_close(noisy1.pytree["w"], noisy2.pytree["w"])

    def test_negative_bound_raises(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": -1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(clipped({"w": torch.zeros(3)}, max_norm=max_norm), state)


class TestEndToEndPerGroup:
    """Integration test: clipped_grad emits clipped values for gaussian_noise."""

    def test_full_pipeline_per_group_bound(self):
        from opake.dpsgd.clipping import clipped_grad, per_group

        def loss(params, data):
            pred = params["attn_w"] * data + params["mlp_w"] * data
            return (pred**2).mean()

        params = {
            "attn_w": torch.tensor(1.0),
            "mlp_w": torch.tensor(2.0),
        }

        clip_bound = per_group(params, attn=1.0, mlp=2.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=clip_bound,
            normalize_by=10.0,
        )

        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))

        grads, clip_state = grad_fn(params, torch.randn(10), state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        assert isinstance(noisy_grads, NoisedPytree)
        assert isinstance(noisy_grads.pytree, dict)
        assert isinstance(noisy_grads.noise_stddev, PerGroup)
        assert noisy_grads.noise_stddev.values == {
            "attn": pytest.approx(1.1 * math.sqrt(0.03)),
            "mlp": pytest.approx(1.1 * math.sqrt(0.06)),
        }
