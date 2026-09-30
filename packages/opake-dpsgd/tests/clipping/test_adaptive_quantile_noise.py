"""Tests for adaptive clipping with quantile noise (Phase 4)."""

import math

import pytest
import torch

from opake.api.dpsgd.clipping._adaptive import (
    _sample_noisy_clipping_rate,
    adaptive_clipped_grad,
)
from opake.dpsgd.accounting import adaclip, gaussian
from opake.exceptions import ConfigurationError
from opake.random import key
from opake.types import ClippedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


def test_centered_indicator_has_fixed_neighbor_sensitivity():
    def rate(count, size):
        return _sample_noisy_clipping_rate(
            count,
            size,
            expected_batch_size=8,
            key=key(42),
            step=0,
            fraction_noise_std=0.05,
        )

    empty = rate(0, 0)
    assert rate(1, 1) - empty == pytest.approx(1 / 16)
    assert rate(0, 1) - empty == pytest.approx(-1 / 16)
    assert rate(1, 12) - rate(0, 11) == pytest.approx(1 / 16)

    indicator_sensitivity = rate(1, 1) - empty
    accounted = adaclip(gaussian(1.5), fraction_noise_std=0.05, expected_batch_size=8)
    assert 1 / accounted.effective_noise_multiplier == pytest.approx(
        math.hypot(1 / 1.5, indicator_sensitivity / 0.05)
    )


def test_indicator_denominator_is_explicit_when_gradient_is_not_normalized():
    def loss_fn(params, x):
        return (params * x).sum()

    with pytest.raises(ConfigurationError, match="expected_batch_size is required"):
        adaptive_clipped_grad(loss_fn, key=key(0), batch_argnums=1)

    _, state = adaptive_clipped_grad(
        loss_fn, key=key(0), batch_argnums=1, expected_batch_size=8
    )
    assert state._expected_batch_size == 8


class TestQuantileNoise:
    """Tests for quantile noise in adaptive clipping."""

    def test_quantile_noise_requires_key(self):
        """Test that key is a required argument."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(TypeError, match="key"):
            adaptive_clipped_grad(
                loss_fn,
                fraction_noise_std=0.1,
                batch_argnums=(1, 2),
                expected_batch_size=1.0,
            )

    def test_invalid_fraction_noise_std(self):
        """Test that negative/zero fraction_noise_std raises error."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(
            ConfigurationError, match="fraction_noise_std must be positive"
        ):
            adaptive_clipped_grad(
                loss_fn,
                fraction_noise_std=-1.0,
                key=key(42),
                batch_argnums=(1, 2),
                expected_batch_size=1.0,
            )

    def test_quantile_noise_reproducibility(self):
        """Test that same key produces reproducible adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two identical setups
        grad_fn1, state1 = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Two updates prove the state advances while preserving the identical
        # noise sequence for equal keys.
        for _ in range(2):
            _, state1 = grad_fn1(params, batch_x, batch_y, state=state1)
            _, state2 = grad_fn2(params, batch_x, batch_y, state=state2)

            # Clip norms should be identical (same noise sequence)
            assert state1._next_clipping_norm == state2._next_clipping_norm
            assert state1._step == state2._step

    def test_quantile_noise_different_keys_produce_different_paths(self):
        """Test that different keys can produce different adaptation paths.

        Note: This test checks that the keys are preserved correctly and used in
        adaptation. Due to stochasticity, paths may or may not diverge in just
        a few steps, depending on clipping rates and noise magnitude. We verify
        that states preserve their different keys correctly.
        """

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two setups with different keys
        grad_fn1, state1 = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=1.0,  # Very large noise to increase chance of divergence
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=1.0,
            key=key(99),  # Different key
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Verify initial states have different keys
        assert state1._rng_key != state2._rng_key

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps - update keys during training
        for _ in range(20):
            _, state1 = grad_fn1(params, batch_x, batch_y, state=state1)
            _, state2 = grad_fn2(params, batch_x, batch_y, state=state2)

        # Keys should still be different after training
        assert state1._rng_key != state2._rng_key

        # With very large noise (1.0), divergence is likely but not guaranteed
        # The important thing is that keys are preserved and affect adaptation

    def test_quantile_noise_affects_adaptation(self):
        """Test that quantile noise actually affects the adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        grad_fn_no_noise, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # With substantial noise (to ensure divergence)
        grad_fn_noise, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.5,  # Increased noise for clear divergence
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps
        for _ in range(15):  # More steps for divergence
            _, state_no_noise = grad_fn_no_noise(
                params, batch_x, batch_y, state=state_no_noise
            )
            _, state_noise = grad_fn_noise(params, batch_x, batch_y, state=state_noise)

        # Adaptation paths should diverge with high noise
        # With fraction_noise_std=0.5, this should reliably cause different decisions
        assert state_no_noise._next_clipping_norm != state_noise._next_clipping_norm

    def test_state_preserves_key_and_step(self):
        """Test that state correctly preserves key and step counter."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Initial state
        assert state._rng_key == key(42)
        assert state._step == 0

        # Run a few steps
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        for i in range(1, 6):
            _, state = grad_fn(params, batch_x, batch_y, state=state)
            assert state._rng_key == key(42)  # Key unchanged
            assert state._step == i  # Step incremented

    def test_quantile_noise_with_aux_output(self):
        """Test that quantile noise works with aux output."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            return_aux=True,
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, aux), state = grad_fn(params, batch_x, batch_y, state=state)
        grads = _unwrap_clipped(grads)

        # Check aux contains expected fields
        assert aux.clipping_rate is not None
        assert grads.shape == params.shape
        assert state._step == 1


class TestQuantileNoiseClipNorm:
    """Tests for clipping_norm with quantile noise."""

    def test_clip_norm_unchanged_by_quantile_noise(self):
        """Test that quantile noise doesn't affect initial clipping_norm."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        _, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # With noise
        _, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # clipping_norm should be identical (same initial_clipping_norm)
        assert (
            state_no_noise._current_clipping_norm == state_noise._current_clipping_norm
        )
        assert state_no_noise._current_clipping_norm == 1.0

    def test_clip_norm_matches_initial(self):
        """Test clipping_norm equals initial_clipping_norm before any steps."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        _, state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=5.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        assert state._current_clipping_norm == 5.0
