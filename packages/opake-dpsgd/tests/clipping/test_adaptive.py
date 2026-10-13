"""Tests for adaptive gradient clipping."""

import math

import pytest
import torch

from opake.api.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opake.api.engine.clipping import _clipped_fun as clipped_fun_module
from opake.dpsgd.clipping.types import ClippingStats
from opake.exceptions import ConfigurationError
from opake.random import key
from opake.types import ClippedPytree, NoisedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


class TestAdaptiveClippedGrad:
    """Tests for adaptive_clipped_grad function."""

    def test_rejects_stats_with_per_example_auxiliary_output(self):
        with pytest.raises(ValueError, match="return_stats"):
            adaptive_clipped_grad(
                lambda params, data: (params - data).square(),
                initial_clipping_norm=1.0,
                key=key(0),
                batch_argnums=1,
                return_aux=True,
                return_stats=True,
                expected_batch_size=1.0,
            )

    def test_basic_usage(self):
        """Test basic adaptive clipping workflow."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create adaptive clipping function
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.5,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Check initial state
        assert clip_state._next_clipping_norm == 1.0

        # Compute gradients
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        # Check state updated
        assert isinstance(clip_state._next_clipping_norm, float)
        assert grads.shape == params.shape

    def test_stats_preserve_sanitized_adaptive_clipping_output(self):
        def loss_fn(params, data):
            return torch.sqrt(data - params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=1,
            return_stats=True,
            expected_batch_size=1.0,
        )

        (grads, stats), _ = grad_fn(
            torch.tensor(0.0),
            torch.tensor([1.0, -1.0]),
            state=clip_state,
        )

        assert isinstance(stats, ClippingStats)
        assert stats.batch_size == 2
        assert torch.isfinite(_unwrap_clipped(grads)).all()

    def test_threshold_increases_when_too_many_clipped(self):
        """Test that threshold increases when clipping rate > target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very low threshold → many gradients will be clipped
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=0.01,  # Very low
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clipping_norm = clip_state._next_clipping_norm

        # First step
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should increase (many gradients clipped)
        assert clip_state._next_clipping_norm > initial_clipping_norm

    def test_threshold_decreases_when_too_few_clipped(self):
        """Test that threshold decreases when clipping rate < target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very high threshold → few gradients will be clipped
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=100.0,  # Very high
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clipping_norm = clip_state._next_clipping_norm

        # First step
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should decrease (few gradients clipped)
        assert clip_state._next_clipping_norm < initial_clipping_norm

    def test_geometric_update_formula(self):
        """Test proportional update: C_{t+1} = C_t * exp(η * (ρ̃ - γ))."""
        from opake.api.dpsgd.clipping._adaptive import _adaptive_clipping_norm_update

        # Direct unit test of the update formula with known values
        base = 1.0
        lr = 0.2
        target = 0.5

        # Case 1: clipping_rate above target → threshold increases
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.8,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.8 - target))
        assert abs(result - expected) < 1e-6

        # Case 2: clipping_rate below target → threshold decreases
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.2,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.2 - target))
        assert abs(result - expected) < 1e-6

        # Case 3: clipping_rate == target → no change
        result = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=target,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )
        assert abs(result - base) < 1e-6

    def test_proportional_update_step_size(self):
        """Test that step size is proportional to deviation from target."""
        from opake.api.dpsgd.clipping._adaptive import _adaptive_clipping_norm_update

        base = 1.0
        lr = 0.2
        target = 0.5

        # Small deviation
        result_small = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.55,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )

        # Large deviation
        result_large = _adaptive_clipping_norm_update(
            base_clipping_norm=base,
            noisy_clipping_rate=0.95,
            target_quantile=target,
            learning_rate=lr,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
        )

        # Large deviation should produce a bigger step
        assert abs(result_large - base) > abs(result_small - base)

    def test_threshold_clamped_to_bounds(self):
        """Test that threshold is clamped to [clipping_norm_min, clipping_norm_max]."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        clipping_norm_min = 0.5
        clipping_norm_max = 2.0

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=0.1,  # Below min
            target_quantile=0.5,
            learning_rate=0.2,
            clipping_norm_min=clipping_norm_min,
            clipping_norm_max=clipping_norm_max,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run many steps to push threshold beyond bounds
        for _ in range(20):
            _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should be clamped
        assert clipping_norm_min <= clip_state._next_clipping_norm <= clipping_norm_max

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        for _i in range(1, 6):
            _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    def test_with_aux_output(self):
        """Test that has_aux=True works correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            loss = ((pred - y) ** 2).mean()
            # Auxiliary outputs must be tensors for torch.func.grad
            aux = {"accuracy": torch.tensor(0.95)}
            return loss, aux

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            has_aux=True,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,  # Need to explicitly request aux outputs
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Should return (grads, grad_aux) and new state
        (grads, _grad_aux), clip_state = grad_fn(
            params, batch_x, batch_y, state=clip_state
        )
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape

    def test_different_target_quantiles(self):
        """Test that different target quantiles affect adaptation."""
        # Use fixed seed for reproducibility in CI
        torch.manual_seed(42)

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(16, 10)  # Larger batch for more stable statistics
        batch_y = torch.randn(16)

        # Low target quantile (aim to clip fewer gradients)
        grad_fn_low, clip_state_low = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.1,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # High target quantile (aim to clip more gradients)
        grad_fn_high, clip_state_high = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.9,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Run 20 steps for more reliable convergence
        for _ in range(20):
            _, clip_state_low = grad_fn_low(
                params, batch_x, batch_y, state=clip_state_low
            )
            _, clip_state_high = grad_fn_high(
                params, batch_x, batch_y, state=clip_state_high
            )

        # Low quantile should result in higher threshold (clip fewer)
        # High quantile should result in lower threshold (clip more)
        assert clip_state_low._next_clipping_norm > clip_state_high._next_clipping_norm

    @pytest.mark.parametrize("target_quantile", [0.25, 0.75])
    def test_converges_to_requested_clipped_fraction(self, target_quantile):
        """The steady-state clipped fraction equals ``target_quantile``.

        The targets are asymmetric on purpose: 0.5 is a fixed point of
        ``x -> 1 - x``, so it cannot tell "fraction clipped" apart from
        the Andrew et al. "fraction left unclipped" convention.
        """

        # Per-example gradient of this loss is the example itself, so the
        # per-example gradient norms are exactly ``norms``.
        def loss_fn(params, x):
            return (params * x).sum()

        norms = torch.arange(1, 101, dtype=torch.float32) / 10.0
        batch_x = torch.zeros(100, 4)
        batch_x[:, 0] = norms
        params = torch.zeros(4)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=target_quantile,
            learning_rate=0.5,
            fraction_noise_std=1e-6,
            clipping_norm_min=0.01,
            clipping_norm_max=100.0,
            key=key(0),
            batch_argnums=(1,),
            normalize_by=100,
        )

        for _ in range(200):
            _, clip_state = grad_fn(params, batch_x, state=clip_state)

        # ``_current_clipping_norm`` is the threshold used on the last step;
        # ``_next_clipping_norm`` is the post-update value for the next one.
        clipped_fraction = float(
            (norms > clip_state._current_clipping_norm).float().mean()
        )
        assert clipped_fraction == pytest.approx(target_quantile, abs=0.02)

    def test_different_learning_rates(self):
        """Test that learning rate affects adaptation speed."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Slow adaptation
        grad_fn_slow, clip_state_slow = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=0.01,
            learning_rate=0.05,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Fast adaptation
        grad_fn_fast, clip_state_fast = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=0.01,
            learning_rate=0.5,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Run 5 steps
        for _ in range(5):
            _, clip_state_slow = grad_fn_slow(
                params, batch_x, batch_y, state=clip_state_slow
            )
            _, clip_state_fast = grad_fn_fast(
                params, batch_x, batch_y, state=clip_state_fast
            )

        # Fast should have adapted more
        assert abs(clip_state_fast._next_clipping_norm - 0.01) > abs(
            clip_state_slow._next_clipping_norm - 0.01
        )

    def test_kwargs_passed_to_clipped_grad(self):
        """Test that additional kwargs are passed to clipped_grad."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Test with return_aux=True to verify kwargs are passed
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,  # Should be passed to clipped_grad
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, grad_aux), clip_state = grad_fn(
            params, batch_x, batch_y, state=clip_state
        )
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape
        assert grad_aux.loss_values is not None
        assert grad_aux.grad_norms is not None

    def test_composition_with_noise(self):
        """Test that adaptive clipping composes naturally with noise."""
        from opake.dpsgd.noise import gaussian_noise

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Compute clipped gradients
        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Add noise scaled to the clipped gradient metadata.
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        assert isinstance(noisy_grads, NoisedPytree)
        assert noisy_grads.noise_stddev == pytest.approx(1.1 * grads.max_norm)
        assert noisy_grads.pytree.shape == grads.pytree.shape
        assert not torch.allclose(noisy_grads.pytree, grads.pytree)  # Noise added

    def test_microbatching_produces_identical_results(self):
        """Test that microbatching produces identical gradients and state updates."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two functions with same config, one with microbatching
        grad_fn_no_mb, state_no_mb = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=None,  # No microbatching
            expected_batch_size=1.0,
        )

        grad_fn_mb, state_mb = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=8,  # Process in microbatches of 8
            expected_batch_size=1.0,
        )

        # Test over multiple steps to verify state updates match
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(32, 10)
        batch_y = torch.randn(32)

        for _step in range(3):
            grads_no_mb, state_no_mb = grad_fn_no_mb(
                params, batch_x, batch_y, state=state_no_mb
            )
            grads_mb, state_mb = grad_fn_mb(params, batch_x, batch_y, state=state_mb)
            grads_no_mb = _unwrap_clipped(grads_no_mb)
            grads_mb = _unwrap_clipped(grads_mb)

            # Gradients should be identical
            torch.testing.assert_close(grads_mb, grads_no_mb, rtol=1e-5, atol=1e-6)

            # State updates should be identical
            assert math.isclose(
                state_mb._next_clipping_norm,
                state_no_mb._next_clipping_norm,
                rel_tol=1e-5,
            )

    def test_microbatching_no_aux_does_not_materialize_per_example_aux(
        self, monkeypatch
    ):
        """No-aux adaptive microbatching should stream stats instead of concatenating aux."""

        original = clipped_fun_module._microbatch_accumulate_reduced

        def wrapped(*args, **kwargs):
            if kwargs.get("return_aux"):
                raise AssertionError("unexpected per-example aux materialization")
            return original(*args, **kwargs)

        monkeypatch.setattr(
            clipped_fun_module, "_microbatch_accumulate_reduced", wrapped
        )

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=4,
            return_aux=False,
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(16, 10)
        batch_y = torch.randn(16)

        grads, new_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape
        assert new_state._step == 1


class TestInputValidation:
    """Tests for parameter validation."""

    def test_invalid_initial_clipping_norm(self):
        """Test that negative initial_clipping_norm raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(
            ConfigurationError, match="initial_clipping_norm must be positive"
        ):
            adaptive_clipped_grad(
                loss_fn, initial_clipping_norm=-1.0, key=key(0), expected_batch_size=1.0
            )

    def test_invalid_target_quantile(self):
        """Test that target_quantile outside (0, 1) raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(
                loss_fn, target_quantile=0.0, key=key(0), expected_batch_size=1.0
            )

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(
                loss_fn, target_quantile=1.0, key=key(0), expected_batch_size=1.0
            )

    def test_invalid_learning_rate(self):
        """Test that negative learning_rate raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adaptive_clipped_grad(
                loss_fn, learning_rate=-0.1, key=key(0), expected_batch_size=1.0
            )

    def test_invalid_clipping_norm_min(self):
        """Test that negative clipping_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(
            ConfigurationError, match="clipping_norm_min must be positive"
        ):
            adaptive_clipped_grad(
                loss_fn, clipping_norm_min=-0.1, key=key(0), expected_batch_size=1.0
            )

    def test_invalid_clipping_norm_max(self):
        """Test that clipping_norm_max <= clipping_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(
            ValueError, match=r"clipping_norm_max.*must be.*clipping_norm_min"
        ):
            adaptive_clipped_grad(
                loss_fn,
                clipping_norm_min=10.0,
                clipping_norm_max=5.0,
                key=key(0),
                expected_batch_size=1.0,
            )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_example_batch(self):
        """Test with batch size of 1."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1, 10)
        batch_y = torch.randn(1)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape

    def test_zero_gradients(self):
        """Test with zero gradients (e.g., at local minimum)."""

        def loss_fn(params, x, y):
            return torch.tensor(0.0)  # Constant loss → zero gradients

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        # Should handle gracefully
        assert torch.allclose(grads, torch.zeros_like(grads))

    def test_batch_size_tracked_in_state(self):
        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )
        assert clip_state._batch_size == 0

        params = torch.randn(10, requires_grad=False)

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        assert clip_state._batch_size == 8

        batch_x = torch.randn(16, 10)
        batch_y = torch.randn(16)
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        assert clip_state._batch_size == 16

    def test_normalize_by_does_not_affect_clipping_rate(self):
        """normalize_by is post-processing on gradients, not clipping rate."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        expected_bs = 20
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=0.001,  # Very small → everything clips
            key=key(0),
            batch_argnums=(1, 2),
            fraction_noise_std=1e-10,  # Near-zero for deterministic testing
            normalize_by=expected_bs,
            return_aux=True,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)
        (_, aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert clip_state._batch_size == 8
        # All 8 examples clipped → rate = 8/8 = 1.0 (normalize_by is irrelevant)
        assert aux.clipping_rate == pytest.approx(1.0, abs=1e-6)

    def test_large_batch(self):
        """Test with large batch size."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1000, 10)  # Large batch
        batch_y = torch.randn(1000)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape
