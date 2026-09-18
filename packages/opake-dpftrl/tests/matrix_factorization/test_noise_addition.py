"""Tests for matrix factorization noise addition (DP-FTRL noise)."""

import pytest
import torch

from opake.api.dpftrl.noise._engine import MFNoiseState, _matrix_factorization_noise
from opake.api.dpftrl.noise._streaming_matrix import (
    StreamingMatrix,
    identity,
    prefix_sum,
)
from opake.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opake.random import key


def _dense_as_streaming(matrix: torch.Tensor) -> StreamingMatrix:
    def init_multiply(_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return ()

    def multiply_next(
        value: torch.Tensor, previous: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        row = matrix[len(previous), : len(previous) + 1]
        inputs = (*previous, value)
        output = sum(
            (
                coefficient * noise
                for coefficient, noise in zip(row, inputs, strict=True)
            ),
            start=torch.zeros_like(value),
        )
        return output, inputs

    return StreamingMatrix(init_multiply, multiply_next)


def _empirical_sequence_covariance(
    noising: torch.Tensor | StreamingMatrix,
    *,
    n_steps: int,
) -> torch.Tensor:
    grad = torch.zeros(512, dtype=torch.float64)
    sequences = []
    for seed in range(64):
        noise_fn, state = _matrix_factorization_noise(
            grad,
            noising,
            key=key(seed),
            compute_dtype=torch.float64,
            n_steps=n_steps,
        )
        rows = []
        for _ in range(n_steps):
            noised, state = noise_fn(grad, state, stddev=1.0)
            rows.append(noised)
        sequences.append(torch.stack(rows, dim=1))
    samples = torch.cat(sequences, dim=0)
    centered = samples - samples.mean(dim=0)
    return centered.T @ centered / (len(samples) - 1)


class TestDenseMatrixFactorizationNoise:
    def test_identity_matrix(self):
        """Identity noising matrix = standard Gaussian."""
        noising = torch.eye(5, dtype=torch.float64)
        grad = torch.zeros(10, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noised, state = noise_fn(grad, state, stddev=1.0)
        assert noised.shape == grad.shape

    def test_stepping(self):
        """State advances through matrix rows."""
        noising = torch.eye(3, dtype=torch.float64) * 2.0
        grad = torch.zeros(5, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        assert state._inner_state.item() == 0
        _, state = noise_fn(grad, state, stddev=1.0)
        assert state._inner_state.item() == 1
        _, state = noise_fn(grad, state, stddev=1.0)
        assert state._inner_state.item() == 2

    def test_invalid_ndim(self):
        grad = torch.zeros(10)
        with pytest.raises(ValueError, match="2D"):
            _matrix_factorization_noise(grad, torch.ones(5), key=key(42))

    def test_sequence_covariance_matches_noising_matrix(self):
        noising = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.25, 0.2, 1.0]],
            dtype=torch.float64,
        )

        observed = _empirical_sequence_covariance(noising, n_steps=len(noising))

        torch.testing.assert_close(observed, noising @ noising.T, atol=0.025, rtol=0)


class TestStreamingMatrixFactorizationNoise:
    def test_identity_streaming(self):
        """Identity StreamingMatrix = standard Gaussian."""
        noising = identity()
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noised, state = noise_fn(grad, state, stddev=1.0)
        assert noised.shape == grad.shape

    def test_adds_noise(self):
        """Noise is actually added to gradients."""
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        noised, _ = noise_fn(grad, state, stddev=1.0)
        assert not torch.allclose(noised, grad)

    def test_stateful(self):
        """Successive calls produce different noise."""
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        noisy1, state = noise_fn(grad, state, stddev=1.0)
        noisy2, state = noise_fn(grad, state, stddev=1.0)
        assert not torch.allclose(noisy1, noisy2)

    def test_noise_scale(self):
        """Noise scale should be proportional to stddev."""
        grad = torch.zeros(1000, dtype=torch.float32)

        # Large stddev
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(0))
        noised, _ = noise_fn(grad, state, stddev=100.0)
        large_std = noised.std().item()

        # Small stddev
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(1))
        noised, _ = noise_fn(grad, state, stddev=1.0)
        small_std = noised.std().item()

        assert large_std > small_std * 10

    def test_toeplitz_noising(self):
        """Test with a BandMF-style Toeplitz noising matrix."""
        coefs = optimal_max_error_strategy_coefs(5)
        noising = inverse_as_streaming_matrix(coefs)
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        for _ in range(5):
            noised, state = noise_fn(grad, state, stddev=1.0)
            assert noised.shape == grad.shape

    def test_pytree_grads(self):
        """Test with dict-structured gradients."""
        noising = identity()
        grad = {
            "weight": torch.zeros(5, 3, dtype=torch.float32),
            "bias": torch.zeros(3, dtype=torch.float32),
        }
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noised, state = noise_fn(grad, state, stddev=1.0)
        assert isinstance(noised, dict)
        assert noised["weight"].shape == (5, 3)
        assert noised["bias"].shape == (3,)

    def test_noise_is_correlated(self):
        """With prefix sum noising, noise should accumulate."""
        noising = prefix_sum()
        grad = torch.zeros(50, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))

        variances = []
        for _ in range(20):
            noised, state = noise_fn(grad, state, stddev=1.0)
            variances.append(noised.var().item())

        # With prefix sum, variance should increase
        assert variances[-1] > variances[0]

    def test_returns_mf_noise_state(self):
        """_matrix_factorization_noise returns MFNoiseState."""
        grad = torch.zeros(10)
        _noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        assert isinstance(state, MFNoiseState)

    def test_sequence_covariance_matches_noising_matrix(self):
        noising = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.25, 0.2, 1.0]],
            dtype=torch.float64,
        )

        observed = _empirical_sequence_covariance(
            _dense_as_streaming(noising), n_steps=len(noising)
        )

        torch.testing.assert_close(observed, noising @ noising.T, atol=0.025, rtol=0)

    def test_continuation_reuses_the_same_next_column_draw(self):
        grad = torch.zeros(32, dtype=torch.float64)
        noising = _dense_as_streaming(torch.tril(torch.ones(4, 4, dtype=torch.float64)))
        noise_fn, state = _matrix_factorization_noise(
            grad,
            noising,
            key=key(42),
            compute_dtype=torch.float64,
            n_steps=4,
        )
        for _ in range(2):
            _, state = noise_fn(grad, state, stddev=1.0)

        continued, _ = noise_fn(grad, state, stddev=1.0)
        resumed, _ = noise_fn(grad, state, stddev=1.0)

        torch.testing.assert_close(resumed, continued)
