"""Tests for key-based RNG in samplers."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from opake.api.dpsgd.sampling._poisson import POISSON_STREAM_FOLD
from opake.dpsgd.sampling import PoissonSampler
from opake.random import fold_in, key


class TestPoissonSamplerKeys:
    """Test PoissonSampler with key-based RNG."""

    def test_requires_key_parameter(self):
        """Should require key parameter (no None fallback)."""
        dataset = TensorDataset(torch.randn(1000, 10))

        with pytest.raises(TypeError, match="key"):
            PoissonSampler(dataset, sample_rate=0.1, n_steps=5)

    def test_reproducibility_with_same_key(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches2 = list(sampler2)

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1 == b2

    def test_different_keys_produce_different_samples(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(43))
        batches2 = list(sampler2)

        assert batches1 != batches2

    def test_rank_shifting_via_fold_in(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        base_key = key(42)

        rank0_key = base_key
        sampler_rank0 = PoissonSampler(
            dataset, sample_rate=0.1, n_steps=5, key=rank0_key
        )
        batches_rank0 = list(sampler_rank0)

        rank1_key = fold_in(base_key, 1)
        sampler_rank1 = PoissonSampler(
            dataset, sample_rate=0.1, n_steps=5, key=rank1_key
        )
        batches_rank1 = list(sampler_rank1)

        assert batches_rank0 != batches_rank1

    def test_fold_in_helper_integration(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        k = fold_in(key(42), 0)
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=k)

        batches = list(sampler)
        assert len(batches) > 0
        assert all(isinstance(b, list) for b in batches)

    def test_sampling_with_key(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler = PoissonSampler(
            dataset,
            sample_rate=0.1,
            n_steps=5,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) > 0


class TestPoissonSamplerTruncatedKeys:
    """Test PoissonSampler with truncation and key-based RNG."""

    def test_requires_key_parameter(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        with pytest.raises(TypeError, match="key"):
            PoissonSampler(dataset, sample_rate=0.1, truncated_batch_size=50, n_steps=5)

    def test_reproducibility_with_same_key(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(
            dataset,
            sample_rate=0.1,
            truncated_batch_size=50,
            n_steps=5,
            key=key(42),
        )
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(
            dataset,
            sample_rate=0.1,
            truncated_batch_size=50,
            n_steps=5,
            key=key(42),
        )
        batches2 = list(sampler2)

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1 == b2

    def test_truncation_respects_max_batch_size(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=50,
            n_steps=10,
            key=key(42),
        )

        for batch in sampler:
            assert len(batch) <= 50


class TestCrossValidationWithNumpy:
    """Test that key-based sampling matches numpy.random.Generator behavior."""

    def test_poisson_matches_numpy_generator(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=1, key=key(42))

        stream_key = fold_in(key(42), POISSON_STREAM_FOLD)
        rng = np.random.default_rng(stream_key.seed)
        expected = np.flatnonzero(rng.random(len(dataset)) < 0.1).tolist()

        assert list(sampler) == [expected]
