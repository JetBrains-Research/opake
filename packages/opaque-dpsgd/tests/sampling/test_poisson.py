"""Tests for the unified PoissonSampler (plain + truncated).

Statistical assertions use exact binomial confidence bands rather than
hand-picked tolerances, so they hold for an arbitrary RNG stream instead of
one lucky seed; seeds are fixed only for reproducible failures.
"""

import numpy as np
import pytest
import scipy.stats
import torch
from torch.utils.data import DataLoader, TensorDataset

from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key

# Total tail mass a statistical assertion is allowed to spend.  Per-record
# assertions split it across records (Bonferroni), so the whole file stays
# below this false-failure probability per run.
_FALSE_FAILURE_PROB = 1e-9


def _inclusion_counts(sampler: PoissonSampler, num_records: int) -> np.ndarray:
    """Count how many of the sampler's batches include each record."""
    counts = np.zeros(num_records, dtype=np.int64)
    for batch in sampler:
        counts += np.bincount(np.asarray(batch, dtype=np.intp), minlength=num_records)
    return counts


class TestPoissonSampler:
    """Tests for plain PoissonSampler."""

    def test_init_basic(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(0))

        assert sampler.sample_rate == 0.1
        assert sampler.n_steps == 5
        assert sampler.truncated_batch_size is None
        assert len(sampler) == 5

    def test_init_invalid_sample_rate(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSampler(dataset, sample_rate=0.0, key=key(0))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSampler(dataset, sample_rate=1.5, key=key(0))

    def test_init_invalid_n_steps(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="n_steps must be"):
            PoissonSampler(dataset, sample_rate=0.1, n_steps=0, key=key(0))

    def test_iteration_produces_variable_batches(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=10, key=key(42))

        batch_sizes = [len(batch) for batch in sampler]

        assert len(set(batch_sizes)) > 1, "Batch sizes should vary"

    @pytest.mark.parametrize(
        ("num_records", "sample_rate", "n_steps", "max_rel_width"),
        [
            (1000, 0.1, 4000, 0.01),
            (1000, 0.5, 1000, 0.01),
            # A 1% band at q=0.001 needs ~4e8 trials; 5% is what is practical.
            (10000, 0.001, 2000, 0.05),
        ],
    )
    def test_pooled_inclusion_rate(
        self, num_records, sample_rate, n_steps, max_rel_width
    ):
        """Pooled inclusions land in a Binomial(records * steps, q) band.

        Every (record, step) pair is an independent Bernoulli(q) trial, so the
        pooled count is exactly binomial and the band is exact.
        """
        dataset = TensorDataset(torch.randn(num_records, 10))
        sampler = PoissonSampler(
            dataset, sample_rate=sample_rate, n_steps=n_steps, key=key(42)
        )

        trials = num_records * n_steps
        expected = trials * sample_rate
        low, high = scipy.stats.binom.interval(
            1 - _FALSE_FAILURE_PROB, trials, sample_rate
        )
        half_width = max(high - expected, expected - low)
        assert half_width / expected <= max_rel_width

        included = sum(len(batch) for batch in sampler)
        assert low <= included <= high

    def test_per_record_inclusion_rate(self):
        """No individual record is systematically over- or under-sampled.

        The pooled band above cannot see this: a sampler that never draws one
        record out of 1000 shifts the pooled rate by only 0.1%.
        """
        num_records, sample_rate, n_steps = 1000, 0.1, 4000
        dataset = TensorDataset(torch.randn(num_records, 10))
        sampler = PoissonSampler(
            dataset, sample_rate=sample_rate, n_steps=n_steps, key=key(7)
        )

        counts = _inclusion_counts(sampler, num_records)
        low, high = scipy.stats.binom.interval(
            1 - _FALSE_FAILURE_PROB / num_records, n_steps, sample_rate
        )
        assert low <= counts.min()
        assert counts.max() <= high

    def test_statistical_properties_variance(self):
        """Batch-size spread matches Binomial(records, q), not a fixed size."""
        num_records, sample_rate, n_steps = 1000, 0.1, 2000
        dataset = TensorDataset(torch.randn(num_records, 10))
        sampler = PoissonSampler(
            dataset, sample_rate=sample_rate, n_steps=n_steps, key=key(42)
        )

        batch_sizes = [len(batch) for batch in sampler]

        # Sampling error of a variance over n iid draws is
        # sqrt((mu4 - var^2) / n), asymptotically normal.
        binomial = scipy.stats.binom(num_records, sample_rate)
        variance = float(binomial.var())
        fourth_moment = (float(binomial.stats(moments="k")) + 3) * variance**2
        sigma = np.sqrt((fourth_moment - variance**2) / n_steps)
        z = scipy.stats.norm.isf(_FALSE_FAILURE_PROB / 2)

        assert abs(np.var(batch_sizes) - variance) <= z * sigma

    def test_no_duplicate_indices_within_batch(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.5, n_steps=20, key=key(42))

        for batch_indices in sampler:
            assert len(batch_indices) == len(set(batch_indices))
            assert all(0 <= idx < 100 for idx in batch_indices)

    def test_indices_in_valid_range(self):
        dataset = TensorDataset(torch.randn(500, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=10, key=key(42))

        for batch_indices in sampler:
            assert all(0 <= idx < 500 for idx in batch_indices)

    def test_expected_batch_size_property(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, key=key(0))

        assert sampler.expected_batch_size == 100.0

    def test_batch_size_variance_property(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, key=key(0))

        expected_var = 1000 * 0.1 * 0.9
        assert sampler.batch_size_variance == expected_var

    def test_reproducibility_with_generator(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches2 = list(sampler2)

        for b1, b2 in zip(batches1, batches2, strict=True):
            assert b1 == b2

    def test_integration_with_dataloader(self):
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))

        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_sizes = []
        for X, y in loader:
            assert X.shape[1] == 10
            assert y.shape[1] == 5
            batch_sizes.append(X.shape[0])

        assert len(batch_sizes) == 5
        assert len(set(batch_sizes)) > 1


class TestPoissonSamplerTruncated:
    """Tests for PoissonSampler with truncated_batch_size set."""

    def test_init_basic(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.1,
            n_steps=5,
            truncated_batch_size=50,
            key=key(0),
        )

        assert sampler.sample_rate == 0.1
        assert sampler.truncated_batch_size == 50
        assert sampler.n_steps == 5

    def test_init_invalid_truncated_batch_size(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="truncated_batch_size must be"):
            PoissonSampler(dataset, sample_rate=0.1, truncated_batch_size=0, key=key(0))

    @pytest.mark.parametrize("cap", [1.5, True])
    def test_init_requires_integral_truncated_batch_size(self, cap):
        dataset = TensorDataset(torch.randn(100, 10))
        with pytest.raises(TypeError, match="truncated_batch_size must be an int"):
            PoissonSampler(
                dataset, sample_rate=0.1, truncated_batch_size=cap, key=key(0)
            )

    def test_truncation_enforced(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=50,
            key=key(42),
        )

        # A Binomial(1000, 0.5) draw below 100 has probability 7e-163, so
        # every step truncates to exactly the cap.
        assert all(len(batch) == 100 for batch in sampler)

    def test_truncation_selects_records_uniformly(self):
        """Truncation subsamples uniformly instead of favouring low indices."""
        num_records, truncated, n_steps = 200, 50, 2000
        dataset = TensorDataset(torch.randn(num_records, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=1.0,
            truncated_batch_size=truncated,
            n_steps=n_steps,
            key=key(3),
        )

        # sample_rate=1 makes each step a uniform draw of ``truncated`` out of
        # ``num_records``, so each record's count is Binomial(steps, B / N).
        counts = _inclusion_counts(sampler, num_records)
        low, high = scipy.stats.binom.interval(
            1 - _FALSE_FAILURE_PROB / num_records,
            n_steps,
            truncated / num_records,
        )
        assert low <= counts.min()
        assert counts.max() <= high

    def test_no_truncation_when_cap_equals_dataset_size(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=1.0,
            truncated_batch_size=100,
            n_steps=5,
            key=key(42),
        )

        for batch in sampler:
            assert batch == list(range(100))

    def test_same_as_plain_when_no_truncation(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler_truncated = PoissonSampler(
            dataset,
            sample_rate=0.05,
            truncated_batch_size=10000,
            n_steps=20,
            key=key(42),
        )

        sampler_regular = PoissonSampler(
            dataset,
            sample_rate=0.05,
            n_steps=20,
            key=key(42),
        )

        batches_truncated = [len(b) for b in sampler_truncated]
        batches_regular = [len(b) for b in sampler_regular]

        assert batches_truncated == batches_regular

    def test_no_duplicate_indices_after_truncation(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=20,
            key=key(42),
        )

        for batch_indices in sampler:
            assert len(batch_indices) == len(set(batch_indices))
            assert all(0 <= idx < 1000 for idx in batch_indices)

    def test_integration_with_dataloader(self):
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=5,
            key=key(42),
        )

        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_sizes = []
        for X, y in loader:
            batch_sizes.append(X.shape[0])
            assert X.shape[0] <= 100
            assert y.shape[0] <= 100

        assert len(batch_sizes) == 5


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_sample_rate_one_yields_every_record(self, seed):
        """``sample_rate=1`` is exhaustive for any stream, not just on average.

        ``rng.random()`` draws from [0, 1), so ``< 1.0`` holds for every draw.
        """
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=1.0, n_steps=5, key=key(seed))

        for batch in sampler:
            assert batch == list(range(100))

    def test_truncated_with_max_batch_size_one(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=1,
            n_steps=20,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        assert all(size <= 1 for size in batch_sizes)
