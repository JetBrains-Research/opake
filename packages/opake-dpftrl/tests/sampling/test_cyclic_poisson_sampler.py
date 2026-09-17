"""Tests for CyclicPoissonSampler with DDP support."""

import pytest
import torch

from opake.api.dpftrl.sampling._partitions import PartitionType
from opake.dpftrl.sampling import CyclicPoissonSampler
from opake.exceptions import ConfigurationError
from opake.random import key


class TestCyclicPoissonSamplerBasic:
    """Basic functionality tests (single device)."""

    def test_standard_poisson(self):
        """bands=1 behaves like standard Poisson sampling."""
        dataset_size = 100
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=1,
            n_steps=10,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 10

        # All indices should be valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

        # In standard Poisson, should get some variation in batch sizes
        batch_sizes = [len(b) for b in batches]
        assert len(set(batch_sizes)) > 1  # Not all same size

    def test_fixed_order(self):
        """sample_rate=1.0 gives deterministic fixed order."""
        dataset_size = 20
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=1.0,
            bands=4,
            n_steps=8,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 8

        # With prob=1, each batch should have all its group's elements
        group_size = dataset_size / 4
        for batch in batches:
            assert len(batch) == int(group_size)

    def test_cyclic_structure(self):
        """Verify cyclic group structure."""
        dataset_size = 15
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=1.0,
            bands=3,
            n_steps=6,
            key=key(42),
        )

        batches = list(sampler)

        # Batches 0 and 3 should have same elements (same group)
        assert set(batches[0]) == set(batches[3])
        # Batches 1 and 4 should have same elements
        assert set(batches[1]) == set(batches[4])
        # Batches 2 and 5 should have same elements
        assert set(batches[2]) == set(batches[5])

    def test_equal_split_partition(self):
        """EQUAL_SPLIT partitions evenly."""
        dataset_size = 100
        bands = 5
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=1.0,
            bands=bands,
            n_steps=bands,
            partition_type=PartitionType.EQUAL_SPLIT,
            key=key(42),
        )

        batches = list(sampler)
        # Each batch should have size dataset_size / bands
        expected_size = dataset_size // bands
        for batch in batches:
            assert len(batch) == expected_size

    def test_independent_partition(self):
        """INDEPENDENT partition works."""
        dataset_size = 100
        bands = 4
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=bands,
            n_steps=8,
            partition_type=PartitionType.INDEPENDENT,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 8

        # Groups may have different sizes with INDEPENDENT
        # Just verify all indices are valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

    def test_reproducibility(self):
        """Same seed produces same batches."""
        dataset_size = 100

        def create_sampler():
            return CyclicPoissonSampler(
                range(dataset_size),
                sample_rate=0.5,
                bands=3,
                n_steps=10,
                key=key(42),
            )

        batches1 = [b.copy() for b in create_sampler()]
        batches2 = [b.copy() for b in create_sampler()]

        assert batches1 == batches2

    def test_different_seeds_different_batches(self):
        """Different seeds produce different batches."""
        dataset_size = 100

        sampler1 = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=3,
            n_steps=10,
            key=key(42),
        )

        sampler2 = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=3,
            n_steps=10,
            key=key(123),
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # With high probability, should get different batches
        assert batches1 != batches2

    def test_key_object(self):
        """Can pass RngKey directly."""
        dataset_size = 100
        k = key(42)

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=2,
            n_steps=5,
            key=k,
        )

        batches = list(sampler)
        assert len(batches) == 5


class TestCyclicPoissonSamplerProperties:
    """Test statistical properties."""

    def test_expected_batch_size(self):
        """expected_batch_size property is reasonable."""
        dataset_size = 1000
        sampling_prob = 0.1
        bands = 5

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=sampling_prob,
            bands=bands,
            n_steps=1,
            key=key(0),
        )

        # Expected: avg_group_size * sampling_prob
        group_size = dataset_size / bands
        expected = group_size * sampling_prob

        assert abs(sampler.expected_batch_size - expected) < 1e-6

    def test_batch_size_variance(self):
        """batch_size_variance property is reasonable."""
        dataset_size = 1000
        sampling_prob = 0.1
        bands = 5

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=sampling_prob,
            bands=bands,
            n_steps=1,
            key=key(0),
        )

        # Variance: avg_group_size * p * (1 - p)
        group_size = dataset_size / bands
        expected_var = group_size * sampling_prob * (1 - sampling_prob)

        assert abs(sampler.batch_size_variance - expected_var) < 1e-6

    def test_len(self):
        """__len__ returns n_steps."""
        dataset_size = 100
        n_steps = 25

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=2,
            n_steps=n_steps,
            key=key(0),
        )

        assert len(sampler) == n_steps


class TestCyclicPoissonSamplerEdgeCases:
    """Test edge cases and error handling."""

    def test_requires_key_parameter(self):
        """`key=` is a required kwarg — guard against silent default-seeding.

        A missing explicit RNG key is a DP correctness hazard (it would
        silently re-seed across ranks / runs), so the constructor must
        raise rather than pick a default.
        """
        with pytest.raises(TypeError, match="key"):
            CyclicPoissonSampler(range(100), sample_rate=0.1, bands=10)

    def test_empty_dataset_raises(self):
        """Empty dataset raises error."""
        with pytest.raises(ValueError, match="data_source must not be empty"):
            CyclicPoissonSampler([], sample_rate=0.5, key=key(0))

    def test_single_example(self):
        """Single example dataset works."""
        sampler = CyclicPoissonSampler(
            [0],
            sample_rate=1.0,
            bands=1,
            n_steps=3,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 3
        # With prob=1.0, should always get [0]
        assert all(batch == [0] for batch in batches)

    def test_bands_larger_than_dataset(self):
        """bands > dataset_size works but gives small groups."""
        dataset_size = 10
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.5,
            bands=20,
            n_steps=20,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 20
        # All indices valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

    def test_sampling_prob_edge_values(self):
        """Test sampling_prob near 0 and 1."""
        dataset_size = 100

        # Near 1
        sampler_high = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.99,
            bands=1,
            n_steps=5,
            key=key(42),
        )
        batches_high = list(sampler_high)
        assert all(len(b) > 0.9 * dataset_size for b in batches_high)

        # Near 0
        sampler_low = CyclicPoissonSampler(
            range(dataset_size),
            sample_rate=0.01,
            bands=1,
            n_steps=20,
            key=key(42),
        )
        batches_low = list(sampler_low)
        assert all(len(b) < 0.1 * dataset_size for b in batches_low)

    def test_invalid_sampling_prob_raises(self):
        """Invalid sampling_prob raises ValueError."""
        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            CyclicPoissonSampler(range(10), sample_rate=0.0, key=key(0))

        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            CyclicPoissonSampler(range(10), sample_rate=1.5, key=key(0))

    def test_invalid_bands_raises(self):
        """Invalid bands raises ValueError."""
        with pytest.raises(ValueError, match="bands must be >= 1"):
            CyclicPoissonSampler(range(10), sample_rate=0.5, bands=0, key=key(0))


class TestCyclicPoissonSamplerDistributedSimulation:
    """Test distributed support via external sharding (inner composition).

    Samplers no longer accept rank/world_size. Instead, the dataset is
    sharded externally using local_shard() (or _local_shard_bounds for
    plain lists), and per-rank keys are derived via fold_in(key, rank).
    """

    def test_external_sharding(self):
        """External sharding produces sampler on a subset."""
        from opake.api.engine.distributed._shard import _local_shard_bounds

        dataset = list(range(100))
        start, end = _local_shard_bounds(len(dataset), rank=0, world_size=4)
        shard = dataset[start:end]

        sampler = CyclicPoissonSampler(
            shard,
            sample_rate=0.5,
            bands=2,
            n_steps=5,
            key=key(0),
        )

        assert sampler.num_examples == 25

    def test_sharding_rank_0(self):
        """Rank 0 gets correct shard via external subsetting."""
        from opake.api.engine.distributed._shard import _local_shard_bounds

        dataset = list(range(100))
        start, end = _local_shard_bounds(len(dataset), rank=0, world_size=4)
        shard = dataset[start:end]

        sampler = CyclicPoissonSampler(
            shard,
            sample_rate=1.0,
            bands=1,
            n_steps=1,
            key=key(42),
        )

        # Shard is [0, 25) so sampler has 25 examples
        assert sampler.num_examples == 25

        batches = list(sampler)
        # Indices are local [0, 25)
        assert all(0 <= idx < 25 for batch in batches for idx in batch)

    def test_sharding_rank_last(self):
        """Last rank gets remainder via external subsetting."""
        from opake.api.engine.distributed._shard import _local_shard_bounds

        dataset = list(range(100))
        start, end = _local_shard_bounds(len(dataset), rank=3, world_size=4)
        shard = dataset[start:end]

        sampler = CyclicPoissonSampler(
            shard,
            sample_rate=1.0,
            bands=1,
            n_steps=1,
            key=key(42),
        )

        # Shard is [75, 100) → 25 local examples
        assert sampler.num_examples == 25

        batches = list(sampler)
        # Indices are local [0, 25)
        assert all(0 <= idx < 25 for batch in batches for idx in batch)

    def test_different_keys_per_rank(self):
        """Different keys per rank produce different sampling."""
        from opake.api.engine.distributed._shard import _local_shard_bounds
        from opake.random import fold_in

        dataset = list(range(100))
        world_size = 4

        # Rank 0
        s0, e0 = _local_shard_bounds(len(dataset), rank=0, world_size=world_size)
        sampler0 = CyclicPoissonSampler(
            dataset[s0:e0],
            sample_rate=0.5,
            bands=1,
            n_steps=5,
            key=fold_in(key(42), 0),
        )
        batches0 = list(sampler0)

        # Rank 1
        s1, e1 = _local_shard_bounds(len(dataset), rank=1, world_size=world_size)
        sampler1 = CyclicPoissonSampler(
            dataset[s1:e1],
            sample_rate=0.5,
            bands=1,
            n_steps=5,
            key=fold_in(key(42), 1),
        )
        batches1 = list(sampler1)

        # Different ranks should get different batches
        assert batches0 != batches1

    def test_cyclic_independence(self):
        """Each rank cycles through its shard independently."""
        from opake.api.engine.distributed._shard import _local_shard_bounds
        from opake.random import fold_in

        dataset = list(range(100))
        start, end = _local_shard_bounds(len(dataset), rank=0, world_size=2)
        shard = dataset[start:end]

        sampler0 = CyclicPoissonSampler(
            shard,
            sample_rate=1.0,
            bands=2,
            n_steps=4,
            key=fold_in(key(42), 0),
        )

        batches0 = list(sampler0)
        # Shard has 50 local examples → groups 0,1,0,1

        # Check all indices are local: [0, 50)
        for batch in batches0:
            assert all(0 <= idx < 50 for idx in batch)


class TestCyclicPoissonSamplerDataLoader:
    """Test integration with PyTorch DataLoader."""

    def test_dataloader_integration(self):
        """Works with PyTorch DataLoader."""
        dataset = list(range(100))
        sampler = CyclicPoissonSampler(
            dataset,
            sample_rate=0.5,
            bands=2,
            n_steps=10,
            key=key(42),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
        )

        batches = list(loader)
        assert len(batches) == 10

        # Each batch should be a tensor of valid indices
        for batch in batches:
            assert isinstance(batch, torch.Tensor)
            assert all(0 <= x < 100 for x in batch)

    def test_dataloader_variable_batch_sizes(self):
        """DataLoader handles variable batch sizes."""
        dataset = list(range(200))
        sampler = CyclicPoissonSampler(
            dataset,
            sample_rate=0.3,
            bands=1,
            n_steps=20,
            key=key(42),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
        )

        batch_sizes = [len(batch) for batch in loader]
        # Poisson property: sizes should vary
        assert len(set(batch_sizes)) > 1


class TestCyclicPoissonSamplerRangeDataset:
    """Test with actual Dataset objects."""

    def test_with_torch_dataset(self):
        """Works with torch.utils.data.Dataset."""

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return idx

        dataset = SimpleDataset(100)
        sampler = CyclicPoissonSampler(
            dataset,
            sample_rate=0.5,
            bands=3,
            n_steps=9,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 9

        # All indices valid
        for batch in batches:
            assert all(0 <= idx < 100 for idx in batch)


class TestCyclicPoissonSamplerTruncated:
    """Tests for CyclicPoissonSampler with ``truncated_batch_size`` set."""

    def test_init_stores_truncated_batch_size(self):
        sampler = CyclicPoissonSampler(
            range(1000),
            sample_rate=0.1,
            bands=1,
            n_steps=5,
            truncated_batch_size=50,
            key=key(0),
        )
        assert sampler.truncated_batch_size == 50

    def test_invalid_truncated_batch_size_raises(self):
        with pytest.raises(ValueError, match="truncated_batch_size"):
            CyclicPoissonSampler(
                range(100), sample_rate=0.1, truncated_batch_size=0, key=key(0)
            )

    def test_cap_enforced(self):
        """``max(batch_sizes) <= truncated_batch_size`` always."""
        sampler = CyclicPoissonSampler(
            range(1000),
            sample_rate=0.5,
            bands=1,
            n_steps=50,
            truncated_batch_size=100,
            key=key(42),
        )

        batch_sizes = [len(b) for b in sampler]
        assert max(batch_sizes) <= 100
        # With p=0.5 over 1000 examples the cap will be hit often.
        assert max(batch_sizes) == 100

    def test_same_as_plain_when_cap_unreachable(self):
        """When the cap is far above the expected draw, the stream is identical."""
        plain = CyclicPoissonSampler(
            range(1000), sample_rate=0.05, bands=1, n_steps=20, key=key(42)
        )
        capped = CyclicPoissonSampler(
            range(1000),
            sample_rate=0.05,
            bands=1,
            n_steps=20,
            truncated_batch_size=10_000,
            key=key(42),
        )
        assert list(plain) == list(capped)

    def test_no_duplicate_indices_after_truncation(self):
        sampler = CyclicPoissonSampler(
            range(1000),
            sample_rate=0.5,
            bands=1,
            n_steps=20,
            truncated_batch_size=100,
            key=key(42),
        )
        for batch in sampler:
            assert len(batch) == len(set(batch))
            assert all(0 <= idx < 1000 for idx in batch)

    def test_cap_applies_with_multiple_bands(self):
        """Sampler caps each step regardless of bands.

        Pairing with privacy accounting (``ftrl_acc.poisson``) for
        ``bands > 1`` is rejected by the accountant, not by the sampler.
        """
        sampler = CyclicPoissonSampler(
            range(1000),
            sample_rate=0.5,
            bands=4,
            n_steps=40,
            truncated_batch_size=20,
            key=key(0),
        )
        for batch in sampler:
            assert len(batch) <= 20
