"""PoissonSampler with external sharding via ``local_shard`` (DP-SGD sampling)."""

from __future__ import annotations

import scipy.stats
import torch
from torch.utils.data import TensorDataset

from opake.distributed import local_shard
from opake.dpsgd.sampling import PoissonSampler
from opake.random import fold_in, key

# Tail mass a statistical assertion may spend, split across shards.
_FALSE_FAILURE_PROB = 1e-9


class TestSingleDeviceMode:
    """Single-device (default) sampling — no sharding."""

    def test_single_device_full_dataset(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=1, key=key(0))
        batches = list(sampler)
        assert len(batches) == 1

    def test_different_keys_produce_different_batches(self) -> None:
        dataset = TensorDataset(torch.randn(1000, 10))
        s0 = PoissonSampler(dataset, sample_rate=0.1, n_steps=10, key=key(42))
        s1 = PoissonSampler(dataset, sample_rate=0.1, n_steps=10, key=key(43))
        assert list(s0) != list(s1)

    def test_same_key_reproduces_batches(self) -> None:
        dataset = TensorDataset(torch.randn(500, 10))
        s1 = PoissonSampler(dataset, sample_rate=0.2, n_steps=5, key=key(12345))
        s2 = PoissonSampler(dataset, sample_rate=0.2, n_steps=5, key=key(12345))
        assert list(s1) == list(s2)


class TestShardedSampling:
    """Sharded sampling via ``local_shard`` + ``fold_in``."""

    def test_shards_correct_sizes(self) -> None:
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        for rank in range(world_size):
            shard = local_shard(dataset, rank=rank, world_size=world_size)
            assert len(shard) == 250

    def test_uneven_split_last_rank_larger(self) -> None:
        dataset = TensorDataset(torch.randn(103, 10))
        world_size = 4

        shard_last = local_shard(dataset, rank=3, world_size=world_size)
        expected = 103 - 3 * (103 // 4)
        assert len(shard_last) == expected

    def test_shards_produce_valid_local_indices(self) -> None:
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        for rank in range(world_size):
            shard = local_shard(dataset, rank=rank, world_size=world_size)
            sampler = PoissonSampler(
                shard, sample_rate=0.5, n_steps=1, key=fold_in(key(42), rank)
            )
            batch = next(iter(sampler))
            shard_size = len(shard)
            assert all(0 <= idx < shard_size for idx in batch)

    def test_statistical_properties_per_shard(self) -> None:
        """Each shard samples at the declared rate, within an exact band."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.1
        world_size = 4
        n_steps = 2000

        for rank in range(world_size):
            shard = local_shard(dataset, rank=rank, world_size=world_size)
            sampler = PoissonSampler(
                shard,
                sample_rate=sample_rate,
                n_steps=n_steps,
                key=fold_in(key(100), rank),
            )
            included = sum(len(b) for b in sampler)
            low, high = scipy.stats.binom.interval(
                1 - _FALSE_FAILURE_PROB / world_size,
                len(shard) * n_steps,
                sample_rate,
            )
            assert low <= included <= high

    def test_different_ranks_independent(self) -> None:
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        shard0 = local_shard(dataset, rank=0, world_size=world_size)
        shard1 = local_shard(dataset, rank=1, world_size=world_size)

        sizes0 = [
            len(b)
            for b in PoissonSampler(
                shard0, sample_rate=0.5, n_steps=20, key=fold_in(key(42), 0)
            )
        ]
        sizes1 = [
            len(b)
            for b in PoissonSampler(
                shard1, sample_rate=0.5, n_steps=20, key=fold_in(key(42), 1)
            )
        ]
        assert sizes0 != sizes1, "Different ranks should sample independently"


class TestPoissonTruncatedDistributed:
    """PoissonSampler truncation under external sharding."""

    def test_truncated_sharded_respects_max_batch_size(self) -> None:
        dataset = TensorDataset(torch.randn(1000, 10))
        max_batch_size = 50
        world_size = 4

        for rank in range(world_size):
            shard = local_shard(dataset, rank=rank, world_size=world_size)
            sampler = PoissonSampler(
                shard,
                sample_rate=0.5,
                truncated_batch_size=max_batch_size,
                n_steps=1,
                key=fold_in(key(100), rank),
            )
            batch = next(iter(sampler))
            assert len(batch) <= max_batch_size

    def test_truncated_single_device(self) -> None:
        dataset = TensorDataset(torch.randn(500, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.2,
            truncated_batch_size=60,
            n_steps=5,
            key=key(999),
        )
        batches = list(sampler)
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) <= 60


class TestEdgeCases:
    """Edge cases for distributed sampling with Poisson."""

    def test_very_small_shards(self) -> None:
        dataset = TensorDataset(torch.randn(50, 10))
        world_size = 10

        for rank in range(world_size):
            shard = local_shard(dataset, rank=rank, world_size=world_size)
            sampler = PoissonSampler(
                shard,
                sample_rate=0.5,
                n_steps=10,
                key=fold_in(key(0), rank),
            )
            batches = list(sampler)
            assert len(batches) == 10
