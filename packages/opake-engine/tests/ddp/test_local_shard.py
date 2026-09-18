"""``local_shard`` and shard bounds (engine distributed data path)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from opake.api.engine.distributed._shard import _local_shard_bounds
from opake.distributed import local_shard


class TestLocalShard:
    """Tests for the local_shard utility."""

    def test_single_device(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        shard = local_shard(dataset)
        assert len(shard) == 100

    def test_even_split(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        for rank in range(4):
            shard = local_shard(dataset, rank=rank, world_size=4)
            assert len(shard) == 25

    def test_uneven_split_last_rank_gets_remainder(self) -> None:
        dataset = TensorDataset(torch.randn(103, 10))
        sizes = [len(local_shard(dataset, rank=r, world_size=4)) for r in range(4)]
        assert sizes == [25, 25, 25, 28]

    def test_returns_subset(self) -> None:
        from torch.utils.data import Subset

        dataset = TensorDataset(torch.randn(100, 10))
        shard = local_shard(dataset, rank=0, world_size=4)
        assert isinstance(shard, Subset)

    def test_invalid_rank_raises(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        with pytest.raises(ValueError, match="rank must be in"):
            local_shard(dataset, rank=4, world_size=4)

    def test_invalid_world_size_raises(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            local_shard(dataset, rank=0, world_size=0)

    def test_negative_dataset_size_raises(self) -> None:
        with pytest.raises(ValueError, match="dataset_size must be >= 0"):
            _local_shard_bounds(-1, rank=0, world_size=1)


class TestLocalShardWorldSizeOne:
    def test_world_size_one_same_as_no_sharding(self) -> None:
        dataset = TensorDataset(torch.randn(100, 10))
        shard = local_shard(dataset, rank=0, world_size=1)
        assert len(shard) == len(dataset)
