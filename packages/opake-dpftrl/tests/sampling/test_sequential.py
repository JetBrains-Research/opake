"""Tests for SequentialBatchSampler."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from opake.dpftrl.sampling import SequentialBatchSampler


class TestSequentialBatchSampler:
    """Tests for SequentialBatchSampler."""

    def test_init_basic(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        assert len(sampler) == 10
        assert sampler.expected_batch_size == 10.0

    def test_init_empty_dataset(self):
        dataset = TensorDataset(torch.randn(0, 10))

        with pytest.raises(ValueError, match="data_source must not be empty"):
            SequentialBatchSampler(dataset, batch_size=10)

    def test_init_invalid_batch_size(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            SequentialBatchSampler(dataset, batch_size=0)

    def test_drops_last_incomplete_batch(self):
        dataset = TensorDataset(torch.randn(105, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        assert len(sampler) == 10  # 105 // 10 = 10 (5 dropped)

        batches = list(sampler)
        assert len(batches) == 10
        assert all(len(b) == 10 for b in batches)

    def test_exact_fit(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        batches = list(sampler)
        assert len(batches) == 10

    def test_sequential_contiguous_indices(self):
        dataset = TensorDataset(torch.randn(30, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        batches = list(sampler)
        assert batches[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert batches[1] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert batches[2] == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    def test_all_indices_covered_once(self):
        n = 100
        dataset = TensorDataset(torch.randn(n, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        all_indices = [idx for batch in sampler for idx in batch]
        assert sorted(all_indices) == list(range(n))

    def test_deterministic_across_iterations(self):
        dataset = TensorDataset(torch.randn(50, 10))
        s1 = SequentialBatchSampler(dataset, batch_size=10)
        s2 = SequentialBatchSampler(dataset, batch_size=10)

        # The sequence is deterministic given inputs — two fresh samplers
        # over the same dataset emit identical batches.  Samplers are
        # single-pass (re-iterating an exhausted sampler yields nothing),
        # so the parity check uses two fresh instances.
        assert list(s1) == list(s2)

    def test_batch_size_one(self):
        dataset = TensorDataset(torch.randn(5, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=1)

        batches = list(sampler)
        assert len(batches) == 5
        assert batches == [[0], [1], [2], [3], [4]]

    def test_batch_size_equals_dataset(self):
        dataset = TensorDataset(torch.randn(10, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == list(range(10))

    def test_batch_size_larger_than_dataset(self):
        dataset = TensorDataset(torch.randn(5, 10))
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        assert len(sampler) == 0
        assert list(sampler) == []

    def test_dataloader_integration(self):
        data = torch.arange(50).unsqueeze(1).float()
        dataset = TensorDataset(data)
        sampler = SequentialBatchSampler(dataset, batch_size=10)

        loader = DataLoader(dataset, batch_sampler=sampler)
        batches = [(t,) for (t,) in loader]

        assert len(batches) == 5
        # First batch should be [0..9]
        assert torch.equal(batches[0][0], torch.arange(10).unsqueeze(1).float())
