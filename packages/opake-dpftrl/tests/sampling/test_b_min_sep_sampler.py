"""Tests for BMinSepSampler."""

import pytest
import torch
from torch.utils.data import TensorDataset

from opake.dpftrl.sampling import BMinSepSampler
from opake.random import fold_in, key


def test_reproducibility():
    ds = TensorDataset(torch.randn(200, 3))
    s1 = BMinSepSampler(ds, bands=4, sampling_prob=0.08, n_steps=20, key=key(7))
    s2 = BMinSepSampler(ds, bands=4, sampling_prob=0.08, n_steps=20, key=key(7))
    assert list(s1) == list(s2)


def test_different_keys():
    ds = TensorDataset(torch.randn(200, 3))
    a = list(BMinSepSampler(ds, bands=4, sampling_prob=0.1, n_steps=30, key=key(1)))
    b = list(BMinSepSampler(ds, bands=4, sampling_prob=0.1, n_steps=30, key=key(2)))
    assert a != b


def test_same_index_respects_min_separation():
    """One example must not appear in two batches fewer than ``bands`` apart."""
    ds = TensorDataset(torch.randn(300, 2))
    bands = 4
    batches = list(
        BMinSepSampler(ds, bands=bands, sampling_prob=0.35, n_steps=120, key=key(101))
    )
    last_seen: dict[int, int] = {}
    for t, batch in enumerate(batches):
        for idx in batch:
            if idx in last_seen:
                assert t - last_seen[idx] >= bands, (idx, last_seen[idx], t)
            last_seen[idx] = t


def test_fold_in_changes_stream():
    ds = TensorDataset(torch.randn(100, 2))
    a = list(BMinSepSampler(ds, bands=2, sampling_prob=0.2, n_steps=15, key=key(0)))
    b = list(
        BMinSepSampler(
            ds, bands=2, sampling_prob=0.2, n_steps=15, key=fold_in(key(0), 1)
        )
    )
    assert a != b


def test_invalid_sampling_prob():
    ds = TensorDataset(torch.randn(10, 1))
    with pytest.raises(ValueError, match="sampling_prob"):
        BMinSepSampler(ds, bands=2, sampling_prob=0.0, n_steps=1, key=key(0))
