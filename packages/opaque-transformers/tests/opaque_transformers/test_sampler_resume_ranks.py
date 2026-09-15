"""Rank-local sampler resume under DDP (T25).

Single-process emulation of a two-rank DDP run: each "rank" builds its
sampler exactly as ``DPTrainer.get_train_dataloader`` does (trimmed
dataset, ``local_shard``, key ``fold_in(key(seed), rank)``), the snapshot
is written by rank 0 only, and every rank restores it through
``_rank_local_sampler_state`` before ``from_state_dict``.  After the
restore, rank 0 must be bit-identical to a continuous rank-0 run (and to a
plain ``from_state_dict`` restore), rank 1 must equal its own continuous
run, and the two ranks' inclusion masks must differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from torch.utils.data import Subset

from opaque.api.transformers.trainer._dp_trainer import _rank_local_sampler_state
from opaque.distributed import local_shard
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.dpsgd.sampling import KOutOfTSampler, PoissonSampler
from opaque.exceptions import CheckpointError
from opaque.random import fold_in, key
from opaque.serialization import from_state_dict, state_dict

if TYPE_CHECKING:
    from collections.abc import Callable

N = 400
WORLD = 2
SEED = 1234
Q = 0.05
BANDS = 4
N_STEPS = 40
STEPS_BEFORE = 7
STEPS_AFTER = 20


def _shard(rank: int):
    dataset: object = list(range(N))
    trimmed = (N // WORLD) * WORLD
    if trimmed < N:
        dataset = Subset(dataset, range(trimmed))
    return local_shard(dataset, rank=rank, world_size=WORLD)


def _rank_key(rank: int):
    # ``DPTrainer.get_train_dataloader``: ``fold_in(sampler_key, self._ddp.rank)``.
    return fold_in(key(SEED), rank)


def _poisson(rank: int):
    return PoissonSampler(
        _shard(rank), sample_rate=Q, n_steps=N_STEPS, key=_rank_key(rank)
    )


def _b_min_sep(rank: int):
    p = Q / (1 - Q * (BANDS - 1))
    return BMinSepSampler(
        _shard(rank), bands=BANDS, sampling_prob=p, n_steps=N_STEPS, key=_rank_key(rank)
    )


def _balls_in_bins(rank: int):
    return BallsInBinsSampler(
        _shard(rank), num_bins=8, n_steps=N_STEPS, key=_rank_key(rank)
    )


def _cyclic_poisson(rank: int):
    return CyclicPoissonSampler(
        _shard(rank),
        sample_rate=Q * BANDS,
        bands=BANDS,
        n_steps=N_STEPS,
        key=_rank_key(rank),
    )


def _k_out_of_t(rank: int):
    return KOutOfTSampler(
        _shard(rank), k=4, t=N_STEPS, allocation="block", key=_rank_key(rank)
    )


FACTORIES: dict[str, Callable[[int], object]] = {
    "poisson": _poisson,
    "b_min_sep": _b_min_sep,
    "balls_in_bins": _balls_in_bins,
    "cyclic_poisson": _cyclic_poisson,
    "k_out_of_t": _k_out_of_t,
}


def _masks(sampler, n: int) -> np.ndarray:
    """Inclusion masks of the next ``n`` batches, one row per step."""
    size = len(sampler.data_source)
    it = iter(sampler)
    rows = []
    for _ in range(n):
        row = np.zeros(size, dtype=bool)
        row[np.asarray(next(it), dtype=int)] = True
        rows.append(row)
    return np.stack(rows)


def _restore(factory, rank: int, snapshot):
    template = factory(rank)
    return from_state_dict(
        template, _rank_local_sampler_state(snapshot, template, WORLD)
    )


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_each_rank_resumes_its_own_stream(name):
    factory = FACTORIES[name]
    rank0, rank1 = factory(0), factory(1)
    _masks(rank0, STEPS_BEFORE)
    _masks(rank1, STEPS_BEFORE)
    snapshot = state_dict(rank0)  # written once, by world rank 0
    assert snapshot["consumed"] == STEPS_BEFORE

    restored0 = _restore(factory, 0, snapshot)
    restored1 = _restore(factory, 1, snapshot)
    assert restored0._stream_key == factory(0)._stream_key
    assert restored1._stream_key == factory(1)._stream_key
    assert restored0.consumed == STEPS_BEFORE
    assert restored1.consumed == STEPS_BEFORE

    continuous0 = _masks(factory(0), STEPS_BEFORE + STEPS_AFTER)[STEPS_BEFORE:]
    continuous1 = _masks(factory(1), STEPS_BEFORE + STEPS_AFTER)[STEPS_BEFORE:]
    after0 = _masks(restored0, STEPS_AFTER)
    after1 = _masks(restored1, STEPS_AFTER)

    assert np.array_equal(after0, continuous0)
    assert np.array_equal(after1, continuous1)
    assert not np.array_equal(after0, after1)
    assert int(np.all(after0 == after1, axis=1).sum()) < STEPS_AFTER


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_rank_zero_restore_is_bit_identical_to_plain_restore(name):
    factory = FACTORIES[name]
    rank0 = factory(0)
    _masks(rank0, STEPS_BEFORE)
    snapshot = state_dict(rank0)

    plain = from_state_dict(factory(0), snapshot)
    rank_local = _restore(factory, 0, snapshot)
    assert state_dict(rank_local) == state_dict(plain)
    assert np.array_equal(_masks(rank_local, STEPS_AFTER), _masks(plain, STEPS_AFTER))


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_plain_restore_on_rank_one_replays_rank_zero_coins(name):
    """The defect the rank-local restore removes: a verbatim snapshot puts
    rank 0's stream on rank 1, co-including records that share a local shard
    index."""
    factory = FACTORIES[name]
    rank0 = factory(0)
    _masks(rank0, STEPS_BEFORE)
    snapshot = state_dict(rank0)

    plain0 = _masks(from_state_dict(factory(0), snapshot), STEPS_AFTER)
    plain1 = _masks(from_state_dict(factory(1), snapshot), STEPS_AFTER)
    assert np.array_equal(plain1, plain0)

    fixed1 = _masks(_restore(factory, 1, snapshot), STEPS_AFTER)
    assert not np.array_equal(fixed1, plain0)


def test_single_process_snapshot_passes_through_unchanged():
    sampler = _poisson(0)
    _masks(sampler, STEPS_BEFORE)
    snapshot = state_dict(sampler)
    assert (
        _rank_local_sampler_state(
            snapshot,
            PoissonSampler(_shard(0), sample_rate=Q, n_steps=N_STEPS, key=key(999)),
            1,
        )
        is snapshot
    )


def test_deterministic_sampler_snapshot_passes_through_unchanged():
    sampler = SequentialBatchSampler(_shard(0), batch_size=10, n_steps=N_STEPS)
    _masks(sampler, STEPS_BEFORE)
    snapshot = state_dict(sampler)
    assert "key_seed" not in snapshot
    assert _rank_local_sampler_state(snapshot, sampler, WORLD) is snapshot


def test_rank_local_state_replaces_only_the_stream_key():
    rank0, rank1 = _poisson(0), _poisson(1)
    _masks(rank0, STEPS_BEFORE)
    snapshot = state_dict(rank0)
    local = _rank_local_sampler_state(snapshot, rank1, WORLD)
    assert local is not snapshot
    assert local["key_seed"] == rank1._stream_key.seed
    assert local["key_impl"] == rank1._stream_key.impl
    assert {k: v for k, v in local.items() if not k.startswith("key_")} == {
        k: v for k, v in snapshot.items() if not k.startswith("key_")
    }
    assert snapshot["key_seed"] == rank0._stream_key.seed


def test_template_without_stream_key_raises():
    rank0 = _poisson(0)
    snapshot = state_dict(rank0)
    with pytest.raises(CheckpointError):
        _rank_local_sampler_state(snapshot, object(), WORLD)
