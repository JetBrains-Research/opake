"""``state_dict`` / ``from_state_dict`` round-trip for DP-FTRL samplers.

Covers all four dp-ftrl samplers:

* :class:`CyclicPoissonSampler` — partition deterministic from key;
  generator replay matches a continuous run.
* :class:`BMinSepSampler` — Markov state (cooldown array + recent deque)
  must be reconstructed by replay; round-trip preserves the b-min-sep
  invariant.
* :class:`BallsInBinsSampler` — bin assignment deterministic; cursor
  restoration alone fixes resume.
* :class:`SequentialBatchSampler` — pure deterministic; cursor only.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from opake.api.dpftrl.sampling._b_min_sep import B_MIN_SEP_STREAM_FOLD
from opake.api.dpftrl.sampling._balls_in_bins import BALLS_IN_BINS_STREAM_FOLD
from opake.api.dpftrl.sampling._poisson import CYCLIC_POISSON_STREAM_FOLD
from opake.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opake.random import fold_in, key
from opake.serialization import from_state_dict, state_dict


def _ds(n: int = 200) -> TensorDataset:
    return TensorDataset(torch.arange(n).reshape(-1, 1))


class TestCyclicPoissonRoundTrip:
    def test_snapshot_resumes_with_matching_remainder(self):
        original = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(42)
        )
        original_batches = list(original)

        fresh = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(42)
        )
        it = iter(fresh)
        K = 9
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        # Template with different key proves the snapshot owns RNG state.
        template = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = CyclicPoissonSampler(
            _ds(),
            sample_rate=0.15,
            bands=5,
            n_steps=25,
            truncated_batch_size=8,
            key=key(7),
        )
        snapshot = state_dict(sampler)
        template = CyclicPoissonSampler(
            _ds(), sample_rate=0.5, bands=2, n_steps=10, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        # Sampling parameters and the domain-separated stream key come from the
        # snapshot.
        assert restored.sample_rate == 0.15
        assert restored.bands == 5
        assert restored.truncated_batch_size == 8
        # ``n_steps`` follows the template (allows resume + extend).
        assert restored.n_steps == 10

    def test_pre_domain_snapshot_uses_saved_seed_without_folding(self):
        snapshot = {
            "key_seed": 17,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_examples": 12,
            "sample_rate": 0.6,
            "bands": 3,
            "n_steps": 6,
            "partition_type": "EQUAL_SPLIT",
            "truncated_batch_size": None,
        }
        template = CyclicPoissonSampler(
            _ds(12), sample_rate=0.6, bands=3, n_steps=6, key=key(0)
        )

        restored = from_state_dict(template, snapshot)
        reference = CyclicPoissonSampler._from_stream_key(
            _ds(12),
            sample_rate=0.6,
            bands=3,
            n_steps=6,
            stream_key=key(17),
        )
        iterator = iter(reference)
        for _ in range(2):
            next(iterator)

        assert list(restored) == list(reference)
        assert state_dict(restored)["key_seed"] == 17
        assert state_dict(restored)["key_impl"] == "opake_threefry_like"

    def test_state_stores_domain_separated_stream_seed(self):
        sampler = CyclicPoissonSampler(
            _ds(12), sample_rate=0.6, bands=3, n_steps=6, key=key(17)
        )
        iterator = iter(sampler)
        for _ in range(2):
            next(iterator)

        snapshot = state_dict(sampler)
        expected_seed = fold_in(key(17), CYCLIC_POISSON_STREAM_FOLD).seed
        expected_tail = list(sampler)
        template = CyclicPoissonSampler(
            _ds(12), sample_rate=0.6, bands=3, n_steps=6, key=key(0)
        )

        assert snapshot == {
            "key_seed": expected_seed,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_examples": 12,
            "sample_rate": 0.6,
            "bands": 3,
            "n_steps": 6,
            "partition_type": "EQUAL_SPLIT",
            "truncated_batch_size": None,
        }
        assert list(from_state_dict(template, snapshot)) == expected_tail

        reader = from_state_dict(template, snapshot)
        assert next(iter(reader)) == expected_tail[0]
        resaved = state_dict(reader)
        assert resaved["key_seed"] == expected_seed
        assert resaved["key_impl"] == snapshot["key_impl"]
        assert list(from_state_dict(template, resaved)) == expected_tail[1:]


class TestBMinSepRoundTrip:
    def test_snapshot_replays_markov_state(self):
        """The cooldown array + recent deque must reconstruct under replay."""
        original = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(13)
        )
        original_batches = list(original)

        fresh = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(13)
        )
        it = iter(fresh)
        K = 5
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        # The Markov state (cooldown + recent deque) must match what a
        # continuous run would have at step K — verified by the fact
        # that the remaining batches match exactly.
        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_pre_domain_snapshot_uses_saved_seed_without_folding(self):
        snapshot = {
            "key_seed": 17,
            "key_impl": "opake_threefry_like",
            "consumed": 3,
            "num_examples": 12,
            "bands": 3,
            "sampling_prob": 0.5,
            "n_steps": 7,
        }
        template = BMinSepSampler(
            _ds(12), bands=3, sampling_prob=0.5, n_steps=7, key=key(0)
        )

        restored = from_state_dict(template, snapshot)
        reference = BMinSepSampler._from_stream_key(
            _ds(12),
            bands=3,
            sampling_prob=0.5,
            n_steps=7,
            stream_key=key(17),
        )
        iterator = iter(reference)
        for _ in range(3):
            next(iterator)

        assert list(restored) == list(reference)
        assert state_dict(restored)["key_seed"] == 17
        assert state_dict(restored)["key_impl"] == "opake_threefry_like"

    def test_state_stores_domain_separated_stream_seed(self):
        sampler = BMinSepSampler(
            _ds(12), bands=3, sampling_prob=0.5, n_steps=7, key=key(17)
        )
        iterator = iter(sampler)
        for _ in range(3):
            next(iterator)

        snapshot = state_dict(sampler)
        expected_seed = fold_in(key(17), B_MIN_SEP_STREAM_FOLD).seed
        expected_tail = list(sampler)
        template = BMinSepSampler(
            _ds(12), bands=3, sampling_prob=0.5, n_steps=7, key=key(0)
        )

        assert snapshot == {
            "key_seed": expected_seed,
            "key_impl": "opake_threefry_like",
            "consumed": 3,
            "num_examples": 12,
            "bands": 3,
            "sampling_prob": 0.5,
            "n_steps": 7,
        }
        assert list(from_state_dict(template, snapshot)) == expected_tail

        reader = from_state_dict(template, snapshot)
        assert next(iter(reader)) == expected_tail[0]
        resaved = state_dict(reader)
        assert resaved["key_seed"] == expected_seed
        assert resaved["key_impl"] == snapshot["key_impl"]
        assert list(from_state_dict(template, resaved)) == expected_tail[1:]


class TestBallsInBinsRoundTrip:
    def test_emits_empty_bin_slots(self):
        sampler = BallsInBinsSampler(_ds(1), num_bins=2, n_steps=4, key=key(5))

        batches = list(sampler)

        assert len(batches) == 4
        assert batches[:2] == batches[2:]
        assert sorted(index for batch in batches[:2] for index in batch) == [0]
        assert any(not batch for batch in batches[:2])

    def test_snapshot_resumes_at_cursor(self):
        original = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        original_batches = list(original)

        fresh = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        it = iter(fresh)
        K = 12
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(99))
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = BallsInBinsSampler(_ds(), num_bins=8, n_steps=16, key=key(1))
        snapshot = state_dict(sampler)
        # ``num_bins`` drives the bin assignment — must match across
        # save/resume to preserve the BnB privacy contract.  Use the
        # same value on the template so the constructor's
        # ``n_steps % num_bins == 0`` check passes; ``n_steps`` itself
        # is the per-run iteration bound and comes from the template.
        template = BallsInBinsSampler(_ds(), num_bins=8, n_steps=24, key=key(0))
        restored = from_state_dict(template, snapshot)

        assert restored.num_bins == 8
        assert restored.n_steps == 24

    def test_pre_domain_snapshot_uses_saved_seed_without_folding(self):
        snapshot = {
            "key_seed": 17,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_samples": 12,
            "num_bins": 3,
            "n_steps": 6,
        }
        template = BallsInBinsSampler(_ds(12), num_bins=3, n_steps=6, key=key(0))

        restored = from_state_dict(template, snapshot)
        reference = BallsInBinsSampler._from_stream_key(
            _ds(12), num_bins=3, n_steps=6, stream_key=key(17)
        )
        iterator = iter(reference)
        for _ in range(2):
            next(iterator)

        assert list(restored) == list(reference)
        assert state_dict(restored)["key_seed"] == 17
        assert state_dict(restored)["key_impl"] == "opake_threefry_like"

    def test_state_stores_domain_separated_stream_seed(self):
        sampler = BallsInBinsSampler(_ds(12), num_bins=3, n_steps=6, key=key(17))
        iterator = iter(sampler)
        for _ in range(2):
            next(iterator)

        snapshot = state_dict(sampler)
        expected_seed = fold_in(key(17), BALLS_IN_BINS_STREAM_FOLD).seed
        expected_tail = list(sampler)
        template = BallsInBinsSampler(_ds(12), num_bins=3, n_steps=6, key=key(0))

        assert snapshot == {
            "key_seed": expected_seed,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_samples": 12,
            "num_bins": 3,
            "n_steps": 6,
        }
        assert list(from_state_dict(template, snapshot)) == expected_tail

        reader = from_state_dict(template, snapshot)
        assert next(iter(reader)) == expected_tail[0]
        resaved = state_dict(reader)
        assert resaved["key_seed"] == expected_seed
        assert resaved["key_impl"] == snapshot["key_impl"]
        assert list(from_state_dict(template, resaved)) == expected_tail[1:]


class TestSequentialRoundTrip:
    def test_snapshot_resumes_at_cursor(self):
        sampler = SequentialBatchSampler(_ds(200), batch_size=10)
        original_batches = list(sampler)

        fresh = SequentialBatchSampler(_ds(200), batch_size=10)
        it = iter(fresh)
        K = 8
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = SequentialBatchSampler(_ds(200), batch_size=10)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = SequentialBatchSampler(_ds(200), batch_size=25)
        snapshot = state_dict(sampler)
        template = SequentialBatchSampler(_ds(200), batch_size=100)
        restored = from_state_dict(template, snapshot)

        assert restored._batch_size == 25

    def test_cycling_snapshot_resumes_past_first_pass(self):
        """A cursor beyond one pass round-trips (``n_steps`` cycles);
        the template's ``n_steps`` fixes the restored horizon."""
        full = list(SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60))

        fresh = SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60)
        it = iter(fresh)
        K = 27  # past the first 20-batch pass
        head = [next(it) for _ in range(K)]
        assert head == full[:K]

        snapshot = state_dict(fresh)
        template = SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert len(restored) == 60 - K
        assert list(restored) == full[K:]


class TestLenReflectsRemaining:
    """``__len__`` reports remaining batches, not the original total."""

    def test_cyclic_poisson_len_drops(self):
        s = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(1)
        )
        assert len(s) == 20
        it = iter(s)
        for _ in range(8):
            next(it)
        assert len(s) == 12

    def test_b_min_sep_len_drops(self):
        s = BMinSepSampler(_ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(1))
        assert len(s) == 12
        it = iter(s)
        for _ in range(5):
            next(it)
        assert len(s) == 7

    def test_balls_in_bins_len_drops(self):
        s = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        total = len(s)
        it = iter(s)
        for _ in range(7):
            next(it)
        assert len(s) == total - 7

    def test_sequential_len_drops(self):
        s = SequentialBatchSampler(_ds(200), batch_size=10)
        assert len(s) == 20
        it = iter(s)
        for _ in range(6):
            next(it)
        assert len(s) == 14


class TestRejectDatasetLengthMismatch:
    """All four ``from_state_dict`` paths validate template dataset length."""

    def test_cyclic_poisson_mismatch_raises(self):
        import pytest

        sampler = CyclicPoissonSampler(
            _ds(200), sample_rate=0.1, bands=4, n_steps=20, key=key(7)
        )
        snapshot = state_dict(sampler)
        template = CyclicPoissonSampler(
            _ds(150), sample_rate=0.1, bands=4, n_steps=20, key=key(0)
        )
        with pytest.raises(ValueError, match="num_examples"):
            from_state_dict(template, snapshot)

    def test_b_min_sep_mismatch_raises(self):
        import pytest

        sampler = BMinSepSampler(
            _ds(200), bands=3, sampling_prob=0.2, n_steps=12, key=key(7)
        )
        snapshot = state_dict(sampler)
        template = BMinSepSampler(
            _ds(150), bands=3, sampling_prob=0.2, n_steps=12, key=key(0)
        )
        with pytest.raises(ValueError, match="num_examples"):
            from_state_dict(template, snapshot)

    def test_balls_in_bins_mismatch_raises(self):
        import pytest

        sampler = BallsInBinsSampler(_ds(200), num_bins=10, n_steps=30, key=key(7))
        snapshot = state_dict(sampler)
        template = BallsInBinsSampler(_ds(150), num_bins=10, n_steps=30, key=key(0))
        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)

    def test_sequential_mismatch_raises(self):
        import pytest

        sampler = SequentialBatchSampler(_ds(200), batch_size=10)
        snapshot = state_dict(sampler)
        template = SequentialBatchSampler(_ds(150), batch_size=10)
        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)
