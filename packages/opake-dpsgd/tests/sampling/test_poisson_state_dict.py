"""``state_dict`` / ``from_state_dict`` round-trip for :class:`PoissonSampler`.

Verifies the registry-based serialization pair behaves correctly:

* A snapshot taken mid-iteration, restored into a freshly-built sampler,
  yields the same remaining batches that the original would have yielded.
* ``consumed`` is honoured on the restored sampler so its iterator
  starts at the saved cursor.
* The ``data_source`` is supplied by the ``template`` (not serialised).
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from opake.api.dpsgd.sampling._poisson import POISSON_STREAM_FOLD
from opake.dpsgd.sampling import PoissonSampler
from opake.random import fold_in, key
from opake.serialization import from_state_dict, state_dict


def _make_sampler(seed: int = 7, n_steps: int = 20) -> PoissonSampler:
    dataset = TensorDataset(torch.arange(200).reshape(-1, 1))
    return PoissonSampler(dataset, sample_rate=0.1, n_steps=n_steps, key=key(seed))


def _small_sampler(seed: int) -> PoissonSampler:
    dataset = TensorDataset(torch.arange(12).reshape(-1, 1))
    return PoissonSampler(dataset, sample_rate=0.35, n_steps=6, key=key(seed))


def _small_stream_sampler(seed: int) -> PoissonSampler:
    dataset = TensorDataset(torch.arange(12).reshape(-1, 1))
    return PoissonSampler._from_stream_key(
        dataset, sample_rate=0.35, n_steps=6, stream_key=key(seed)
    )


class TestPoissonStateDictRoundTrip:
    def test_snapshot_at_step_k_resumes_with_matching_remainder(self):
        """Snapshot mid-iter → load → remaining yields match the original tail."""
        original = _make_sampler(seed=42)
        original_batches = list(original)

        # Build a fresh sampler, iterate up to step K, snapshot.
        fresh = _make_sampler(seed=42)
        it = iter(fresh)
        K = 7
        first_k = [next(it) for _ in range(K)]
        assert first_k == original_batches[:K]
        snapshot = state_dict(fresh)

        # Load into a NEW template (different RNG to prove the snapshot,
        # not the template's key, drives the restored state).
        template = _make_sampler(seed=99)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_snapshot_at_zero_is_full_run(self):
        original = _make_sampler(seed=11)
        original_batches = list(original)

        fresh = _make_sampler(seed=11)
        snapshot = state_dict(fresh)
        template = _make_sampler(
            seed=0
        )  # different seed; template only supplies dataset
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == 0
        assert list(restored) == original_batches

    def test_snapshot_at_completion_yields_empty(self):
        sampler = _make_sampler(seed=1, n_steps=5)
        _ = list(sampler)
        assert sampler.consumed == 5

        snapshot = state_dict(sampler)
        template = _make_sampler(seed=999, n_steps=5)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == 5
        assert list(restored) == []

    def test_round_trip_preserves_truncated_batch_size(self):
        dataset = TensorDataset(torch.arange(300).reshape(-1, 1))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.2,
            n_steps=10,
            truncated_batch_size=15,
            key=key(7),
        )
        snapshot = state_dict(sampler)
        # Sampling parameters and the domain-separated stream key come from the
        # snapshot. The template supplies ``n_steps`` so the run may be extended.
        template = PoissonSampler(dataset, sample_rate=0.5, n_steps=99, key=key(0))
        restored = from_state_dict(template, snapshot)

        assert restored.sample_rate == 0.2
        assert restored.truncated_batch_size == 15
        # n_steps follows the template (allows resume + extend).
        assert restored.n_steps == 99


class TestPoissonStateCompatibility:
    def test_pre_domain_snapshot_uses_saved_seed_without_folding(self):
        snapshot = {
            "key_seed": 17,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_samples": 12,
            "sample_rate": 0.35,
            "n_steps": 6,
            "truncated_batch_size": None,
        }

        restored = from_state_dict(_small_sampler(0), snapshot)
        reference = _small_stream_sampler(17)
        iterator = iter(reference)
        for _ in range(2):
            next(iterator)

        assert list(restored) == list(reference)
        assert state_dict(restored)["key_seed"] == 17
        assert state_dict(restored)["key_impl"] == "opake_threefry_like"

    def test_state_stores_domain_separated_stream_seed(self):
        sampler = _small_sampler(17)
        iterator = iter(sampler)
        for _ in range(2):
            next(iterator)

        snapshot = state_dict(sampler)
        expected_seed = fold_in(key(17), POISSON_STREAM_FOLD).seed
        expected_tail = list(sampler)

        assert snapshot == {
            "key_seed": expected_seed,
            "key_impl": "opake_threefry_like",
            "consumed": 2,
            "num_samples": 12,
            "sample_rate": 0.35,
            "n_steps": 6,
            "truncated_batch_size": None,
        }
        assert list(from_state_dict(_small_sampler(0), snapshot)) == expected_tail

        reader = from_state_dict(_small_sampler(0), snapshot)
        assert next(iter(reader)) == expected_tail[0]
        resaved = state_dict(reader)
        assert resaved["key_seed"] == expected_seed
        assert resaved["key_impl"] == snapshot["key_impl"]
        assert list(from_state_dict(_small_sampler(0), resaved)) == expected_tail[1:]


class TestPoissonLenReflectsRemaining:
    """``__len__`` reports batches the iterator will yield, not total."""

    def test_len_drops_with_consumption(self):
        sampler = _make_sampler(seed=42, n_steps=20)
        assert len(sampler) == 20

        it = iter(sampler)
        for _ in range(7):
            next(it)
        assert len(sampler) == 13

    def test_len_after_restore_matches_remaining(self):
        fresh = _make_sampler(seed=42, n_steps=20)
        it = iter(fresh)
        for _ in range(11):
            next(it)
        snapshot = state_dict(fresh)

        template = _make_sampler(seed=0, n_steps=20)
        restored = from_state_dict(template, snapshot)
        assert len(restored) == 20 - 11


class TestPoissonRejectsDatasetLengthMismatch:
    """``from_state_dict`` validates the template dataset length."""

    def test_mismatch_raises(self):
        sampler = PoissonSampler(
            TensorDataset(torch.arange(200).reshape(-1, 1)),
            sample_rate=0.1,
            n_steps=5,
            key=key(7),
        )
        snapshot = state_dict(sampler)

        # Template over a different-sized dataset — would silently
        # produce a different Poisson stream.
        template = PoissonSampler(
            TensorDataset(torch.arange(150).reshape(-1, 1)),
            sample_rate=0.1,
            n_steps=5,
            key=key(0),
        )
        import pytest

        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)
