"""Contract tests for full-run sampler streams (issue #357 review).

The DP-FTRL example builds ONE sampler with ``n_steps=total_steps`` and
lets it span every epoch boundary.  These tests pin the properties that
make a single stream the correct sampler lifetime — and that per-epoch
recreation silently violates for the randomized samplers:

* BnB / Sequential: the single stream is byte-identical to same-key
  per-epoch recreation, so switching lifetimes changes no batches.
* BMinSep: min-separation holds across the WHOLE stream.  Recreation
  redraws the warm-start cooldown at each epoch boundary, letting an
  example participate again fewer than ``bands`` steps after its last
  participation of the previous epoch — violating the ``min_sep``
  contract the band-MF accounting sizes noise against.
* CyclicPoisson: the band partition and phase stay fixed across the
  stream; recreation redraws the partition and resets the phase.
"""

from itertools import pairwise

import pytest
import torch
from torch.utils.data import TensorDataset

from opake.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opake.random import key


def _ds(n=64):
    return TensorDataset(torch.arange(n).unsqueeze(1).float())


def _participation_steps(stream):
    """Map example index -> ordered list of steps it participated in."""
    seen = {}
    for t, batch in enumerate(stream):
        for idx in batch:
            seen.setdefault(idx, []).append(t)
    return seen


class TestSingleStreamMatchesRecreation:
    """Deterministic samplers: one stream ≡ same-key per-epoch rebuilds."""

    def test_bnb_single_stream_equals_same_key_recreation(self):
        num_epochs, k = 3, 8
        ds = _ds(64)
        single = list(
            BallsInBinsSampler(ds, num_bins=k, n_steps=num_epochs * k, key=key(7))
        )
        per_epoch = [
            batch
            for _ in range(num_epochs)
            for batch in BallsInBinsSampler(ds, num_bins=k, n_steps=k, key=key(7))
        ]
        assert single == per_epoch

    def test_sequential_n_steps_cycles_fixed_order(self):
        num_epochs = 3
        ds = _ds(64)
        one_pass = list(SequentialBatchSampler(ds, batch_size=8))
        cycled = list(SequentialBatchSampler(ds, batch_size=8, n_steps=num_epochs * 8))
        assert cycled == one_pass * num_epochs

    def test_sequential_rejects_partial_final_pass(self):
        with pytest.raises(ValueError, match="multiple of the per-pass"):
            SequentialBatchSampler(_ds(64), batch_size=8, n_steps=12)


class TestFullStreamInvariants:
    """Properties the accounting assumes over ``total_steps`` — and that
    per-epoch recreation breaks at epoch boundaries."""

    def test_b_min_sep_holds_across_former_epoch_boundaries(self):
        bands, steps_per_epoch, num_epochs = 4, 10, 5
        for seed in range(10):
            sampler = BMinSepSampler(
                _ds(64),
                bands=bands,
                sampling_prob=0.5,
                n_steps=steps_per_epoch * num_epochs,
                key=key(seed),
            )
            for steps in _participation_steps(sampler).values():
                gaps = [b - a for a, b in pairwise(steps)]
                assert all(g >= bands for g in gaps)

    def test_cyclic_poisson_band_phase_fixed_across_stream(self):
        bands, steps_per_epoch, num_epochs = 4, 12, 3
        for seed in range(10):
            sampler = CyclicPoissonSampler(
                _ds(64),
                sample_rate=0.5,
                bands=bands,
                n_steps=steps_per_epoch * num_epochs,
                key=key(seed),
            )
            for steps in _participation_steps(sampler).values():
                # Fixed partition + phase: an example only ever appears
                # at steps congruent to its band's slot.
                assert len({t % bands for t in steps}) == 1
