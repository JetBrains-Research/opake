"""Block k-out-of-t allocation and balls-in-bins are different schemes.

Both samplers emit ``num_bins`` batches per epoch whose union is the dataset,
so a reader skimming their output could easily believe they are
interchangeable.
They are not: DP-FTRL's balls-in-bins fixes the assignment once (the
matrix-mechanism dominating pair needs a known participation separation),
while DP-SGD's block k-out-of-t allocation redraws it every block (valid only because
DP-SGD noise is uncorrelated across steps, and strictly better there).

Swapping one for the other silently changes the mechanism being accounted,
so this asserts the observable difference rather than trusting docstrings.
"""

from __future__ import annotations

from itertools import chain

from opake.dpftrl.sampling import BallsInBinsSampler
from opake.dpsgd.sampling import KOutOfTSampler
from opake.random import key

_N = 120
_BINS = 6
_EPOCHS = 4


def _epochs(sampler) -> list[list[list[int]]]:
    batches = list(sampler)
    return [batches[e * _BINS : (e + 1) * _BINS] for e in range(_EPOCHS)]


def test_balls_in_bins_repeats_its_partition_every_epoch():
    """Fixed assignment: every epoch is the same partition, in the same order."""
    epochs = _epochs(
        BallsInBinsSampler(
            list(range(_N)), num_bins=_BINS, n_steps=_BINS * _EPOCHS, key=key(11)
        )
    )
    assert all(e == epochs[0] for e in epochs[1:])


def test_block_k_out_of_t_redraws_its_partition_every_block():
    """Block redraw: the partition changes from one block to the next."""
    epochs = _epochs(
        KOutOfTSampler(
            list(range(_N)),
            k=_EPOCHS,
            t=_BINS * _EPOCHS,
            allocation="block",
            key=key(11),
        )
    )
    assert any(e != epochs[0] for e in epochs[1:])


def test_same_key_gives_different_streams():
    """The two schemes must not coincide even when handed the same key."""
    args = (list(range(_N)),)
    kwargs = {"num_bins": _BINS, "n_steps": _BINS * _EPOCHS, "key": key(11)}
    bnb = list(BallsInBinsSampler(*args, **kwargs))
    block = list(
        KOutOfTSampler(
            *args,
            k=_EPOCHS,
            t=_BINS * _EPOCHS,
            allocation="block",
            key=key(11),
        )
    )
    assert bnb != block


def test_both_partition_each_epoch():
    """The property they *do* share — so the difference above is genuinely
    the redraw, not a different notion of an epoch."""
    for sampler in (
        BallsInBinsSampler(
            list(range(_N)), num_bins=_BINS, n_steps=_BINS * _EPOCHS, key=key(3)
        ),
        KOutOfTSampler(
            list(range(_N)),
            k=_EPOCHS,
            t=_BINS * _EPOCHS,
            allocation="block",
            key=key(3),
        ),
    ):
        for epoch in _epochs(sampler):
            assert sorted(chain(*epoch)) == list(range(_N))
