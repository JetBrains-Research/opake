"""MfAmplification Protocol conformance — every amplifier exposes ``n_steps``,
``min_sep``, and ``max_participations`` with the documented formulas.
"""

from __future__ import annotations

import pytest

import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.dpftrl.amplification.types import MfAmplification
from opake.dpftrl.noise import band_mf_strategy, identity_strategy


def _bnb():
    return ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        num_bins=10,
        n_steps=100,
    )


def _bms():
    return ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=8)),
        n_steps=80,
        p0=0.05,
    )


def _poisson():
    return ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        sample_rate=0.01,
        n_steps=200,
    )


@pytest.mark.parametrize("make", [_bnb, _bms, _poisson], ids=["bnb", "bms", "poisson"])
def test_protocol_conformance(make):
    proc = make()
    assert isinstance(proc, MfAmplification)
    assert isinstance(proc.n_steps, int)
    assert proc.n_steps >= 1
    assert isinstance(proc.min_sep, int)
    assert proc.min_sep >= 1
    assert isinstance(proc.max_participations, int)
    assert proc.max_participations >= 1


def test_balls_in_bins_formulas():
    proc = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        num_bins=10,
        n_steps=100,
    )
    assert proc.n_steps == 100
    assert proc.min_sep == 10
    assert proc.max_participations == 10  # n_steps // num_bins


def test_b_min_sep_formulas():
    proc = ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=8)),
        n_steps=80,
        p0=0.05,
    )
    assert proc.n_steps == 80
    assert proc.min_sep == 8  # = bands
    assert proc.max_participations == 10  # ceil(n_steps / bands) = 10


def test_cyclic_poisson_degenerate_limits():
    proc = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        sample_rate=0.01,
        n_steps=200,
    )
    assert proc.n_steps == 200
    # No worst-case separation / participation guarantee.
    assert proc.min_sep == 1
    assert proc.max_participations == 200
