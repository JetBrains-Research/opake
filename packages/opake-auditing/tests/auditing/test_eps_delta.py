"""Tests for the (ε, δ)-DP audit method.

Covers ``methods/_eps_delta.py`` — the analytical Xiang p-value and the
``OneRunEstimate.eps_delta()`` factory.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special
import scipy.stats

from opake.api.auditing.one_run._eps_delta import _p_value
from opake.auditing import one_run
from opake.auditing.types import CanaryScores, CoinFlip


def _make_estimate(in_scores, out_scores):
    """Build a OneRunEstimate from raw in/out score arrays."""
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    canary_indices = np.arange(n_in + n_out)
    mask = np.array([True] * n_in + [False] * n_out)
    cf = CoinFlip(
        num_canaries=n_in + n_out,
        canary_indices=canary_indices,
        _in_mask=mask,
        in_indices=canary_indices[mask],
        out_indices=canary_indices[~mask],
    )
    scores = np.empty(n_in + n_out)
    scores[mask] = in_scores
    scores[~mask] = out_scores
    return one_run(CanaryScores(scores, canary_indices=canary_indices), coin_flip=cf)


# ---- _p_value --------------------------------------------------------------


class TestPValue:
    """Unit tests for the (ε, δ)-DP analytical p-value."""

    def test_matches_binom_cdf_delta_zero(self):
        r, u, eps = 1000, 400, 1.0
        p = scipy.special.expit(-eps)
        expected = scipy.stats.binom.cdf(u, r, p)
        actual = _p_value(r, u, eps, 0.0)
        assert abs(actual - expected) < 1e-10

    def test_matches_binom_cdf_delta_positive(self):
        r, u, eps, delta = 1000, 400, 1.0, 0.01
        p = scipy.special.expit(-eps)
        n_eff = r - round(r * delta)
        expected = scipy.stats.binom.cdf(u, n_eff, p)
        actual = _p_value(r, u, eps, delta)
        assert abs(actual - expected) < 1e-10

    def test_monotone_in_eps(self):
        p1 = _p_value(1000, 300, 0.5, 0.0)
        p2 = _p_value(1000, 300, 1.0, 0.0)
        p3 = _p_value(1000, 300, 2.0, 0.0)
        assert p1 <= p2 <= p3

    def test_perfect_attack(self):
        p = _p_value(1000, 0, 0.5, 0.0)
        assert p < 1e-10

    def test_random_guessing(self):
        p = _p_value(1000, 500, 0.0, 0.0)
        assert 0.4 < p < 0.6

    def test_delta_one_returns_one(self):
        assert _p_value(100, 10, 1.0, 1.0) == 1.0

    def test_worse_than_expected(self):
        p = _p_value(100, 80, 1.0, 0.0)
        assert p > 0.5


# ---- OneRunEstimate.eps_delta() --------------------------------------------


class TestEpsDeltaMethod:
    """Behavioural tests for OneRunEstimate.eps_delta()."""

    def test_separated_delta_zero(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.eps_delta().epsilon_at(delta=0.0)
        assert eps > 0

    def test_separated_delta_positive(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.eps_delta().epsilon_at(delta=1e-5)
        assert eps > 0

    def test_no_separation(self):
        est = _make_estimate(np.arange(100), np.arange(100))
        eps = est.eps_delta().epsilon_at(delta=0.0)
        assert eps < 1.0

    def test_threshold_works(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.eps_delta().epsilon_at(delta=0.0, threshold=50.0)
        assert eps > 0

    def test_returns_float(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert isinstance(est.eps_delta().epsilon_at(delta=0.0), float)
        assert isinstance(est.eps_delta().epsilon_at(delta=0.1), float)

    def test_invalid_significance(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="significance"):
            est.eps_delta().epsilon_at(significance=0.0)

    def test_invalid_delta(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta"):
            est.eps_delta().epsilon_at(delta=-0.1)

    def test_large_eps(self):
        """1000 canaries with perfect separation → large ε."""
        est = _make_estimate(np.arange(1000, 2000), np.arange(0, 1000))
        eps = est.eps_delta().epsilon_at(delta=0.0, threshold=1000)
        assert eps > 5.0


# ---- delta_at --------------------------------------------------------------


class TestEpsDeltaDeltaAt:
    """Tests for EpsDeltaMethod.delta_at()."""

    def test_inverts_epsilon_at(self):
        """delta_at and epsilon_at are inverses up to n_eff discretization (1/m)."""
        m = 200
        est = _make_estimate(np.arange(m // 2, m), np.arange(0, m // 2))
        method = est.eps_delta()
        delta_input = 0.1
        eps = method.epsilon_at(delta=delta_input)
        d_out = method.delta_at(epsilon=eps)
        # n_eff = m − round(m·δ) is a step function; round-trip lands within 2/m
        assert abs(d_out - delta_input) < 2.0 / m

    def test_unreachable_eps_returns_zero(self):
        """Asking for ε > ε̂(0) → return 0 (no δ certifies)."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.eps_delta()
        eps0 = method.epsilon_at(delta=0.0)
        d = method.delta_at(epsilon=eps0 + 5.0)
        assert d == 0.0

    def test_negative_epsilon_raises(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            est.eps_delta().delta_at(epsilon=-0.1)


# ---- Contract: torch-free --------------------------------------------------
