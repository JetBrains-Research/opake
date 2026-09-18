"""Tests for the μ-GDP audit method.

Covers ``one_run/_gdp.py`` — the method class, GDP↔(ε,δ) conversion,
discretised base pair, and order-statistics p-value.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import scipy.stats

from opaque.api.auditing.one_run._eps_delta import _p_value as _eps_delta_p_value
from opaque.api.auditing.one_run._gdp import (
    GdpMethod,
    _chernoff_lower_tail,
    _compute_v_k,
    _gdp_base_pair_grid,
    _gdp_to_eps_delta,
    _p_value,
)
from opaque.auditing import one_run
from opaque.auditing.types import CanaryScores, CoinFlip

gdp_module = importlib.import_module("opaque.api.auditing.one_run._gdp")


class _StubEstimate:
    """Minimal estimate stub that returns fixed (r, u) for ``_mu_at`` tests."""

    def __init__(self, n_in: int, n_out: int, u: int):
        self.n_in = n_in
        self.n_out = n_out
        self._u = u

    def _best_r_u(self, threshold: float | None = None) -> tuple[int, int]:
        return self.n_in + self.n_out, self._u


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


def _delta_at(mu: float, eps: float) -> float:
    """Analytical δ(ε) for μ-GDP."""
    a = mu / 2.0 - eps / mu
    b = -mu / 2.0 - eps / mu
    return scipy.stats.norm.cdf(a) - math.exp(eps) * scipy.stats.norm.cdf(b)


# ---- _gdp_to_eps_delta -----------------------------------------------------


class TestGdpToEpsDelta:
    """Unit tests for μ-GDP → (ε, δ) conversion."""

    def test_mu_zero_returns_zero(self):
        assert _gdp_to_eps_delta(0.0, 1e-5) == 0.0

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 5.0])
    def test_cross_check(self, mu):
        delta = 1e-5
        eps = _gdp_to_eps_delta(mu, delta)
        actual_delta = _delta_at(mu, eps)
        assert abs(actual_delta - delta) < 1e-7

    def test_monotone_in_mu(self):
        delta = 1e-5
        prev = 0.0
        for mu in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            eps = _gdp_to_eps_delta(mu, delta)
            assert eps >= prev
            prev = eps

    def test_large_delta(self):
        eps = _gdp_to_eps_delta(1.0, 0.5)
        assert eps < _gdp_to_eps_delta(1.0, 1e-5)

    def test_negative_mu_raises(self):
        with pytest.raises(ValueError, match="mu must be >= 0"):
            _gdp_to_eps_delta(-1.0, 1e-5)

    def test_delta_out_of_range_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            _gdp_to_eps_delta(1.0, 0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            _gdp_to_eps_delta(1.0, -0.1)

    def test_delta_one_returns_zero(self):
        """δ = 1 ⇒ no privacy constraint ⇒ smallest ε is 0 (no exception)."""
        assert _gdp_to_eps_delta(1.0, 1.0) == 0.0


# ---- _gdp_base_pair_grid ---------------------------------------------------


class TestGdpBasePairGrid:
    """Unit tests for the discretised μ-GDP base pair."""

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_total_mass_near_one(self, mu):
        grid = _gdp_base_pair_grid(mu, 10_000)
        total = float(np.sum(grid.mass))
        assert abs(total - 1.0) < 0.01

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
    def test_F_y_monotone(self, mu):
        grid = _gdp_base_pair_grid(mu, 10_000)
        assert np.all(np.diff(grid.F_y) >= -1e-10)

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
    def test_abs_pl_sorted(self, mu):
        grid = _gdp_base_pair_grid(mu, 10_000)
        assert np.all(np.diff(grid.abs_privacy_loss) >= -1e-10)

    def test_min_abs_pl_near_zero(self):
        grid = _gdp_base_pair_grid(1.0, 10_000)
        assert grid.abs_privacy_loss[0] < 0.05

    def test_mass_non_negative(self):
        grid = _gdp_base_pair_grid(1.0, 10_000)
        assert np.all(grid.mass >= -1e-15)

    def test_mu_zero_raises(self):
        with pytest.raises(ValueError, match="mu must be > 0"):
            _gdp_base_pair_grid(0.0, 100)


# ---- v_k grid convergence (Eq. 12 identity) ----------------------------------


class TestVKGridConvergence:
    """At accepted grid sizes the discretised v_k must match closed form.

    For n = 1, Eq. 12 of Xiang et al. (2025) reduces to the Bayes error
    of the shift test, v_1 = Phi(-mu/2).  The grid floor accepted by
    ``OneRunEstimate.gdp`` is validated against this identity.
    """

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("grid_size", [1_000, 10_000])
    def test_n_one_matches_bayes_error(self, mu, grid_size):
        v_1 = _compute_v_k(1, 1, _gdp_base_pair_grid(mu, grid_size))
        assert abs(float(v_1[0]) - scipy.stats.norm.cdf(-mu / 2)) < 1e-5


# ---- _p_value --------------------------------------------------------------


class TestPValue:
    """Unit tests for the μ-GDP grid-based p-value."""

    def test_perfect_attack(self):
        p = _p_value(100, 100, 0, 1.0, 5000)
        assert 0 < p < 0.01

    def test_near_random(self):
        p = _p_value(100, 100, 45, 1.0, 5000)
        assert p > 0.3

    def test_mu_zero_returns_one(self):
        assert _p_value(100, 100, 10, 0.0, 5000) == 1.0

    def test_monotone_in_mu(self):
        prev = 0.0
        for mu in [0.3, 0.5, 1.0, 2.0, 3.0]:
            p = _p_value(100, 100, 20, mu, 5000)
            assert p >= prev - 1e-10
            prev = p

    def test_tighter_than_eps_delta(self):
        """At the same effective (ε, δ), μ-GDP p-value ≤ (ε, δ) p-value."""
        mu = 1.0
        delta = 1e-5
        eps = _gdp_to_eps_delta(mu, delta)
        p_gdp = _p_value(200, 200, 30, mu, 5000)
        p_epsd = _eps_delta_p_value(200, 30, eps, delta)
        assert p_gdp <= p_epsd + 1e-10

    def test_scaling_with_n(self):
        p50 = _p_value(50, 50, 0, 1.0, 5000)
        p200 = _p_value(200, 200, 0, 1.0, 5000)
        assert p200 < p50

    def test_returns_float(self):
        p = _p_value(100, 100, 20, 1.0, 5000)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    @pytest.mark.parametrize("mu", [math.inf, math.nan])
    def test_non_finite_mu_returns_one(self, mu):
        assert _p_value(100, 100, 0, mu, 64) == 1.0

    def test_no_guesses_returns_one(self):
        assert _p_value(100, 0, 0, 1.0, 64) == 1.0

    def test_truncated_rank_bound_is_conservative(self, monkeypatch):
        """The boundary substitution must upper-bound the exact-rank p-value."""
        kwargs = {"n": 500, "r": 500, "u": 100, "mu": 1.0, "grid_size": 256}
        monkeypatch.setattr(gdp_module, "_MAX_EXACT_RANKS", 50)
        truncated = _p_value(**kwargs)
        monkeypatch.setattr(gdp_module, "_MAX_EXACT_RANKS", 500)
        exact = _p_value(**kwargs)
        assert truncated >= exact

    def test_truncated_rank_bound_respects_rank_monotonicity(self):
        all_v = _compute_v_k(100, 100, _gdp_base_pair_grid(1.0, 1_000))
        boundary_v = all_v[-20]
        assert np.all(all_v[:-20] >= boundary_v)

    def test_truncated_p_value_is_monotone_in_mu(self, monkeypatch):
        monkeypatch.setattr(gdp_module, "_MAX_EXACT_RANKS", 50)
        p_values = [_p_value(500, 500, 100, mu, 256) for mu in (0.5, 1.0, 2.0, 4.0)]
        assert np.all(np.diff(p_values) >= -1e-10)

    def test_truncation_boundary_decays_with_mu(self):
        bounds = [
            _compute_v_k(500, 50, _gdp_base_pair_grid(mu, 1_000))[0]
            for mu in (0.5, 1.0, 2.0, 4.0)
        ]
        assert np.all(np.diff(bounds) < 0.0)

    def test_chernoff_requires_truncated_rank_bound(self):
        with pytest.raises(ValueError, match="trunc_v is required"):
            _chernoff_lower_tail(np.array([0.1]), n_trunc=1, u=0)


# ---- GdpMethod._mu_at bracket / bisection caps -----------------------------


class TestMuAtTermination:
    """Regression: strong attacks past rank truncation remain invertible.

    These build ``GdpMethod`` directly with deliberately tiny grids to keep
    the test fast; the public factory enforces ``_MIN_GRID_SIZE`` separately.
    """

    @pytest.mark.parametrize(("n_half", "u"), [(1500, 0), (2500, 1400)])
    def test_strong_attack_past_truncation_inverts(self, n_half, u):
        method = GdpMethod(_estimate=_StubEstimate(n_half, n_half, u), grid_size=64)
        mu = method._mu_at(0.05, None)
        assert math.isfinite(mu)
        assert mu > 0.0

    def test_below_truncation_still_inverts(self):
        method = GdpMethod(_estimate=_StubEstimate(500, 500, u=0), grid_size=256)
        mu = method._mu_at(0.05, None)
        assert math.isfinite(mu)
        assert mu > 0.0


# ---- OneRunEstimate.gdp() --------------------------------------------------


class TestGdpMethod:
    """Tests for OneRunEstimate.gdp()."""

    def test_basic(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.gdp().epsilon_at(delta=1e-5)
        assert eps > 0

    def test_tighter_than_eps_delta(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eg = est.gdp().epsilon_at(delta=1e-5)
        ed = est.eps_delta().epsilon_at(delta=1e-5)
        assert eg >= ed * 0.9, f"gdp ({eg:.4f}) << eps_delta ({ed:.4f})"

    def test_delta_zero_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta > 0"):
            est.gdp().epsilon_at(delta=0.0)

    def test_delta_negative_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta > 0"):
            est.gdp().epsilon_at(delta=-1.0)

    def test_delta_gt_one_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            est.gdp().epsilon_at(delta=2.0)

    def test_threshold_works(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.gdp().epsilon_at(delta=1e-5, threshold=50.0)
        assert eps > 0

    def test_returns_float(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert isinstance(est.gdp().epsilon_at(delta=1e-5), float)

    def test_grid_size_config(self):
        """grid_size on the factory propagates to the method."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.gdp(grid_size=5_000)
        assert method.grid_size == 5_000
        eps_5k = method.epsilon_at(delta=1e-5)
        eps_10k = est.gdp().epsilon_at(delta=1e-5)
        assert abs(eps_5k - eps_10k) < 0.5

    def test_grid_size_below_floor_raises(self):
        """Public factory rejects grids below the validated floor."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(ValueError, match=r"grid_size must be >= 1000, got 999"):
            est.gdp(grid_size=999)
        # The floor itself is accepted.
        assert est.gdp(grid_size=1_000).grid_size == 1_000


class TestCoarseGridCannotEraseLeakage:
    """Regression #829: a known-leaky config never audits eps=0 at an
    accepted grid size, even though a coarse grid below the floor did."""

    @staticmethod
    def _leaky_estimate():
        """500/500 split whose best threshold makes exactly 250 errors.

        A coarse grid (grid_size=16) resolved this as eps=0 (no leak); the
        accepted floor must report a positive privacy lower bound.
        """
        in_scores = np.concatenate([np.arange(1_000, 1_250), np.arange(500, 750)])
        out_scores = np.concatenate([np.arange(0, 250), np.arange(500, 750)])
        n = 1_000
        idx = np.arange(n)
        mask = np.array([True] * 500 + [False] * 500)
        cf = CoinFlip(
            num_canaries=n,
            canary_indices=idx,
            _in_mask=mask,
            in_indices=idx[mask],
            out_indices=idx[~mask],
        )
        scores = np.empty(n)
        scores[:500] = in_scores
        scores[500:] = out_scores
        return one_run(CanaryScores(scores, canary_indices=idx), coin_flip=cf)

    @pytest.mark.parametrize("grid_size", [1_000, 2_500, 10_000])
    def test_known_leak_positive_epsilon(self, grid_size):
        est = self._leaky_estimate()
        # Confirm this is a leaky config: the order-statistics test sees far
        # fewer errors than the coin-flip baseline (u << m/2).
        m, u = est._best_r_u(None)
        assert m == 1_000
        assert u == 250
        eps = est.gdp(grid_size=grid_size).epsilon_at(delta=1e-5)
        assert eps > 0.0


# ---- Pld-mirror surface: delta_at, beta_at, advantage ----------------------


class TestGdpPldSurface:
    """Tests for delta_at / beta_at / advantage on GdpMethod."""

    def test_delta_at_round_trips_with_epsilon_at(self):
        """gdp(): δ̂ = delta_at(epsilon_at(δ)) ≈ δ (closed-form inversion)."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.gdp()
        delta_input = 1e-5
        eps = method.epsilon_at(delta=delta_input)
        d_out = method.delta_at(epsilon=eps)
        assert abs(d_out - delta_input) < 1e-6

    def test_delta_at_monotone_in_epsilon(self):
        """Larger ε ⇒ smaller δ along the μ̂-GDP curve."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.gdp()
        d_small = method.delta_at(epsilon=1.0)
        d_large = method.delta_at(epsilon=4.0)
        assert d_large <= d_small

    def test_delta_at_negative_epsilon_raises(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            est.gdp().delta_at(epsilon=-0.1)

    def test_beta_at_endpoints(self):
        """β(0) ≤ 1, β(1) ≥ 0; perfect attack has β well below 1 − α."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.gdp()
        assert 0.0 <= method.beta_at(alpha=0.0) <= 1.0
        assert 0.0 <= method.beta_at(alpha=1.0) <= 1.0
        # Strong separation → β at α=0.5 substantially below 0.5 (random)
        assert method.beta_at(alpha=0.5) < 0.5

    def test_beta_at_no_separation_near_random(self):
        """μ̂ ≈ 0 → β(α) ≈ 1 − α."""
        est = _make_estimate(np.arange(100), np.arange(100))
        beta = est.gdp().beta_at(alpha=0.3)
        assert abs(beta - 0.7) < 0.1

    def test_beta_at_invalid_alpha_raises(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(ValueError, match="alpha must be in"):
            est.gdp().beta_at(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            est.gdp().beta_at(alpha=1.5)

    def test_advantage_in_unit_interval(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        adv = est.gdp().advantage()
        assert 0.0 <= adv <= 1.0

    def test_advantage_no_separation_near_zero(self):
        est = _make_estimate(np.arange(100), np.arange(100))
        adv = est.gdp().advantage()
        assert adv < 0.1

    def test_advantage_matches_closed_form(self):
        """advantage() ≡ 2·Φ(μ̂/2) − 1 at the inferred μ̂."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        method = est.gdp()
        mu = method._mu_at(0.05, None)
        expected = 2.0 * scipy.stats.norm.cdf(mu / 2.0) - 1.0
        assert abs(method.advantage() - expected) < 1e-10
