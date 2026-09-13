"""Cross-validation: opaque.accounting vs dp_accounting + riskcal.

These tests verify numerical agreement between opaque's Rust-backed PLD
implementation and Google's dp_accounting library (the reference).  The
riskcal library provides additional f-DP / error-rate metrics on top of
dp_accounting PLDs, enabling triple validation.

**Tolerance**: < 1e-6 for direct Gaussian comparisons and committed numerical
vectors. Generic Poisson comparisons use per-metric tolerances measured against
the independent oracle grid below.

Requires optional deps — install with ``uv sync --group cross-validation``.
The ``pytest.importorskip`` calls below gate the entire module automatically.

Run selectively with::

    uv run --group cross-validation pytest packages/opaque-dpsgd/tests/accounting/test_cross_validation.py -v
"""

from __future__ import annotations

import math
from itertools import product

import pytest

dp_accounting = pytest.importorskip("dp_accounting")
riskcal = pytest.importorskip("riskcal")

import riskcal.analysis as rc_analysis  # noqa: E402
from dp_accounting.pld import privacy_loss_distribution as pld_lib  # noqa: E402

import opaque.accounting as acc  # noqa: E402

# Expensive reference-library cases below are scoped to their individual test
# functions. Fast cross-checks stay in the normal suite so accounting behavior
# remains covered pre-merge without running the costly pure-Python PLD
# convolutions.
import opaque.dpsgd.accounting as dpsgd_acc  # noqa: E402
from opaque.accounting import calibration as cal  # noqa: E402
from opaque.api.accounting.core.discretization import get_discretization  # noqa: E402

# ============================================================================
# Helpers
# ============================================================================

# The generic pointwise path is compared directly with dp_accounting/riskcal
# over the parameter grids below. These envelopes exceed the measured maxima
# by a small margin; exact generic-output vectors retain the 1e-6 drift check.
ATOL = 1e-6
POISSON_EPSILON_ATOL = 6e-5
POISSON_DELTA_ATOL = 3e-8
POISSON_ADVANTAGE_ATOL = 1.3e-5
POISSON_BETA_ATOL = 2.2e-4

# NOTE: opaque and dp_accounting agree to ~1e-9 relative error.
# For high-epsilon regimes (small σ), absolute error can be ~1e-7.
# We use rel=1e-8 for single Gaussian (high ε) and abs=1e-6 elsewhere.

# User-specified parameter grids (σ must be in [0.1, 1.2] per Rust crate)
SIGMAS = [0.1, 0.3, 0.5, 0.8, 1.2]
SAMPLE_RATES = [0.001, 0.0005, 0.0001]
STEPS = [10, 50, 200, 500, 1000, 3000]
DELTAS = [1e-5, 1e-6, 1e-8]


def _slow_params(cases, slow_cases):
    return [
        pytest.param(
            *case,
            marks=pytest.mark.slow if case in slow_cases else (),
        )
        for case in cases
    ]


def _ref_gaussian_pld(sigma: float, sampling_prob: float = 1.0):
    """Build a dp_accounting PLD for Gaussian (± Poisson subsampling)."""
    return pld_lib.from_gaussian_mechanism(sigma, sampling_prob=sampling_prob)


def _ref_epsilon(
    sigma: float, delta: float, sampling_prob: float = 1.0, steps: int = 1
):
    """Reference epsilon from dp_accounting."""
    pld = _ref_gaussian_pld(sigma, sampling_prob)
    if steps > 1:
        pld = pld.self_compose(steps)
    return pld.get_epsilon_for_delta(delta)


def _ref_delta(
    sigma: float, epsilon: float, sampling_prob: float = 1.0, steps: int = 1
):
    """Reference delta from dp_accounting."""
    pld = _ref_gaussian_pld(sigma, sampling_prob)
    if steps > 1:
        pld = pld.self_compose(steps)
    return pld.get_delta_for_epsilon(epsilon)


_POISSON_EPSILON_GOLDENS: dict[tuple[float, float, int], float] = {
    (0.5, 0.001, 10): 1.0860492909460648,
    (0.5, 0.001, 100): 2.1086898552170803,
    (0.5, 0.001, 200): 2.43399001267914,
    (0.5, 0.001, 500): 2.877723018444483,
    (0.5, 0.001, 1000): 3.2389339479391825,
    (0.5, 0.0001, 10): 0.03943809854733671,
    (0.5, 0.0001, 100): 0.17936544758819523,
    (0.5, 0.0001, 200): 0.2601922038995986,
    (0.5, 0.0001, 500): 0.40375338457776483,
    (0.5, 0.0001, 1000): 0.5421242776418086,
    (0.8, 0.001, 10): 0.06144713575743773,
    (0.8, 0.001, 100): 0.14097516463864504,
    (0.8, 0.001, 200): 0.17738406244303395,
    (0.8, 0.001, 500): 0.2398344201560575,
    (0.8, 0.001, 1000): 0.30358601971364896,
    (0.8, 0.001, 3000): 0.45888699888296003,
    (0.8, 0.0001, 10): 0.0023467920178165467,
    (0.8, 0.0001, 100): 0.007360893019880962,
    (0.8, 0.0001, 200): 0.010070421322347263,
    (0.8, 0.0001, 500): 0.015192956530648017,
    (0.8, 0.0001, 1000): 0.02082692729696504,
    (0.8, 0.0001, 3000): 0.034925433345447704,
    (1.2, 0.001, 10): 0.013182306680728033,
    (1.2, 0.001, 100): 0.03429607770742004,
    (1.2, 0.001, 200): 0.047249089757492306,
    (1.2, 0.001, 500): 0.07378327716983124,
    (1.2, 0.001, 1000): 0.10469894176313432,
    (1.2, 0.001, 3000): 0.1850295122634436,
    (1.2, 0.0001, 10): 0.00068660034862896,
    (1.2, 0.0001, 100): 0.002443806869778121,
    (1.2, 0.0001, 200): 0.003559861166450515,
    (1.2, 0.0001, 500): 0.005868520833600029,
    (1.2, 0.0001, 1000): 0.008580386619970966,
    (1.2, 0.0001, 3000): 0.015689593788266062,
}

_POISSON_DELTA_GOLDENS: dict[tuple[float, float, int], tuple[float, float]] = {
    (0.5, 0.001, 100): (1.6869518841736644, 2.5322151238227026e-05),
    (0.5, 0.001, 500): (2.3021784147555864, 3.5463424127813845e-05),
    (0.5, 0.0001, 100): (0.1434923580705562, 1.4745948609814662e-05),
    (0.5, 0.0001, 500): (0.3230027076622119, 1.622049016238609e-05),
    (1.2, 0.001, 100): (0.027436862165936035, 4.0098017078646245e-05),
    (1.2, 0.001, 500): (0.059026621735864995, 5.917849784753735e-05),
    (1.2, 0.0001, 100): (0.0019550454958224966, 2.4543791978496092e-05),
    (1.2, 0.0001, 500): (0.0046948166668800235, 3.231825057037382e-05),
}

_POISSON_ADVANTAGE_GOLDENS: dict[tuple[float, float, int], float] = {
    (0.5, 0.001, 100): 0.018588339518244953,
    (0.5, 0.001, 500): 0.04851396870881404,
    (0.5, 0.0001, 100): 0.0018816756388983582,
    (0.5, 0.0001, 500): 0.0049713898803606564,
    (0.8, 0.001, 100): 0.007352244641155427,
    (0.8, 0.001, 500): 0.01696024954311608,
    (0.8, 0.0001, 100): 0.0007545658901290552,
    (0.8, 0.0001, 500): 0.0017403349502866613,
    (1.2, 0.001, 100): 0.003967596131708196,
    (1.2, 0.001, 500): 0.008927947536175752,
    (1.2, 0.0001, 100): 0.0004314263850629069,
    (1.2, 0.0001, 500): 0.0009704172619880299,
}

_POISSON_BETA_GOLDENS: dict[tuple[float, float, int, float], float] = {
    (0.8, 0.001, 100, 0.01): 0.9891997358156981,
    (0.8, 0.001, 100, 0.1): 0.8959669533545063,
    (0.8, 0.001, 500, 0.01): 0.9884657040417835,
    (0.8, 0.001, 500, 0.1): 0.891369523457133,
    (0.8, 0.0001, 100, 0.01): 0.9899156762736153,
    (0.8, 0.0001, 100, 0.1): 0.8995519070673036,
    (0.8, 0.0001, 500, 0.01): 0.9898422436590798,
    (0.8, 0.0001, 500, 0.1): 0.8990530506911125,
    (1.2, 0.001, 100, 0.01): 0.9896686719291854,
    (1.2, 0.001, 100, 0.1): 0.897995611775056,
    (1.2, 0.001, 500, 0.01): 0.9893108486219238,
    (1.2, 0.001, 500, 0.1): 0.895658391579177,
    (1.2, 0.0001, 100, 0.01): 0.9899532062356198,
    (1.2, 0.0001, 100, 0.1): 0.8997090808060371,
    (1.2, 0.0001, 500, 0.01): 0.9899022067770801,
    (1.2, 0.0001, 500, 0.1): 0.8993718952935553,
}

# ============================================================================
# 1. Gaussian mechanism — epsilon_at vs dp_accounting
# ============================================================================


class TestGaussianEpsilon:
    """Single Gaussian mechanism: opaque vs dp_accounting epsilon_at."""

    @pytest.mark.parametrize(
        ("sigma", "delta"),
        _slow_params(
            product(SIGMAS, DELTAS),
            {
                (0.1, 1e-5),
                (0.3, 1e-5),
                (0.1, 1e-6),
                (0.3, 1e-6),
                (0.1, 1e-8),
            },
        ),
    )
    def test_epsilon(self, sigma, delta):
        ours = dpsgd_acc.gaussian(sigma).epsilon_at(delta)
        ref = _ref_epsilon(sigma, delta)
        assert ours == pytest.approx(ref, rel=1e-8), (
            f"Gaussian(σ={sigma}) eps@δ={delta}: ours={ours}, ref={ref}"
        )


class TestGaussianDelta:
    """Single Gaussian mechanism: opaque vs dp_accounting delta_at."""

    @pytest.mark.parametrize(
        "sigma",
        _slow_params([(sigma,) for sigma in [0.3, 0.5, 0.8, 1.2]], {(0.3,)}),
    )
    def test_delta(self, sigma):
        # Pick an epsilon value in a reasonable range
        eps = dpsgd_acc.gaussian(sigma).epsilon_at(1e-5) * 0.8
        ours = dpsgd_acc.gaussian(sigma).delta_at(eps)
        ref = _ref_delta(sigma, eps)
        # delta_at has slightly lower precision than epsilon_at due to
        # different internal search/interpolation paths; rel~1e-7 is expected.
        assert ours == pytest.approx(ref, rel=1e-7), (
            f"Gaussian(σ={sigma}) delta@ε={eps}: ours={ours}, ref={ref}"
        )


# ============================================================================
# 2. Poisson-subsampled Gaussian — epsilon_at vs dp_accounting
# ============================================================================


class TestPoissonEpsilon:
    """Poisson epsilon goldens and dp_accounting comparisons."""

    # Grid spans the regimes that matter for the opaque-vs-dp_accounting
    # cross-check (low/high σ, two orders of magnitude in q and in steps).
    # Adjacent middle points (q=5e-4, steps=50/500) were dropped — they sit
    # between covered points and never caught a discrepancy the endpoints
    # missed.
    @pytest.mark.parametrize(
        ("sigma", "q", "steps"),
        _slow_params(
            product([0.5, 0.8, 1.2], [0.001, 0.0001], [10, 200, 1000]),
            {
                (0.5, 0.0001, 10),
                (0.5, 0.0001, 200),
                (0.5, 0.001, 200),
                (0.8, 0.001, 200),
                (0.8, 0.0001, 200),
                (1.2, 0.001, 1000),
                (0.5, 0.001, 1000),
            },
        ),
    )
    def test_epsilon(self, sigma, q, steps):
        ours = (dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps).epsilon_at(
            1e-5
        )
        ref = _ref_epsilon(sigma, 1e-5, sampling_prob=q, steps=steps)
        expected = _POISSON_EPSILON_GOLDENS[(sigma, q, steps)]
        assert ours == pytest.approx(expected, abs=ATOL)
        assert ours == pytest.approx(ref, abs=POISSON_EPSILON_ATOL)


class TestPoissonHighSteps:
    """Long Poisson runs: golden values and dp_accounting comparisons."""

    @pytest.mark.parametrize("sigma", [0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    def test_3000_steps(self, sigma, q):
        ours = (dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * 3000).epsilon_at(1e-5)
        ref = _ref_epsilon(sigma, 1e-5, sampling_prob=q, steps=3000)
        expected = _POISSON_EPSILON_GOLDENS[(sigma, q, 3000)]
        assert ours == pytest.approx(expected, abs=ATOL)
        assert ours == pytest.approx(ref, abs=POISSON_EPSILON_ATOL)


class TestPoissonDelta:
    """Poisson delta goldens and dp_accounting comparisons."""

    @pytest.mark.parametrize(
        ("sigma", "q", "steps"),
        _slow_params(
            product([0.5, 1.2], [0.001, 0.0001], [100, 500]),
            {
                (0.5, 0.0001, 100),
                (0.5, 0.001, 100),
                (0.5, 0.0001, 500),
                (0.5, 0.001, 500),
                (1.2, 0.001, 500),
            },
        ),
    )
    def test_delta(self, sigma, q, steps):
        eps = (dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps).epsilon_at(
            1e-5
        ) * 0.8
        ours = (dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps).delta_at(eps)
        ref = _ref_delta(sigma, eps, sampling_prob=q, steps=steps)
        expected_epsilon, expected_delta = _POISSON_DELTA_GOLDENS[(sigma, q, steps)]
        assert eps == pytest.approx(expected_epsilon, abs=ATOL)
        assert ours == pytest.approx(expected_delta, abs=ATOL)
        assert ours == pytest.approx(ref, abs=POISSON_DELTA_ATOL)


# ============================================================================
# 3. Triple validation: opaque vs dp_accounting vs riskcal
# ============================================================================


class TestTripleEpsilon:
    """Poisson epsilon and advantage goldens with external comparisons."""

    @pytest.mark.parametrize(
        ("sigma", "q", "steps"),
        _slow_params(
            product([0.5, 0.8, 1.2], [0.001, 0.0001], [100, 500]),
            {
                (0.5, 0.0001, 100),
                (0.5, 0.001, 100),
                (0.5, 0.0001, 500),
                (0.8, 0.0001, 500),
                (0.8, 0.001, 500),
                (1.2, 0.001, 500),
            },
        ),
    )
    def test_triple_epsilon(self, sigma, q, steps):
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps
        eps_ours = proc.epsilon_at(1e-5)

        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        eps_ref = ref_pld.get_epsilon_for_delta(1e-5)
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)
        adv_ours = proc.advantage()

        key = (sigma, q, steps)
        assert eps_ours == pytest.approx(_POISSON_EPSILON_GOLDENS[key], abs=ATOL)
        assert adv_ours == pytest.approx(_POISSON_ADVANTAGE_GOLDENS[key], abs=ATOL)
        assert eps_ours == pytest.approx(eps_ref, abs=POISSON_EPSILON_ATOL)
        assert adv_ours == pytest.approx(adv_riskcal, abs=POISSON_ADVANTAGE_ATOL)


class TestTripleBeta:
    """Beta (Type-II error): Gaussian cross-validation and Poisson goldens."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_beta_single_gaussian(self, sigma, alpha):
        proc = dpsgd_acc.gaussian(sigma)
        beta_ours = proc.beta_at(alpha)

        ref_pld = _ref_gaussian_pld(sigma)
        beta_riskcal = float(rc_analysis.get_beta_from_pld(ref_pld, alpha=alpha))

        assert beta_ours == pytest.approx(beta_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) beta@α={alpha}: ours={beta_ours}, riskcal={beta_riskcal}"
        )

    @pytest.mark.parametrize(
        ("sigma", "q", "steps", "alpha"),
        _slow_params(
            _POISSON_BETA_GOLDENS,
            {
                (0.8, 0.001, 500, 0.01),
                (0.8, 0.001, 500, 0.1),
                (0.8, 0.0001, 500, 0.01),
                (0.8, 0.0001, 500, 0.1),
                (1.2, 0.001, 500, 0.01),
            },
        ),
    )
    def test_beta_poisson(self, sigma, q, steps, alpha):
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps
        beta_ours = proc.beta_at(alpha)
        expected = _POISSON_BETA_GOLDENS[(sigma, q, steps, alpha)]
        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        beta_riskcal = float(rc_analysis.get_beta_from_pld(ref_pld, alpha=alpha))

        assert beta_ours == pytest.approx(expected, abs=ATOL), (
            f"Poisson(G({sigma}),{q})*{steps} beta@α={alpha}: "
            f"ours={beta_ours}, expected={expected}"
        )
        assert beta_ours == pytest.approx(beta_riskcal, abs=POISSON_BETA_ATOL)


class TestTripleAdvantage:
    """Advantage goldens with riskcal comparisons."""

    @pytest.mark.parametrize(
        "sigma",
        _slow_params([(sigma,) for sigma in SIGMAS], {(0.1,)}),
    )
    def test_gaussian_advantage(self, sigma):
        proc = dpsgd_acc.gaussian(sigma)
        adv_ours = proc.advantage()

        ref_pld = _ref_gaussian_pld(sigma)
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)

        assert adv_ours == pytest.approx(adv_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) advantage: ours={adv_ours}, riskcal={adv_riskcal}"
        )

    @pytest.mark.parametrize(
        ("sigma", "q", "steps"),
        _slow_params(
            product([0.8, 1.2], [0.001, 0.0001], [100, 500]),
            {
                (0.8, 0.001, 100),
                (0.8, 0.001, 500),
                (0.8, 0.0001, 500),
                (1.2, 0.001, 500),
            },
        ),
    )
    def test_poisson_advantage(self, sigma, q, steps):
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q) * steps
        adv_ours = proc.advantage()

        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)

        expected = _POISSON_ADVANTAGE_GOLDENS[(sigma, q, steps)]
        assert adv_ours == pytest.approx(expected, abs=ATOL)
        assert adv_ours == pytest.approx(adv_riskcal, abs=POISSON_ADVANTAGE_ATOL)


class TestTripleRisk:
    """Bayes risk: opaque vs riskcal.get_bayes_risk_from_pld."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("prior", [0.3, 0.5, 0.7])
    def test_gaussian_risk(self, sigma, prior):
        proc = dpsgd_acc.gaussian(sigma)
        risk_ours = proc.risk_at(prior)

        ref_pld = _ref_gaussian_pld(sigma)
        risk_riskcal = float(rc_analysis.get_bayes_risk_from_pld(ref_pld, prior))

        assert risk_ours == pytest.approx(risk_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) risk@prior={prior}: ours={risk_ours}, riskcal={risk_riskcal}"
        )


# ============================================================================
# 4. Truncated Poisson — validity checks (no dp_accounting equivalent)
# ============================================================================


class TestTruncatedPoissonValidity:
    """Truncated Poisson: internal consistency and fallback behavior.

    Note: Truncated Poisson uses the asymmetric mixture from Ganesh (2025),
    Theorem 3.1 (https://arxiv.org/abs/2508.15089), with a
    doubled-sensitivity component, so it does NOT guarantee lower epsilon
    than standard Poisson for the same sample_rate.
    The benefit is a more accurate model of real production systems.
    """

    @pytest.mark.parametrize(
        "sigma",
        _slow_params([(sigma,) for sigma in [0.5, 0.8, 1.2]], {(0.5,), (0.8,)}),
    )
    def test_larger_cap_lower_epsilon(self, sigma):
        """Larger truncated_batch_size → less truncation → lower epsilon."""
        n = 100_000
        q = 0.001
        steps = 500
        g = dpsgd_acc.gaussian(sigma)
        # cap=50 (heavy truncation: expected_batch=100, cap < expected)
        eps_small_cap = (
            dpsgd_acc.poisson(g, q, truncated_batch_size=50, dataset_size=n) * steps
        ).epsilon_at(1e-5)
        # cap=500 (light truncation: expected_batch=100, cap >> expected)
        eps_large_cap = (
            dpsgd_acc.poisson(g, q, truncated_batch_size=500, dataset_size=n) * steps
        ).epsilon_at(1e-5)
        assert eps_large_cap <= eps_small_cap + 1e-10, (
            f"Larger cap should give ≤ epsilon: cap=500 → {eps_large_cap}, cap=50 → {eps_small_cap}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_monotone_in_steps(self, sigma):
        """More steps → higher epsilon."""
        g = dpsgd_acc.gaussian(sigma)
        epsilons = []
        for steps in [10, 100, 500, 1000]:
            eps = (
                dpsgd_acc.poisson(
                    g, 0.005, truncated_batch_size=250, dataset_size=50_000
                )
                * steps
            ).epsilon_at(1e-5)
            epsilons.append(eps)
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i + 1]

    def test_positive_epsilon(self):
        """Truncated poisson epsilon > 0 for non-trivial mechanism."""
        g = dpsgd_acc.gaussian(0.8)
        proc = (
            dpsgd_acc.poisson(g, 0.005, truncated_batch_size=250, dataset_size=50_000)
            * 100
        )
        assert proc.epsilon_at(1e-5) > 0

    @pytest.mark.slow
    def test_fallback_when_no_truncation(self):
        """When truncated_batch_size >> expected batch, result ≈ standard Poisson."""
        g = dpsgd_acc.gaussian(0.8)
        steps = 500
        eps_trunc = (
            dpsgd_acc.poisson(
                g, 0.001, truncated_batch_size=100_000, dataset_size=100_000
            )
            * steps
        ).epsilon_at(1e-5)
        eps_poisson = (dpsgd_acc.poisson(g, 0.001) * steps).epsilon_at(1e-5)
        assert eps_trunc == pytest.approx(eps_poisson, rel=1e-6)


# ============================================================================
# 5. Parallel Poisson — cross-validation
# ============================================================================


class TestParallelPoissonCrossValidation:
    """ParallelPoisson mechanism: verify native worker-count behavior."""

    @pytest.mark.slow
    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0005])
    def test_parallel_poisson_matches_native_worker_behavior(self, sigma, q):
        """One worker is Poisson; additional workers strictly increase epsilon."""
        inner = dpsgd_acc.gaussian(sigma)
        eps_poisson = (dpsgd_acc.poisson(inner, q) * 500).epsilon_at(1e-5)
        eps_by_workers = {
            num_workers: (
                dpsgd_acc.parallel_poisson(inner, q, num_workers) * 500
            ).epsilon_at(1e-5)
            for num_workers in (1, 2, 4, 8)
        }

        assert eps_by_workers[1] == pytest.approx(eps_poisson, abs=1e-6)
        assert (
            eps_by_workers[1]
            < eps_by_workers[2]
            < eps_by_workers[4]
            < eps_by_workers[8]
        )

    @pytest.mark.slow
    def test_parallel_poisson_matches_pinned_reference(self):
        """Worker-count privacy loss matches the native reference calculation."""
        inner = dpsgd_acc.gaussian(0.8)
        epsilons = tuple(
            (dpsgd_acc.parallel_poisson(inner, 0.001, num_workers) * 500).epsilon_at(
                1e-5
            )
            for num_workers in (1, 2, 4, 8)
        )

        assert epsilons == pytest.approx(
            (
                0.2398344201560575,
                0.5325659019424767,
                1.124570816812394,
                2.255585330074185,
            ),
            rel=1e-6,
        )


# ============================================================================
# 6. AdaClip — cross-validation
# ============================================================================


class TestAdaClipCrossValidation:
    """AdaClip: verify adaclip_sensitivity formula matches expected z_eff."""

    @pytest.mark.parametrize(
        ("sigma", "batch_size"),
        [
            (1.0, 1000),
            (1.1, 1000),
            (0.5, 200),
            (1.2, 2000),
        ],
    )
    def test_adaclip_effective_noise(self, sigma, batch_size):
        from opaque.api.accounting.core import _native

        proc = dpsgd_acc.adaclip(
            dpsgd_acc.gaussian(sigma), expected_batch_size=batch_size
        )
        sigma_b = batch_size * 0.05
        s = _native.adaclip_sensitivity(sigma, sigma_b)
        z_eff = 1.0 / s

        # Verify effective noise via PLD
        config = get_discretization()
        ref = _native.gaussian_pld(z_eff, config.to_native())
        assert proc.epsilon_at(1e-5) == pytest.approx(ref.epsilon_at(1e-5), rel=1e-12)

        from opaque.dpsgd.accounting.mechanisms.types import AdaClip

        assert isinstance(proc, AdaClip)

    @pytest.mark.parametrize("batch_size", [200, 1000, 2000])
    def test_adaclip_increases_privacy_cost(self, batch_size):
        g = dpsgd_acc.gaussian(1.0)
        a = dpsgd_acc.adaclip(g, expected_batch_size=batch_size)
        assert a.epsilon_at(1e-5) > g.epsilon_at(1e-5)

    def test_adaclip_composed_with_poisson(self):
        """AdaClip result composes with poisson() normally."""
        step = (
            dpsgd_acc.poisson(
                dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.1), expected_batch_size=1000),
                0.01,
            )
            * 1000
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


# ============================================================================
# 7. Metrics consistency — internal invariants
# ============================================================================


class TestMetricsConsistency:
    """Verify internal consistency of privacy metrics (roundtrips, identities)."""

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.2])
    def test_epsilon_delta_roundtrip(self, sigma):
        """epsilon_at(δ) -> delta_at(ε) ≈ δ."""
        proc = dpsgd_acc.gaussian(sigma)
        delta = 1e-5
        eps = proc.epsilon_at(delta)
        delta_back = proc.delta_at(eps)
        assert delta_back == pytest.approx(delta, abs=ATOL), (
            f"Roundtrip failed: δ={delta} → ε={eps} → δ'={delta_back}"
        )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.2])
    def test_advantage_equals_delta_at_zero(self, sigma):
        """advantage() == delta_at(0) (by definition of f-DP advantage)."""
        proc = dpsgd_acc.gaussian(sigma)
        adv = proc.advantage()
        d0 = proc.delta_at(0.0)
        assert adv == pytest.approx(d0, abs=1e-10), (
            f"advantage={adv} != delta_at(0)={d0}"
        )

    @pytest.mark.parametrize(
        "sigma",
        _slow_params([(sigma,) for sigma in [0.5, 0.8, 1.2]], {(1.2,)}),
    )
    def test_advantage_poisson_equals_delta_at_zero(self, sigma):
        """advantage() == delta_at(0) for Poisson-subsampled too."""
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), 0.01) * 500
        adv = proc.advantage()
        d0 = proc.delta_at(0.0)
        assert adv == pytest.approx(d0, abs=1e-10)

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_risk_bounds(self, sigma):
        """Risk must be in [0, min(prior, 1-prior)] for optimal adversary."""
        proc = dpsgd_acc.gaussian(sigma)
        for prior in [0.3, 0.5, 0.7]:
            r = proc.risk_at(prior)
            assert 0 <= r <= min(prior, 1 - prior) + 1e-10, (
                f"Risk out of bounds: risk={r}, prior={prior}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_risk_symmetry(self, sigma):
        """risk_at(p) == risk_at(1-p) for symmetric mechanisms."""
        proc = dpsgd_acc.gaussian(sigma)
        for prior in [0.2, 0.3, 0.4]:
            r1 = proc.risk_at(prior)
            r2 = proc.risk_at(1.0 - prior)
            assert r1 == pytest.approx(r2, abs=1e-10), (
                f"Risk asymmetry: risk({prior})={r1}, risk({1 - prior})={r2}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_beta_monotone_in_alpha(self, sigma):
        """beta_at(α) must be non-increasing in α."""
        proc = dpsgd_acc.gaussian(sigma)
        alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        betas = [proc.beta_at(a) for a in alphas]
        for i in range(len(betas) - 1):
            assert betas[i] >= betas[i + 1] - 1e-10, (
                f"Beta not monotone: β({alphas[i]})={betas[i]} > β({alphas[i + 1]})={betas[i + 1]}"
            )

    def test_identity_zero_epsilon(self):
        """Identity mechanism should have ε≈0 for any δ."""
        proc = acc.identity()
        assert proc.epsilon_at(1e-5) == pytest.approx(0.0, abs=1e-8)
        assert proc.epsilon_at(1e-10) == pytest.approx(0.0, abs=1e-8)


# ============================================================================
# 8. Composition cross-validation
# ============================================================================


class TestCompositionCrossValidation:
    """Verify composed mechanisms match dp_accounting composition."""

    @pytest.mark.slow
    @pytest.mark.slow
    def test_heterogeneous_compose(self):
        """Compose different mechanisms: (G(0.5)*100 | G(1.0)*200)."""
        # opaque
        proc = (dpsgd_acc.gaussian(0.5) * 100) | (dpsgd_acc.gaussian(1.0) * 200)
        eps_ours = proc.epsilon_at(1e-5)

        # dp_accounting
        pld_a = _ref_gaussian_pld(0.5).self_compose(100)
        pld_b = _ref_gaussian_pld(1.0).self_compose(200)
        pld_ref = pld_a.compose(pld_b)
        eps_ref = pld_ref.get_epsilon_for_delta(1e-5)

        # For large epsilon (~400), use relative tolerance
        assert eps_ours == pytest.approx(eps_ref, rel=1e-6)

    @pytest.mark.slow
    def test_compose_poisson_different_rates(self):
        """Compose Poisson steps with different sample rates."""
        # opaque
        p1 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.001) * 500
        p2 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.0005) * 500
        proc = p1 | p2
        eps_ours = proc.epsilon_at(1e-5)

        # dp_accounting
        pld1 = _ref_gaussian_pld(0.8, 0.001).self_compose(500)
        pld2 = _ref_gaussian_pld(0.8, 0.0005).self_compose(500)
        pld_ref = pld1.compose(pld2)
        eps_ref = pld_ref.get_epsilon_for_delta(1e-5)

        assert eps_ours == pytest.approx(0.2489660893634798, abs=ATOL)
        assert eps_ours == pytest.approx(eps_ref, abs=POISSON_EPSILON_ATOL)

    def test_compose_same_mechanism(self):
        """Compose same Poisson steps — should equal repeat."""
        proc_compose = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8), 0.001
        ) | dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.001)
        proc_repeat = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.001) * 2
        eps_compose = proc_compose.epsilon_at(1e-5)
        eps_repeat = proc_repeat.epsilon_at(1e-5)
        assert eps_compose == pytest.approx(eps_repeat, abs=1e-10)


# ============================================================================
# 9. Calibration cross-validation
# ============================================================================


class TestCalibrationCrossValidation:
    """Verify calibration results match dp_accounting reference."""

    @pytest.mark.slow
    def test_calibrate_epsilon_roundtrip(self):
        """Calibrated noise → check epsilon against reference."""
        target_eps = 5.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        result = cal.calibrate(
            cal.epsilon_budget(target_eps, delta=delta),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), q) * steps,
            0.1,
            1.2,
            tolerance=1e-4,
        )
        assert result.converged

        # Verify with dp_accounting
        ref_eps = _ref_epsilon(result.param, delta, sampling_prob=q, steps=steps)
        assert ref_eps == pytest.approx(target_eps, abs=1e-3)

    @pytest.mark.slow
    def test_calibrate_advantage_roundtrip(self):
        """Calibrated noise → check advantage against riskcal."""
        result = cal.calibrate(
            cal.advantage_budget(0.15),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 500,
            0.3,
            1.2,
            tolerance=1e-4,
        )
        assert result.converged

        # Verify with riskcal
        ref_pld = _ref_gaussian_pld(result.param, 0.01).self_compose(500)
        adv_ref = rc_analysis.get_advantage_from_pld(ref_pld)
        assert adv_ref == pytest.approx(0.15, abs=1e-3)


# ============================================================================
# 10. Numerical stability regressions
# ============================================================================


class TestNumericalStability:
    """Edge cases and regression tests for numerical stability."""

    def test_smallest_sigma(self):
        """σ=0.1 (boundary, high privacy loss) — should compute without overflow."""
        proc = dpsgd_acc.gaussian(0.1)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 50  # very high epsilon expected

    def test_largest_sigma(self):
        """σ=1.2 (boundary, low privacy loss) — should compute without underflow."""
        proc = dpsgd_acc.gaussian(1.2)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps < 5  # relatively low epsilon

    def test_very_small_delta(self):
        """δ=1e-10 — tight delta should work."""
        proc = dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-10)
        assert math.isfinite(eps)
        assert eps > proc.epsilon_at(1e-5)  # tighter delta → higher epsilon

    def test_very_small_sample_rate(self):
        """q=1e-5 — very small batches."""
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), 1e-5) * 1000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_many_steps(self):
        """3000 steps — verify no accumulation of error."""
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.001) * 3000
        eps_ours = proc.epsilon_at(1e-5)
        eps_ref = _ref_epsilon(0.8, 1e-5, sampling_prob=0.001, steps=3000)
        assert eps_ours == pytest.approx(
            _POISSON_EPSILON_GOLDENS[(0.8, 0.001, 3000)],
            abs=ATOL,
        )
        assert eps_ours == pytest.approx(eps_ref, abs=POISSON_EPSILON_ATOL)

    def test_compose_many_steps(self):
        """Build up 1000 steps via loop composition vs single repeat."""
        g = dpsgd_acc.gaussian(0.8)
        step = dpsgd_acc.poisson(g, 0.001)
        repeated = step * 1000
        eps_repeated = repeated.epsilon_at(1e-5)

        # Same via composition loop (small)
        composed = step | step | step | step | step  # 5 steps
        composed = composed * 200  # 1000 steps
        eps_composed = composed.epsilon_at(1e-5)

        assert eps_repeated == pytest.approx(eps_composed, abs=ATOL)

    @pytest.mark.slow
    def test_epsilon_monotone_in_steps(self):
        """More steps → weakly higher epsilon (monotonicity)."""
        g = dpsgd_acc.gaussian(0.8)
        epsilons = []
        for steps in [1, 10, 50, 100, 500, 1000]:
            proc = dpsgd_acc.poisson(g, 0.001) * steps
            epsilons.append(proc.epsilon_at(1e-5))
        for i in range(len(epsilons) - 1):
            assert epsilons[i] <= epsilons[i + 1] + 1e-10

    def test_epsilon_monotone_in_noise(self):
        """More noise → lower epsilon (monotonicity)."""
        sigmas = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
        epsilons = [dpsgd_acc.gaussian(s).epsilon_at(1e-5) for s in sigmas]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i + 1] - 1e-10
