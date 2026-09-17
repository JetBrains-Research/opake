"""Integration tests for privacy auditing workflow."""

import numpy as np

from opake.auditing import one_run
from opake.auditing.types import CanaryScores, CoinFlip
from opake.random import key


def _make_estimate(in_scores, out_scores):
    """Helper: build a OneRunEstimate from raw in/out score arrays."""
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in = len(in_scores)
    n_out = len(out_scores)
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


def test_basic_audit_workflow():
    """Test basic privacy auditing workflow."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=100)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=100)
    result = _make_estimate(in_scores, out_scores)

    # Test epsilon estimation with explicit threshold
    eps = result.eps_delta().epsilon_at(significance=0.05, delta=1e-5, threshold=4.0)
    assert eps > 0, "Should detect privacy leakage"

    # Test epsilon with the threshold-free (Pareto-argmax) path
    eps_threshold_free = result.eps_delta().epsilon_at(significance=0.05, delta=1e-5)
    assert eps_threshold_free > 0, "Threshold-free estimation should detect leakage"

    # Test utility metrics
    assert 0.5 < result.attack_auc() < 1.0, "AUC should be above random"
    assert result.attack_beta_at(alpha=0.05) < 0.95, (
        "Beta should be below 1 for detectable leakage"
    )


def test_audit_with_auc_ci():
    """Test auditing with AUC confidence intervals."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=50)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=50)

    result = _make_estimate(in_scores, out_scores)

    auc_ci = result.attack_auc(confidence=0.95, num_samples=20, key=key(42))
    assert len(auc_ci) == 2
    assert auc_ci[0] <= auc_ci[1]


def test_no_privacy_leakage():
    """Test auditing when there's no privacy leakage."""
    np.random.seed(42)
    scores = np.random.normal(loc=3.0, scale=1.0, size=100)
    result = _make_estimate(scores[:50], scores[50:])

    eps = result.eps_delta().epsilon_at(significance=0.05, delta=0)
    assert eps < 0.5, "Should detect minimal leakage"

    assert 0.4 < result.attack_auc() < 0.6, "AUC should be near random"


def test_perfect_attack():
    """Test auditing with perfect attack separation."""
    result = _make_estimate(
        np.arange(50, 100, dtype=float), np.arange(0, 50, dtype=float)
    )

    assert result.attack_auc() > 0.99, "Perfect attack should have AUC ~1.0"

    eps = result.eps_delta().epsilon_at(significance=0.05, delta=0, threshold=50)
    assert eps > 2.5, "Perfect attack should give large epsilon"


def test_real_world_scenario():
    """Test realistic privacy auditing scenario."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=0.6, scale=0.3, size=500)
    out_scores = np.random.normal(loc=0.4, scale=0.3, size=500)
    result = _make_estimate(in_scores, out_scores)

    eps = result.eps_delta().epsilon_at(delta=1e-5)
    assert eps > 0, "Should detect some privacy leakage"

    assert 0.5 < result.attack_auc() < 0.85, "AUC should show modest attack"

    beta_at_1pct = result.attack_beta_at(alpha=0.01)
    assert beta_at_1pct > 0.7, (
        "Beta should be high at low alpha (weak attack at strict threshold)"
    )


def test_one_run_audit():
    """Test one-run privacy auditing method."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=100)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=100)
    result = _make_estimate(in_scores, out_scores)

    eps = result.eps_delta().epsilon_at(significance=0.05, delta=1e-5)
    assert eps > 0, "Should detect privacy leakage"


def test_all_metrics_on_single_result():
    """Test that all metrics work on a single OneRunEstimate instance."""
    np.random.seed(42)
    result = _make_estimate(
        np.random.normal(loc=5.0, scale=1.0, size=100),
        np.random.normal(loc=3.0, scale=1.0, size=100),
    )

    # Epsilon
    assert result.eps_delta().epsilon_at(significance=0.05, delta=1e-5) > 0
    assert result.eps_delta().epsilon_at(delta=1e-5) > 0

    # All utility metrics
    assert 0.5 < result.attack_auc() < 1.0
    assert result.attack_beta_at(alpha=0.05) <= 1.0
