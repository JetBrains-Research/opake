"""Tests for auditing ROC helpers (one_run/roc.py)."""

import numpy as np
import pytest

from opake.api.auditing.one_run._roc import (
    get_tn_fn_counts,
    pareto_frontier,
    tpr_at_given_fpr,
)


class TestParetoFrontier:
    def test_simple_frontier(self):
        points = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
        indices = pareto_frontier(points)
        expected = np.array([0, 1, 3])
        np.testing.assert_array_equal(indices, expected)

    def test_all_on_line(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        indices = pareto_frontier(points)
        expected = np.array([0, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_only_two_points(self):
        points = np.array([[0, 0], [1, 1]])
        indices = pareto_frontier(points)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(indices, expected)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Expected at least two 2D points"):
            pareto_frontier(np.array([[0, 0, 0]]))

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="Expected points to be sorted"):
            pareto_frontier(np.array([[1, 1], [0, 0]]))


class TestGetTnFnCounts:
    def test_perfect_separation(self):
        """The ideal corner (FN=0, TN=n_out) is on the threshold grid exactly once."""
        in_scores, out_scores = [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]
        thresholds, tn, fn = get_tn_fn_counts(in_scores, out_scores)
        grid = f"thresholds={thresholds} tn={tn} fn={fn}"

        # Only a threshold in (max(out), min(in)] classifies every canary correctly.
        corner = np.flatnonzero((fn == 0) & (tn == len(out_scores)))
        assert corner.size == 1, grid
        assert thresholds[corner[0]] == min(in_scores), grid

        np.testing.assert_array_equal(
            thresholds, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf]
        )
        np.testing.assert_array_equal(tn, [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5])
        np.testing.assert_array_equal(fn, [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

    def test_perfect_separation_hull(self):
        """The frontier of a separated ROC is the ideal corner plus both endpoints."""
        thresholds, tn, fn = get_tn_fn_counts(
            [5, 6, 7, 8, 9], [0, 1, 2, 3, 4], hull=True
        )
        np.testing.assert_array_equal(thresholds, [0, 5, np.inf])
        np.testing.assert_array_equal(tn, [0, 5, 5])
        np.testing.assert_array_equal(fn, [0, 0, 5])

    def test_complete_overlap(self):
        _thresholds, tn, fn = get_tn_fn_counts([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(tn, fn)

    def test_empty_out_scores(self):
        _thresholds, tn, _fn = get_tn_fn_counts([1, 2, 3], [])
        assert np.all(tn == 0)

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            get_tn_fn_counts([], [])


class TestTprAtGivenFpr:
    def test_perfect_classifier(self):
        """A separated ROC attains TPR=1 at FPR=0."""
        _thresholds, tn, fn = get_tn_fn_counts([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
        tp_counts = (fn[-1] - fn)[::-1]
        fp_counts = (tn[-1] - tn)[::-1]
        assert tpr_at_given_fpr(0.0, tp_counts, fp_counts) == 1.0

    def test_degenerate_roc(self):
        """A two-point ROC without separation has no true positives at FPR=0."""
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 1])
        assert tpr_at_given_fpr(0.0, tp_counts, fp_counts) == 0.0

    def test_random_classifier(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        tpr = tpr_at_given_fpr(0.5, tp_counts, fp_counts)
        assert np.isclose(tpr, 0.5)

    def test_vectorized_fpr(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        fprs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tprs = tpr_at_given_fpr(fprs, tp_counts, fp_counts)
        assert len(tprs) == len(fprs)
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 100])
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(-0.1, tp_counts, fp_counts)
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(1.5, tp_counts, fp_counts)


def test_raw_auc_is_unbiased_under_null():
    """#378: raw-ROC AUC recovers ~0.5 under the null; the hull basis is biased high."""
    from opake.api.auditing.one_run._estimate import _auc_from_counts

    rng = np.random.default_rng(0)
    n, trials = 64, 300
    raw, hull = [], []
    for _ in range(trials):
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        _, tn, fn = get_tn_fn_counts(a, b)
        raw.append(_auc_from_counts(tn, fn))
        _, htn, hfn = get_tn_fn_counts(a, b, hull=True)
        hull.append(_auc_from_counts(htn, hfn))
    raw_mean, hull_mean = float(np.mean(raw)), float(np.mean(hull))
    assert abs(raw_mean - 0.5) < 0.02, raw_mean
    assert hull_mean > raw_mean + 0.01, (raw_mean, hull_mean)


def test_infinite_scores_counted_in_denominators():
    """#378: +inf scores are included in the n_in / n_out ROC denominators."""
    _, tn, fn = get_tn_fn_counts([1.0, 2.0, np.inf], [0.5, 1.5, 2.5])
    assert fn[-1] == 3  # n_in (includes the +inf in-score), not the finite-only 2
    assert tn[-1] == 3  # n_out


def test_tpr_at_alpha_one_is_one_not_nan():
    """#378 review: repeated terminal FP counts on the raw ROC made ``alpha=1``
    interpolate a zero-width segment to NaN; the endpoint must be TPR=1 (β=0)."""
    _thresholds, tn, fn = get_tn_fn_counts([0.0, 2.0], [1.0])
    tp = (fn[-1] - fn)[::-1]
    fp = (tn[-1] - tn)[::-1]
    assert tpr_at_given_fpr(1.0, tp, fp) == 1.0
    arr = tpr_at_given_fpr(np.array([0.0, 0.5, 1.0]), tp, fp)
    assert not np.isnan(arr).any()
    assert arr[-1] == 1.0
