"""Tests for opake.accounting.composition — nodes + algebra."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.api.accounting.core._base import DpProcess
from opake.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)

# ── Node dataclass tests ─────────────────────────────────────────────


class TestComposed:
    """Composed node — heterogeneous composition."""

    def test_fields(self):
        a = dpsgd_acc.gaussian(0.8)
        b = dpsgd_acc.gaussian(0.5)
        c = Composed(a, b)
        assert c.left is a
        assert c.right is b

    def test_frozen(self):
        c = Composed(dpsgd_acc.gaussian(0.8), dpsgd_acc.gaussian(0.5))
        with pytest.raises(FrozenInstanceError):
            c.left = dpsgd_acc.gaussian(1.0)  # type: ignore[misc]

    def test_is_dp_process(self):
        c = Composed(dpsgd_acc.gaussian(0.8), dpsgd_acc.gaussian(0.5))
        assert isinstance(c, DpProcess)

    def test_pld_composes(self):
        """PLD should compose left and right."""
        a = dpsgd_acc.gaussian(0.8)
        b = dpsgd_acc.gaussian(0.5)
        composed = Composed(a, b)
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0
        # eps of composition > eps of each part
        assert eps > a.epsilon_at(1e-5)
        assert eps > b.epsilon_at(1e-5)


class TestRepeated:
    """Repeated node — homogeneous k-fold composition."""

    def test_fields(self):
        step = dpsgd_acc.gaussian(0.8)
        r = Repeated(step, 100)
        assert r.inner is step
        assert r.count == 100

    def test_frozen(self):
        r = Repeated(dpsgd_acc.gaussian(0.8), 100)
        with pytest.raises(FrozenInstanceError):
            r.count = 200  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Repeated(dpsgd_acc.gaussian(0.8), 10), DpProcess)

    def test_leaf_and_count(self):
        step = dpsgd_acc.gaussian(0.8)
        r = Repeated(step, 100)
        leaf, count = r._leaf_and_count()
        assert leaf is step
        assert count == 100

    def test_pld_self_composes(self):
        step = dpsgd_acc.gaussian(0.8)
        r = Repeated(step, 10)
        eps = r.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0
        # 10x composition > single step
        assert eps > step.epsilon_at(1e-5)


class TestCachedProcess:
    """CachedProcess — caching wrapper."""

    def test_identity_equality(self):
        """Two CachedProcesses with same inner are equal."""
        inner = dpsgd_acc.gaussian(0.8)
        a = CachedProcess(inner)
        b = CachedProcess(inner)
        assert a == b

    def test_frozen(self):
        cp = CachedProcess(dpsgd_acc.gaussian(0.8))
        with pytest.raises(FrozenInstanceError):
            cp.inner = dpsgd_acc.gaussian(1.0)  # type: ignore[misc]

    def test_caches_pld(self):
        """Second pld() call returns cached result."""
        cp = CachedProcess(dpsgd_acc.gaussian(0.8))
        pld1 = cp.pld()
        pld2 = cp.pld()
        assert pld1 is pld2

    def test_pld_returns_valid(self):
        cp = CachedProcess(dpsgd_acc.gaussian(0.8))
        eps = cp.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_opake_merge_barrier(self):
        """CachedProcess merges equal inner via structural equality."""
        step = dpsgd_acc.gaussian(0.8)
        cached_a = CachedProcess(step)
        cached_b = CachedProcess(step)
        result = cached_a | cached_b
        assert isinstance(result, Repeated)
        assert result.count == 2

    def test_repr(self):
        cp = CachedProcess(dpsgd_acc.gaussian(0.8))
        assert "CachedProcess" in repr(cp)


class TestCachedFunction:
    """acc.cached() convenience function."""

    def test_wraps_process(self):
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01) * 100
        cached_step = acc.cached(step)
        assert isinstance(cached_step, CachedProcess)

    def test_idempotent(self):
        """cached(cached(x)) returns the same CachedProcess."""
        step = dpsgd_acc.gaussian(0.8)
        c1 = acc.cached(step)
        c2 = acc.cached(c1)
        assert c2 is c1


# ── Composition algebra tests ────────────────────────────────────────


class TestRepeatOperator:
    """step * k and k * step produce Repeated."""

    def test_mul_returns_repeated(self):
        step = dpsgd_acc.gaussian(0.8)
        result = step * 100
        assert isinstance(result, Repeated)
        assert result.count == 100

    def test_rmul_returns_repeated(self):
        step = dpsgd_acc.gaussian(0.8)
        result = 100 * step
        assert isinstance(result, Repeated)
        assert result.count == 100

    def test_mul_and_rmul_agree(self):
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        a = step * 100
        b = 100 * step
        assert a.epsilon_at(1e-5) == pytest.approx(b.epsilon_at(1e-5))

    def test_nested_repeat_flattens(self):
        """(step * 3) * 4 → Repeated(step, 12)."""
        step = dpsgd_acc.gaussian(0.8)
        nested = (step * 3) * 4
        assert isinstance(nested, Repeated)
        assert nested.count == 12
        assert nested.inner is step

    def test_repeat_single(self):
        step = dpsgd_acc.gaussian(0.8)
        r = step * 1
        assert isinstance(r, Repeated)
        assert r.count == 1


class TestComposeOperator:
    """a | b produces Composed (with optimizations)."""

    def test_or_returns_composed(self):
        a = dpsgd_acc.gaussian(0.8)
        b = dpsgd_acc.gaussian(0.5)
        result = a | b
        assert isinstance(result, Composed)

    def test_identity_elision_left(self):
        step = dpsgd_acc.gaussian(0.8)
        result = acc.identity() | step
        assert result is step

    def test_identity_elision_right(self):
        step = dpsgd_acc.gaussian(0.8)
        result = step | acc.identity()
        assert result is step

    def test_direct_merge_same(self):
        """a | a → Repeated(a, 2)."""
        step = dpsgd_acc.gaussian(0.8)
        result = step | step
        assert isinstance(result, Repeated)
        assert result.count == 2

    def test_merge_repeated_same(self):
        """a * n | a * m → Repeated(a, n + m)."""
        step = dpsgd_acc.gaussian(0.8)
        result = (step * 3) | (step * 4)
        assert isinstance(result, Repeated)
        assert result.count == 7

    def test_merge_repeated_left_single_right(self):
        """a * n | a → Repeated(a, n + 1)."""
        step = dpsgd_acc.gaussian(0.8)
        result = (step * 5) | step
        assert isinstance(result, Repeated)
        assert result.count == 6

    def test_merge_single_left_repeated_right(self):
        """a | a * n → Repeated(a, n + 1)."""
        step = dpsgd_acc.gaussian(0.8)
        result = step | (step * 5)
        assert isinstance(result, Repeated)
        assert result.count == 6

    def test_right_spine_merge(self):
        """(X | a * n) | a * m → Composed(X, Repeated(a, n + m))."""
        a = dpsgd_acc.gaussian(0.8)
        x = dpsgd_acc.gaussian(0.5)
        result = (x | (a * 3)) | (a * 4)
        assert isinstance(result, Composed)
        assert isinstance(result.right, Repeated)
        assert result.right.count == 7

    def test_no_merge_different_leaves(self):
        """Different leaves → plain Composed, no merge."""
        a = dpsgd_acc.gaussian(0.8)
        b = dpsgd_acc.gaussian(0.5)
        result = a | b
        assert isinstance(result, Composed)
        assert result.left is a
        assert result.right is b


class TestComposeFunctions:
    """acc.repeat() and acc.compose() match operators."""

    def test_repeat_matches_mul(self):
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        via_op = (step * 100).epsilon_at(1e-5)
        via_fn = acc.repeat(step, 100).epsilon_at(1e-5)
        assert via_op == pytest.approx(via_fn)

    def test_compose_matches_or(self):
        a = dpsgd_acc.gaussian(0.8)
        b = dpsgd_acc.gaussian(0.5)
        via_op = (a | b).epsilon_at(1e-5)
        via_fn = acc.compose(a, b).epsilon_at(1e-5)
        assert via_op == pytest.approx(via_fn)


class TestCompositionProperties:
    """Privacy properties of composition."""

    def test_epsilon_increases_with_composition(self):
        """More steps → higher epsilon."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        deltas = [1e-5]
        step_counts = [1, 10, 100, 500]
        for d in deltas:
            epsilons = [(step * k).epsilon_at(d) for k in step_counts]
            for i in range(1, len(epsilons)):
                assert epsilons[i] > epsilons[i - 1]

    @pytest.mark.slow
    def test_epsilon_decreases_with_noise(self):
        """More noise → lower epsilon."""
        sigmas = [0.3, 0.5, 0.8, 1.2]
        epsilons = [
            (dpsgd_acc.poisson(dpsgd_acc.gaussian(s), 0.01) * 100).epsilon_at(1e-5)
            for s in sigmas
        ]
        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1]

    def test_lower_sample_rate_better_privacy(self):
        """Lower q → lower epsilon (privacy amplification)."""
        rates = [0.01, 0.001, 0.0001]
        epsilons = [
            (dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), q) * 100).epsilon_at(1e-5)
            for q in rates
        ]
        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1]

    def test_sublinear_composition_growth(self):
        """10x more steps → < 10x more epsilon (sublinear composition)."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)
        eps_1k = (step * 1000).epsilon_at(1e-5)
        eps_10k = (step * 10000).epsilon_at(1e-5)
        growth = eps_10k / eps_1k
        assert 1.5 < growth < 10.0

    @pytest.mark.slow
    def test_heterogeneous_composition(self):
        """Heterogeneous composition ε > max single phase ε."""
        phase1 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01) * 100
        phase2 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01) * 100
        combined = phase1 | phase2
        eps_combined = combined.epsilon_at(1e-5)
        eps1 = phase1.epsilon_at(1e-5)
        eps2 = phase2.epsilon_at(1e-5)
        assert eps_combined > max(eps1, eps2)

    def test_accumulate_in_loop(self):
        """Simulates training loop: repeatedly composing same step merges."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        training = acc.identity()
        for _ in range(100):
            training = training | step
        # After identity elision + merge, should be Repeated(step, 100)
        assert isinstance(training, Repeated)
        assert training.count == 100
