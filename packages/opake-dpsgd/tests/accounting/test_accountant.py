"""Tests for the Accountant class.

Covers construction, metrics, budget tracking, functional properties,
realistic training-loop patterns, calibration integration, and API consistency.
"""

import pytest

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.api.accounting.core._accountant import Accountant
from opake.api.accounting.core.calibration import epsilon_budget

# ============================================================================
# Construction & composition
# ============================================================================


class TestAccountantBasics:
    """Test basic Accountant construction and composition."""

    def test_init_default(self):
        """Accountant initializes with identity process."""
        acct = Accountant()

        # Should have zero privacy cost (identity process)
        eps = acct.epsilon_at(1e-5)
        assert eps < 1e-10

    def test_init_with_budget(self):
        """Accountant accepts optional budget."""
        budget = epsilon_budget(3.0, delta=1e-5)
        acct = Accountant(budget=budget)
        assert acct._budget is not None

    def test_composition_via_or(self):
        """Composing processes via | returns new Accountant."""
        acct1 = Accountant()
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)

        acct2 = acct1 | step

        # acct2 is a different object
        assert acct2 is not acct1

        # acct2 has accumulated privacy
        eps1 = acct1.epsilon_at(1e-5)
        eps2 = acct2.epsilon_at(1e-5)
        assert eps2 > eps1

    def test_composition_returns_accountant(self):
        """Composition always returns Accountant instance."""
        acct = Accountant()
        result = acct | dpsgd_acc.gaussian(1.0)
        assert isinstance(result, Accountant)


class TestAccountantPrefix:
    """Test seeding an Accountant with an already-executed process."""

    def test_prefix_seeds_process(self):
        """Accountant(prefix=p) starts at p's privacy cost."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100
        acct = Accountant(prefix=step)
        assert acct.process is step
        assert acct.epsilon_at(1e-5) == step.epsilon_at(1e-5)

    def test_prefix_default_is_identity(self):
        """Without a prefix the accountant still starts at zero cost."""
        assert Accountant().epsilon_at(1e-5) < 1e-10

    def test_prefix_equivalent_to_composing(self):
        """Accountant(prefix=p) | q matches Accountant() | p | q."""
        sft = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 50
        dpo_step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.02)

        seeded = Accountant(prefix=sft)
        plain = Accountant() | sft
        for _ in range(10):
            seeded = seeded | dpo_step
            plain = plain | dpo_step

        assert abs(seeded.epsilon_at(1e-5) - plain.epsilon_at(1e-5)) < 1e-10

    def test_prefix_counts_against_budget(self):
        """budget_exceeded accounts for the prefix."""
        budget = epsilon_budget(0.1, delta=1e-5)
        prefix = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), 0.01) * 1000
        acct = Accountant(budget=budget, prefix=prefix)
        assert acct.budget_exceeded

    def test_prefix_from_executed_accountant(self):
        """A finished run's process seeds a new accountant exactly."""
        sft = Accountant() | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100)

        dpo = Accountant(prefix=sft.process)
        assert dpo.epsilon_at(1e-5) == sft.epsilon_at(1e-5)

    def test_prefix_survives_serialization(self):
        """state_dict round-trip preserves a prefixed process tree."""
        from opake.serialization import from_state_dict, state_dict

        prefix = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100
        acct = Accountant(prefix=prefix) | dpsgd_acc.gaussian(0.5)

        restored = from_state_dict(Accountant(), state_dict(acct))
        assert restored.epsilon_at(1e-5) == acct.epsilon_at(1e-5)


# ============================================================================
# Metric queries
# ============================================================================


class TestAccountantMetrics:
    """Test privacy metric queries on Accountant."""

    def test_epsilon_at(self):
        """epsilon_at() returns reasonable values."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100)

        eps = acct.epsilon_at(1e-5)
        assert eps > 0
        assert eps < 100  # Sanity check

    def test_delta_at(self):
        """delta_at() returns reasonable values."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100)

        delta = acct.delta_at(1.0)
        assert 0 <= delta <= 1

    def test_advantage(self):
        """advantage() returns f-DP advantage."""
        acct = Accountant()
        acct = acct | dpsgd_acc.gaussian(1.0)

        adv = acct.advantage()
        assert 0 <= adv <= 1

    def test_beta_at(self):
        """beta_at() returns Type-II error rate."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.gaussian(1.0) * 10)

        beta = acct.beta_at(0.05)
        assert 0 <= beta <= 1

    def test_risk_at(self):
        """risk_at() returns Bayes risk."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.gaussian(1.0) * 10)

        risk = acct.risk_at(0.5)
        assert 0 <= risk <= 0.5

    def test_metrics_delegate_to_process(self):
        """Verify metrics delegate to underlying process."""
        acct = Accountant()
        acct = acct | dpsgd_acc.gaussian(1.0)

        eps_from_acct = acct.epsilon_at(1e-5)
        eps_from_process = acct.process.epsilon_at(1e-5)

        assert eps_from_acct == eps_from_process


# ============================================================================
# Budget tracking
# ============================================================================


class TestAccountantBudget:
    """Test budget tracking in Accountant."""

    def test_no_budget(self):
        """Accountant without budget never exceeds."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.gaussian(0.5) * 100)

        assert not acct.budget_exceeded

    def test_budget_not_exceeded(self):
        """budget_exceeded is False when within budget."""
        # Create a very loose budget with high epsilon
        budget = epsilon_budget(100.0, delta=1e-5)
        acct = Accountant(budget=budget)
        # Use gaussian with reasonable noise and few steps
        acct = acct | (dpsgd_acc.gaussian(1.0) * 5)

        assert not acct.budget_exceeded

    def test_budget_exceeded(self):
        """budget_exceeded is True when privacy cost exceeds budget."""
        budget = epsilon_budget(0.1, delta=1e-5)  # Very tight budget
        acct = Accountant(budget=budget)
        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), 0.01) * 1000)

        # Should exceed the budget
        assert acct.budget_exceeded

    @pytest.mark.slow
    def test_non_epsilon_budget(self):
        """Budget works with non-epsilon budgets (advantage)."""
        budget = acc.advantage_budget(0.5)
        acct = Accountant(budget=budget)
        acct = acct | (dpsgd_acc.gaussian(0.5) * 50)

        # Should not raise
        _ = acct.budget_exceeded

    @pytest.mark.parametrize(
        ("evaluate", "make_budget", "metric_max"),
        [
            (
                lambda process: process.beta_at(0.1),
                lambda value: acc.beta_budget(value, alpha=0.1),
                1.0,
            ),
            (
                lambda process: process.risk_at(0.5),
                lambda value: acc.risk_budget(value, prior=0.5),
                0.5,
            ),
        ],
        ids=["beta", "risk"],
    )
    def test_privacy_gain_budget_direction(self, evaluate, make_budget, metric_max):
        """Beta and risk budgets are exceeded when achieved privacy is too low."""
        process = dpsgd_acc.gaussian(1.0)
        achieved = evaluate(process)

        stricter_budget = make_budget(achieved + (metric_max - achieved) / 2.0)
        assert Accountant(budget=stricter_budget, prefix=process).budget_exceeded

        equal_budget = make_budget(achieved)
        assert not Accountant(budget=equal_budget, prefix=process).budget_exceeded

        looser_budget = make_budget(achieved / 2.0)
        assert not Accountant(budget=looser_budget, prefix=process).budget_exceeded


# ============================================================================
# Functional properties & API consistency
# ============================================================================


class TestAccountantFunctional:
    """Test functional properties of Accountant."""

    def test_composition_immutability(self):
        """Composing doesn't mutate original Accountant."""
        acct1 = Accountant()
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)

        eps1_before = acct1.epsilon_at(1e-5)

        _ = acct1 | step

        eps1_after = acct1.epsilon_at(1e-5)

        # Original should be unchanged
        assert eps1_before == eps1_after

    def test_chained_composition(self):
        """Can chain multiple compositions."""
        acct = Accountant()
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)

        for _ in range(10):
            acct = acct | step

        eps = acct.epsilon_at(1e-5)

        # Should have accumulated privacy loss
        assert eps > 0

    def test_different_mechanisms_compose(self):
        """Can compose different mechanism types."""
        acct = Accountant()
        acct = acct | dpsgd_acc.gaussian(1.0)
        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 10)
        acct = acct | dpsgd_acc.gaussian(0.5)

        eps = acct.epsilon_at(1e-5)
        assert eps > 0

    def test_budget_persists_through_composition(self):
        """Budget is preserved when creating new Accountant via composition."""
        budget = epsilon_budget(5.0, delta=1e-5)
        acct1 = Accountant(budget=budget)

        # Compose multiple times
        acct2 = acct1 | dpsgd_acc.gaussian(0.1)
        acct3 = acct2 | dpsgd_acc.gaussian(0.1)

        # All should have the same budget reference
        assert acct1._budget is budget
        assert acct2._budget is budget
        assert acct3._budget is budget

    def test_or_operator_associativity(self):
        """| operator is associative for privacy accounting."""
        acct1 = Accountant()
        step1 = dpsgd_acc.gaussian(1.0)
        step2 = dpsgd_acc.gaussian(0.5)

        acct2 = (acct1 | step1) | step2
        acct3 = acct1 | (step1 | step2)

        eps2 = acct2.epsilon_at(1e-5)
        eps3 = acct3.epsilon_at(1e-5)

        assert abs(eps2 - eps3) < 1e-10


# ============================================================================
# Realistic training-loop patterns
# ============================================================================


class TestAccountantTrainingLoop:
    """Realistic end-to-end Accountant usage patterns."""

    def test_batch_composition(self):
        """Simulate training loop: compose all steps at once."""
        budget = acc.epsilon_budget(5.0, delta=1e-5)
        acct = Accountant(budget=budget)

        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 10)

        assert not acct.budget_exceeded

    def test_incremental_composition(self):
        """Simulate training loop: compose one step at a time."""
        budget = acc.epsilon_budget(10.0, delta=1e-5)
        acct = Accountant(budget=budget)

        step_process = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)

        for _ in range(5):
            acct_new = acct | step_process
            if acct_new.budget_exceeded:
                break
            acct = acct_new

        assert not acct.budget_exceeded

    @pytest.mark.slow
    def test_calibration_then_train(self):
        """Calibrate noise, then use Accountant to track budget."""
        budget = acc.epsilon_budget(2.0, delta=1e-5)
        steps = 25

        result = acc.calibrate(
            budget=budget,
            process=lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * steps,
            param_min=0.1,
            param_max=3.5,
            tolerance=1e-3,
        )

        acct = Accountant(budget=budget)
        acct = acct | (
            dpsgd_acc.poisson(dpsgd_acc.gaussian(result.param), 0.01) * steps
        )

        achieved = acct.epsilon_at(1e-5)
        assert abs(achieved - budget.value) < 0.5

    def test_mixed_mechanisms(self):
        """Accountant with mixed mechanism types."""
        budget = acc.epsilon_budget(5.0, delta=1e-5)
        acct = Accountant(budget=budget)

        acct = acct | dpsgd_acc.gaussian(0.5)
        acct = acct | (dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), 0.01) * 20)
        acct = acct | dpsgd_acc.gaussian(0.3)

        eps = acct.epsilon_at(1e-5)
        assert eps > 0.5


# ============================================================================
# Incremental caching via acc.cached(accountant)
# ============================================================================


class TestAccountantCached:
    """Test acc.cached() on Accountant for incremental PLD reuse."""

    def test_cached_matches_no_cache(self):
        """cached(acct) must produce identical epsilon as without caching."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)
        delta = 1e-5

        # Without caching
        plain = Accountant()
        for _ in range(20):
            plain = plain | step
        eps_plain = plain.epsilon_at(delta)

        # With cached() at step 10
        acct = Accountant()
        for i in range(20):
            acct = acct | step
            if i == 9:
                acct = acc.cached(acct)
                _ = acct.epsilon_at(delta)  # populate PLD cache
        eps_cached = acct.epsilon_at(delta)

        assert abs(eps_plain - eps_cached) < 1e-10

    def test_cached_returns_accountant(self):
        """cached() on Accountant returns an Accountant."""
        acct = Accountant()
        acct = acct | dpsgd_acc.gaussian(1.0)
        result = acc.cached(acct)
        assert isinstance(result, Accountant)

    def test_cached_preserves_budget(self):
        """Budget is preserved through cached()."""
        budget = epsilon_budget(5.0, delta=1e-5)
        acct = Accountant(budget=budget)
        acct = acct | dpsgd_acc.gaussian(1.0)
        acct = acc.cached(acct)
        assert acct._budget is budget

    def test_cached_does_not_mutate(self):
        """cached() does not mutate the original Accountant."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.gaussian(1.0) * 5)
        eps_before = acct.epsilon_at(1e-5)

        _ = acc.cached(acct)

        eps_after = acct.epsilon_at(1e-5)
        assert eps_before == eps_after

    def test_multiple_cached_calls(self):
        """Multiple cached() calls produce correct results."""
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01)
        delta = 1e-5

        # Without caching
        plain = Accountant()
        for _ in range(30):
            plain = plain | step
        eps_plain = plain.epsilon_at(delta)

        # With cached() every 10 steps
        acct = Accountant()
        for i in range(30):
            acct = acct | step
            if (i + 1) % 10 == 0:
                acct = acc.cached(acct)
                _ = acct.epsilon_at(delta)
        eps_cached = acct.epsilon_at(delta)

        assert abs(eps_plain - eps_cached) < 1e-10

    def test_cached_heterogeneous_steps(self):
        """cached() works with heterogeneous (varying) steps."""
        delta = 1e-5

        steps = [
            dpsgd_acc.poisson(
                dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.1), expected_batch_size=bs), 0.01
            )
            for bs in [120, 130, 125, 128, 135, 122, 131, 127, 129, 126]
        ]

        # Without caching
        plain = Accountant()
        for s in steps:
            plain = plain | s
        eps_plain = plain.epsilon_at(delta)

        # With cached() at step 5
        acct = Accountant()
        for i, s in enumerate(steps):
            acct = acct | s
            if i == 4:
                acct = acc.cached(acct)
                _ = acct.epsilon_at(delta)
        eps_cached = acct.epsilon_at(delta)

        assert abs(eps_plain - eps_cached) < 1e-10

    def test_cached_on_identity(self):
        """Caching empty accountant works."""
        acct = Accountant()
        acct = acc.cached(acct)
        eps = acct.epsilon_at(1e-5)
        assert eps < 1e-10

    def test_cached_idempotent(self):
        """Double-caching an accountant is idempotent."""
        acct = Accountant()
        acct = acct | (dpsgd_acc.gaussian(1.0) * 5)
        acct1 = acc.cached(acct)
        acct2 = acc.cached(acct1)
        # Inner process should be the same CachedProcess (not double-wrapped)
        assert acct1.process is acct2.process

    def test_cached_k_out_of_t_preserves_full_horizon(self):
        horizon = dpsgd_acc.k_out_of_t(
            dpsgd_acc.gaussian(1.0),
            k=2,
            t=4,
            allocation="block",
        )
        cached_horizon = acc.cached(horizon)

        assert cached_horizon is not horizon
        assert cached_horizon.pld() is horizon.pld()

    def test_cached_k_out_of_t_accountant_wraps_horizon(self):
        horizon = dpsgd_acc.k_out_of_t(
            dpsgd_acc.gaussian(1.0),
            k=2,
            t=4,
            allocation="block",
        )
        accountant = Accountant(prefix=horizon)
        cached_accountant = acc.cached(accountant)

        assert cached_accountant is not accountant
        assert cached_accountant.epsilon_at(1e-5) == accountant.epsilon_at(1e-5)
