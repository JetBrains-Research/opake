"""Tests for opake.accounting.calibration — binary search calibration framework."""

import math
from dataclasses import dataclass
from functools import lru_cache

import pytest

import opake.dpsgd.accounting as dpsgd_acc
from opake.accounting import calibration as cal
from opake.api.accounting.core.calibration import (
    AdvantageBudget,
    BetaBudget,
    CalibrateResult,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
)
from opake.api.accounting.core.discretization import (
    get_discretization,
    set_discretization,
)
from opake.exceptions import CalibrationError, PrivacyBudgetError


@dataclass(frozen=True)
class _MetricProcess:
    metric: float


@dataclass(frozen=True)
class _MetricBudget:
    value: float
    decreasing: bool
    name: str = "test metric"

    def evaluate(self, process: _MetricProcess) -> float:
        return process.metric


_CALIBRATION_STEPS = 100
_INTEGRATION_TOLERANCE = 1e-3
_PREFIX_TOLERANCE = 1e-4


def _assert_safe_result(
    result: CalibrateResult,
    *,
    target: float,
    decreasing: bool,
    tolerance: float = 1e-6,
) -> None:
    assert result.converged is True
    if decreasing:
        assert result.achieved <= target
    else:
        assert result.achieved >= target
    assert math.isclose(
        result.achieved,
        target,
        rel_tol=tolerance,
        abs_tol=0.0,
    )


@lru_cache
def _calibrate_epsilon(
    target: float, param_min: float, param_max: float
) -> CalibrateResult:
    """Reuse identical expensive PLD calibration requests within this module."""
    return cal.calibrate(
        cal.epsilon_budget(target, delta=1e-5),
        lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * _CALIBRATION_STEPS,
        param_min,
        param_max,
        tolerance=_PREFIX_TOLERANCE,
    )


# -- Budget validation -------------------------------------------------------


class TestEpsilonBudget:
    def test_valid(self):
        t = cal.epsilon_budget(3.0, delta=1e-5)
        assert isinstance(t, EpsilonBudget)
        assert t.value == 3.0
        assert "epsilon" in t.name

    def test_decreasing(self):
        """Epsilon decreases with noise → decreasing=True."""
        assert cal.epsilon_budget(3.0, delta=1e-5).decreasing is True

    def test_evaluate(self):
        t = cal.epsilon_budget(3.0, delta=1e-5)
        proc = dpsgd_acc.gaussian(0.8)
        val = t.evaluate(proc)
        assert math.isfinite(val)

    def test_rejects_negative_epsilon(self):
        with pytest.raises(PrivacyBudgetError, match="epsilon"):
            cal.epsilon_budget(-1.0, delta=1e-5)

    def test_rejects_delta_out_of_range(self):
        with pytest.raises(PrivacyBudgetError, match="delta"):
            cal.epsilon_budget(3.0, delta=0.0)
        with pytest.raises(PrivacyBudgetError, match="delta"):
            cal.epsilon_budget(3.0, delta=1.0)


class TestDeltaBudget:
    def test_valid(self):
        t = cal.delta_budget(1e-5, epsilon=3.0)
        assert isinstance(t, DeltaBudget)
        assert t.value == pytest.approx(1e-5)

    def test_decreasing(self):
        assert cal.delta_budget(1e-5, epsilon=3.0).decreasing is True

    def test_rejects_invalid(self):
        with pytest.raises(PrivacyBudgetError, match="delta"):
            cal.delta_budget(0.0, epsilon=3.0)
        with pytest.raises(PrivacyBudgetError, match="epsilon"):
            cal.delta_budget(1e-5, epsilon=-1.0)


class TestAdvantageBudget:
    def test_valid(self):
        t = cal.advantage_budget(0.1)
        assert isinstance(t, AdvantageBudget)
        assert t.value == pytest.approx(0.1)

    def test_decreasing(self):
        assert cal.advantage_budget(0.1).decreasing is True

    def test_rejects_out_of_range(self):
        with pytest.raises(PrivacyBudgetError, match="advantage"):
            cal.advantage_budget(0.0)
        with pytest.raises(PrivacyBudgetError, match="advantage"):
            cal.advantage_budget(1.0)


class TestBetaBudget:
    def test_valid(self):
        t = cal.beta_budget(0.05, alpha=0.01)
        assert isinstance(t, BetaBudget)
        assert t.value == pytest.approx(0.05)

    def test_decreasing(self):
        """Beta increases with noise → decreasing=False."""
        assert cal.beta_budget(0.05, alpha=0.01).decreasing is False

    def test_rejects_invalid(self):
        with pytest.raises(PrivacyBudgetError, match="beta"):
            cal.beta_budget(0.0, alpha=0.01)
        with pytest.raises(PrivacyBudgetError, match="alpha"):
            cal.beta_budget(0.5, alpha=0.0)


class TestRiskBudget:
    def test_valid(self):
        t = cal.risk_budget(0.1, prior=0.5)
        assert isinstance(t, RiskBudget)
        assert t.value == pytest.approx(0.1)

    def test_decreasing(self):
        """Risk increases with noise → decreasing=False."""
        assert cal.risk_budget(0.1, prior=0.5).decreasing is False

    def test_rejects_invalid(self):
        with pytest.raises(PrivacyBudgetError, match="risk"):
            cal.risk_budget(0.0, prior=0.5)
        with pytest.raises(PrivacyBudgetError, match="prior"):
            cal.risk_budget(0.1, prior=0.0)


# -- Calibration errors -------------------------------------------------------


class TestCalibrateErrors:
    def _process(self, nm):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000

    def test_param_min_ge_max(self):
        with pytest.raises(CalibrationError, match="param_min"):
            cal.calibrate(cal.epsilon_budget(3.0, delta=1e-5), self._process, 0.8, 0.5)

    def test_budget_outside_bracket(self):
        """Target not achievable within bounds → ValueError."""
        with pytest.raises(CalibrationError, match="unreachable"):
            cal.calibrate(
                cal.epsilon_budget(0.001, delta=1e-5), self._process, 0.5, 0.6
            )

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_invalid_tolerance_rejected_before_process_probe(self, tolerance):
        calls = []

        def process(param):
            calls.append(param)
            return _MetricProcess(param)

        with pytest.raises(CalibrationError, match="tolerance"):
            cal.calibrate(
                _MetricBudget(1.0, decreasing=False),
                process,
                0.0,
                2.0,
                tolerance=tolerance,
            )

        assert calls == []

    @pytest.mark.parametrize("max_iterations", [0, -1])
    def test_invalid_iteration_limit_rejected_before_process_probe(
        self, max_iterations
    ):
        calls = []

        def process(param):
            calls.append(param)
            return _MetricProcess(param)

        with pytest.raises(CalibrationError, match="max_iterations"):
            cal.calibrate(
                _MetricBudget(1.0, decreasing=False),
                process,
                0.0,
                2.0,
                max_iterations=max_iterations,
            )

        assert calls == []


class TestCalibrateSafeEndpoint:
    @pytest.mark.parametrize(
        ("decreasing", "metric"),
        [
            pytest.param(True, lambda param: 2.05 - param, id="decreasing"),
            pytest.param(False, lambda param: param - 0.05, id="increasing"),
        ],
    )
    def test_unsafe_first_midpoint_is_not_returned(self, decreasing, metric):
        target = 1.0
        tolerance = 0.1
        first_midpoint = metric(1.0)
        assert math.isclose(
            first_midpoint,
            target,
            rel_tol=tolerance,
            abs_tol=0.0,
        )
        assert (first_midpoint > target) == decreasing

        result = cal.calibrate(
            _MetricBudget(target, decreasing=decreasing),
            lambda param: _MetricProcess(metric(param)),
            0.0,
            2.0,
            tolerance=tolerance,
        )

        assert result.param != 1.0
        _assert_safe_result(
            result,
            target=target,
            decreasing=decreasing,
            tolerance=tolerance,
        )

    def test_small_target_uses_relative_tolerance(self):
        target = 1e-9
        tolerance = 1e-6

        result = cal.calibrate(
            _MetricBudget(target, decreasing=True),
            lambda param: _MetricProcess(1.8e-9 - param * 1e-9),
            0.0,
            1.0,
            tolerance=tolerance,
        )

        _assert_safe_result(
            result,
            target=target,
            decreasing=True,
            tolerance=tolerance,
        )

    def test_iteration_exhaustion_raises_without_an_extra_probe(self):
        calls = []

        def process(param):
            calls.append(param)
            return _MetricProcess(2.0 if param < 1.0 else 0.5)

        with pytest.raises(CalibrationError) as exc_info:
            cal.calibrate(
                _MetricBudget(1.0, decreasing=True, name="step metric"),
                process,
                0.0,
                2.0,
                tolerance=1e-6,
                max_iterations=4,
            )

        message = str(exc_info.value)
        assert "step metric" in message
        assert "target=1.0" in message
        assert "relative tolerance=1e-06" in message
        assert "iterations=4" in message
        assert "final bracket=(unsafe=" in message
        assert "last safe achieved=0.5" in message
        assert len(calls) == 2 + 4


# -- Prefix (sequential composition across runs) ------------------------------


class TestCalibratePrefix:
    """Calibrate a second stage over an already-executed prefix."""

    # Composition counts are kept small on purpose: these tests assert
    # step-count-independent invariants (prefix== closure equivalence,
    # budget round-trip, prefix→more-noise monotonicity), so a short
    # composition exercises the same calibration logic at a fraction of the
    # PLD self-compose cost.
    def _stage(self, nm):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.02) * 25

    def _prefix(self):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 12

    @pytest.mark.slow
    def test_prefix_calibration_contracts(self):
        """A prefixed calibration composes correctly and consumes budget."""
        prefix = self._prefix()
        budget = cal.epsilon_budget(6.0, delta=1e-5)

        via_param = cal.calibrate(
            budget,
            self._stage,
            0.3,
            3.0,
            prefix=prefix,
            tolerance=_PREFIX_TOLERANCE,
        )
        via_closure = cal.calibrate(
            budget,
            lambda nm: prefix | self._stage(nm),
            0.3,
            3.0,
            tolerance=_PREFIX_TOLERANCE,
        )
        assert via_param.param == pytest.approx(via_closure.param)
        assert via_param.achieved == pytest.approx(via_closure.achieved)
        _assert_safe_result(
            via_param,
            target=6.0,
            decreasing=True,
            tolerance=_PREFIX_TOLERANCE,
        )

        total = prefix | self._stage(via_param.param)
        achieved = total.epsilon_at(1e-5)
        assert achieved <= 6.0
        assert math.isclose(achieved, 6.0, rel_tol=_PREFIX_TOLERANCE, abs_tol=0.0)

        without = cal.calibrate(
            budget,
            self._stage,
            0.3,
            3.0,
            tolerance=_PREFIX_TOLERANCE,
        )
        assert via_param.param > without.param

    @pytest.mark.slow
    def test_prefix_exhausting_budget_raises(self):
        """Prefix alone above the target → bounds can't bracket."""
        # Low-noise, high-sampling prefix that blows past ε=0.5 in few steps
        # so the bracket can't be established — cheap to compose.
        prefix = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.3), 0.1) * 100
        with pytest.raises(CalibrationError, match="unreachable"):
            cal.calibrate(
                cal.epsilon_budget(0.5, delta=1e-5),
                self._stage,
                0.3,
                3.0,
                prefix=prefix,
            )


# -- Calibration roundtrip ---------------------------------------------------


@pytest.mark.slow
class TestCalibrateEpsilon:
    """Calibrate noise multiplier for target epsilon — verify roundtrip."""

    def test_basic(self):
        result = _calibrate_epsilon(5.0, 0.3, 1.2)
        assert isinstance(result, CalibrateResult)
        _assert_safe_result(
            result,
            target=5.0,
            decreasing=True,
            tolerance=_INTEGRATION_TOLERANCE,
        )

    def test_strict_epsilon(self):
        result = _calibrate_epsilon(4.0, 0.3, 1.2)
        _assert_safe_result(
            result,
            target=4.0,
            decreasing=True,
            tolerance=_INTEGRATION_TOLERANCE,
        )

    def test_loose_epsilon(self):
        result = _calibrate_epsilon(8.0, 0.1, 1.0)
        _assert_safe_result(
            result,
            target=8.0,
            decreasing=True,
            tolerance=_INTEGRATION_TOLERANCE,
        )

    def test_monotonicity(self):
        """Stricter target → more noise."""
        result_loose = _calibrate_epsilon(8.0, 0.1, 1.0)
        result_strict = _calibrate_epsilon(4.0, 0.3, 1.2)
        # stricter (lower) epsilon requires higher noise_multiplier
        assert result_strict.param > result_loose.param


@pytest.mark.slow
class TestCalibrateDifferentBatchSizes:
    """Calibration converges at the smallest and largest batch/dataset ratios."""

    @pytest.mark.parametrize("batch_size", [8, 128])
    def test_converges(self, batch_size):
        n = 10_000
        q = batch_size / n

        result = cal.calibrate(
            cal.epsilon_budget(5.0, delta=1e-4),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), q) * 250,
            0.1,
            1.2,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        _assert_safe_result(
            result,
            target=5.0,
            decreasing=True,
            tolerance=_INTEGRATION_TOLERANCE,
        )


@pytest.mark.slow
class TestCalibrateAdvantage:
    """Calibrate for f-DP advantage target."""

    def test_roundtrip(self):
        result = cal.calibrate(
            cal.advantage_budget(0.1),
            lambda nm: (
                dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * _CALIBRATION_STEPS
            ),
            0.3,
            1.2,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        _assert_safe_result(
            result,
            target=0.1,
            decreasing=True,
            tolerance=_INTEGRATION_TOLERANCE,
        )


@pytest.mark.slow
class TestCalibrateBeta:
    """Calibrate for (α, β) error rate target.

    beta INCREASES with noise_multiplier (more noise → more private → higher
    Type-II error).  The target declares decreasing=False so calibrate()
    handles the direction automatically.
    """

    def test_roundtrip_and_monotonicity(self):
        """Stricter (higher) beta target produces a valid, noisier result."""

        def process(nm):
            return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * _CALIBRATION_STEPS

        result_low = cal.calibrate(
            cal.beta_budget(0.6, alpha=0.1),
            process,
            0.3,
            1.2,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        result_high = cal.calibrate(
            cal.beta_budget(0.8, alpha=0.1),
            process,
            0.3,
            1.2,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        _assert_safe_result(
            result_low,
            target=0.6,
            decreasing=False,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        _assert_safe_result(
            result_high,
            target=0.8,
            decreasing=False,
            tolerance=_INTEGRATION_TOLERANCE,
        )
        # Higher beta (more privacy) requires more noise
        assert result_high.param > result_low.param


class TestCalibrateResult:
    """CalibrateResult repr."""

    def test_repr_converged(self):
        r = CalibrateResult(1.0, 3.0, 3.0, 10, True)
        assert "converged" in repr(r)

    def test_repr_not_converged(self):
        r = CalibrateResult(1.0, 3.1, 3.0, 100, False)
        assert "max iterations" in repr(r)


class TestCalibrateDirectionIntegration:
    """PLD-mechanism regressions for probe-derived search direction (#333).

    The fast quadrant/rejection coverage lives in
    ``packages/opake-accounting/tests/test_calibrate_direction.py``; these
    pin the real Poisson/Gaussian shapes end-to-end (PLD-heavy, slow-marked).
    """

    def test_sample_rate_calibration_converges(self):
        # epsilon INCREASES with sample rate; raised ValueError before #333.
        budget = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            budget,
            lambda sr: dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sr) * 1000,
            param_min=0.001,
            param_max=0.05,
            tolerance=1e-4,
        )
        assert result.converged
        assert result.achieved <= 3.0
        check = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), result.param) * 1000
        assert check.epsilon_at(1e-5) == pytest.approx(3.0, rel=1e-3)

    @pytest.mark.slow
    def test_noise_multiplier_regression_value_unchanged(self):
        # The classic decreasing direction still lands the same parameter.
        result = cal.calibrate(
            cal.epsilon_budget(3.0, delta=1e-5),
            lambda nm: (
                dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.032) * _CALIBRATION_STEPS
            ),
            param_min=0.1,
            param_max=10.0,
            tolerance=1e-4,
        )
        assert result.achieved <= 3.0
        assert result.param == pytest.approx(0.8865, rel=1e-2)


class TestCalibrateRuntimeDiscretization:
    @pytest.fixture(autouse=True)
    def _reset_discretization(self):
        from opake.accounting import discretization

        original = discretization._default_config
        yield
        discretization._default_config = original

    def test_reported_achieved_is_runtime_upper_bound(self):
        set_discretization(
            discretization=1e-3,
            mc_resolution=1e-3,
            mc_failure_probability=1e-6,
        )
        result = cal.calibrate(
            cal.epsilon_budget(3.0, delta=1e-5),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.032) * 100,
            param_min=0.1,
            param_max=10.0,
            tolerance=1e-4,
        )

        runtime_achieved = (
            dpsgd_acc.poisson(dpsgd_acc.gaussian(result.param), 0.032) * 100
        ).epsilon_at(1e-5)
        assert runtime_achieved == pytest.approx(result.achieved, rel=1e-9, abs=0.0)
        assert result.achieved <= 3.0

    def test_recalibrates_when_runtime_evaluation_is_unsafe(self):
        @dataclass(frozen=True)
        class _ConfigProcess:
            param: float

        @dataclass(frozen=True)
        class _ConfigBudget:
            value: float = 1.0
            decreasing: bool = True
            name: str = "config metric"

            def evaluate(self, process: _ConfigProcess) -> float:
                return 2.0 - process.param + get_discretization().mc_failure_probability

        set_discretization(mc_failure_probability=0.1)
        result = cal.calibrate(
            _ConfigBudget(),
            _ConfigProcess,
            param_min=0.0,
            param_max=2.0,
            tolerance=1e-6,
            max_iterations=30,
        )

        assert result.param == pytest.approx(1.1, abs=1e-6)
        assert result.achieved <= 1.0
