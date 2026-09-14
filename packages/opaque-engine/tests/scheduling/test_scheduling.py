"""Tests for opaque.scheduling.

Cross-checks each public curve against HuggingFace's reference lambdas
pointwise — the multiplier returned by HF's LambdaLR scaled by
``BASE_LR`` must match our schedule at every step in a sweep.
"""

import math

import pytest
import torch

transformers = pytest.importorskip(
    "transformers", reason="transformers required for HF cross-check"
)

from torch.optim import SGD  # noqa: E402
from transformers.optimization import (  # noqa: E402
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from opaque.scheduling import (  # noqa: E402
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    warmup_stable_decay,
    with_restarts,
    with_warmup,
)

BASE_LR = 1e-3


def _hf_lambda(make_scheduler):
    """Extract HF's LambdaLR multiplier function (step -> multiplier)."""
    p = torch.zeros(1, requires_grad=True)
    return make_scheduler(SGD([p], lr=BASE_LR)).lr_lambdas[0]


def _assert_pointwise(ours, hf_lam, steps, *, tol=1e-9):
    """Our schedule must equal hf_lam(step) * BASE_LR at every step."""
    for s in steps:
        assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=tol), (
            f"step={s}: ours={ours(s)} hf={hf_lam(s) * BASE_LR}"
        )


# ---------------------------------------------------------------------------
# constant_schedule
# ---------------------------------------------------------------------------


class TestConstantSchedule:
    def test_returns_value_at_every_step(self):
        sched = constant_schedule(0.5)
        for s in (0, 1, 100, 10_000):
            assert sched(s) == 0.5

    def test_matches_hf_get_constant_schedule(self):
        sched = constant_schedule(BASE_LR)
        hf = _hf_lambda(get_constant_schedule)
        _assert_pointwise(sched, hf, range(0, 1100, 13))


# ---------------------------------------------------------------------------
# linear_schedule
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    def test_starts_at_init(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_reaches_end(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(100) == pytest.approx(0.0, abs=1e-12)

    def test_midpoint(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(50) == pytest.approx(0.5)

    def test_clamps_after_transition(self):
        s = linear_schedule(1.0, 0.2, transition_steps=100)
        assert s(100) == pytest.approx(0.2)
        assert s(500) == pytest.approx(0.2)

    def test_transition_begin_holds_init(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert s(0) == pytest.approx(1.0)
        assert s(49) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# polynomial_schedule
# ---------------------------------------------------------------------------


class TestPolynomialSchedule:
    def test_power_one_matches_linear(self):
        poly = polynomial_schedule(1.0, 0.0, power=1.0, transition_steps=100)
        lin = linear_schedule(1.0, 0.0, transition_steps=100)
        for s in (0, 25, 50, 75, 100, 200):
            assert poly(s) == pytest.approx(lin(s), abs=1e-12)

    def test_power_two_quadratic(self):
        # frac = 1 - count/T; value = (init - end) * frac^2 + end.
        s = polynomial_schedule(1.0, 0.0, power=2.0, transition_steps=100)
        # At progress=0.5: frac=0.5, frac^2 = 0.25.
        assert s(50) == pytest.approx(0.25)
        # At progress=1: frac=0, value=end.
        assert s(100) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# exponential_schedule
# ---------------------------------------------------------------------------


class TestExponentialSchedule:
    def test_starts_at_init_value(self):
        s = exponential_schedule(1.0, decay_rate=0.5, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_decays_geometrically(self):
        # decayed = init * decay_rate^(step / transition_steps)
        s = exponential_schedule(1.0, decay_rate=0.5, transition_steps=100)
        assert s(100) == pytest.approx(0.5)
        assert s(200) == pytest.approx(0.25)
        assert s(300) == pytest.approx(0.125)

    def test_transition_begin_holds_init(self):
        s = exponential_schedule(
            1.0,
            decay_rate=0.5,
            transition_begin=50,
            transition_steps=100,
        )
        assert s(0) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.5)

    def test_staircase(self):
        # With staircase=True, exponent is floored.
        s = exponential_schedule(
            1.0,
            decay_rate=0.5,
            transition_steps=100,
            staircase=True,
        )
        # step=99: floor(0.99) = 0 -> decay_rate^0 = 1.
        assert s(99) == pytest.approx(1.0)
        # step=100: floor(1.0) = 1 -> 0.5.
        assert s(100) == pytest.approx(0.5)
        # step=199: floor(1.99) = 1 -> still 0.5.
        assert s(199) == pytest.approx(0.5)
        # step=200: floor(2.0) = 2 -> 0.25.
        assert s(200) == pytest.approx(0.25)

    def test_end_value_clamps(self):
        # decay_rate < 1 -> clamp at max(decayed, end_value).
        s = exponential_schedule(
            1.0,
            decay_rate=0.5,
            transition_steps=100,
            end_value=0.1,
        )
        # Without clamp at step=400: 0.5^4 = 0.0625 < 0.1.
        assert s(400) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# cosine_schedule
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    def test_starts_at_init_value(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100)
        assert sched(0) == pytest.approx(1.0)

    def test_reaches_end_value_at_transition_steps(self):
        # num_cycles=0.5: half cosine, bottoms at progress=1.
        sched = cosine_schedule(1.0, 0.0, transition_steps=100, num_cycles=0.5)
        assert sched(100) == pytest.approx(0.0, abs=1e-12)

    def test_monotone_decreasing_for_default_cycles(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100)
        prev = sched(0)
        for s in range(1, 101):
            cur = sched(s)
            assert cur <= prev + 1e-12, f"non-monotone at step {s}"
            prev = cur

    def test_transition_begin_holds_init_before_offset(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert sched(0) == pytest.approx(1.0)
        assert sched(49) == pytest.approx(1.0)
        assert sched(50) == pytest.approx(1.0)
        assert sched(150) == pytest.approx(0.0, abs=1e-12)

    def test_matches_hf_no_warmup(self):
        sched = cosine_schedule(BASE_LR, 0.0, transition_steps=1000)
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(
                o, num_warmup_steps=0, num_training_steps=1000
            )
        )
        _assert_pointwise(sched, hf, range(0, 1100, 7))

    def test_matches_hf_num_cycles_one(self):
        # Multi-cycle: progress past 1 keeps oscillating.
        sched = cosine_schedule(BASE_LR, 0.0, transition_steps=1000, num_cycles=1.0)
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(
                o, num_warmup_steps=0, num_training_steps=1000, num_cycles=1.0
            )
        )
        _assert_pointwise(sched, hf, range(0, 1500, 11))


# ---------------------------------------------------------------------------
# inverse_sqrt_schedule
# ---------------------------------------------------------------------------


class TestInverseSqrtSchedule:
    def test_starts_at_init_value(self):
        sched = inverse_sqrt_schedule(1.0, transition_steps=100)
        assert sched(0) == pytest.approx(1.0)

    def test_decays_as_one_over_sqrt(self):
        T = 100
        sched = inverse_sqrt_schedule(1.0, transition_steps=T)
        # At s=T, multiplier = 1/sqrt(2).
        assert sched(T) == pytest.approx(1.0 / math.sqrt(2))
        # At s=3T, multiplier = 1/sqrt(4) = 1/2.
        assert sched(3 * T) == pytest.approx(0.5)

    def test_transition_begin_holds_init(self):
        sched = inverse_sqrt_schedule(1.0, transition_steps=100, transition_begin=50)
        assert sched(0) == pytest.approx(1.0)
        assert sched(50) == pytest.approx(1.0)
        assert sched(150) == pytest.approx(1.0 / math.sqrt(2))


# ---------------------------------------------------------------------------
# one_minus_sqrt_schedule
# ---------------------------------------------------------------------------


class TestOneMinusSqrtSchedule:
    def test_starts_at_init(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_reaches_end_at_transition_steps(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        assert s(100) == pytest.approx(0.0, abs=1e-12)

    def test_concave_decreasing(self):
        # Drops faster early than late.
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        # At 25% progress: 1 - sqrt(0.25) = 0.5
        assert s(25) == pytest.approx(0.5)
        # At 50% progress: 1 - sqrt(0.5) ≈ 0.293
        assert s(50) == pytest.approx(1.0 - math.sqrt(0.5))

    def test_clamps_after_transition_steps(self):
        s = one_minus_sqrt_schedule(1.0, 0.2, transition_steps=100)
        assert s(100) == pytest.approx(0.2)
        assert s(500) == pytest.approx(0.2)

    def test_transition_begin_holds_init(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert s(0) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# with_warmup
# ---------------------------------------------------------------------------


class TestWithWarmup:
    def test_zero_warmup_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_warmup(constant_schedule(0.5), transition_steps=0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_warmup(constant_schedule(0.5), transition_steps=-1)

    def test_warmup_ramp_starts_at_zero(self):
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        assert sched(0) == pytest.approx(0.0)

    def test_warmup_reaches_inner_value_at_transition_steps(self):
        # At step==transition_steps, ramp = 1, so result = inner(step).
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        assert sched(10) == pytest.approx(1.0)

    def test_multiplicative_composition(self):
        # Inner returns float(step); ramp(step) = step/10 for step < 10.
        # So sched(step) = (step/10) * step for step < 10, step otherwise.
        def inner(s):
            return float(s)

        sched = with_warmup(inner, transition_steps=10)
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.5 * 5)  # 0.5 * 5.0 = 2.5
        assert sched(10) == pytest.approx(10.0)  # ramp == 1, inner(10) = 10.0
        assert sched(100) == pytest.approx(100.0)

    def test_warmup_ramp_linear_with_constant_inner(self):
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        for k in range(10):
            assert sched(k) == pytest.approx(k / 10)

    def test_accepts_float_as_constant(self):
        # `with_warmup(0.5, ...)` is shorthand for `with_warmup(constant_schedule(0.5), ...)`.
        sched = with_warmup(0.5, transition_steps=10)
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.25)  # 0.5 ramp * 0.5 const
        assert sched(10) == pytest.approx(0.5)
        assert sched(100) == pytest.approx(0.5)

    def test_cosine_ramp(self):
        # ramp(p) = 0.5 * (1 - cos(pi*p)). At p=0: 0; p=0.5: 0.5; p=1: 1.
        sched = with_warmup(constant_schedule(1.0), transition_steps=10, ramp="cosine")
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.5, abs=1e-12)
        assert sched(10) == pytest.approx(1.0)

    def test_one_minus_sqrt_ramp(self):
        # ramp(p) = 1 - sqrt(1 - p). At p=0: 0; p=1: 1.
        sched = with_warmup(constant_schedule(1.0), transition_steps=100, ramp="1-sqrt")
        assert sched(0) == pytest.approx(0.0)
        assert sched(100) == pytest.approx(1.0)
        # Concave (drops slowly early, fast late).
        # At p=0.5: 1 - sqrt(0.5) ~= 0.293
        assert sched(50) == pytest.approx(1.0 - math.sqrt(0.5))

    def test_callable_ramp(self):
        # Quadratic ramp: progress**2.
        sched = with_warmup(
            constant_schedule(1.0), transition_steps=10, ramp=lambda p: p * p
        )
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.25)  # (5/10)**2 = 0.25
        assert sched(10) == pytest.approx(1.0)

    def test_unknown_ramp_string_raises(self):
        with pytest.raises(ValueError, match="Unknown ramp"):
            with_warmup(constant_schedule(1.0), transition_steps=10, ramp="bogus")


class TestWithRestarts:
    def test_zero_cycles_raises(self):
        with pytest.raises(ValueError, match="num_cycles > 0"):
            with_restarts(constant_schedule(1.0), transition_steps=10, num_cycles=0)

    def test_zero_transition_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_restarts(constant_schedule(1.0), transition_steps=0, num_cycles=2)

    def test_single_cycle_is_identity_within_window(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=100)
        sched = with_restarts(decay, transition_steps=100, num_cycles=1)
        for s in (0, 25, 50, 75, 100):
            assert sched(s) == pytest.approx(decay(s), abs=1e-12)

    def test_two_cycles_replays_at_half(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(decay, transition_steps=100, num_cycles=2)
        # Cycle length = 50; second cycle starts at step 50 from peak.
        assert sched(0) == pytest.approx(1.0)  # first cycle peak
        assert sched(50) == pytest.approx(1.0)  # second cycle peak (restart)
        assert sched(25) == pytest.approx(decay(25))
        assert sched(75) == pytest.approx(decay(25))  # same point in cycle 2

    def test_transition_begin_holds_at_init(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(
            decay, transition_steps=100, num_cycles=2, transition_begin=10
        )
        assert sched(0) == pytest.approx(1.0)  # before begin -> schedule(0)
        assert sched(9) == pytest.approx(1.0)
        assert sched(10) == pytest.approx(1.0)  # cycle starts here

    def test_after_window_returns_cycle_end(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(decay, transition_steps=100, num_cycles=2)
        assert sched(100) == pytest.approx(
            0.0
        )  # cycle_length = 50; schedule(50) = end_value
        assert sched(500) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Composed: pointwise HF parity for each supported scheduler type.
# ---------------------------------------------------------------------------


class TestHFParity:
    """Cross-check each composed schedule pointwise vs HF for >1000 steps."""

    @pytest.fixture(params=[(0, 1000), (50, 1000), (100, 1000), (500, 2000)])
    def warmup_total(self, request):
        return request.param

    def _check(self, ours, hf, num_total, *, beyond=200):
        steps = range(0, num_total + beyond, max(1, (num_total + beyond) // 200))
        _assert_pointwise(ours, hf, steps)

    def test_linear(self, warmup_total):
        W, N = warmup_total
        decay = linear_schedule(BASE_LR, 0.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_cosine(self, warmup_total):
        W, N = warmup_total
        decay = cosine_schedule(BASE_LR, 0.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_constant_with_warmup(self, warmup_total):
        W, _ = warmup_total
        if W == 0:
            pytest.skip("constant_with_warmup is just constant; tested elsewhere")
        ours = with_warmup(BASE_LR, transition_steps=W)
        hf = _hf_lambda(lambda o: get_constant_schedule_with_warmup(o, W))
        self._check(ours, hf, max(W * 3, 100))

    def test_polynomial(self, warmup_total):
        W, N = warmup_total
        # HF defaults: lr_end=1e-7, power=1.0.
        decay = polynomial_schedule(BASE_LR, 1e-7, 1.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_polynomial_decay_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_polynomial_custom_power(self):
        W, N = 100, 1000
        decay = polynomial_schedule(BASE_LR, 1e-7, 2.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W)
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(o, W, N, power=2.0)
        )
        self._check(ours, hf, N)

    def test_inverse_sqrt(self, warmup_total):
        W, _ = warmup_total
        # HF default: timescale = num_warmup_steps (or 10000 if W==0; but HF
        # raises ValueError for W==0 since timescale defaults None -> int(None)).
        if W == 0:
            pytest.skip("HF inverse_sqrt requires num_warmup_steps > 0 by default")
        decay = inverse_sqrt_schedule(BASE_LR, transition_steps=W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W)
        hf = _hf_lambda(lambda o: get_inverse_sqrt_schedule(o, num_warmup_steps=W))
        self._check(ours, hf, W * 5)

    @pytest.mark.parametrize("num_cycles", [1, 2, 4])
    def test_cosine_with_restarts(self, warmup_total, num_cycles):
        W, N = warmup_total
        if (N - W) % num_cycles != 0:
            # ``with_restarts`` itself no longer requires divisibility (it
            # uses a real-valued cycle length).  This manual composition,
            # however, pins the inner half-cosine span to the *integer*
            # ``(N-W)//num_cycles``; matching HF for a non-divisible window
            # needs the fractional span the Trainer dispatch builds (see
            # test_scheduler_dispatch.test_cosine_with_restarts_non_divisible).
            pytest.skip(
                f"integer inner-cosine span needs num_cycles | (N-W); "
                f"got N-W={N - W}, num_cycles={num_cycles}"
            )
        cycle_length = (N - W) // num_cycles
        inner = cosine_schedule(BASE_LR, 0.0, transition_steps=cycle_length)
        decay = with_restarts(
            inner,
            transition_steps=N - W,
            num_cycles=num_cycles,
            transition_begin=W,
        )
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(
            lambda o: get_cosine_with_hard_restarts_schedule_with_warmup(
                o,
                W,
                N,
                num_cycles=num_cycles,
            )
        )
        self._check(ours, hf, N)


# ---------------------------------------------------------------------------
# with_warmup(init_value=...) — non-zero "floor → 1" ramp
# ---------------------------------------------------------------------------


class TestWithWarmupInitValue:
    """``init_value`` interpolates the ramp factor from ``init_value`` to 1.0.

    Equivalent to multiplying the inner schedule by
    ``init_value + (1 - init_value) * ramp(progress)`` over the warmup
    window, then by 1.0 afterwards.
    """

    def test_default_init_value_is_zero(self):
        """Omitting ``init_value`` is identical to the historical 0 → 1 ramp."""
        a = with_warmup(constant_schedule(1.0), transition_steps=10)
        b = with_warmup(constant_schedule(1.0), transition_steps=10, init_value=0.0)
        for s in range(20):
            assert a(s) == pytest.approx(b(s))

    def test_floor_to_one_starts_at_init_value(self):
        sched = with_warmup(
            constant_schedule(1.0), transition_steps=10, init_value=0.25
        )
        assert sched(0) == pytest.approx(0.25)

    def test_floor_to_one_reaches_inner_at_transition_steps(self):
        sched = with_warmup(
            constant_schedule(1.0), transition_steps=10, init_value=0.25
        )
        assert sched(10) == pytest.approx(1.0)

    def test_floor_to_one_linear_interpolation(self):
        """``init_value + (1 - init_value) * progress`` shape at the midpoint."""
        sched = with_warmup(constant_schedule(1.0), transition_steps=10, init_value=0.2)
        # Midpoint factor = 0.2 + 0.8 * 0.5 = 0.6.
        assert sched(5) == pytest.approx(0.6)

    def test_floor_to_one_with_cosine_ramp(self):
        """At the midpoint cosine ramp = 0.5 → factor = init + (1-init)*0.5."""
        sched = with_warmup(
            constant_schedule(1.0),
            transition_steps=10,
            ramp="cosine",
            init_value=0.1,
        )
        # ramp(0.5) = 0.5 * (1 - cos(pi/2)) = 0.5
        # factor = 0.1 + 0.9 * 0.5 = 0.55
        assert sched(5) == pytest.approx(0.55, abs=1e-12)

    def test_floor_to_one_multiplies_inner(self):
        """The floor scales the inner schedule's value at step 0, not the ramp output."""
        # Inner returns float(step), so at step 0 inner=0 and we expect 0 even with floor.
        # Use a constant inner to isolate the factor behavior.
        sched = with_warmup(constant_schedule(2.0), transition_steps=4, init_value=0.5)
        # factor at step 0 = 0.5; inner = 2.0; result = 1.0.
        assert sched(0) == pytest.approx(1.0)
        # factor at step 2 = 0.5 + 0.5 * 0.5 = 0.75; inner = 2.0; result = 1.5.
        assert sched(2) == pytest.approx(1.5)
        # After warmup, factor = 1.0; result = 2.0.
        assert sched(4) == pytest.approx(2.0)
        assert sched(100) == pytest.approx(2.0)

    def test_init_value_one_is_constant_inner(self):
        """``init_value=1.0`` collapses to the inner schedule unchanged."""
        inner = cosine_schedule(1.0, 0.0, transition_steps=20)
        sched = with_warmup(inner, transition_steps=10, init_value=1.0)
        for s in (0, 1, 5, 10, 15, 20, 30):
            assert sched(s) == pytest.approx(inner(s))

    def test_negative_init_value_raises(self):
        with pytest.raises(ValueError, match="init_value"):
            with_warmup(constant_schedule(1.0), transition_steps=10, init_value=-0.1)

    def test_init_value_above_one_raises(self):
        with pytest.raises(ValueError, match="init_value"):
            with_warmup(constant_schedule(1.0), transition_steps=10, init_value=1.1)


# ---------------------------------------------------------------------------
# warmup_stable_decay (WSD)
# ---------------------------------------------------------------------------


class TestWarmupStableDecay:
    """Three-phase warmup → constant → decay schedule (Hägele et al. 2024)."""

    def test_phase_lengths_validated(self):
        with pytest.raises(ValueError, match="num_warmup_steps > 0"):
            warmup_stable_decay(
                1.0, num_warmup_steps=0, num_stable_steps=10, num_decay_steps=10
            )
        with pytest.raises(ValueError, match="num_stable_steps >= 0"):
            warmup_stable_decay(
                1.0, num_warmup_steps=10, num_stable_steps=-1, num_decay_steps=10
            )
        with pytest.raises(ValueError, match="num_decay_steps > 0"):
            warmup_stable_decay(
                1.0, num_warmup_steps=10, num_stable_steps=10, num_decay_steps=0
            )

    def test_unknown_warmup_ramp_raises(self):
        with pytest.raises(ValueError, match="warmup_ramp"):
            warmup_stable_decay(
                1.0,
                num_warmup_steps=10,
                num_stable_steps=10,
                num_decay_steps=10,
                warmup_ramp="bogus",
            )

    def test_unknown_decay_shape_raises(self):
        with pytest.raises(ValueError, match="decay_shape"):
            warmup_stable_decay(
                1.0,
                num_warmup_steps=10,
                num_stable_steps=10,
                num_decay_steps=10,
                decay_shape="bogus",
            )

    def test_warmup_phase_starts_at_zero(self):
        sched = warmup_stable_decay(
            1.0, num_warmup_steps=10, num_stable_steps=20, num_decay_steps=30
        )
        assert sched(0) == pytest.approx(0.0)

    def test_warmup_phase_linear_interpolation(self):
        sched = warmup_stable_decay(
            1.0, num_warmup_steps=10, num_stable_steps=20, num_decay_steps=30
        )
        assert sched(5) == pytest.approx(0.5)
        # At the end of warmup we're still on the warmup branch (step <
        # num_warmup_steps is the boundary), so at step 10 we've crossed
        # into the stable phase = init_value.
        assert sched(10) == pytest.approx(1.0)

    def test_stable_phase_is_constant(self):
        sched = warmup_stable_decay(
            1.0, num_warmup_steps=10, num_stable_steps=20, num_decay_steps=30
        )
        for s in (10, 15, 20, 25, 29):
            assert sched(s) == pytest.approx(1.0)

    def test_decay_phase_one_minus_sqrt_default(self):
        sched = warmup_stable_decay(
            1.0,
            end_value=0.0,
            num_warmup_steps=10,
            num_stable_steps=20,
            num_decay_steps=100,
        )
        # Stable ends at step 30; decay runs steps 30..130.
        # At step 30: factor = 1 - sqrt(0) = 1 → value = init = 1.0.
        assert sched(30) == pytest.approx(1.0)
        # At step 30 + 100 * 0.25 = 55: progress = 0.25,
        # factor = 1 - sqrt(0.25) = 0.5 → value = 0 + 1 * 0.5 = 0.5.
        assert sched(55) == pytest.approx(0.5)
        # After decay: value = end_value.
        assert sched(130) == pytest.approx(0.0)
        assert sched(500) == pytest.approx(0.0)

    def test_decay_phase_linear_shape(self):
        sched = warmup_stable_decay(
            1.0,
            end_value=0.0,
            num_warmup_steps=10,
            num_stable_steps=20,
            num_decay_steps=100,
            decay_shape="linear",
        )
        # Linear: factor(p) = 1 - p → value = end + (init - end) * (1 - p).
        # At step 30 (p=0): 1.0.
        # At step 80 (p=0.5): 0.5.
        # At step 130 (p=1): 0.0.
        assert sched(30) == pytest.approx(1.0)
        assert sched(80) == pytest.approx(0.5)
        assert sched(130) == pytest.approx(0.0)

    def test_decay_phase_cosine_shape(self):
        sched = warmup_stable_decay(
            1.0,
            end_value=0.0,
            num_warmup_steps=10,
            num_stable_steps=20,
            num_decay_steps=100,
            decay_shape="cosine",
        )
        # Cosine: factor(p) = 0.5 * (1 + cos(pi * p)).
        # At step 30 (p=0): factor=1 → value=1.0.
        # At step 80 (p=0.5): factor=0.5 → value=0.5.
        # At step 130 (p=1): factor=0 → value=0.0.
        assert sched(30) == pytest.approx(1.0)
        assert sched(80) == pytest.approx(0.5, abs=1e-12)
        assert sched(130) == pytest.approx(0.0, abs=1e-12)

    def test_non_zero_end_value_for_min_lr_floor(self):
        sched = warmup_stable_decay(
            1.0,
            end_value=0.1,
            num_warmup_steps=10,
            num_stable_steps=20,
            num_decay_steps=100,
            decay_shape="linear",
        )
        # End of decay clamps at end_value.
        assert sched(130) == pytest.approx(0.1)
        assert sched(500) == pytest.approx(0.1)
        # Midpoint: 0.1 + 0.9 * 0.5 = 0.55.
        assert sched(80) == pytest.approx(0.55)

    def test_zero_stable_steps_collapses_to_warmup_then_decay(self):
        sched = warmup_stable_decay(
            1.0,
            num_warmup_steps=10,
            num_stable_steps=0,
            num_decay_steps=100,
            decay_shape="linear",
        )
        # Decay starts immediately after warmup ends.
        assert sched(0) == pytest.approx(0.0)  # start of warmup
        assert sched(10) == pytest.approx(1.0)  # peak (start of decay)
        assert sched(60) == pytest.approx(0.5)  # midpoint of decay
        assert sched(110) == pytest.approx(0.0)  # end of decay

    def test_callable_decay_shape(self):
        # Quadratic decay: factor(p) = (1 - p) ** 2.
        sched = warmup_stable_decay(
            1.0,
            end_value=0.0,
            num_warmup_steps=10,
            num_stable_steps=0,
            num_decay_steps=10,
            decay_shape=lambda p: (1.0 - p) ** 2,
        )
        # At step 10 (p=0): factor=1 → 1.0.
        # At step 15 (p=0.5): factor=0.25 → 0.25.
        # At step 20 (p=1): factor=0 → 0.0.
        assert sched(10) == pytest.approx(1.0)
        assert sched(15) == pytest.approx(0.25)
        assert sched(20) == pytest.approx(0.0)

    def test_callable_warmup_ramp(self):
        # Square ramp: ramp(p) = p**2.
        sched = warmup_stable_decay(
            1.0,
            num_warmup_steps=10,
            num_stable_steps=10,
            num_decay_steps=10,
            warmup_ramp=lambda p: p * p,
        )
        # At step 5 (p=0.5): ramp=0.25 → value = 0.25 * 1.0 = 0.25.
        assert sched(5) == pytest.approx(0.25)
