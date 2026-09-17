"""Regression tests for DP-FTRL PLD cache fingerprints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import opake.accounting as acc
import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.core.composition.types import Composed
from opake.dpftrl.noise import band_mf_strategy, blt_strategy, identity_strategy
from opake.scheduling import constant_schedule, linear_schedule


def _horizon_process(schedule, *, momentum: float = 1.0):
    strategy = band_mf_strategy(bands=4, momentum=momentum, lr_schedule=schedule)
    process = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, strategy),
        sample_rate=0.01,
        n_steps=16,
    )
    return process


def test_distinct_materialized_schedules_do_not_share_cached_plds() -> None:
    constant = _horizon_process(lambda _step: 1.0)
    ramp = _horizon_process(lambda step: 1.0 + 0.01 * step)

    assert constant.pld(discretization=0.1) is not ramp.pld(discretization=0.1)


def test_structurally_distinct_equivalent_schedules_reuse_cached_pld() -> None:
    first = _horizon_process(lambda step: 1.0 + 0.01 * step)
    second = _horizon_process(lambda step: float(1.0 + 0.01 * step))

    assert first.pld(discretization=0.1) is second.pld(discretization=0.1)


def test_distinct_mf_gaussian_parameters_do_not_share_cached_plds() -> None:
    first = ftrl_acc.mf_gaussian(0.0, identity_strategy(), n_steps=1)
    second = ftrl_acc.mf_gaussian(1.0, identity_strategy(), n_steps=1)
    first.pld.cache_clear()

    assert first.pld(discretization=0.1) is not second.pld(discretization=0.1)


def test_distinct_strategy_parameters_do_not_share_cached_plds() -> None:
    first = _horizon_process(lambda _step: 1.0, momentum=0.9)
    second = _horizon_process(lambda _step: 1.0, momentum=0.8)
    first.pld.cache_clear()

    assert first.pld(discretization=0.1) is not second.pld(discretization=0.1)


def test_composed_processes_preserve_schedule_cache_identity() -> None:
    constant = acc.eps_delta(0.1) | _horizon_process(lambda _step: 1.0)
    ramp = acc.eps_delta(0.1) | _horizon_process(lambda step: 1.0 + 0.01 * step)

    assert constant.pld(discretization=0.1) is not ramp.pld(discretization=0.1)


def test_blt_horizon_fingerprint_uses_the_full_schedule() -> None:
    constant = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, blt_strategy(lr_schedule=lambda _step: 1.0)),
        num_bins=4,
        n_steps=16,
    )
    changed_after_prefix = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(
            1.0,
            blt_strategy(lr_schedule=lambda step: 1.0 if step < 4 else 0.5),
        ),
        num_bins=4,
        n_steps=16,
    )

    assert constant._pld_cache_key() != changed_after_prefix._pld_cache_key()


def _blt_phase(schedule):
    return ftrl_acc.mf_gaussian(
        2.0,
        blt_strategy(max_buffers=1, lr_schedule=schedule),
        n_steps=4,
        min_sep=2,
        max_participations=2,
    )


@dataclass(frozen=True)
class _ScheduleWithIncompleteEquality:
    value: float = field(compare=False)

    def __call__(self, _step: int) -> float:
        return self.value


def test_distinct_schedule_phases_remain_heterogeneous() -> None:
    first = _blt_phase(linear_schedule(1.0, 0.1, 2, transition_begin=2))
    second = _blt_phase(constant_schedule(1.0))

    combined = first | second
    assert isinstance(combined, Composed)

    expected = Composed(first, second)
    homogeneous = first * 2

    actual_epsilon = combined.epsilon_at(1e-5, discretization=0.1)
    expected_epsilon = expected.epsilon_at(1e-5, discretization=0.1)
    homogeneous_epsilon = homogeneous.epsilon_at(1e-5, discretization=0.1)

    assert actual_epsilon == expected_epsilon
    assert homogeneous_epsilon < expected_epsilon


def test_callable_equality_cannot_merge_distinct_schedules() -> None:
    first = _blt_phase(_ScheduleWithIncompleteEquality(1.0))
    second = _blt_phase(_ScheduleWithIncompleteEquality(0.5))

    assert first.strategy == second.strategy
    assert isinstance(first | second, Composed)


def test_right_spine_does_not_merge_distinct_schedule_phases() -> None:
    first = _blt_phase(linear_schedule(1.0, 0.1, 2, transition_begin=2))
    second = _blt_phase(constant_schedule(1.0))

    combined = (acc.eps_delta(0.1) | first) | second

    assert isinstance(combined, Composed)
    assert combined.right is second


def test_repeated_whole_horizon_uses_the_full_schedule_fingerprint() -> None:
    first = _blt_phase(linear_schedule(1.0, 0.1, 2, transition_begin=2)) * 2
    second = _blt_phase(constant_schedule(1.0)) * 2
    first.pld.cache_clear()

    assert first._pld_cache_key() != second._pld_cache_key()
    first_pld = first.pld(discretization=0.1)
    second_pld = second.pld(discretization=0.1)

    assert first_pld is not second_pld
    assert not math.isclose(first_pld.epsilon_at(1e-5), second_pld.epsilon_at(1e-5))
