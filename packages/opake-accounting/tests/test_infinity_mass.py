"""Tests for ``Pld.infinity_mass`` — the δ floor observable.

After enough compositions, ``delta_at(large_epsilon) == infinity_mass``
because the tail-truncation budget (``tail_mass_truncation/2``) is absorbed
into ``infinity_mass`` by successive ``self_compose`` calls.  Tests cover
both symmetric (``gaussian_pld``) and asymmetric (``poisson_pld``)
PLDs to exercise the worst-case-over-adjacencies path.
"""

from __future__ import annotations

import pytest

TAIL_MASS_TRUNCATION = 1e-15
EXPECTED_FLOOR = TAIL_MASS_TRUNCATION / 2  # 5e-16


def _gaussian_pld(sigma: float = 1.1):
    from opake.api.accounting.core.opake_accounting import (
        DiscretizationConfig,
        gaussian_pld,
    )

    config = DiscretizationConfig(tail_mass_truncation=TAIL_MASS_TRUNCATION)
    return gaussian_pld(sigma, config)


@pytest.mark.parametrize("sigma", [1.0, 1.1])
def test_infinity_mass_single_pld_is_nonnegative(sigma: float):
    pld = _gaussian_pld(sigma)
    assert pld.infinity_mass >= 0.0


def test_infinity_mass_equals_expected_floor_after_many_compositions():
    """After ~20 compositions the infinity_mass saturates to tail_mass_truncation/2."""
    pld = _gaussian_pld().self_compose(20)
    assert pld.infinity_mass == pytest.approx(EXPECTED_FLOOR, rel=1e-6)


def test_delta_at_large_epsilon_equals_infinity_mass():
    """For ε large enough that all grid mass decays, delta_at(ε) == infinity_mass."""
    pld = _gaussian_pld().self_compose(20)
    floor = pld.infinity_mass
    # At ε=100 all grid contributions are negligible
    assert pld.delta_at(100.0) == pytest.approx(floor, rel=1e-6)


def test_delta_at_always_ge_infinity_mass():
    """delta_at(ε) ≥ infinity_mass for all ε, including ε=0."""
    pld = _gaussian_pld().self_compose(5)
    floor = pld.infinity_mass
    for epsilon in [0.0, 0.5, 1.0, 5.0, 50.0]:
        assert pld.delta_at(epsilon) >= floor - 1e-30


def test_infinity_mass_increases_monotonically_with_composition():
    base = _gaussian_pld()
    prev = base.infinity_mass
    for k in [2, 5, 10, 20]:
        composed = base.self_compose(k)
        assert composed.infinity_mass >= prev
        prev = composed.infinity_mass


def test_infinity_mass_asymmetric_pld_worst_case():
    """Poisson-subsampled PLD is asymmetric; infinity_mass must reflect the
    worst-case adjacency (max over remove/add) and still saturate at the floor."""
    from opake.api.accounting.core.opake_accounting import (
        DiscretizationConfig,
        gaussian_pld,
        poisson_pld,
    )

    config = DiscretizationConfig(tail_mass_truncation=TAIL_MASS_TRUNCATION)
    pld = poisson_pld(gaussian_pld(1.1, config), 0.01)
    composed = pld.self_compose(20)
    floor = composed.infinity_mass
    assert floor == pytest.approx(EXPECTED_FLOOR, rel=1e-3)
    # delta_at a large epsilon equals infinity_mass
    assert composed.delta_at(100.0) == pytest.approx(floor, rel=1e-6)
    # delta_at(ε) >= infinity_mass for all ε
    for epsilon in [0.0, 0.5, 2.0, 50.0]:
        assert composed.delta_at(epsilon) >= floor - 1e-30
