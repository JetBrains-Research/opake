"""End-to-end accounting tests for ``opake.dpftrl.accounting``."""

from __future__ import annotations

import pytest

import opake.dpftrl.accounting as ftrl_acc
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    lambda_cgd_strategy,
)


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_band_mf_poisson(self):
        import math

        import opake.dpftrl.accounting as ftrl_acc
        from opake.dpftrl.noise import band_mf_strategy

        strategy = band_mf_strategy(bands=2)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, strategy),
            sample_rate=0.01,
            n_steps=20,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_blt_standalone_uses_full_horizon(self):
        import pytest

        import opake.dpftrl.accounting as ftrl_acc
        from opake.api.accounting.core import _native
        from opake.api.accounting.core.discretization import get_discretization
        from opake.dpftrl.noise import blt_strategy

        strategy = blt_strategy(max_buffers=3)
        context = {"n_steps": 32, "min_sep": 1, "max_participations": 1}
        coefficients = strategy.coefficients(**context)
        # The first Toeplitz column has the largest norm for one participation.
        sensitivity = coefficients.norm().item()
        reference = _native.gaussian_pld(
            2.0 / sensitivity, get_discretization().to_native()
        ).epsilon_at(1e-5)
        eps = ftrl_acc.mf_gaussian(2.0, strategy, **context).epsilon_at(1e-5)
        one_step = ftrl_acc.mf_gaussian(
            2.0, strategy, n_steps=1, min_sep=1, max_participations=1
        ).epsilon_at(1e-5)
        assert eps == pytest.approx(reference, rel=1e-9)
        assert eps > one_step


@pytest.mark.parametrize(
    ("strategy", "amplification", "expected"),
    [
        pytest.param(
            blt_strategy(max_buffers=2, momentum=0.9),
            "balls_in_bins",
            (5.319030807298099, 3.0725352777481323),
            id="balls-in-bins-blt",
        ),
        pytest.param(
            bsr_strategy(bandwidth=4, alpha=1.0, beta=0.5),
            "balls_in_bins",
            (7.088073030236192, 4.2439714264428146),
            id="balls-in-bins-bsr",
        ),
        pytest.param(
            bisr_strategy(bandwidth=4, momentum=0.5, normalized=True),
            "balls_in_bins",
            (5.679465488219542, 3.5405959846930015),
            id="balls-in-bins-bisr",
        ),
        pytest.param(
            lambda_cgd_strategy(lambda_=0.7, normalized=True),
            "balls_in_bins",
            (4.530464625990122, 2.7022440276102166),
            id="balls-in-bins-lambda-cgd",
        ),
        pytest.param(
            band_mf_strategy(bands=4),
            "b_min_sep",
            (3.816001776216504, 2.1351874122380825),
            id="b-min-sep-band-mf",
        ),
    ],
)
def test_monte_carlo_epsilon_regression(strategy, amplification, expected):
    """Pin conservative MC estimates, not just two equivalent computations."""
    inner = ftrl_acc.mf_gaussian(1.5, strategy)
    process = (
        ftrl_acc.balls_in_bins(inner, num_bins=8, n_steps=32)
        if amplification == "balls_in_bins"
        else ftrl_acc.b_min_sep(inner, n_steps=32, p0=0.125)
    )
    # Fixed seeds and estimator settings make these numerical regression values.
    # Compute one PLD: epsilon_at on the process may adjust MC resolution to delta.
    pld = process.pld(
        seed=42,
        discretization=1e-4,
        max_grid_size=10_000_000,
        mc_resolution=1e-4,
        mc_failure_probability=1e-6,
    )
    assert pld.mc_failure_probability == 1e-6
    assert pld.mc_confidence == 1 - 1e-6
    assert 0 < pld.mc_resolution <= 1e-4 + 1e-12
    assert pld.infinity_mass == pytest.approx(pld.mc_resolution, rel=0, abs=1e-12)
    observed = tuple(pld.epsilon_at(delta) for delta in (2e-4, 1e-2))
    # Allows measured floating-point variation between CPU numerical libraries.
    assert observed == pytest.approx(expected, rel=0, abs=1e-7)
