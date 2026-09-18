"""Tests for shared test infrastructure that combines package capabilities."""

from __future__ import annotations

from dataclasses import replace

from opake_test_support import fast_mc_accounting

import opake.accounting as acc


def test_fast_mc_accounting_preserves_other_discretization_settings() -> None:
    """Only Monte Carlo controls change inside the temporary test configuration."""
    from opake.accounting import discretization

    original = discretization._default_config
    try:
        acc.set_discretization(
            discretization=2e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=50_000,
            tail_mass_truncation=1e-10,
            seed=7,
            max_conv_grid=512,
            mc_resolution=1e-4,
            mc_failure_probability=1e-3,
        )
        configured = acc.get_discretization()

        with fast_mc_accounting():
            assert acc.get_discretization() == replace(
                configured,
                mc_resolution=5e-3,
                mc_failure_probability=1e-2,
            )

        assert acc.get_discretization() == configured
    finally:
        discretization._default_config = original
