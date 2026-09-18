"""Tests for opake.accounting.discretization — module-level PLD config management."""

from dataclasses import asdict

import pytest

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.api.accounting.core.discretization import (
    DiscretizationConfig,
    get_discretization,
    set_discretization,
)


@pytest.fixture(autouse=True)
def _reset_discretization():
    """Reset module-level default after each test."""
    from opake.accounting import discretization

    original = discretization._default_config
    yield
    discretization._default_config = original


class TestSetGetDiscretization:
    """set_discretization / get_discretization roundtrip."""

    def test_default_returns_library_default(self):
        """Before set_discretization(), returns library default (1e-4)."""
        from opake.accounting import discretization

        discretization._default_config = None
        cfg = get_discretization()
        assert cfg is not None
        assert cfg.discretization == pytest.approx(1e-4)  # Library default
        assert cfg.log_x_mass_truncation_bound == pytest.approx(
            -50.0
        )  # Library default
        assert cfg.max_grid_size == 10_000_000

    def test_set_and_get(self):
        set_discretization(discretization=1e-3)
        cfg = get_discretization()
        assert cfg is not None
        assert cfg.discretization == pytest.approx(1e-3)

    def test_set_all_params(self):
        set_discretization(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            seed=7,
            mc_resolution=1e-3,
            mc_failure_probability=1e-4,
        )
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-3)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-40.0)
        assert cfg.max_grid_size == 500_000
        assert cfg.tail_mass_truncation == pytest.approx(1e-12)
        assert cfg.seed == 7
        assert cfg.mc_resolution == pytest.approx(1e-3)
        assert cfg.mc_failure_probability == pytest.approx(1e-4)

    def test_overwrite(self):
        set_discretization(discretization=1e-3)
        set_discretization(discretization=1e-5)
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-5)

    def test_partial_update_preserves_other_params(self):
        """Setting one parameter keeps previously set parameters (issue #784)."""
        set_discretization(mc_failure_probability=1e-10, mc_resolution=1e-8)
        set_discretization(discretization=1e-3)
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-3)
        assert cfg.mc_resolution == pytest.approx(1e-8)
        assert cfg.mc_failure_probability == pytest.approx(1e-10)

    def test_bare_call_is_noop(self):
        """``set_discretization()`` without arguments leaves the default intact."""
        set_discretization(discretization=1e-3, seed=7)
        before = get_discretization()
        set_discretization()
        assert get_discretization() == before
        # A bare call must not materialize a config when none is set.
        from opake.accounting import discretization

        discretization._default_config = None
        set_discretization()
        assert discretization._default_config is None

    def test_full_set_restores_library_defaults(self):
        """Explicitly naming every field overrides earlier partial updates."""
        set_discretization(discretization=1e-3, mc_resolution=1e-8)
        set_discretization(**asdict(DiscretizationConfig()))
        assert get_discretization() == DiscretizationConfig()

    def test_params_are_keyword_only(self):
        with pytest.raises(TypeError):
            set_discretization(1e-3)


class TestDiscretizationAffectsResults:
    """Changing discretization affects computed privacy metrics."""

    def test_coarser_grid_changes_epsilon(self):
        """Coarser discretization should produce a different (less precise) epsilon."""
        proc = dpsgd_acc.gaussian(0.8)
        eps_fine = proc.epsilon_at(1e-5, discretization=1e-4)
        eps_coarse = proc.epsilon_at(1e-5, discretization=2e-1)
        # Both valid, but coarser grid = different result
        assert eps_fine != pytest.approx(eps_coarse, rel=1e-3)

    def test_mc_params_accepted_and_ignored_by_analytic_plds(self):
        """Analytic mechanisms accept MC confidence settings and ignore them."""
        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01) * 100
        kw = {"seed": 1, "mc_resolution": 1e-3, "mc_failure_probability": 1e-4}
        assert proc.epsilon_at(1e-5) == proc.epsilon_at(1e-5, **kw)
        assert proc.delta_at(2.0) == proc.delta_at(2.0, **kw)


class TestQueryTimeOverrides:
    """Test query-time parameter overrides via get_discretization()."""

    def test_discretization_override(self):
        """Query-time discretization override takes precedence."""

        # Set global default
        set_discretization(discretization=1e-3)

        # Query-time override should take precedence
        cfg = get_discretization(discretization=1e-5)
        assert cfg.discretization == pytest.approx(1e-5)

        # Global default unchanged
        global_cfg = get_discretization()
        assert global_cfg.discretization == pytest.approx(1e-3)

    def test_log_x_mass_truncation_bound_override(self):
        """Query-time log_x_mass_truncation_bound override works."""
        # Set global default
        set_discretization(log_x_mass_truncation_bound=-50.0)

        # Query-time override
        cfg = get_discretization(log_x_mass_truncation_bound=-30.0)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)

        # Global default unchanged
        global_cfg = get_discretization()
        assert global_cfg.log_x_mass_truncation_bound == pytest.approx(-50.0)

    def test_max_grid_size_override(self):
        """Query-time max_grid_size override works."""
        set_discretization(max_grid_size=10_000_000)

        cfg = get_discretization(max_grid_size=5_000_000)
        assert cfg.max_grid_size == 5_000_000

        global_cfg = get_discretization()
        assert global_cfg.max_grid_size == 10_000_000

    def test_mc_resolution_override(self):
        """Query-time MC resolution override works."""
        set_discretization(mc_resolution=1e-4)

        cfg = get_discretization(mc_resolution=1e-3)
        assert cfg.mc_resolution == pytest.approx(1e-3)

        global_cfg = get_discretization()
        assert global_cfg.mc_resolution == pytest.approx(1e-4)

    def test_seed_override(self):
        """Query-time seed override works."""
        set_discretization(seed=42)

        cfg = get_discretization(seed=7)
        assert cfg.seed == 7

        global_cfg = get_discretization()
        assert global_cfg.seed == 42

    def test_multiple_overrides(self):
        """Multiple query-time overrides work together."""
        set_discretization(
            discretization=1e-3,
            log_x_mass_truncation_bound=-50.0,
            max_grid_size=10_000_000,
        )

        cfg = get_discretization(
            discretization=1e-5,
            log_x_mass_truncation_bound=-30.0,
        )

        # Overridden values
        assert cfg.discretization == pytest.approx(1e-5)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)

        # Non-overridden values from global default
        assert cfg.max_grid_size == 10_000_000

    def test_override_on_library_default(self):
        """Query-time override works even without set_discretization."""
        from opake.accounting import discretization

        discretization._default_config = None  # Clear global default

        cfg = get_discretization(
            discretization=1e-5,
            log_x_mass_truncation_bound=-30.0,
        )

        assert cfg.discretization == pytest.approx(1e-5)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)
        # Library defaults for rest
        assert cfg.max_grid_size == 10_000_000


class TestDiscretizationBehavior:
    """Native configuration conversion and exact-grid behavior."""

    def test_config_converts_native_controls(self):
        config = DiscretizationConfig(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            seed=7,
            mc_resolution=1e-3,
            mc_failure_probability=1e-4,
        )

        native = config.to_native()
        assert native.discretization == pytest.approx(1e-3)
        assert native.log_mass_truncation_bound == pytest.approx(-40.0)
        assert native.max_grid_size == 500_000
        assert native.tail_mass_truncation == pytest.approx(1e-12)
        assert native.seed == 7
        assert native.mc_resolution == pytest.approx(1e-3)
        assert native.mc_failure_probability == pytest.approx(1e-4)
        assert config == DiscretizationConfig(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            seed=7,
            mc_resolution=1e-3,
            mc_failure_probability=1e-4,
        )
        assert hash(config) == hash(config)

    def test_exact_atoms_and_composition_round_up_to_the_grid(self):
        process = acc.eps_delta(0.11, 0.0)
        pld = process.pld(discretization=0.1)

        assert pld.delta_at(0.11) > 0.0
        assert pld.delta_at(0.2) == pytest.approx(0.0)
        composed = pld.compose(pld)
        assert composed.delta_at(0.22) > 0.0
        assert composed.delta_at(0.4) == pytest.approx(0.0)
