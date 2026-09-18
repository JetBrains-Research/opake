"""Bounds in ``__post_init__`` must fire for direct construction and
deserialization, not only through the factories."""

import pytest

import opake.accounting as acc
import opake.dpftrl.accounting as ftrl_acc
from opake.dpftrl.accounting.types import BallsInBins, MfGaussian
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opake.dpftrl.noise.types import BandMfStrategy, BltStrategy, BsrStrategy
from opake.exceptions import ConfigurationError
from opake.serialization import from_state_dict, state_dict


def _band_mf(nm: float = 1.0) -> MfGaussian:
    return ftrl_acc.mf_gaussian(nm, band_mf_strategy(bands=2))


def _identity_mf(nm: float = 1.0) -> MfGaussian:
    return ftrl_acc.mf_gaussian(nm, identity_strategy())


class TestStrategyRecipeValidation:
    def test_band_mf_rejects_invalid_recipe(self):
        with pytest.raises(ConfigurationError, match="bands must be >= 1"):
            BandMfStrategy(bands=0)
        with pytest.raises(ConfigurationError, match="momentum must be >= 0"):
            BandMfStrategy(bands=2, momentum=-0.1)
        with pytest.raises(ValueError, match="bands must be >= 1"):
            band_mf_strategy(bands=0)

    def test_blt_rejects_invalid_recipe(self):
        with pytest.raises(ConfigurationError, match="max_buffers must be >= 1"):
            BltStrategy(max_buffers=0)
        with pytest.raises(ConfigurationError, match="momentum must be >= 0"):
            BltStrategy(max_buffers=2, momentum=-1.0)
        with pytest.raises(ValueError, match="max_buffers must be >= 1"):
            blt_strategy(max_buffers=0)

    def test_bsr_rejects_invalid_recipe(self):
        with pytest.raises(ConfigurationError, match="bandwidth must be >= 1"):
            BsrStrategy(bandwidth=0, alpha=0.5, beta=0.2)
        with pytest.raises(ConfigurationError, match=r"α in \(0, 1\]"):
            BsrStrategy(bandwidth=4, alpha=0.0, beta=0.2)
        with pytest.raises(ConfigurationError, match=r"β in \[0, 1\)"):
            bsr_strategy(bandwidth=4, alpha=0.9, beta=1.0)

    def test_bisr_rejects_invalid_momentum(self):
        with pytest.raises(
            ConfigurationError, match=r"momentum must be finite and in \[0, 1\)"
        ):
            bisr_strategy(bandwidth=3, momentum=1.0)

    def test_lambda_cgd_rejects_invalid_lambda(self):
        with pytest.raises(
            ConfigurationError, match=r"lambda_ must be finite and in \[0, 1\)"
        ):
            lambda_cgd_strategy(lambda_=1.0)

    def test_tampered_strategy_state_fails_on_load(self):
        proc = ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2))
        state = state_dict(proc)
        state["strategy"]["bands"] = 0
        with pytest.raises(ValueError, match="bands must be >= 1"):
            from_state_dict(acc.identity(), state)


class TestMfGaussian:
    @pytest.mark.parametrize("constructor", [MfGaussian, ftrl_acc.mf_gaussian])
    @pytest.mark.parametrize("noise_multiplier", [0.0, 1.0])
    def test_recipe_requires_a_horizon_for_bare_accounting(
        self, constructor, noise_multiplier
    ):
        recipe = constructor(noise_multiplier, blt_strategy(max_buffers=3))
        assert recipe.n_steps is None
        for process in (recipe, acc.cached(recipe), recipe * 2):
            with pytest.raises(ConfigurationError, match="explicit n_steps"):
                process.pld()
            with pytest.raises(ConfigurationError, match="explicit n_steps"):
                process.epsilon_at(1e-5)

    @pytest.mark.parametrize("missing_horizon", [False, True])
    def test_restoring_a_recipe_does_not_invent_a_horizon(self, missing_horizon):
        state = state_dict(_identity_mf())
        if missing_horizon:
            del state["n_steps"]
        restored = from_state_dict(acc.identity(), state)
        assert restored.n_steps is None
        with pytest.raises(ConfigurationError, match="explicit n_steps"):
            restored.epsilon_at(1e-5)

        amplified = ftrl_acc.poisson(restored, sample_rate=0.1, n_steps=10)
        explicit = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy(), n_steps=1),
            sample_rate=0.1,
            n_steps=10,
        )
        assert amplified.epsilon_at(1e-5) == explicit.epsilon_at(1e-5)

    @pytest.mark.parametrize("n_steps", [1, 32])
    def test_explicit_horizons_roundtrip_unchanged(self, n_steps):
        process = ftrl_acc.mf_gaussian(
            2.0,
            blt_strategy(max_buffers=3),
            n_steps=n_steps,
            min_sep=1,
            max_participations=1,
        )
        state = state_dict(process)
        restored = from_state_dict(acc.identity(), state)
        assert state_dict(restored) == state
        assert restored.epsilon_at(1e-5) == process.epsilon_at(1e-5)

    def test_rejects_negative_noise_multiplier(self):
        with pytest.raises(
            ConfigurationError, match="noise_multiplier must be non-negative"
        ):
            MfGaussian(-1.0, band_mf_strategy(bands=2))

    def test_factory_rejects_negative_noise_multiplier(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            ftrl_acc.mf_gaussian(-1.0, band_mf_strategy(bands=2))

    def test_rejects_invalid_horizon_params(self):
        s = band_mf_strategy(bands=2)
        with pytest.raises(ConfigurationError, match="n_steps must be >= 1"):
            ftrl_acc.mf_gaussian(1.0, s, n_steps=0)
        with pytest.raises(ConfigurationError, match="min_sep must be >= 1"):
            ftrl_acc.mf_gaussian(1.0, s, min_sep=0)
        with pytest.raises(ConfigurationError, match="max_participations must be"):
            ftrl_acc.mf_gaussian(1.0, s, max_participations=0)

    def test_negative_sigma_state_dict_fails_on_load(self):
        state = state_dict(_band_mf())
        state["noise_multiplier"] = -1.0
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            from_state_dict(acc.identity(), state)


class TestCyclicPoisson:
    def test_rejects_empty_horizon(self):
        with pytest.raises(ConfigurationError, match="n_steps must be >= 1"):
            ftrl_acc.poisson(_band_mf(), sample_rate=0.01, n_steps=0)

    def test_rejects_sample_rate_out_of_range(self):
        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            ftrl_acc.poisson(_band_mf(), sample_rate=0.0, n_steps=10)


class TestBallsInBins:
    def test_rejects_num_bins_below_two(self):
        with pytest.raises(ConfigurationError, match="num_bins must be >= 2"):
            BallsInBins(_band_mf(), num_bins=1, n_steps=10)

    @pytest.mark.parametrize("n_steps", [0, -5])
    def test_rejects_empty_horizon(self, n_steps):
        with pytest.raises(ConfigurationError, match="n_steps must be >= 1"):
            BallsInBins(_band_mf(), num_bins=2, n_steps=n_steps)

    def test_rejects_fractional_epochs(self):
        with pytest.raises(ConfigurationError, match="multiple of"):
            ftrl_acc.balls_in_bins(_identity_mf(), num_bins=3, n_steps=10)

    def test_tampered_state_dict_fails_on_load(self):
        state = state_dict(
            ftrl_acc.balls_in_bins(_identity_mf(), num_bins=2, n_steps=10)
        )
        state["n_steps"] = 3
        with pytest.raises(ValueError, match="multiple of"):
            from_state_dict(acc.identity(), state)


class TestBMinSep:
    def test_rejects_empty_horizon(self):
        with pytest.raises(ConfigurationError, match="n_steps must be >= 1"):
            ftrl_acc.b_min_sep(_band_mf(), n_steps=0, p0=0.02)

    @pytest.mark.parametrize("p0", [0.0, 1.0, -0.1, 1.2])
    def test_rejects_participation_rate_out_of_range(self, p0):
        with pytest.raises(ConfigurationError, match=r"p_0 must be in \(0, 1\)"):
            ftrl_acc.b_min_sep(_band_mf(), n_steps=10, p0=p0)

    def test_tampered_state_dict_fails_on_load(self):
        state = state_dict(ftrl_acc.b_min_sep(_band_mf(), n_steps=10, p0=0.02))
        state["p0"] = 1.5
        with pytest.raises(ValueError, match=r"p_0 must be in \(0, 1\)"):
            from_state_dict(acc.identity(), state)
