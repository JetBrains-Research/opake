"""Tests for ``MfGaussian(IdentityStrategy)`` and the FTRL amplifications
dispatching on it (``poisson``, ``balls_in_bins``)."""

import math
from functools import lru_cache

import pytest

import opake.accounting as acc
import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.core import _native
from opake.api.accounting.core.discretization import get_discretization
from opake.dpftrl.accounting.types import MfGaussian
from opake.dpftrl.noise import band_mf_strategy, identity_strategy
from opake.dpftrl.noise.types import BandMfStrategy, IdentityStrategy
from opake.exceptions import ConfigurationError

_DELTA = 1e-5


@lru_cache
def _identity_bnb_epsilon(
    noise_multiplier: float, num_bins: int, num_epochs: int
) -> float:
    return ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(noise_multiplier, identity_strategy()),
        num_bins=num_bins,
        n_steps=num_bins * num_epochs,
    ).epsilon_at(_DELTA)


# ---------------------------------------------------------------------------
# Mechanism: opake.dpftrl.accounting.identity_mf
# ---------------------------------------------------------------------------


class TestIdentityMfMechanism:
    def test_factory_returns_mf_gaussian_with_identity_strategy(self):
        proc = ftrl_acc.mf_gaussian(1.0, identity_strategy())
        assert isinstance(proc, MfGaussian)
        assert isinstance(proc.strategy, IdentityStrategy)
        assert proc.noise_multiplier == 1.0

    def test_pld_matches_unsubsampled_gaussian(self):
        nm = 1.5
        proc = ftrl_acc.mf_gaussian(nm, identity_strategy(), n_steps=1)

        cfg = get_discretization()
        ref = _native.gaussian_pld(nm, cfg.to_native())
        assert math.isclose(
            proc.pld().epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_zero_noise_is_non_private(self):
        assert math.isinf(
            ftrl_acc.mf_gaussian(0.0, identity_strategy(), n_steps=1).epsilon_at(_DELTA)
        )

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ftrl_acc.mf_gaussian(-0.1, identity_strategy())

    def test_self_compose_matches_repeated_gaussian(self):
        nm = 2.0
        T = 50
        proc = ftrl_acc.mf_gaussian(nm, identity_strategy(), n_steps=1) * T

        cfg = get_discretization()
        ref = _native.gaussian_pld(nm, cfg.to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# poisson(IdentityMf(...), sample_rate, n_steps)
# ---------------------------------------------------------------------------


class TestPoissonIdentity:
    @pytest.mark.slow
    def test_pld_matches_self_composed_generic_poisson(self):
        nm, p, T = 1.1, 0.01, 500
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()), sample_rate=p, n_steps=T
        )
        cfg = get_discretization()
        ref = _native.poisson_pld(
            _native.gaussian_pld(nm, cfg.to_native()),
            p,
        ).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_requires_n_steps(self):
        with pytest.raises(TypeError):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), sample_rate=0.1
            )

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.1,
                n_steps=0,
            )

    def test_rejects_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=1.5,
                n_steps=10,
            )

    def test_accepts_full_participation_rate(self):
        nm, T = 1.1, 50
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()), sample_rate=1.0, n_steps=T
        )
        assert proc.sample_rate == pytest.approx(1.0)
        # q=1 ⇒ no amplification: Gaussian self-composed T times.
        ref = _native.gaussian_pld(nm, get_discretization().to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_rejects_full_rate_with_truncation(self):
        nm = 1.1
        with pytest.raises(ConfigurationError, match=r"sample_rate=1\.0"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(nm, identity_strategy()),
                sample_rate=1.0,
                n_steps=10,
                truncated_batch_size=64,
                dataset_size=10_000,
            )

    def test_rejects_zero_rate_on_direct_construction(self):
        from opake.dpftrl.accounting.types import CyclicPoisson

        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            CyclicPoisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), 0.0, n_steps=10
            )


class TestPoissonBandMf:
    def test_requests_single_participation_sensitivity(self, monkeypatch):
        captured = {}

        def sensitivity(
            _strategy,
            *,
            n_steps: int,
            min_sep: int,
            max_participations: int | None,
        ) -> float:
            captured.update(
                n_steps=n_steps,
                min_sep=min_sep,
                max_participations=max_participations,
            )
            return 1.0

        monkeypatch.setattr(BandMfStrategy, "sensitivity", sensitivity)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(0.0, band_mf_strategy(bands=2)),
            sample_rate=0.01,
            n_steps=6,
        )

        proc.pld()

        # Theorem 4 of arXiv:2306.08153 prices the repeat participations by
        # the ``self_compose(num_groups)``, so this site must ask for the
        # single-participation column norm explicitly.
        assert captured == {
            "n_steps": 6,
            "min_sep": 6,
            "max_participations": 1,
        }

    def test_pld_matches_self_composed_with_bands(self):
        """For BandMf: num_groups = ceil(n_steps / bands)."""
        nm, p = 1.1, 0.01
        bands = 2
        n_steps = 100
        strategy = band_mf_strategy(bands=bands)
        # Match the normaliser the amplifier applies: the single-participation
        # column norm, not the schema sensitivity at the default min_sep=1.
        sens = strategy.sensitivity(
            n_steps=n_steps, min_sep=n_steps, max_participations=1
        )
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm / sens, strategy),
            sample_rate=p,
            n_steps=n_steps,
        )
        cfg = get_discretization()
        num_groups = math.ceil(n_steps / bands)
        ref = _native.poisson_pld(
            _native.gaussian_pld(nm, cfg.to_native()),
            p,
        ).self_compose(num_groups)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# balls_in_bins(IdentityMf(...), num_bins, n_steps)  — tight reduction
# ---------------------------------------------------------------------------


class TestBallsInBinsIdentity:
    @pytest.mark.slow
    def test_pld_agrees_with_generic_bnb_mc(self):
        """The deterministic identity path must agree with the generic MC
        accountant on the same dominating pair.

        With ``C = I`` the Gram is exactly ``E·I_b``, so ``bnb_mc_pld`` on that
        matrix samples the very pair the deterministic transform evaluates in
        closed form. MC is the oracle here; agreement within its sampling noise
        is the cross-check that the ``σ/√E`` reduction is right.
        """
        from opake.api.accounting.core.discretization import DiscretizationConfig

        nm, k, E = 1.5, 32, 4
        mc_delta = 1e-2
        cfg_native = DiscretizationConfig(
            seed=2024,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        ).to_native()

        ref_eps = _native.bnb_mc_pld(
            [E if i == j else 0.0 for i in range(k) for j in range(k)],
            k,
            nm,
            cfg_native,
        ).epsilon_at(mc_delta)

        det_eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            num_bins=k,
            n_steps=k * E,
        ).epsilon_at(mc_delta)

        assert ref_eps >= det_eps

    @pytest.mark.slow
    def test_factory_path_finite(self):
        """The default ``balls_in_bins`` factory produces a finite, positive ε."""
        nm, k, E = 1.5, 32, 4
        eps = _identity_bnb_epsilon(nm, k, E)
        assert math.isfinite(eps)
        assert eps > 0

    @pytest.mark.slow
    def test_strictly_tighter_than_unamplified_composition(self):
        """Amplification (factor ~1/num_bins) must beat unamplified composition."""
        nm, k, E = 1.5, 32, 4
        amplified = _identity_bnb_epsilon(nm, k, E)
        cfg = get_discretization()
        unamplified = (
            _native.gaussian_pld(nm, cfg.to_native())
            .self_compose(k * E)
            .epsilon_at(_DELTA)
        )
        assert amplified < unamplified

    def test_zero_noise_non_private(self):
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(0.0, identity_strategy()), num_bins=10, n_steps=20
        )
        assert math.isinf(proc.epsilon_at(_DELTA))

    def test_rejects_invalid_num_bins(self):
        with pytest.raises(ValueError, match="num_bins"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=1, n_steps=20
            )

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=10, n_steps=0
            )

    def test_rejects_n_steps_not_multiple_of_num_bins(self):
        with pytest.raises(ValueError, match="multiple of"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=10, n_steps=15
            )


# ---------------------------------------------------------------------------
# Calibration smoke
# ---------------------------------------------------------------------------


class TestTruncatedPoissonIdentity:
    """``ftrl_acc.poisson(identity_mf(...), ..., truncated_batch_size=, dataset_size=)``."""

    def test_rejects_unpaired_truncated_batch_size(self):
        with pytest.raises(
            ConfigurationError, match="truncated_batch_size and dataset_size"
        ):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
            )

    def test_rejects_unpaired_dataset_size(self):
        with pytest.raises(
            ConfigurationError, match="truncated_batch_size and dataset_size"
        ):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                dataset_size=10_000,
            )

    def test_rejects_truncated_batch_size_below_one(self):
        with pytest.raises(
            ConfigurationError, match="truncated_batch_size must be >= 1"
        ):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=0,
                dataset_size=10_000,
            )

    def test_rejects_dataset_size_below_one(self):
        with pytest.raises(ValueError, match="dataset_size must be >= 1"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
                dataset_size=0,
            )

    def test_rejects_band_mf_with_truncation(self):
        with pytest.raises(ValueError, match="IdentityStrategy"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
                dataset_size=10_000,
            )

    def test_pld_matches_self_composed_truncated_poisson_gaussian(self):
        """Truncated path is the per-step PLD composed ``n_steps`` times."""
        nm, p, T = 1.1, 0.01, 200
        cap, n = 64, 50_000
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=p,
            n_steps=T,
            truncated_batch_size=cap,
            dataset_size=n,
        )
        cfg = get_discretization()
        ref = _native.truncated_poisson_gaussian_pld(
            nm, p, cap, n, cfg.to_native()
        ).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_truncated_finite_and_positive(self):
        eps = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.1, identity_strategy()),
            sample_rate=0.01,
            n_steps=200,
            truncated_batch_size=64,
            dataset_size=50_000,
        ).epsilon_at(_DELTA)
        assert math.isfinite(eps)
        assert eps > 0


@pytest.mark.slow
def test_identity_mf_calibrates_through_poisson():
    n_steps = 100
    cal = acc.calibrate(
        acc.epsilon_budget(3.0, delta=_DELTA),
        lambda nm: ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        ),
        param_min=0.1,
        param_max=10.0,
        tolerance=1e-3,
    )
    assert cal.param > 0
    assert cal.achieved <= 3.0 + 1e-6
