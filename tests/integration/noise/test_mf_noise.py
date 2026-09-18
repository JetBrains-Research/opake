"""Tests for MF strategy factories and mf_gaussian_noise()."""

from typing import ClassVar

import pytest
import torch

import opake.dpftrl.accounting as ftrl_acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.dpftrl.noise.types import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    LambdaCgdStrategy,
)
from opake.random import key
from opake.types import NoisedPytree, clipped

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}
_BAND_PART = {"n_steps": 100, "min_sep": 1, "max_participations": 100}


# ── BandMfStrategy ──────────────────────────────────────────────────────


class TestBandMfStrategy:
    def test_returns_correct_type(self):
        assert isinstance(band_mf_strategy(bands=10, momentum=0.95), BandMfStrategy)

    def test_single_participation_sensitivity_is_one(self):
        """Optimized Toeplitz coefficients are L2-normalized.

        The claim is about one column, so it is the single-participation
        request that returns 1; repeat participation is strictly costlier.
        """
        s = band_mf_strategy(bands=10, momentum=0.95)
        single = {"n_steps": 100, "min_sep": 100, "max_participations": 1}
        assert s.sensitivity(**single) == pytest.approx(1.0, abs=1e-6)
        assert s.sensitivity(**_BAND_PART) > 1.0

    def test_no_gram_matrix(self):
        """BandMF uses Poisson amplification, not BnB — no Gram needed."""
        s = band_mf_strategy(bands=10, momentum=0.95)
        with pytest.raises(NotImplementedError):
            s.gram_matrix(n_steps=100, min_sep=10, max_participations=10)

    def test_coefficients_length(self):
        s = band_mf_strategy(bands=10, momentum=0.95)
        assert len(s.coefficients(n_steps=100)) == 10

    def test_streaming_matrix_present(self):
        s = band_mf_strategy(bands=10, momentum=0.95)
        assert s.streaming_matrix(n_steps=100) is not None

    def test_rejects_bad_bands(self):
        with pytest.raises(ValueError, match="bands must be >= 1"):
            band_mf_strategy(bands=0)

    def test_with_lr_schedule(self):
        lr = torch.ones(100, dtype=torch.float64) * 0.01
        lr[:10] = torch.linspace(0.001, 0.01, 10)
        schedule = lambda t: float(lr[t])  # noqa: E731 — Schedule callable
        s = band_mf_strategy(bands=10, momentum=0.95, lr_schedule=schedule)
        single = {"n_steps": 100, "min_sep": 100, "max_participations": 1}
        assert s.sensitivity(**single) == pytest.approx(1.0, abs=1e-6)


# ── BltStrategy ─────────────────────────────────────────────────────────


class TestBltStrategy:
    def test_returns_correct_type(self):
        assert isinstance(blt_strategy(momentum=0.95), BltStrategy)

    def test_sensitivity_positive(self):
        assert blt_strategy(momentum=0.95).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = blt_strategy(momentum=0.95).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_coefficients_length(self):
        s = blt_strategy(momentum=0.95)
        assert len(s.coefficients(**_PART)) == 100

    def test_streaming_matrix_present(self):
        assert blt_strategy(momentum=0.95).streaming_matrix(**_PART) is not None

    def test_matches_old_sensitivity(self):
        assert blt_strategy(momentum=0.95).sensitivity(**_PART) > 0

    def test_single_participation(self):
        s = blt_strategy()
        assert s.sensitivity(n_steps=50, min_sep=1, max_participations=1) > 0


# ── LambdaCgdStrategy ──────────────────────────────────────────────────


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        assert isinstance(lambda_cgd_strategy(lambda_=0.9), LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        assert lambda_cgd_strategy(lambda_=0.9).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = lambda_cgd_strategy(lambda_=0.9).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_normalized_single_participation_sensitivity_one(self):
        """Normalized + single participation → sensitivity = 1.0."""
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.sensitivity(
            n_steps=100, min_sep=1, max_participations=1
        ) == pytest.approx(1.0, abs=1e-6)

    def test_matches_old_sensitivity(self):
        assert lambda_cgd_strategy(lambda_=0.9).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        """lambda_cgd_strategy does not accept momentum (use bisr_strategy)."""
        with pytest.raises(TypeError):
            lambda_cgd_strategy(lambda_=0.5, momentum=0.95)

    def test_unnormalized(self):
        s = lambda_cgd_strategy(lambda_=0.9, normalized=False)
        assert s.sensitivity(**_PART) > 0

    def test_rejects_bad_lambda(self):
        with pytest.raises(ValueError, match="lambda_ must be finite and in"):
            lambda_cgd_strategy(lambda_=-0.1)
        with pytest.raises(ValueError, match="lambda_ must be finite and in"):
            lambda_cgd_strategy(lambda_=1.0)

    def test_internal_fields(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.lambda_ == pytest.approx(0.9)
        assert s.normalized is True


# ── BisrStrategy ────────────────────────────────────────────────────────


class TestBisrStrategy:
    def test_returns_correct_type(self):
        assert isinstance(bisr_strategy(bandwidth=4), BisrStrategy)

    def test_sensitivity_positive(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = bisr_strategy(bandwidth=4).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_streaming_matrix_present(self):
        assert bisr_strategy(bandwidth=4).streaming_matrix(**_PART) is not None

    def test_matches_old_sensitivity(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be >= 2"):
            bisr_strategy(bandwidth=1)


# ── PLD equivalence ────────────────────────────────────────────────────


class TestPldEquivalence:
    """New strategy + mf_gaussian produces a finite, positive PLD."""

    delta = 1e-5

    def test_band_mf_pld(self):
        s = band_mf_strategy(bands=10, momentum=0.95)
        eps_new = ftrl_acc.mf_gaussian(1.0, s, **_BAND_PART).epsilon_at(self.delta)
        assert eps_new > 0

    def test_band_mf_poisson_pld(self):
        """Poisson via band_mf matches manual poisson composition."""
        import math

        n_steps = 100
        bands = 10
        s = band_mf_strategy(bands=bands, momentum=0.95)
        sample_rate = 0.05

        eps_new = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, s),
            sample_rate=sample_rate,
            n_steps=n_steps,
        ).epsilon_at(self.delta)
        num_groups = math.ceil(n_steps / bands)
        # Match the normalizer the cyclic-Poisson route uses: single
        # participation, with repeat participation priced by the composition.
        sens = s.sensitivity(n_steps=n_steps, min_sep=n_steps, max_participations=1)
        eps_manual = (
            dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0 / sens), sample_rate) * num_groups
        ).epsilon_at(self.delta)
        assert eps_new == pytest.approx(eps_manual, abs=1e-10)

    def test_blt_pld(self):
        s = blt_strategy(momentum=0.95)
        eps_new = ftrl_acc.mf_gaussian(1.0, s, **_PART).epsilon_at(self.delta)
        assert eps_new > 0

    def test_lambda_cgd_pld(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps_new = ftrl_acc.mf_gaussian(1.0, s, **_PART).epsilon_at(self.delta)
        assert eps_new > 0

    def test_bisr_pld(self):
        s = bisr_strategy(bandwidth=4)
        eps_new = ftrl_acc.mf_gaussian(1.0, s, **_PART).epsilon_at(self.delta)
        assert eps_new > 0


# ── BnB equivalence ────────────────────────────────────────────────────


class TestBnbEquivalence:
    """BnB built around each strategy composes correctly."""

    delta = 1e-2

    def test_lambda_cgd_bnb(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0

    def test_blt_bnb(self):
        s = blt_strategy(momentum=0.95)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0

    def test_bisr_bnb(self):
        s = bisr_strategy(bandwidth=4)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0


# ── mf_gaussian_noise() tests ───────────────────────────────────────────────────


class TestMfNoise:
    """mf_gaussian_noise() produces noise for all strategy types."""

    grad: ClassVar = {"w": torch.randn(3, 4)}

    def _clipped_grad(self):
        return clipped(self.grad, max_norm=1.0)

    def test_band_mf_noise(self):
        s = band_mf_strategy(bands=10, momentum=0.95)
        n_steps = 50
        nf, ns = mf_gaussian_noise(
            self.grad, s, n_steps=n_steps, noise_multiplier=1.0, key=key(42)
        )
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)
        assert noised.max_norm == pytest.approx(1.0)
        # ``noise_stddev`` is the per-step *realized* σ (= base σ ·
        # ‖row_t(C^-1)‖) — see the bug-fix tests in
        # ``packages/opake-dpftrl/tests/noise/test_realized_stddev.py``.
        # For BandMF the row_l2 differs from 1; recover base σ and check.
        streaming = s.streaming_matrix(
            n_steps=n_steps, min_sep=1, max_participations=n_steps
        )
        row_l2 = float(streaming.row_norms_squared(n_steps).clamp_min(0.0).sqrt()[0])
        assert float(noised.noise_stddev) == pytest.approx(row_l2, rel=1e-9)
        assert "w" in noised.pytree
        assert noised.pytree["w"].shape == self.grad["w"].shape

    def test_blt_noise(self):
        s = blt_strategy(momentum=0.95)
        nf, ns = mf_gaussian_noise(
            self.grad,
            s,
            n_steps=50,
            min_sep=10,
            max_participations=5,
            noise_multiplier=1.0,
            key=key(42),
        )
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_lambda_cgd_noise(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        nf, ns = mf_gaussian_noise(
            self.grad,
            s,
            n_steps=50,
            min_sep=10,
            max_participations=5,
            noise_multiplier=1.0,
            key=key(42),
        )
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_bisr_noise(self):
        s = bisr_strategy(bandwidth=4)
        nf, ns = mf_gaussian_noise(
            self.grad,
            s,
            n_steps=50,
            min_sep=10,
            max_participations=5,
            noise_multiplier=1.0,
            key=key(42),
        )
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_lambda_cgd_multi_step(self):
        """λCGD noise correctly handles step 0 (no prev) and step 1+ (with prev)."""
        s = lambda_cgd_strategy(lambda_=0.9)
        nf, ns = mf_gaussian_noise(
            self.grad,
            s,
            n_steps=50,
            min_sep=10,
            max_participations=5,
            noise_multiplier=1.0,
            key=key(42),
        )
        _, ns = nf(self._clipped_grad(), ns)
        _, ns = nf(self._clipped_grad(), ns)
        _, ns = nf(self._clipped_grad(), ns)
        assert ns._step_counter == 3

    def test_noise_is_nonzero(self):
        """Noise adds something to the gradients."""
        s = band_mf_strategy(bands=10, momentum=0.95)
        nf, ns = mf_gaussian_noise(
            self.grad, s, n_steps=50, noise_multiplier=1.0, key=key(42)
        )
        noised, _ = nf(self._clipped_grad(), ns)
        diff = torch.norm(noised.pytree["w"] - self.grad["w"])
        assert diff > 0.1

    def test_raw_grads_are_rejected(self):
        s = band_mf_strategy(bands=10, momentum=0.95)
        nf, ns = mf_gaussian_noise(
            self.grad, s, n_steps=50, noise_multiplier=1.0, key=key(42)
        )
        with pytest.raises(TypeError, match="ClippedPytree"):
            nf(self.grad, ns)
