"""Tests for AdaClip — now in opake.dpsgd.accounting.mechanisms."""

import math

import pytest

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.dpsgd.accounting.mechanisms.types import AdaClip, Gaussian
from opake.serialization import from_state_dict, state_dict

# ── Constructor function tests ───────────────────────────────────────


class TestAdaclipConstructor:
    """dpsgd_acc.adaclip() returns AdaClip with effective noise multiplier."""

    def test_returns_adaclip(self):
        result = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        assert isinstance(result, AdaClip)

    def test_effective_noise_differs_from_base(self):
        base = dpsgd_acc.gaussian(0.8)
        result = dpsgd_acc.adaclip(base, expected_batch_size=1000)
        # Effective noise should reduce, so epsilon should increase.
        assert result.epsilon_at(1e-5) > base.epsilon_at(1e-5)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.adaclip(acc.eps_delta(1.0), expected_batch_size=100)  # type: ignore[arg-type]

    def test_more_privacy_cost_than_base(self):
        """AdaClip ε ≥ base ε (extra cost from quantile noise)."""
        base = dpsgd_acc.gaussian(0.8)
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        eps_base = base.epsilon_at(1e-5)
        eps_ac = ac.epsilon_at(1e-5)
        assert eps_ac >= eps_base - 1e-6


class TestEffectiveNoiseMultiplier:
    """Tests for AdaClip.effective_noise_multiplier property."""

    def test_effective_noise_multiplier_is_less_than_base(self):
        """z_eff < z because extra sensitivity from quantile estimator."""
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.0), expected_batch_size=100)
        assert ac.effective_noise_multiplier < 1.0
        assert ac.effective_noise_multiplier > 0.0

    def test_effective_noise_multiplier_formula(self):
        z = 1.1
        multiplier = 0.05
        batch_size = 200
        sigma_b = batch_size * multiplier

        ac = dpsgd_acc.adaclip(
            dpsgd_acc.gaussian(z),
            fraction_noise_std=multiplier,
            expected_batch_size=batch_size,
        )
        expected = 1.0 / math.sqrt(1.0 / z**2 + 1.0 / (4.0 * sigma_b**2))
        assert abs(ac.effective_noise_multiplier - expected) < 1e-10

    def test_large_batch_size_negligible_cost(self):
        z = 1.0
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(z), expected_batch_size=100_000)
        assert abs(ac.effective_noise_multiplier - z) < 1e-4

    def test_small_batch_size_significant_cost(self):
        z = 1.0
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(z), expected_batch_size=5)
        assert ac.effective_noise_multiplier < 0.5 * z

    def test_poisson_uses_effective_noise_multiplier(self):
        """Poisson wrapper should use effective_noise_multiplier, not inner.noise_multiplier."""
        base = dpsgd_acc.gaussian(1.0)
        ac = dpsgd_acc.adaclip(base, expected_batch_size=100)

        # Poisson wrapping AdaClip should produce higher epsilon than Poisson wrapping Gaussian(z_eff)
        step_ac = dpsgd_acc.poisson(ac, sample_rate=0.01)
        step_eff = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(ac.effective_noise_multiplier), sample_rate=0.01
        )

        eps_ac = step_ac.epsilon_at(1e-5)
        eps_eff = step_eff.epsilon_at(1e-5)
        # Should match: both use the same z_eff
        assert abs(eps_ac - eps_eff) < 1e-6


# ── Parameter validation ────────────────────────────────────────────────────────


class TestAdaclipValidation:
    """Bounds live in ``AdaClip.__post_init__`` so direct construction, the
    factory, and codec deserialization all reject a mis-priced process (e.g.
    ``num_groups=0``, which used to price the quantile release as free)."""

    def test_direct_construction_rejects_zero_num_groups(self):
        with pytest.raises(ValueError, match="num_groups"):
            AdaClip(Gaussian(1.1), 0.05, 250.0, num_groups=0)

    def test_direct_construction_rejects_fractional_num_groups(self):
        with pytest.raises(ValueError, match="num_groups"):
            AdaClip(Gaussian(1.1), 0.05, 250.0, num_groups=2.5)

    def test_direct_construction_rejects_non_positive_noise_std(self):
        with pytest.raises(ValueError, match="fraction_noise_std"):
            AdaClip(Gaussian(1.1), 0.0, 250.0)
        with pytest.raises(ValueError, match="fraction_noise_std"):
            AdaClip(Gaussian(1.1), -0.05, 250.0)

    def test_direct_construction_rejects_non_positive_batch_size(self):
        with pytest.raises(ValueError, match="expected_batch_size"):
            AdaClip(Gaussian(1.1), 0.05, 0.0)
        with pytest.raises(ValueError, match="expected_batch_size"):
            AdaClip(Gaussian(1.1), 0.05, -250.0)

    def test_factory_still_rejects_invalid_params(self):
        with pytest.raises(ValueError, match="fraction_noise_std"):
            dpsgd_acc.adaclip(
                dpsgd_acc.gaussian(1.1),
                fraction_noise_std=0.0,
                expected_batch_size=250,
            )
        with pytest.raises(ValueError, match="expected_batch_size"):
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.1), expected_batch_size=0)
        with pytest.raises(ValueError, match="num_groups"):
            dpsgd_acc.adaclip(
                dpsgd_acc.gaussian(1.1), expected_batch_size=250, num_groups=0
            )


class TestGaussianValidation:
    """A negative multiplier fails on construction, not later inside the
    native PLD call.  ``0.0`` stays valid: documented non-private value."""

    def test_negative_multiplier_rejected_on_construction(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            Gaussian(-1.0)

    def test_factory_rejects_negative_multiplier(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            dpsgd_acc.gaussian(-0.5)

    def test_zero_multiplier_stays_valid(self):
        assert math.isinf(Gaussian(0.0).epsilon_at(1e-5))


class TestCodecCannotProduceUnvalidated:
    """The generic DpProcess codec rebuilds with ``cls(**kwargs)``, so
    ``__post_init__`` validation fires on deserialization."""

    def _adaclip_state(self, **overrides):
        state = {
            "type": "AdaClip",
            "inner": {"type": "Gaussian", "noise_multiplier": 1.1},
            "fraction_noise_std": 0.05,
            "expected_batch_size": 250.0,
            "num_groups": 1,
        }
        state.update(overrides)
        return state

    def test_num_groups_zero_rejected_on_load(self):
        with pytest.raises(ValueError, match="num_groups"):
            from_state_dict(acc.identity(), self._adaclip_state(num_groups=0))

    def test_fraction_noise_std_zero_rejected_on_load(self):
        with pytest.raises(ValueError, match="fraction_noise_std"):
            from_state_dict(acc.identity(), self._adaclip_state(fraction_noise_std=0.0))

    def test_expected_batch_size_negative_rejected_on_load(self):
        with pytest.raises(ValueError, match="expected_batch_size"):
            from_state_dict(
                acc.identity(), self._adaclip_state(expected_batch_size=-5.0)
            )

    def test_nested_negative_gaussian_rejected_on_load(self):
        state = self._adaclip_state(
            inner={"type": "Gaussian", "noise_multiplier": -1.0},
        )
        with pytest.raises(ValueError, match="noise_multiplier"):
            from_state_dict(acc.identity(), state)

    def test_standalone_negative_gaussian_rejected_on_load(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            from_state_dict(
                acc.identity(), {"type": "Gaussian", "noise_multiplier": -2.0}
            )

    def test_valid_adaclip_round_trips(self):
        """Valid checkpoints keep loading unchanged (no over-eager rejection)."""
        proc = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        restored = from_state_dict(acc.identity(), state_dict(proc))
        assert restored == proc

    def test_whole_numbered_float_groups_normalized_to_int(self):
        """``2.0`` is accepted but normalized so the field matches its ``int``
        annotation, including after a codec round-trip."""
        proc = AdaClip(Gaussian(1.1), 0.05, 250.0, num_groups=2.0)
        assert isinstance(proc.num_groups, int)
        restored = from_state_dict(acc.identity(), state_dict(proc))
        assert restored == proc
        assert isinstance(restored.num_groups, int)
