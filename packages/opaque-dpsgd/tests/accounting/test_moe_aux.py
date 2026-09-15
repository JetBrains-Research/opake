"""``moe_aux``: the joint gradient-plus-load Gaussian of the MoE release."""

from __future__ import annotations

import math

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal
from opaque.dpsgd.accounting.mechanisms.types import MoeAux
from opaque.serialization import from_state_dict, state_dict

_DELTA = 1e-5


class TestConstructor:
    def test_returns_moe_aux_with_default_ratio(self):
        result = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(0.8))
        assert isinstance(result, MoeAux)
        assert result.ratio == pytest.approx(0.02)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.moe_aux(acc.eps_delta(1.0))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Gaussian"):
            MoeAux(inner=acc.eps_delta(1.0))  # type: ignore[arg-type]

    @pytest.mark.parametrize("ratio", [0.0, -0.1, math.inf, math.nan])
    def test_rejects_bad_ratio(self, ratio):
        with pytest.raises(ValueError, match="ratio"):
            dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.0), ratio=ratio)

    def test_rejects_non_numeric_ratio(self):
        with pytest.raises(TypeError, match="ratio"):
            dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.0), ratio="0.02")  # type: ignore[arg-type]

    def test_nonprivate_inner_stays_nonprivate(self):
        m = dpsgd_acc.moe_aux(acc.nonprivate())
        assert m.effective_noise_multiplier == 0.0
        assert (
            dpsgd_acc.moe_aux(dpsgd_acc.gaussian(0.0)).effective_noise_multiplier == 0.0
        )


class TestEffectiveNoiseMultiplier:
    def test_formula(self):
        for nm, ratio in [(1.1, 0.02), (0.5622, 0.02), (2.0, 1.0)]:
            m = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(nm), ratio=ratio)
            assert m.effective_noise_multiplier == pytest.approx(
                nm / math.sqrt(1 + ratio)
            )

    def test_whitened_sensitivities_add_in_quadrature(self):
        """``1/nm_eff² = 1/nm² + ratio/nm²``: the load half costs ``ratio`` of the gradient half."""
        nm, ratio = 0.9, 0.3
        m = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(nm), ratio=ratio)
        assert 1 / m.effective_noise_multiplier**2 == pytest.approx(
            1 / nm**2 + ratio / nm**2
        )

    def test_default_costs_one_percent_of_gradient_noise(self):
        m = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.0))
        assert 1.0 / m.effective_noise_multiplier == pytest.approx(math.sqrt(1.02))

    def test_vanishing_ratio_recovers_inner(self):
        base = dpsgd_acc.gaussian(1.0)
        m = dpsgd_acc.moe_aux(base, ratio=1e-9)
        assert m.effective_noise_multiplier == pytest.approx(1.0, abs=1e-8)
        assert m.epsilon_at(_DELTA) == pytest.approx(base.epsilon_at(_DELTA), rel=1e-6)

    def test_costs_more_than_inner(self):
        base = dpsgd_acc.gaussian(0.8)
        m = dpsgd_acc.moe_aux(base, ratio=0.5)
        assert m.epsilon_at(_DELTA) > base.epsilon_at(_DELTA)


class TestAmplifiers:
    """Every amplifier prices MoeAux as the Gaussian at the joint multiplier."""

    def _pair(self, ratio=0.1):
        m = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.0), ratio=ratio)
        g = dpsgd_acc.gaussian(m.effective_noise_multiplier)
        return m, g

    def test_poisson(self):
        m, g = self._pair()
        assert dpsgd_acc.poisson(m, 0.01).epsilon_at(_DELTA) == pytest.approx(
            dpsgd_acc.poisson(g, 0.01).epsilon_at(_DELTA), abs=1e-9
        )

    def test_truncated_poisson(self):
        m, g = self._pair()
        kwargs = {"truncated_batch_size": 64, "dataset_size": 10_000}
        assert dpsgd_acc.poisson(m, 0.005, **kwargs).epsilon_at(
            _DELTA
        ) == pytest.approx(
            dpsgd_acc.poisson(g, 0.005, **kwargs).epsilon_at(_DELTA), abs=1e-9
        )

    def test_truncated_poisson_nonprivate_inner(self):
        m = dpsgd_acc.moe_aux(acc.nonprivate())
        step = dpsgd_acc.poisson(m, 0.005, truncated_batch_size=64, dataset_size=10_000)
        assert math.isinf(step.epsilon_at(_DELTA))

    def test_parallel_poisson(self):
        m, g = self._pair()
        assert dpsgd_acc.parallel_poisson(m, 0.01, 4).epsilon_at(
            _DELTA
        ) == pytest.approx(
            dpsgd_acc.parallel_poisson(g, 0.01, 4).epsilon_at(_DELTA), abs=1e-9
        )

    def test_k_out_of_t(self):
        m, g = self._pair()
        assert dpsgd_acc.k_out_of_t(m, k=2, t=16, allocation="block").epsilon_at(
            _DELTA
        ) == pytest.approx(
            dpsgd_acc.k_out_of_t(g, k=2, t=16, allocation="block").epsilon_at(_DELTA),
            abs=1e-9,
        )

    def test_composition_over_steps(self):
        m, g = self._pair()
        run_m = dpsgd_acc.poisson(m, 0.01) * 200
        run_g = dpsgd_acc.poisson(g, 0.01) * 200
        assert run_m.epsilon_at(_DELTA) == pytest.approx(
            run_g.epsilon_at(_DELTA), abs=1e-8
        )


class TestCalibration:
    def test_calibrated_multiplier_is_inflated_by_sqrt_one_plus_ratio(self):
        """At a fixed budget the gradient multiplier grows by exactly ``sqrt(1 + ratio)``."""
        ratio = 0.5
        budget = cal.epsilon_budget(4.0, delta=_DELTA)
        plain = cal.calibrate(
            budget,
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.02) * 100,
            0.3,
            5.0,
            tolerance=1e-4,
        ).param
        joint = cal.calibrate(
            budget,
            lambda nm: (
                dpsgd_acc.poisson(
                    dpsgd_acc.moe_aux(dpsgd_acc.gaussian(nm), ratio=ratio), 0.02
                )
                * 100
            ),
            0.3,
            5.0,
            tolerance=1e-4,
        ).param
        assert joint == pytest.approx(plain * math.sqrt(1 + ratio), rel=2e-3)


class TestCodec:
    def test_round_trip(self):
        m = dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.1), ratio=0.05)
        restored = from_state_dict(acc.identity(), state_dict(m))
        assert restored == m
        assert restored.effective_noise_multiplier == pytest.approx(
            m.effective_noise_multiplier
        )

    def test_zero_ratio_rejected_on_load(self):
        state = {
            "type": "MoeAux",
            "inner": {"type": "Gaussian", "noise_multiplier": 1.1},
            "ratio": 0.0,
        }
        with pytest.raises(ValueError, match="ratio"):
            from_state_dict(acc.identity(), state)
