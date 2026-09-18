"""Tests for opake.optimizers._adagrad with DP-aware Φ subtraction."""

from __future__ import annotations

import math

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import adagrad
from opake.optimizers.types import AdagradState
from opake.types import noised

# Closed-form oracle for the DP tests below.  Under a constant gradient
# Adagrad's accumulators are exact: ``v_acc = t·g²`` and ``phi_acc = t·sigma²``,
# so with ``weight_decay=0`` the update is ``−lr·g/(√v_eff + eps)`` for
# ``v_eff = t(g² − sigma²)`` while the signal dominates and ``t·g²`` once it
# does not.
_DP_LR = 0.1
_DP_EPS = 1e-10

# The σ=0.95 case cancels two ~20.0 values down to ~1.95; measured worst-case
# float32 error is 2.2e-07, well inside this bound and well below the 15%
# separation the assertions rely on.
_DP_RTOL = 1e-5


def _expected_update(
    g: float, sigma: float, steps: int, *, lr=_DP_LR, eps=_DP_EPS
) -> float:
    """Adagrad's update after ``steps`` constant-gradient steps."""
    v_acc = steps * g * g
    corrected = v_acc - steps * sigma * sigma
    v_eff = corrected if corrected > 0 else v_acc
    return -lr * g / (math.sqrt(v_eff) + eps)


def _run_constant(opt, g: float, steps: int, sigma: float | None):
    """Drive ``opt`` for ``steps`` steps on a constant gradient ``g``.

    ``sigma=None`` feeds the bare pytree (no DP metadata); otherwise the
    gradients are wrapped with a realized noise stddev.  Returns the final
    update pytree.
    """
    params = {"w": torch.ones(4)}
    grads = {"w": torch.full((4,), g)}
    state = opt.init(params)
    updates = None
    for _ in range(steps):
        step_grads = (
            grads if sigma is None else noised(grads, max_norm=1.0, noise_stddev=sigma)
        )
        updates, state = opt.update(step_grads, state, params=params)
    return updates


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _ada_state(chain_state) -> AdagradState:
    for entry in chain_state:
        if isinstance(entry, AdagradState):
            return entry
    raise AssertionError(f"AdagradState not found in {chain_state!r}")


class TestVanilla:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "eps": 1e-10, "initial_accumulator_value": 0.0},
            {"lr": 5e-3, "eps": 1e-8, "initial_accumulator_value": 0.1},
            {"lr": 0.1, "eps": 1e-6, "initial_accumulator_value": 1.0},
        ],
        ids=["default", "warm_accumulator", "large_accumulator"],
    )
    def test_matches_torchopt_adagrad(self, params, kwargs):
        """Vanilla Adagrad is numerically identical to torchopt.adagrad."""
        opt_opake = adagrad(**kwargs)
        opt_ref = torchopt.adagrad(**kwargs)
        state_opake = opt_opake.init(params)
        state_ref = opt_ref.init(params)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates_opake, state_opake = opt_opake.update(
                step_grads, state_opake, params=params
            )
            updates_ref, state_ref = opt_ref.update(
                step_grads, state_ref, params=params
            )
            for k in params:
                torch.testing.assert_close(updates_opake[k], updates_ref[k])

    def test_v_acc_starts_at_initial_value(self, params):
        opt = adagrad(lr=1e-2, initial_accumulator_value=0.5)
        st = _ada_state(opt.init(params))
        for k in params:
            torch.testing.assert_close(st.v_acc[k], torch.full_like(params[k], 0.5))

    def test_v_acc_accumulates_without_decay(self, params, grads):
        opt = adagrad(lr=1e-2, initial_accumulator_value=0.0)
        state = opt.init(params)
        for _ in range(3):
            _, state = opt.update(grads, state, params=params)
        st = _ada_state(state)
        # After 3 identical-grad steps: v_acc = 3 * g²
        for k in grads:
            torch.testing.assert_close(st.v_acc[k], 3 * grads[k] * grads[k])

    def test_phi_acc_zero_without_noise(self, params, grads):
        opt = adagrad(lr=1e-2)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(grads, state, params=params)
        assert _ada_state(state).phi_acc == 0.0


class TestDPCorrection:
    def test_phi_acc_accumulates_cumulatively(self, params, grads):
        sigma = 0.5
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for t in range(1, 6):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            # Cumulative — every step adds σ², no decay.
            expected = t * (sigma**2)
            phi_acc = _ada_state(state).phi_acc
            assert isinstance(phi_acc, dict)
            assert all(v == pytest.approx(expected) for v in phi_acc.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        expected = 0.0
        for sigma in [0.1, 0.2, 0.3]:
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected += sigma**2
        phi_acc = _ada_state(state).phi_acc
        assert isinstance(phi_acc, dict)
        assert all(v == pytest.approx(expected) for v in phi_acc.values())

    @pytest.mark.parametrize("steps", [5, 20])
    @pytest.mark.parametrize("sigma", [0.5, 0.8, 0.95])
    def test_correction_removes_noise_variance_from_denominator(self, sigma, steps):
        """``Φ_acc`` subtraction removes the noise variance from the denominator.

        Adagrad never decays ``v_acc``, so the denominator would otherwise
        accumulate ``t·σ²`` forever.  The corrected/uncorrected ratio is
        exactly ``√(g²/(g²−σ²))`` — 1.15 at σ=0.5, 3.20 at σ=0.95.

        ``σ`` comparable to ``|g|`` is the ordinary DP regime: per-coordinate
        noise is ``σ_dp·C/B`` ≈ 3.9e-3 at ``C=1, B=256, σ_dp=1``, against
        clipped gradient means of 1e-3–1e-2.
        """
        g = 1.0
        opt_on = adagrad(lr=_DP_LR, eps=_DP_EPS, noise_bias_correction=True)
        opt_off = adagrad(lr=_DP_LR, eps=_DP_EPS, noise_bias_correction=False)
        u_on = _run_constant(opt_on, g, steps, sigma)
        u_off = _run_constant(opt_off, g, steps, sigma)

        # Corrected: denominator tracks t(g² − σ²).
        torch.testing.assert_close(
            u_on["w"],
            torch.full_like(u_on["w"], _expected_update(g, sigma, steps)),
            rtol=_DP_RTOL,
            atol=0,
        )
        # Uncorrected: the same stream still carries t·σ² in the denominator.
        torch.testing.assert_close(
            u_off["w"],
            torch.full_like(u_off["w"], _expected_update(g, 0.0, steps)),
            rtol=_DP_RTOL,
            atol=0,
        )
        # The resulting gain is exactly √(g²/(g²−σ²)), independent of t.
        expected_gain = math.sqrt(g**2 / (g**2 - sigma**2))
        assert (u_on["w"][0] / u_off["w"][0]).item() == pytest.approx(
            expected_gain, rel=_DP_RTOL
        )

    def test_noise_dominant_regime_falls_back_to_uncorrected_v(self):
        """A non-positive correction reverts the coordinate to plain Adagrad.

        ``v_acc − Φ_acc ≤ 0`` carries no signal, so
        ``torch.where(corrected > 0, corrected, v_acc)`` returns the
        uncorrected accumulator — hence bit-equality with plain Adagrad.  The
        trailing magnitude bound separates that from a floor clamp, which
        would divide by ``≈ eps`` and inflate the coordinate by ~1e8.
        """
        g, sigma, steps, lr = 0.01, 1e3, 10, 1e-3
        opt_bc = adagrad(lr=lr, eps=_DP_EPS, noise_bias_correction=True)
        opt_plain = adagrad(lr=lr, eps=_DP_EPS, noise_bias_correction=False)
        u_bc = _run_constant(opt_bc, g, steps, sigma)
        u_plain = _run_constant(opt_plain, g, steps, None)

        assert torch.isfinite(u_bc["w"]).all()
        # Fallback, not floor: bit-identical to plain Adagrad on this stream.
        torch.testing.assert_close(u_bc["w"], u_plain["w"], rtol=0, atol=0)
        torch.testing.assert_close(
            u_bc["w"],
            torch.full_like(u_bc["w"], _expected_update(g, 0.0, steps, lr=lr)),
            rtol=_DP_RTOL,
            atol=0,
        )
        # A clamp to eps² would divide by √eps² + eps = 2·eps.
        floored = lr * g / (2 * _DP_EPS)
        ratio = floored / u_bc["w"].abs().max().item()
        assert ratio > 1e3, (
            f"update is only {ratio:.1e}x smaller than a floor clamp would give; "
            f"the non-positive branch must fall back to the uncorrected v_acc"
        )

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=False)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _ada_state(state).phi_acc == 0.0


class TestWeightDecay:
    def test_decoupled_with_zero_grad(self):
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}
        opt = adagrad(lr=0.1, weight_decay=0.1, initial_accumulator_value=1.0)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # g = 0 → moment-scaled = 0; only WD survives.
        # update = -lr * (0 + wd * params) = -0.01 * 2.0
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)


class TestValidation:
    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps"):
            adagrad(eps=0.0)

    def test_negative_initial_accumulator_raises(self):
        with pytest.raises(ValueError, match="initial_accumulator_value"):
            adagrad(initial_accumulator_value=-1.0)

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValueError, match="weight_decay"):
            adagrad(weight_decay=-0.1)
