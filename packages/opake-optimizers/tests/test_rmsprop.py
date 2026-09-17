"""Tests for opake.optimizers._rmsprop."""

from __future__ import annotations

import math

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import rmsprop
from opake.optimizers.types import RMSpropState
from opake.types import (
    SecondMomentNoiseOutput,
    noised,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _rms_state(chain_state) -> RMSpropState:
    for entry in chain_state:
        if isinstance(entry, RMSpropState):
            return entry
    raise AssertionError(f"RMSpropState not found in {chain_state!r}")


class TestVanilla:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
            {"lr": 5e-3, "alpha": 0.95, "eps": 1e-6, "weight_decay": 0.0},
            {"lr": 0.1, "alpha": 0.9, "eps": 1e-4, "weight_decay": 0.0},
        ],
        ids=["default", "alpha_095", "high_lr"],
    )
    def test_matches_torchopt_rmsprop(self, params, kwargs):
        """Vanilla RMSprop is numerically identical to torchopt.rmsprop."""
        opt_opake = rmsprop(**kwargs)
        opt_ref = torchopt.rmsprop(**kwargs)
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

    def test_state_carries_phi_at_zero(self, params):
        opt = rmsprop(lr=1e-2)
        st = _rms_state(opt.init(params))
        assert isinstance(st, RMSpropState)
        assert st.phi == 0.0
        assert st.step == 0

    def test_v_advances_via_ema(self, params, grads):
        alpha = 0.99
        opt = rmsprop(lr=1e-2, alpha=alpha)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        st = _rms_state(state)
        for k in grads:
            torch.testing.assert_close(st.nu[k], (1 - alpha) * grads[k] * grads[k])

    def test_step_increments(self, params, grads):
        opt = rmsprop(lr=1e-2)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        assert _rms_state(state).step == 1
        _, state = opt.update(grads, state, params=params)
        assert _rms_state(state).step == 2

    def test_apply_updates_changes_params(self, params, grads):
        opt = rmsprop(lr=1e-1)
        orig = {k: v.clone() for k, v in params.items()}
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        new = torchopt.apply_updates(params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in params)


class TestBCMode:
    def test_phi_advances_under_noisy_metadata(self, params, grads):
        alpha = 0.99
        sigma = 0.5
        opt = rmsprop(lr=1e-2, alpha=alpha, noise_bias_correction=True)
        state = opt.init(params)
        expected_phi = 0.0
        for _ in range(10):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = alpha * expected_phi + (1 - alpha) * (sigma**2)
        phi = _rms_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        alpha = 0.99
        opt = rmsprop(lr=1e-2, alpha=alpha, noise_bias_correction=True)
        state = opt.init(params)
        expected_phi = 0.0
        for sigma in [0.1, 0.2, 0.3, 0.2, 0.1]:
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = alpha * expected_phi + (1 - alpha) * (sigma**2)
        phi = _rms_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    @pytest.mark.parametrize("steps", [5, 20])
    @pytest.mark.parametrize("sigma", [0.5, 0.8, 0.95])
    def test_correction_removes_noise_variance_from_denominator(self, sigma, steps):
        """``φ`` subtraction removes the noise variance from the denominator.

        Under a constant gradient ``ν_t = (1−α^t)·g²`` and
        ``φ_t = (1−α^t)·σ²``, so with ``weight_decay=0`` the update is
        ``−lr·g/(√ν_eff + eps)`` and the corrected/uncorrected ratio is
        exactly ``√(g²/(g²−σ²))`` — 1.15 at ``σ/|g| = 0.5``, 3.20 at 0.95.
        Ratios near zero carry no signal: at 1e-3 the correction moves the
        update by ~5e-07.
        """
        g, lr, eps, alpha = 1.0, 1e-2, 1e-8, 0.9
        params = {"w": torch.ones(4)}
        grads = {"w": torch.full((4,), g)}
        opt_on = rmsprop(lr=lr, alpha=alpha, eps=eps, noise_bias_correction=True)
        opt_off = rmsprop(lr=lr, alpha=alpha, eps=eps, noise_bias_correction=False)
        s_on, s_off = opt_on.init(params), opt_off.init(params)
        for _ in range(steps):
            u_on, s_on = opt_on.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma), s_on, params=params
            )
            u_off, s_off = opt_off.update(grads, s_off, params=params)

        # rtol=1e-5: worst-case float32 error here is 1.5e-07, well inside the
        # 15% separation between the corrected and uncorrected forms.
        bc = 1.0 - alpha**steps
        expected_on = -lr * g / (math.sqrt(bc * (g**2 - sigma**2)) + eps)
        expected_off = -lr * g / (math.sqrt(bc * g**2) + eps)
        torch.testing.assert_close(
            u_on["w"], torch.full_like(u_on["w"], expected_on), rtol=1e-5, atol=0
        )
        torch.testing.assert_close(
            u_off["w"], torch.full_like(u_off["w"], expected_off), rtol=1e-5, atol=0
        )
        # The gain is exactly √(g²/(g²−σ²)), independent of t and α.
        assert (u_on["w"][0] / u_off["w"][0]).item() == pytest.approx(
            math.sqrt(g**2 / (g**2 - sigma**2)), rel=1e-5
        )

    def test_noise_dominant_regime_falls_back_to_uncorrected_v(self):
        """A non-positive correction reverts the coordinate to plain RMSprop.

        ``ν − φ ≤ 0`` carries no signal, so
        ``torch.where(corrected > 0, corrected, ν)`` returns the uncorrected
        accumulator — hence bit-equality with plain RMSprop.  The trailing
        magnitude bound separates that from a ``bc_floor`` clamp, which would
        divide by ``≈ eps``.
        """
        lr, eps, sigma, steps = 1e-3, 1e-8, 1e6, 5
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        opt_bc = rmsprop(lr=lr, eps=eps, noise_bias_correction=True)
        opt_plain = rmsprop(lr=lr, eps=eps, noise_bias_correction=False)
        st_bc, st_plain = opt_bc.init(params), opt_plain.init(params)
        for _ in range(steps):
            u_bc, st_bc = opt_bc.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                st_bc,
                params=params,
            )
            u_plain, st_plain = opt_plain.update(grads, st_plain, params=params)

        assert torch.isfinite(u_bc["w"]).all()
        # Fallback, not floor: bit-identical to plain RMSprop on this stream.
        torch.testing.assert_close(u_bc["w"], u_plain["w"], rtol=0, atol=0)
        # A floor at bc_floor = eps² would divide by √eps² + eps = 2·eps.
        floored = lr * grads["w"].abs().max().item() / (2 * eps)
        ratio = floored / u_bc["w"].abs().max().item()
        assert ratio > 1e3, (
            f"update is only {ratio:.1e}x smaller than what a bc_floor clamp "
            f"would produce; the non-positive branch should fall back to the "
            f"uncorrected nu, not clamp to a floor"
        )

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        opt = rmsprop(lr=1e-2, noise_bias_correction=False)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _rms_state(state).phi == 0.0


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_consumes_external_g_squared(self, params, grads, sq_grads):
        alpha = 0.99
        opt = rmsprop(lr=1e-2, alpha=alpha)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = opt.update(output, state, params=params)
        st = _rms_state(state)
        for k in params:
            torch.testing.assert_close(st.nu[k], (1 - alpha) * sq_grads[k])

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        opt = rmsprop(lr=1e-2)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = opt.update(output, state, params=params)
        for k in updates:
            assert torch.isfinite(updates[k]).all()
            assert updates[k].abs().max().item() < 10.0

    def test_explicit_second_moment_kwarg_rejected(self, params, grads, sq_grads):
        opt = rmsprop(lr=1e-2)
        state = opt.init(params)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(
                grads,
                state,
                params=params,
                noisy_squared_grads=sq_grads,
            )


class TestValidation:
    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            rmsprop(alpha=-0.1)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            rmsprop(alpha=1.0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps"):
            rmsprop(eps=0.0)
