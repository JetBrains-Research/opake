"""Tests for opaque.optimizers._adadelta — Adadelta with two-EMA DP BC.

Adadelta's two EMAs both pick up DP noise:

- ``E[g²]`` inherits the same per-step σ² offset as Adam's ``v``.
- ``E[Δx²]`` accumulates ``coef² · σ²`` per element because
  ``Δx = -coef · g̃`` is linear in the noised gradient.

The opaque-built variant maintains two parallel φ-EMAs at the same
decay ``ρ`` to subtract both biases.  Tests assert:

- φ_g converges to σ² (the homogeneous-σ steady state).
- φ_dx is non-negative and per-element (tensor-shaped state).
- Per-group σ routes correctly into per-leaf φ_g.
- BC subtraction actually changes the resulting update vs vanilla.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import adadelta
from opaque.optimizers.types import AdadeltaState
from opaque.pytree import tree_map
from opaque.types import (
    PerGroup,
    SecondMomentNoiseOutput,
    clipped,  # noqa: F401
    noised,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _state(chain_state) -> AdadeltaState:
    for entry in chain_state:
        if isinstance(entry, AdadeltaState):
            return entry
    raise AssertionError(f"AdadeltaState not found in {chain_state!r}")


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------


class TestVanillaAdadelta:
    def test_state_init(self, params):
        opt = adadelta()
        state = opt.init(params)
        s = _state(state)
        assert s.step == 0
        # When noise_bias_correction=False (default), phi fields should be None
        assert s.phi_g is None
        assert s.phi_dx is None

    def test_state_init_with_bc(self, params):
        """When noise_bias_correction=True, phi fields are allocated."""
        opt = adadelta(noise_bias_correction=True)
        state = opt.init(params)
        s = _state(state)
        assert s.step == 0
        # phi_g should be allocated as a dict (per-leaf)
        assert isinstance(s.phi_g, dict)
        assert set(s.phi_g.keys()) == {(k,) for k in params}
        # phi_dx is per-leaf tensor matching params shape
        assert set(s.phi_dx) == set(params)
        for k in params:
            assert s.phi_dx[k].shape == params[k].shape
            assert torch.all(s.phi_dx[k] == 0)

    def test_step_increments(self, params, grads):
        opt = adadelta()
        state = opt.init(params)
        for expected in [1, 2, 3]:
            _, state = opt.update(grads, state, params=params)
            assert _state(state).step == expected

    def test_apply_updates_changes_params(self, params, grads):
        opt = adadelta(lr=1.0)
        orig = {k: v.clone() for k, v in params.items()}
        state = opt.init(params)
        # Adadelta needs a few steps to build up RMS[Δx]; first update
        # is small (RMS[Δx]_0 = √eps).
        for _ in range(10):
            updates, state = opt.update(grads, state, params=params)
        new = torchopt.apply_updates(params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in params)

    def test_v_g_accumulates(self, params, grads):
        """``E[g²]`` accumulates with decay ρ."""
        opt = adadelta(rho=0.9)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        s = _state(state)
        # After one step: v_g = (1-ρ)·g²
        for k in params:
            expected = 0.1 * grads[k].pow(2)
            torch.testing.assert_close(s.v_g[k], expected, rtol=1e-6, atol=1e-7)

    def test_finite_updates(self, params, grads):
        opt = adadelta()
        state = opt.init(params)
        for _ in range(5):
            updates, state = opt.update(grads, state, params=params)
            for k in params:
                assert torch.isfinite(updates[k]).all()


# ---------------------------------------------------------------------------
# DP bias correction
# ---------------------------------------------------------------------------


class TestNoiseBiasCorrection:
    """``noise_bias_correction=True`` activates both φ-EMAs."""

    def test_phi_g_converges_to_sigma_sq(self, params):
        """Homogeneous σ → φ_g approaches σ² in steady state."""
        sigma = 0.5
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(200):
            torch.manual_seed(0)
            grads = {k: torch.randn_like(v) for k, v in params.items()}
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
        s = _state(state)
        # 200 steps at ρ=0.9 → 1-0.9^200 ≈ 1.0; should be very close to σ².
        assert isinstance(s.phi_g, dict)
        assert all(v == pytest.approx(sigma**2, rel=1e-3) for v in s.phi_g.values())

    def test_phi_dx_is_nonneg_per_element(self, params, grads):
        """φ_dx is a per-element tensor and stays non-negative."""
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=params,
            )
        s = _state(state)
        for k in params:
            assert s.phi_dx[k].shape == params[k].shape
            assert torch.all(s.phi_dx[k] >= 0)
            # phi_dx must be > 0 somewhere by step 20 with σ > 0.
            assert s.phi_dx[k].max() > 0

    def test_phi_g_stays_none_when_bc_off(self, params, grads):
        opt = adadelta(rho=0.9, noise_bias_correction=False)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=params,
            )
        s = _state(state)
        assert s.phi_g is None
        assert s.phi_dx is None

    def test_phi_stays_none_under_clean_grads_when_bc_off(self, params, grads):
        opt = adadelta(rho=0.9, noise_bias_correction=False)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(grads, state, params=params)
        s = _state(state)
        # BC off: phi fields should remain None
        assert s.phi_g is None
        assert s.phi_dx is None

    def test_phi_stays_zero_under_clean_grads_when_bc_on(self, params, grads):
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(grads, state, params=params)
        s = _state(state)
        # BC on but clean grads: phi should be initialized but stay at zero
        assert isinstance(s.phi_g, dict)
        assert all(v == 0.0 for v in s.phi_g.values())
        for k in params:
            assert torch.all(s.phi_dx[k] == 0)

    def test_bc_changes_update(self, params, grads):
        """BC subtraction actually changes the resulting update."""
        sigma = 0.5
        opt_bc = adadelta(rho=0.9, noise_bias_correction=True)
        opt_no = adadelta(rho=0.9, noise_bias_correction=False)
        s_bc = opt_bc.init(params)
        s_no = opt_no.init(params)
        # Build up state.
        for _ in range(30):
            _, s_bc = opt_bc.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                s_bc,
                params=params,
            )
            _, s_no = opt_no.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                s_no,
                params=params,
            )
        u_bc, _ = opt_bc.update(
            noised(grads, max_norm=1.0, noise_stddev=sigma), s_bc, params=params
        )
        u_no, _ = opt_no.update(
            noised(grads, max_norm=1.0, noise_stddev=sigma), s_no, params=params
        )
        any_diff = any(not torch.allclose(u_bc[k], u_no[k]) for k in params)
        assert any_diff


# ---------------------------------------------------------------------------
# Per-group BC
# ---------------------------------------------------------------------------


class TestPerGroupBC:
    """Per-group σ routes to per-leaf φ_g entries."""

    def test_phi_g_dict_per_leaf(self, params, grads):
        pg = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 0.2, "mlp": 0.8},
        )
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(50):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=params,
            )
        s = _state(state)
        assert isinstance(s.phi_g, dict)
        # Steady-state ratio: (0.8/0.2)² = 16.
        assert s.phi_g[("weight",)] > 0
        assert s.phi_g[("bias",)] / s.phi_g[("weight",)] == pytest.approx(
            16.0, rel=1e-3
        )


# ---------------------------------------------------------------------------
# Second-moment substitution
# ---------------------------------------------------------------------------


class TestSecondMomentSubstitution:
    """``SecondMomentNoiseOutput`` substitutes the privatised g² into v_g.

    φ_dx is intentionally frozen on this branch — σ isn't carried with
    the second-moment stream, so the update-noise EMA cannot advance
    consistently.  Documented trade-off (use ``NoisedPytree`` for full
    BC).
    """

    def test_substitution_runs_and_freezes_phi(self, params, grads):
        sq = {k: v.pow(2) for k, v in grads.items()}
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        first = noised(grads, max_norm=1.0, noise_stddev=0.5)
        second = noised(sq, max_norm=1.0, noise_stddev=0.5)
        out = SecondMomentNoiseOutput(noisy_grads=first, noisy_squared_grads=second)
        for _ in range(20):
            _, state = opt.update(out, state, params=params)
        s = _state(state)
        # Both phi EMAs stay at zero in the substitution branch.
        assert isinstance(s.phi_g, dict)
        assert all(v == 0.0 for v in s.phi_g.values())
        for k in params:
            assert torch.all(s.phi_dx[k] == 0)

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        lr = 1.0
        opt = adadelta(lr=lr, rho=0.9)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        for _ in range(8):
            updates, state = opt.update(output, state, params=params)
            for k in updates:
                assert torch.isfinite(updates[k]).all()
                atol = torch.finfo(updates[k].dtype).eps
                assert torch.all(updates[k].abs() <= lr * grads[k].abs() + atol)

    def test_mode_switch_does_not_double_correct(self, params, grads):
        """``NoisedPytree`` → ``SecondMomentNoiseOutput`` mid-run must not
        subtract a stale φ_g/φ_dx from the already-debiased v_g/v_dx.

        Regression for Copilot review #3191963511 — without the
        ``noisy_squared_grads is None`` guards, the carried-over EMAs
        would silently apply on top of the post-processing-debiased
        second moment, producing a wrong update direction.  Bit-equality
        against a φ-zeroed clone of the same state (identical v_g/v_dx/
        step) pins this: any guard removal makes the two updates differ.
        """
        sigma = 0.5
        opt = adadelta(rho=0.9, noise_bias_correction=True)
        state = opt.init(params)
        # Build up phi_g and phi_dx via NoisedPytree updates.
        for _ in range(30):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
        s_before_switch = _state(state)
        assert isinstance(s_before_switch.phi_g, dict)
        assert all(v > 0 for v in s_before_switch.phi_g.values())
        assert all(s_before_switch.phi_dx[k].max() > 0 for k in params)
        # Same state with φ forced to zero — identical v_g/v_dx/step, only
        # the bias EMAs differ.  If the substitution branch honours its
        # guards, φ is never consulted and the two updates are bit-equal;
        # if a guard is removed, the live run subtracts the stale φ from
        # the already-debiased accumulators and the updates diverge.
        mom_zeroed = dataclasses.replace(
            s_before_switch,
            phi_g=dict.fromkeys(s_before_switch.phi_g, 0.0),
            phi_dx=tree_map(torch.zeros_like, s_before_switch.phi_dx),
        )
        state_zeroed = tuple(
            mom_zeroed if entry is s_before_switch else entry for entry in state
        )
        sq = {k: v.pow(2) for k, v in grads.items()}
        first = noised(grads, max_norm=1.0, noise_stddev=sigma)
        second = noised(sq, max_norm=1.0, noise_stddev=sigma)
        out = SecondMomentNoiseOutput(noisy_grads=first, noisy_squared_grads=second)
        updates_live, state_after_live = opt.update(out, state, params=params)
        updates_zeroed, _ = opt.update(out, state_zeroed, params=params)
        for k in params:
            assert torch.isfinite(updates_live[k]).all()
            assert torch.equal(updates_live[k], updates_zeroed[k])
        # The substitution step must also freeze φ (documented branch
        # contract): carried-over EMAs survive untouched, not decayed.
        s_after_switch = _state(state_after_live)
        assert s_after_switch.phi_g == s_before_switch.phi_g
        for k in params:
            assert torch.equal(s_after_switch.phi_dx[k], s_before_switch.phi_dx[k])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_eps_positive(self):
        with pytest.raises(ValueError, match="eps"):
            adadelta(eps=0.0)

    def test_rho_range(self):
        with pytest.raises(ValueError, match="rho"):
            adadelta(rho=1.0)

    def test_negative_weight_decay(self):
        with pytest.raises(ValueError, match="weight_decay"):
            adadelta(weight_decay=-1.0)

    def test_update_rms_clip_positive(self):
        with pytest.raises(ValueError, match="update_rms_clip"):
            adadelta(update_rms_clip=-0.5)
