"""Tests for opaque.optimizers._radam — Rectified Adam with DP bias correction.

Two execution branches gate the optimizer:

- Early phase (``ρ_t ≤ 5``): SGD-of-momentum.  No second-moment scaling,
  naturally DP-robust.
- Rectified phase (``ρ_t > 5``): Adam-shaped update with the
  rectification factor ``r_t`` and (optionally) φ-EMA bias correction.

Tests assert that the φ-EMA advances *through* the early branch so the
correction at the first rectified step reflects all prior noise
contributions to ``v``.
"""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.api.optimizers._radam import _rectification, _rho_t
from opaque.optimizers import radam
from opaque.optimizers.types import RAdamState
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


def _state(chain_state) -> RAdamState:
    for entry in chain_state:
        if isinstance(entry, RAdamState):
            return entry
    raise AssertionError(f"RAdamState not found in chain state {chain_state!r}")


# ---------------------------------------------------------------------------
# Rectification math
# ---------------------------------------------------------------------------


class TestRectificationMath:
    """``ρ_t`` and ``r_t`` are pure functions of ``(β₂, t)``."""

    def test_rho_inf_matches_paper(self):
        """``ρ_∞ = 2/(1−β₂) − 1``; β₂=0.999 → ρ_∞ = 1999."""
        # The implementation embeds ρ_∞ in _rho_t indirectly; test by
        # taking t→large where ρ_t → ρ_∞.
        b2 = 0.999
        rho_at_huge_t = _rho_t(b2, 10_000)
        expected_rho_inf = 2.0 / (1.0 - b2) - 1.0
        assert rho_at_huge_t == pytest.approx(expected_rho_inf, rel=1e-3)

    def test_rectification_warmup_transition(self):
        """The default β₂ enters the rectified branch at step 6."""
        for t in range(1, 6):
            assert _rectification(b2=0.999, t=t) is None
        assert _rectification(b2=0.999, t=6) is not None

    def test_rectification_defined_long_term(self):
        """For ``t`` past warmup, ``r_t`` is real and converges to 1."""
        b2 = 0.999
        # ρ_t > 5 well before t=20 for β₂=0.999.
        r_20 = _rectification(b2, 20)
        assert r_20 is not None
        assert 0 < r_20 < 1
        # As t→∞, r_t should approach 1.
        r_huge = _rectification(b2, 10_000)
        assert r_huge == pytest.approx(1.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Vanilla RAdam
# ---------------------------------------------------------------------------


class TestVanillaRAdam:
    def test_state_init(self, params):
        opt = radam(lr=1e-3)
        state = opt.init(params)
        s = _state(state)
        assert s.step == 0
        assert s.phi == 0.0
        assert set(s.mu) == set(params)
        assert set(s.nu) == set(params)

    def test_step_increments(self, params, grads):
        opt = radam(lr=1e-3)
        state = opt.init(params)
        for expected in [1, 2, 3]:
            _, state = opt.update(grads, state, params=params)
            assert _state(state).step == expected

    def test_apply_updates_changes_params(self, params, grads):
        opt = radam(lr=1e-1)
        orig = {k: v.clone() for k, v in params.items()}
        state = opt.init(params)
        # Run past the warmup so the update is non-trivial.
        for _ in range(20):
            updates, state = opt.update(grads, state, params=params)
        new = torchopt.apply_updates(params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in params)

    def test_finite_updates_through_warmup(self, params, grads):
        """Updates stay finite across the warmup→rectified transition."""
        opt = radam(lr=1e-3)
        state = opt.init(params)
        for _ in range(20):
            updates, state = opt.update(grads, state, params=params)
            for k in params:
                assert torch.isfinite(updates[k]).all(), f"non-finite update for {k}"


# ---------------------------------------------------------------------------
# DP bias correction
# ---------------------------------------------------------------------------


class TestNoiseBiasCorrection:
    """``noise_bias_correction=True`` advances φ through the warmup."""

    def test_phi_advances_in_warmup(self, params, grads):
        """φ accumulates through the SGD-of-momentum branch.

        v advances every step (we always update it) so φ must too — the
        correction at the first rectified step has to reflect all prior
        noise.  This is the property that distinguishes opaque-built
        DP-RAdam from a naive port that only tracks φ when r_t is real.
        """
        sigma = 0.5
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        # Five warmup steps at β₂=0.999.
        for _ in range(5):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
        s = _state(state)
        # φ must be > 0 even in warmup; expected = (1-(β₂)^5)·σ²
        b2 = 0.999
        expected_phi = (1.0 - b2**5) * sigma**2
        assert isinstance(s.phi, dict)
        assert all(v == pytest.approx(expected_phi, rel=1e-4) for v in s.phi.values())

    def test_phi_stays_zero_when_bc_off(self, params, grads):
        """φ does not advance when ``noise_bias_correction=False``."""
        opt = radam(lr=1e-3, noise_bias_correction=False)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=params,
            )
        assert _state(state).phi == 0.0

    def test_phi_stays_zero_under_clean_grads(self, params, grads):
        """φ stays at zero when no NoisedPytree updates flow."""
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(20):
            _, state = opt.update(grads, state, params=params)
        phi = _state(state).phi
        assert isinstance(phi, dict)
        assert all(v == 0.0 for v in phi.values())

    def test_bc_changes_rectified_update(self, params, grads):
        """With non-zero σ past warmup, BC actually changes the update."""
        sigma = 0.5
        opt_bc = radam(lr=1e-3, noise_bias_correction=True)
        opt_no = radam(lr=1e-3, noise_bias_correction=False)
        s_bc = opt_bc.init(params)
        s_no = opt_no.init(params)
        # Warm up to the rectified branch.
        for _ in range(20):
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
    """Per-group ``noise_stddev`` routes to per-leaf φ."""

    def test_phi_dict_diverges_per_group(self, params, grads):
        pg = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 0.2, "mlp": 0.8},
        )
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(50):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=params,
            )
        s = _state(state)
        assert isinstance(s.phi, dict)
        # Variance ratio is (0.8/0.2)² = 16.
        assert s.phi[("weight",)] > 0
        assert s.phi[("bias",)] / s.phi[("weight",)] == pytest.approx(16.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Private second-moment substitution
# ---------------------------------------------------------------------------


class TestSecondMomentSubstitution:
    """``SecondMomentNoiseOutput`` substitutes the privatised g² stream."""

    def test_substitution_path_runs(self, params, grads):
        sq = {k: v.pow(2) for k, v in grads.items()}
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        first = noised(grads, max_norm=1.0, noise_stddev=0.5)
        second = noised(sq, max_norm=1.0, noise_stddev=0.5)
        out = SecondMomentNoiseOutput(noisy_grads=first, noisy_squared_grads=second)
        for _ in range(20):
            _, state = opt.update(out, state, params=params)
        # Substitution branch leaves φ at zero (post-processing already
        # debiased v).
        phi = _state(state).phi
        assert isinstance(phi, dict)
        assert all(v == 0.0 for v in phi.values())

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        lr = 1e-3
        betas = (0.9, 0.999)
        opt = radam(lr=lr, betas=betas)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        for step in range(1, 9):
            updates, state = opt.update(output, state, params=params)
            r_t = _rectification(b2=betas[1], t=step)
            for update in updates.values():
                assert torch.isfinite(update).all()
                if r_t is not None:
                    atol = torch.finfo(update.dtype).eps
                    assert update.abs().max().item() <= lr * r_t + atol


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_eps_positive(self):
        with pytest.raises(ValueError, match="eps"):
            radam(eps=0.0)

    def test_betas_two_values(self):
        with pytest.raises(ValueError, match="exactly two"):
            radam(betas=(0.9,))  # type: ignore[arg-type]

    def test_beta_range(self):
        with pytest.raises(ValueError, match="beta_2"):
            radam(betas=(0.9, 1.0))

    def test_negative_weight_decay(self):
        with pytest.raises(ValueError, match="weight_decay"):
            radam(weight_decay=-1.0)

    def test_update_rms_clip_positive(self):
        with pytest.raises(ValueError, match="update_rms_clip"):
            radam(update_rms_clip=0.0)
