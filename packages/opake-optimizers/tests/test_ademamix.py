"""Tests for opake.optimizers._ademamix."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import ademamix
from opake.optimizers.types import AdEMAMixState
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


def _ame(chain_state) -> AdEMAMixState:
    return chain_state[0]


class TestVanillaAdEMAMix:
    def test_state_has_two_first_moments(self, params):
        opt = ademamix(lr=1e-3)
        st = _ame(opt.init(params))
        assert isinstance(st, AdEMAMixState)
        for k in params:
            assert torch.equal(st.m_fast[k], torch.zeros_like(params[k]))
            assert torch.equal(st.m_slow[k], torch.zeros_like(params[k]))
            assert torch.equal(st.nu[k], torch.zeros_like(params[k]))

    def test_emas_advance_with_distinct_betas(self, params, grads):
        b1, b2, b3 = 0.9, 0.999, 0.9999
        opt = ademamix(lr=1e-3, betas=(b1, b2, b3), alpha=5.0, weight_decay=0.0)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        st = _ame(state)
        for k in grads:
            torch.testing.assert_close(st.m_fast[k], (1 - b1) * grads[k])
            torch.testing.assert_close(st.m_slow[k], (1 - b3) * grads[k])
            torch.testing.assert_close(st.nu[k], (1 - b2) * grads[k] * grads[k])

    def test_alpha_zero_reduces_to_adam(self, params, grads):
        """With α=0 and matched β's, AdEMAMix's update equals AdamW's."""
        from opake.optimizers import adamw

        opt_ame = ademamix(
            lr=1e-3,
            betas=(0.9, 0.999, 0.9999),
            alpha=0.0,
            weight_decay=0.01,
        )
        opt_adam = adamw(
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        s_ame = opt_ame.init(params)
        s_adam = opt_adam.init(params)
        torch.manual_seed(7)
        for _ in range(5):
            g = {k: torch.randn_like(v) for k, v in params.items()}
            u_ame, s_ame = opt_ame.update(g, s_ame, params=params)
            u_adam, s_adam = opt_adam.update(g, s_adam, params=params)
            for k in params:
                torch.testing.assert_close(u_ame[k], u_adam[k])


class TestBCMode:
    def test_phi_advances_under_noisy_metadata(self, params, grads):
        b2 = 0.999
        sigma = 0.4
        opt = ademamix(lr=1e-3, betas=(0.9, b2, 0.9999), noise_bias_correction=True)
        state = opt.init(params)
        expected_phi = 0.0
        for _ in range(8):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        phi = _ame(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        opt = ademamix(lr=1e-3, noise_bias_correction=False)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.4),
            state,
            params=params,
        )
        assert _ame(state).phi == 0.0


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.05 for k, v in grads.items()}

    def test_consumes_external_g_squared(self, params, grads, sq_grads):
        b2 = 0.999
        opt = ademamix(lr=1e-3, betas=(0.9, b2, 0.9999))
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = opt.update(output, state, params=params)
        st = _ame(state)
        for k in params:
            torch.testing.assert_close(st.nu[k], (1 - b2) * sq_grads[k])

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        opt = ademamix(lr=1e-3)
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
        opt = ademamix(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(
                grads,
                state,
                params=params,
                noisy_squared_grads=sq_grads,
            )


class TestValidation:
    def test_three_betas_required(self):
        with pytest.raises(ValueError, match="three"):
            ademamix(betas=(0.9, 0.999))  # type: ignore[arg-type]

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ademamix(alpha=-1.0)
