"""Tests for opaque.optimizers._adamw — universal Adam / AdamW.

Covers all four orthogonal modes:

- vanilla AdamW (no DP kwargs at update)
- DP-AdamW-BC (``NoisedPytree`` metadata)
- DP-Adam with projected JME moments (``SecondMomentNoiseOutput``)
- StableAdamW (``update_rms_clip`` constructor knob)

Plus the L2 weight decay branch (``decoupled_weight_decay=False``).
"""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import adam, adamw
from opaque.optimizers.types import AdamState
from opaque.types import (
    PerGroup,
    SecondMomentNoiseOutput,
    clipped,
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


def _adam_state(chain_state) -> AdamState:
    """Pull out the ``AdamState`` from the chain tuple.

    Position depends on the chain layout: position 0 in the decoupled
    chain (``moment → wd → lr``) and position 1 in the L2 chain
    (``wd → moment → lr``).
    """
    for entry in chain_state:
        if isinstance(entry, AdamState):
            return entry
    raise AssertionError(f"AdamState not found in chain state {chain_state!r}")


# ---------------------------------------------------------------------------
# Vanilla AdamW
# ---------------------------------------------------------------------------


class TestVanillaAdamW:
    def test_returns_gradient_transformation(self):
        opt = adamw(lr=1e-3)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_state_carries_phi_even_when_unused(self, params):
        opt = adamw(lr=1e-3)
        adam = _adam_state(opt.init(params))
        assert isinstance(adam, AdamState)
        assert adam.phi == 0.0
        assert adam.step == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
            {"lr": 0.1, "betas": (0.85, 0.99), "eps": 1e-6, "weight_decay": 0.0},
            {"lr": 5e-4, "weight_decay": 0.1},
        ],
        ids=["default", "high_lr_no_wd", "heavy_wd"],
    )
    def test_matches_torchopt_adamw(self, params, kwargs):
        """``adamw`` (no DP kwargs) is numerically identical to torchopt.adamw."""
        opt_dp = adamw(**kwargs)
        opt_ref = torchopt.adamw(**kwargs)
        s_dp = opt_dp.init(params)
        s_ref = opt_ref.init(params)
        torch.manual_seed(42)
        for _ in range(10):
            g = {k: torch.randn_like(v) for k, v in params.items()}
            u_dp, s_dp = opt_dp.update(g, s_dp, params=params)
            u_ref, s_ref = opt_ref.update(g, s_ref, params=params)
            for k in params:
                torch.testing.assert_close(u_dp[k], u_ref[k])

    def test_apply_updates_compatible(self, params, grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        new_params = torchopt.apply_updates(params, updates)
        for k in params:
            assert new_params[k].shape == params[k].shape
            assert torch.isfinite(new_params[k]).all()


# ---------------------------------------------------------------------------
# Decoupled vs L2 weight decay
# ---------------------------------------------------------------------------


class TestWeightDecayMode:
    def test_decoupled_with_zero_grad_updates_only_wd(self):
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}
        opt = adamw(lr=0.1, weight_decay=0.1)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # Standard m̂/v̂ at zero grad still need bias correction; m̂ = 0
        # so the moment-scaled update is 0.  Only WD survives.
        # update = -lr * (0 + wd * params) = -0.01 * 2.0
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)


class TestAdamAlias:
    def test_adam_uses_l2_weight_decay_branch(self, params, grads):
        opt = adam(lr=1e-3, weight_decay=0.01)
        ref = adamw(lr=1e-3, weight_decay=0.01, decoupled_weight_decay=False)
        state = opt.init(params)
        ref_state = ref.init(params)

        updates, _ = opt.update(grads, state, params=params)
        ref_updates, _ = ref.update(grads, ref_state, params=params)

        for name in updates:
            torch.testing.assert_close(updates[name], ref_updates[name])

    def test_adam_consumes_noisy_metadata(self, params, grads):
        opt = adam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _adam_state(state).phi != 0.0

    def test_l2_includes_wd_in_moments(self, params, grads):
        """L2 form (decoupled=False): the m,v moments capture wd*params."""
        opt_l2 = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=False)
        opt_dec = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=True)
        s_l2 = opt_l2.init(params)
        s_dec = opt_dec.init(params)
        # After one step on the same grads, the inner Adam state must
        # differ — L2 absorbed the wd*params term into m and v.
        _, s_l2 = opt_l2.update(grads, s_l2, params=params)
        _, s_dec = opt_dec.update(grads, s_dec, params=params)
        adam_l2 = _adam_state(s_l2)
        adam_dec = _adam_state(s_dec)
        # mu must differ between the two (wd was added to the gradient
        # under L2, so the first moment EMA picked up wd * params).
        assert any(not torch.allclose(adam_l2.mu[k], adam_dec.mu[k]) for k in params)


# ---------------------------------------------------------------------------
# DP-AdamW-BC: NoisedPytree metadata
# ---------------------------------------------------------------------------


class TestBCMode:
    def test_raw_pytree_keeps_phi_zero(self, params, grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        for _ in range(10):
            _, state = opt.update(grads, state, params=params)
        assert _adam_state(state).phi == 0.0

    def test_bc_init_accepts_leaf_tensor(self):
        """Single-Tensor params are a valid pytree (empty ParamPath)."""
        params = torch.randn(4, 3)
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        phi = _adam_state(state).phi
        assert isinstance(phi, dict)
        assert set(phi) == {()}
        assert phi[()] == 0.0
        grads = torch.randn_like(params)
        updates, new_state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert updates.shape == params.shape
        assert _adam_state(new_state).phi[()] > 0.0

    def test_phi_advances_under_noisy_metadata(self, params, grads):
        b2 = 0.999
        sigma = 0.5
        opt = adamw(lr=1e-3, betas=(0.9, b2), noise_bias_correction=True)
        state = opt.init(params)
        expected_phi = 0.0
        for _ in range(10):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        phi = _adam_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        b2 = 0.999
        opt = adamw(lr=1e-3, betas=(0.9, b2), noise_bias_correction=True)
        state = opt.init(params)
        expected_phi = 0.0
        sigmas = [0.1, 0.2, 0.3, 0.2, 0.1]
        for sigma in sigmas:
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        phi = _adam_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_noisy_updates_route_stddev_metadata(self, params, grads):
        b2 = 0.999
        sigma = 0.25
        opt = adamw(lr=1e-3, betas=(0.9, b2), noise_bias_correction=True)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=sigma),
            state,
            params=params,
        )
        phi = _adam_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx((1 - b2) * sigma**2) for v in phi.values())

    def test_explicit_noise_stddev_kwarg_rejected(self, params, grads):
        """``optimizer.update()`` does not take a per-step ``noise_stddev``
        kwarg; metadata travels via ``NoisedPytree``.  Stray kwargs surface
        as a Python ``TypeError`` from the unknown-keyword check."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(TypeError, match="noise_stddev"):
            opt.update(grads, state, params=params, noise_stddev=0.5)

    def test_clipped_updates_are_rejected(self, params, grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            opt.update(clipped(grads, max_norm=1.0), state, params=params)

    def test_raw_pytree_matches_torchopt(self, params, grads):
        """Raw pytrees are non-private and match torchopt.adamw."""
        opt_bc = adamw(lr=1e-3, weight_decay=0.01)
        opt_ref = torchopt.adamw(lr=1e-3, weight_decay=0.01)
        s_bc = opt_bc.init(params)
        s_ref = opt_ref.init(params)
        for _ in range(5):
            u_bc, s_bc = opt_bc.update(grads, s_bc, params=params)
            u_ref, s_ref = opt_ref.update(grads, s_ref, params=params)
            for k in params:
                torch.testing.assert_close(u_bc[k], u_ref[k])

    def test_bc_increases_effective_lr(self, params, grads):
        """Subtracting φ̂ from v̂ shrinks the denom → larger updates."""
        big = {k: v * 10 for k, v in grads.items()}
        opt_std = adamw(lr=1e-3)
        opt_bc = adamw(lr=1e-3, noise_bias_correction=True)
        s_std = opt_std.init(params)
        s_bc = opt_bc.init(params)
        for _ in range(10):
            u_std, s_std = opt_std.update(big, s_std, params=params)
            u_bc, s_bc = opt_bc.update(
                noised(big, max_norm=1.0, noise_stddev=0.01),
                s_bc,
                params=params,
            )
        norm_std = sum(u.norm() for u in u_std.values())
        norm_bc = sum(u.norm() for u in u_bc.values())
        assert norm_bc >= norm_std

    def test_bc_floor_keeps_denom_positive(self):
        """Huge σ pushes v̂ − φ̂ < 0; the floor saves us."""
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        updates, _ = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=1e6),
            state,
            params=params,
        )
        assert torch.isfinite(updates["w"]).all()

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        opt = adamw(lr=1e-3, noise_bias_correction=False)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _adam_state(state).phi == 0.0


# ---------------------------------------------------------------------------
# Per-group BC (PerGroup NoisedPytree metadata)
# ---------------------------------------------------------------------------


class TestPerGroupBC:
    @pytest.fixture
    def pg_params(self):
        return {"q_proj.weight": torch.randn(4, 3), "mlp.weight": torch.randn(3, 2)}

    @pytest.fixture
    def pg_grads(self, pg_params):
        return {k: torch.randn_like(v) for k, v in pg_params.items()}

    @pytest.fixture
    def pg_stddev(self):
        return PerGroup(
            groups={"q_proj.weight": "attn", "mlp.weight": "mlp"},
            values={"attn": 0.3, "mlp": 0.8},
        )

    def test_per_group_state_phi_is_dict(self, pg_params, pg_stddev):
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(pg_params)
        _, state = opt.update(
            noised(pg_params, max_norm=1.0, noise_stddev=pg_stddev),
            state,
            params=pg_params,
        )
        adam = _adam_state(state)
        assert isinstance(adam.phi, dict)
        assert set(adam.phi.keys()) == {(k,) for k in pg_params}

    def test_per_group_correction_is_per_key(self, pg_params, pg_grads, pg_stddev):
        opt_pg = adamw(lr=1e-3, noise_bias_correction=True)
        opt_scalar = adamw(lr=1e-3, noise_bias_correction=True)  # matches "attn" group
        s_pg = opt_pg.init(pg_params)
        s_sc = opt_scalar.init(pg_params)
        for _ in range(5):
            u_pg, s_pg = opt_pg.update(
                noised(pg_grads, max_norm=1.0, noise_stddev=pg_stddev),
                s_pg,
                params=pg_params,
            )
            u_sc, s_sc = opt_scalar.update(
                noised(pg_grads, max_norm=1.0, noise_stddev=0.3),
                s_sc,
                params=pg_params,
            )
        torch.testing.assert_close(u_pg["q_proj.weight"], u_sc["q_proj.weight"])
        assert not torch.equal(u_pg["mlp.weight"], u_sc["mlp.weight"])

    def test_per_group_with_nested_params(self):
        """Nested param pytrees: PerGroup looks up by dotted leaf path,
        and BC must walk leaves the same way (regression test for the
        review comment about iterating top-level dict keys only)."""
        torch.manual_seed(0)
        nested_params = {
            "layer1": {
                "weight": torch.randn(4, 3),
                "bias": torch.randn(3),
            },
            "layer2": {"weight": torch.randn(3, 2)},
        }
        nested_grads = {
            "layer1": {
                "weight": torch.randn_like(nested_params["layer1"]["weight"]),
                "bias": torch.randn_like(nested_params["layer1"]["bias"]),
            },
            "layer2": {"weight": torch.randn_like(nested_params["layer2"]["weight"])},
        }
        # PerGroup keyed by nested optree paths.
        pg = PerGroup(
            groups={
                ("layer1", "weight"): "g_a",
                ("layer1", "bias"): "g_a",
                ("layer2", "weight"): "g_b",
            },
            values={"g_a": 0.2, "g_b": 0.7},
        )
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(nested_params)
        init_phi = _adam_state(state).phi
        assert isinstance(init_phi, dict)
        assert set(init_phi) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        assert all(v == 0.0 for v in init_phi.values())
        # Update should not raise (the old code crashed on the
        # ``resolve_noise_variance(pg, "layer1")`` lookup).
        updates, new_state = opt.update(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=nested_params,
        )
        new_adam = _adam_state(new_state)
        assert set(new_adam.phi.keys()) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        # Different groups → different φ-EMA values.
        assert new_adam.phi[("layer1", "weight")] == pytest.approx(
            new_adam.phi[("layer1", "bias")]
        )
        assert new_adam.phi[("layer2", "weight")] != pytest.approx(
            new_adam.phi[("layer1", "weight")]
        )
        # Updates have matching nested shape.
        assert updates["layer1"]["weight"].shape == (4, 3)
        assert updates["layer1"]["bias"].shape == (3,)
        assert updates["layer2"]["weight"].shape == (3, 2)


# ---------------------------------------------------------------------------
# Private second-moment stream
# ---------------------------------------------------------------------------


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        # Projected JME supplies a separately noised clean aggregate square; this synthetic stream
        # is enough to exercise the v-update branch.
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_second_moment_consumes_external_g_squared(self, params, grads, sq_grads):
        """The v EMA must use ``SecondMomentNoiseOutput`` instead of g·g."""
        b2 = 0.999
        opt = adamw(lr=1e-3, betas=(0.9, b2))
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = opt.update(output, state, params=params)
        adam = _adam_state(state)
        # v-update without bias correction in the kept state:
        for k in params:
            expected_v = (1 - b2) * sq_grads[k]
            torch.testing.assert_close(adam.nu[k], expected_v)

    def test_second_moment_output_unwraps_noisy_streams(self, params, grads, sq_grads):
        b2 = 0.999
        opt = adamw(lr=1e-3, betas=(0.9, b2))
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = opt.update(output, state, params=params)
        adam = _adam_state(state)
        assert adam.phi == 0.0
        for k in params:
            expected_v = (1 - b2) * sq_grads[k]
            torch.testing.assert_close(adam.nu[k], expected_v)

    def test_coupled_weight_decay_updates_both_jme_moments(self, params, grads):
        b2 = 0.999
        weight_decay = 0.2
        raw_opt = adam(
            lr=1e-3,
            betas=(0.9, b2),
            weight_decay=weight_decay,
        )
        jme_opt = adam(
            lr=1e-3,
            betas=(0.9, b2),
            weight_decay=weight_decay,
        )
        raw_state = raw_opt.init(params)
        jme_state = jme_opt.init(params)
        expected_updates, raw_state = raw_opt.update(
            grads,
            raw_state,
            params=params,
        )
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.0),
            noised(
                {name: grad.square() for name, grad in grads.items()},
                max_norm=1.0,
                noise_stddev=0.0,
            ),
        )

        actual_updates, jme_state = jme_opt.update(
            output,
            jme_state,
            params=params,
        )

        torch.testing.assert_close(actual_updates, expected_updates)
        raw_adam = _adam_state(raw_state)
        jme_adam = _adam_state(jme_state)
        for name in params:
            torch.testing.assert_close(jme_adam.mu[name], raw_adam.mu[name])
            torch.testing.assert_close(jme_adam.nu[name], raw_adam.nu[name])

    def test_second_moment_output_rejects_clipped_stream(self, params, grads, sq_grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            clipped(sq_grads, max_norm=1.0),
        )
        with pytest.raises(
            TypeError, match=r"SecondMomentNoiseOutput.noisy_squared_grads"
        ):
            opt.update(output, state, params=params)

    def test_second_moment_phi_unchanged(self, params, grads, sq_grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        # External second-moment path explicitly bypasses φ; phi stays at 0.
        for _ in range(3):
            output = SecondMomentNoiseOutput(
                noised(grads, max_norm=1.0, noise_stddev=0.1),
                noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
            )
            _, state = opt.update(output, state, params=params)
        assert _adam_state(state).phi == 0.0

    def test_second_moment_negative_squared_stream_bounded(self, params, grads):
        """Private g² streams are noised and can be negative.
        Updates must be finite AND bounded — the denominator must not collapse
        to bc_floor (eps²), which would cause ~1e6× explosion."""
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = opt.update(output, state, params=params)
        for k in updates:
            assert torch.isfinite(updates[k]).all()
            # Fallback to m_hat² keeps update magnitude ≈ 1; the old bc_floor clamp
            # would collapse the denominator to ~2e-8, giving magnitudes > 1e5.
            assert updates[k].abs().max().item() < 10.0

    def test_explicit_second_moment_kwarg_rejected(self, params, grads, sq_grads):
        """The optimizer surface no longer takes a per-step
        ``noisy_squared_grads`` kwarg; the privatised stream travels via
        ``SecondMomentNoiseOutput`` only."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(
                grads,
                state,
                params=params,
                noisy_squared_grads=sq_grads,
            )


# ---------------------------------------------------------------------------
# StableAdamW (update_rms_clip)
# ---------------------------------------------------------------------------


class TestStableAdamW:
    def test_clip_bounds_update_rms(self, params):
        """RMS of the update should stay below threshold (within FP)."""
        # Construct grads designed to produce a large-RMS update before clip.
        grads = {k: torch.full_like(v, 100.0) for k, v in params.items()}
        opt = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=0.5)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # The chain applies neg-LR after the clip, so the LR scaling
        # divides the magnitude back; we read the *clip's* effect by
        # comparing to the no-clip variant.
        opt_no = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=None)
        s_no = opt_no.init(params)
        u_no, _ = opt_no.update(grads, s_no, params=params)
        n_clip = sum(u.pow(2).sum() for u in updates.values())
        n_no = sum(u.pow(2).sum() for u in u_no.values())
        assert n_clip < n_no  # clip should shrink the update.

    def test_no_clip_below_threshold(self, params, grads):
        """Updates whose RMS is below threshold are passed through unchanged."""
        opt_clip = adamw(lr=1e-3, weight_decay=0.0, update_rms_clip=10.0)
        opt_none = adamw(lr=1e-3, weight_decay=0.0, update_rms_clip=None)
        s_c = opt_clip.init(params)
        s_n = opt_none.init(params)
        u_c, _ = opt_clip.update(grads, s_c, params=params)
        u_n, _ = opt_none.update(grads, s_n, params=params)
        for k in params:
            torch.testing.assert_close(u_c[k], u_n[k])

    def test_clip_uses_single_global_scale_not_per_leaf(self):
        """RMS clipping uses one model-wide scale across all tensor leaves."""
        params = {
            "big": torch.zeros(16),
            "small": torch.zeros(16),
        }
        grads = {
            "big": torch.full((16,), 10.0),
            "small": torch.full((16,), 1.0),
        }
        threshold = 0.85
        opt = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=threshold)
        opt_no = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=None)
        state = opt.init(params)
        state_no = opt_no.init(params)
        updates, _ = opt.update(grads, state, params=params)
        updates_no, _ = opt_no.update(grads, state_no, params=params)

        global_rms = torch.sqrt(
            (updates_no["big"].pow(2).sum() + updates_no["small"].pow(2).sum())
            / (updates_no["big"].numel() + updates_no["small"].numel())
        )
        expected_scale = torch.clamp(global_rms / threshold, min=1.0).item()
        # With a global scale both leaves are divided by the same factor.
        torch.testing.assert_close(
            updates["big"], updates_no["big"] / expected_scale, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            updates["small"], updates_no["small"] / expected_scale, atol=1e-6, rtol=0
        )

        # The chosen threshold makes the full-pytree RMS clip active.
        assert expected_scale > 1.0
        assert not torch.allclose(updates["small"], updates_no["small"])


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_vanilla_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        opt = adamw(lr=0.05, weight_decay=0.0)
        state = opt.init(params)
        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, updates)
        torch.testing.assert_close(params["x"], target, atol=0.05, rtol=0)

    def test_bc_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        opt = adamw(lr=0.05, weight_decay=0.0, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.01),
                state,
                params=params,
            )
            params = torchopt.apply_updates(params, updates)
        torch.testing.assert_close(params["x"], target, atol=0.1, rtol=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            adamw(weight_decay=-1.0)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            adamw(eps=0.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta_1"):
            adamw(betas=(1.0, 0.999))
        with pytest.raises(ValueError, match="beta_2"):
            adamw(betas=(0.9, 1.5))

    def test_zero_rms_clip_raises(self):
        with pytest.raises(ValueError, match="update_rms_clip"):
            adamw(update_rms_clip=0.0)
