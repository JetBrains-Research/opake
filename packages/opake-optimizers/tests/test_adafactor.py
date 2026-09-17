"""Tests for opake.optimizers._adafactor."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import adafactor
from opake.optimizers.types import AdafactorState
from opake.types import (
    PerGroup,
    noised,
)


@pytest.fixture
def matrix_params():
    torch.manual_seed(0)
    return {
        "fc1.weight": torch.randn(8, 4),  # factored
        "fc2.weight": torch.randn(4, 2),  # factored
        "bias": torch.randn(2),  # scalar v
    }


@pytest.fixture
def matrix_grads(matrix_params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in matrix_params.items()}


def _af_state(chain_state) -> AdafactorState:
    return chain_state[0]


class TestVanilla:
    def test_state_factored_for_matrices_scalar_for_vectors(self, matrix_params):
        opt = adafactor(lr=1e-3)
        st = _af_state(opt.init(matrix_params))
        assert isinstance(st, AdafactorState)
        # Three flat leaves; per-leaf v_state tuple of length 2 (factored)
        # for matrices, length 1 (scalar) for the bias vector.
        assert len(st.v_flat) == 3
        v_lengths = [len(v) for v in st.v_flat]
        assert sorted(v_lengths) == [1, 2, 2]

    def test_factored_v_shapes_match_axes(self):
        """For a (rows, cols) matrix, v_row has shape (rows,) and v_col (cols,)."""
        params = {"w": torch.randn(8, 4)}
        opt = adafactor(lr=1e-3)
        st = _af_state(opt.init(params))
        v_row, v_col = st.v_flat[0]
        assert v_row.shape == (8,)
        assert v_col.shape == (4,)

    def test_step_increments(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        _, state = opt.update(matrix_grads, state, params=matrix_params)
        assert _af_state(state).step == 1
        _, state = opt.update(matrix_grads, state, params=matrix_params)
        assert _af_state(state).step == 2

    def test_apply_updates_changes_params(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        orig = {k: v.clone() for k, v in matrix_params.items()}
        state = opt.init(matrix_params)
        updates, _ = opt.update(matrix_grads, state, params=matrix_params)
        new = torchopt.apply_updates(matrix_params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in matrix_params)

    def test_first_moment_optional(self, matrix_params, matrix_grads):
        opt0 = adafactor(lr=1e-3, beta1=0.0)
        opt1 = adafactor(lr=1e-3, beta1=0.9)
        st0 = _af_state(opt0.init(matrix_params))
        st1 = _af_state(opt1.init(matrix_params))
        assert st0.m is None
        assert st1.m is not None

    def test_decoupled_wd_zero_grad(self):
        params = {"w": torch.ones(3) * 2.0}
        grads = {"w": torch.zeros(3)}
        opt = adafactor(lr=0.1, weight_decay=0.5, decoupled_weight_decay=True)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # update = -lr * (0 + wd * params)
        expected = -0.1 * 0.5 * params["w"]
        torch.testing.assert_close(updates["w"], expected, atol=1e-6, rtol=0.0)

    def test_finite_updates(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        updates, _ = opt.update(matrix_grads, state, params=matrix_params)
        for k in matrix_params:
            assert torch.isfinite(updates[k]).all()


class TestUpdateRmsClip:
    def test_clip_uses_single_global_scale_not_per_leaf(self):
        """A large leaf can trigger clipping for an otherwise unscaled leaf."""
        params = {
            "large": torch.zeros(4, 4),
            "small": torch.zeros(4),
        }
        grads = {
            "large": torch.ones(4, 4),
            "small": torch.tensor([10.0, 0.0, 0.0, 0.0]),
        }
        threshold = 0.9
        common = {"lr": 1.0, "eps_root": 1.0, "weight_decay": 0.0}
        opt = adafactor(**common, update_rms_clip=threshold)
        opt_no_clip = adafactor(**common, update_rms_clip=1e9)

        updates, _ = opt.update(grads, opt.init(params), params=params)
        unclipped, _ = opt_no_clip.update(
            grads, opt_no_clip.init(params), params=params
        )

        global_rms = torch.sqrt(
            sum(update.pow(2).sum() for update in unclipped.values())
            / sum(update.numel() for update in unclipped.values())
        )
        expected_scale = torch.clamp(global_rms / threshold, min=1.0).item()
        small_rms = unclipped["small"].pow(2).mean().sqrt()

        assert small_rms < threshold < global_rms
        assert expected_scale > 1.0
        for name in params:
            torch.testing.assert_close(
                updates[name],
                unclipped[name] / expected_scale,
                atol=1e-6,
                rtol=0,
            )

    def test_clip_precedes_first_moment_ema(self):
        """The global clip retains Adafactor's pre-EMA update-clipping order."""
        params = {
            "large": torch.zeros(4, 4),
            "small": torch.zeros(4),
        }
        grads = {
            "large": torch.ones(4, 4),
            "small": torch.tensor([10.0, 0.0, 0.0, 0.0]),
        }
        threshold = 0.9
        common = {"lr": 1.0, "eps_root": 1.0, "weight_decay": 0.0}
        no_momentum = adafactor(**common, update_rms_clip=1e9)
        clipped = adafactor(**common, beta1=0.5, update_rms_clip=threshold)
        momentum_unclipped = adafactor(**common, beta1=0.5, update_rms_clip=1e9)

        normalized, _ = no_momentum.update(
            grads, no_momentum.init(params), params=params
        )
        updates, _ = clipped.update(grads, clipped.init(params), params=params)
        unclipped, _ = momentum_unclipped.update(
            grads, momentum_unclipped.init(params), params=params
        )

        global_rms = torch.sqrt(
            sum(update.pow(2).sum() for update in normalized.values())
            / sum(update.numel() for update in normalized.values())
        )
        expected_scale = torch.clamp(global_rms / threshold, min=1.0).item()
        momentum_rms = torch.sqrt(
            sum(update.pow(2).sum() for update in unclipped.values())
            / sum(update.numel() for update in unclipped.values())
        )

        assert momentum_rms < threshold < global_rms
        for name in params:
            torch.testing.assert_close(
                updates[name],
                unclipped[name] / expected_scale,
                atol=1e-6,
                rtol=0,
            )


class TestExplicitKwargsRejected:
    """``optimizer.update()`` does not take per-step metadata kwargs;
    metadata travels via ``NoisedPytree``.  Stray kwargs surface as a
    Python ``TypeError`` from the unknown-keyword check."""

    def test_noise_stddev_rejected(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(TypeError, match="noise_stddev"):
            opt.update(matrix_grads, state, params=matrix_params, noise_stddev=0.5)

    def test_noisy_squared_grads_rejected(self, matrix_params, matrix_grads):
        sq = {k: v.pow(2) for k, v in matrix_grads.items()}
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(
                matrix_grads, state, params=matrix_params, noisy_squared_grads=sq
            )


class TestBCMode:
    """DP noise-variance bias correction on the row/col factors."""

    def test_default_keeps_phi_zero(self, matrix_params, matrix_grads):
        """Default ``noise_bias_correction=False``: φ stays at 0."""
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        for _ in range(5):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=matrix_params,
            )
        assert all(v == 0.0 for v in _af_state(state).phi_flat)

    def test_explicit_true_advances_phi(self, matrix_params, matrix_grads):
        """Explicit ``noise_bias_correction=True``: φ advances under NoisedPytree updates."""
        opt = adafactor(lr=1e-3, noise_bias_correction=True)
        state = opt.init(matrix_params)
        for _ in range(5):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=matrix_params,
            )
        assert all(v > 0.0 for v in _af_state(state).phi_flat)

    def test_explicit_false_keeps_phi_zero(self, matrix_params, matrix_grads):
        """Explicit ``noise_bias_correction=False``: φ stays at 0 (same as default)."""
        opt = adafactor(lr=1e-3, noise_bias_correction=False)
        state = opt.init(matrix_params)
        for _ in range(5):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=matrix_params,
            )
        assert all(v == 0.0 for v in _af_state(state).phi_flat)

    def test_phi_advances_under_noisy_metadata(self, matrix_params, matrix_grads):
        """With BC on, φ tracks the β₂_t-EMA of σ² per leaf."""
        sigma = 0.5
        opt = adafactor(lr=1e-3, noise_bias_correction=True)
        state = opt.init(matrix_params)
        # Drive 8 steps of constant σ; phi should approach σ² steady state.
        for _ in range(8):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=matrix_params,
            )
        # All leaves have the same scalar σ → all phi entries equal.
        phi = _af_state(state).phi_flat
        assert all(p == pytest.approx(phi[0]) for p in phi)
        assert phi[0] > 0.0
        # Steady-state target: σ² (mostly converged after 8 steps).
        assert phi[0] == pytest.approx(sigma**2, rel=0.2)

    def test_per_group_routes_to_per_leaf_phi(self, matrix_params, matrix_grads):
        """PerGroup noise_stddev with σ varying per group → phi varies per leaf
        according to the group its dotted-path key resolves to."""
        pg = PerGroup(
            groups={
                "fc1.weight": "attn",
                "fc2.weight": "mlp",
                "bias": "mlp",
            },
            values={"attn": 0.2, "mlp": 0.8},
        )
        opt = adafactor(lr=1e-3, noise_bias_correction=True)
        state = opt.init(matrix_params)
        _, state = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=matrix_params,
        )
        af = _af_state(state)
        path_to_phi = dict(zip(af.paths, af.phi_flat, strict=False))
        # Group "attn" → σ=0.2 → variance 0.04 (× one-step EMA factor)
        # Group "mlp"  → σ=0.8 → variance 0.64
        # The two should differ proportionally to (0.04, 0.64).
        attn_phi = path_to_phi[("fc1.weight",)]
        mlp_phi = path_to_phi[("fc2.weight",)]
        bias_phi = path_to_phi[("bias",)]
        assert mlp_phi > attn_phi
        # Bias is in the same group as fc2.weight → same phi.
        assert mlp_phi == pytest.approx(bias_phi)
        # Variance ratio is (0.8/0.2)² = 16.
        assert mlp_phi / attn_phi == pytest.approx(16.0, rel=1e-4)

    def test_bc_changes_updates(self, matrix_params, matrix_grads):
        """With non-zero σ, BC actually changes the update vs vanilla."""
        sigma = 0.5
        opt_bc = adafactor(lr=1e-3, noise_bias_correction=True)
        opt_no = adafactor(lr=1e-3, noise_bias_correction=False)
        s_bc = opt_bc.init(matrix_params)
        s_no = opt_no.init(matrix_params)
        # Run a few warmup steps so phi has built up.
        for _ in range(3):
            _, s_bc = opt_bc.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                s_bc,
                params=matrix_params,
            )
            _, s_no = opt_no.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                s_no,
                params=matrix_params,
            )
        u_bc, _ = opt_bc.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
            s_bc,
            params=matrix_params,
        )
        u_no, _ = opt_no.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
            s_no,
            params=matrix_params,
        )
        # On at least one leaf the BC and no-BC updates differ.
        any_diff = any(not torch.allclose(u_bc[k], u_no[k]) for k in matrix_params)
        assert any_diff


class TestValidation:
    def test_negative_decay_rate_required(self):
        with pytest.raises(ValueError, match="decay_rate"):
            adafactor(decay_rate=0.5)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            adafactor(eps_grad=0.0)


class TestScaleAware:
    """Verify that eps_root floors are scale-relative, not absolute.

    At DP-clipped gradient magnitudes (~1e-4 per element) the old absolute
    eps_root=1e-3 floor dominated, collapsing Adafactor to RMS-normalised SGD.
    The corrected implementation uses eps_root * sqrt(mean(v)) so the floor
    tracks gradient scale and only fires on genuine numerical underflow.
    """

    @pytest.fixture
    def small_params(self):
        torch.manual_seed(42)
        return {
            "fc1.weight": torch.randn(8, 4),
            "bias": torch.randn(4),
        }

    @pytest.fixture
    def dp_scale_grads(self, small_params):
        """Gradients at typical DP-clipped scale: per-element ~1e-4."""
        torch.manual_seed(7)
        return {k: torch.randn_like(v) * 1e-4 for k, v in small_params.items()}

    def test_floor_does_not_fire_at_dp_gradient_scale(
        self, small_params, dp_scale_grads
    ):
        """With g ~ 1e-4, the update should not collapse to ``g / eps_root``.

        Under the broken absolute-floor behavior, ``sqrt(v̂)`` was clamped to
        ``eps_root = 1e-3`` for nearly every coordinate, so the update reduced
        to ``g / eps_root``.  The scale-relative floor should keep the
        denominator near ``sqrt(v̂) ≈ |g|`` instead, producing a materially
        different update.
        """
        opt = adafactor(lr=1.0, eps_root=1e-3, update_rms_clip=1e9)
        state = opt.init(small_params)
        for _ in range(5):
            _, state = opt.update(dp_scale_grads, state, params=small_params)
        updates, _ = opt.update(dp_scale_grads, state, params=small_params)

        for name, grad in dp_scale_grads.items():
            floored = grad / 1e-3
            actual_norm = updates[name].norm().item()
            floored_norm = floored.norm().item()
            assert actual_norm > 4.0 * floored_norm, (
                f"Leaf {name!r}: update norm {actual_norm:.3e} is too close to "
                f"the broken absolute-floor behavior {floored_norm:.3e}"
            )

    def test_distinguishable_from_rms_sgd(self, small_params, dp_scale_grads):
        """Corrected Adafactor must NOT reduce to RMS-normalised SGD at DP scale.

        RMS-normalised SGD: update = g / rms(g) * lr (constant unit-norm step).
        If absolute eps_root fires: update = g / eps_root → after RMS clip =
        g / rms(g) * update_rms_clip * lr  (same as RMS-SGD).

        The corrected implementation's update should be proportional to
        approximately sign(g) (element-wise), NOT to a scalar multiple of g.
        Specifically, the update magnitudes per element should vary in a way
        that is anti-correlated with |g| (large gradient → smaller relative
        scaling because v̂ tracked it), unlike sign(g) where all magnitudes
        are equal after RMS clipping.
        """
        # Compare: scale-aware Adafactor vs update = lr * g / rms(g) (rms-sgd).
        opt = adafactor(lr=1e-3, eps_root=1e-3)
        state = opt.init(small_params)
        for _ in range(10):
            _, state = opt.update(dp_scale_grads, state, params=small_params)
        updates, _ = opt.update(dp_scale_grads, state, params=small_params)

        # Construct the "RMS-SGD" reference for the matrix leaf.
        g_mat = dp_scale_grads["fc1.weight"]
        rms_sgd = g_mat / g_mat.pow(2).mean().sqrt()

        u_mat = updates["fc1.weight"]
        # Adafactor update (before lr scaling) should NOT equal lr * rms_sgd.
        # Check that the normalised directions differ significantly.
        u_norm = u_mat / u_mat.pow(2).mean().sqrt()
        rms_norm = rms_sgd / rms_sgd.pow(2).mean().sqrt()
        cosine_sim = (u_norm * rms_norm).sum() / (u_norm.norm() * rms_norm.norm())
        # With scale-aware floors the update is not a scalar multiple of g,
        # so the cosine similarity to RMS-SGD should be well below 1.
        assert cosine_sim.abs().item() < 0.999, (
            f"Update too similar to RMS-SGD (cosine={cosine_sim.item():.4f}); "
            "scale-aware floor may not be active"
        )

    def test_scale_invariance_of_updates(self, small_params):
        """Adafactor updates should be scale-invariant: scaling g by k should NOT
        scale the update norm by k (it normalises by √v̂ ≈ |g| so the output
        is approximately sign(g) in all cases).

        This is the intended Adafactor behavior.  With broken absolute floors,
        the denominator is clamped to eps_root for small g, producing updates
        of magnitude g/eps_root << 1 — a sub-linear (not scale-invariant) map.
        With scale-relative floors, the denominator tracks g and the update norm
        stays close to √d (sign-vector norm) at all gradient scales.
        """
        torch.manual_seed(99)
        g_base = {k: torch.randn_like(v) for k, v in small_params.items()}

        scale = 1e-3  # DP-clipped magnitude

        def run_steps(grads, n=10):
            opt = adafactor(lr=1.0, update_rms_clip=1e9)  # disable rms clip
            state = opt.init(small_params)
            for _ in range(n):
                _, state = opt.update(grads, state, params=small_params)
            updates, _ = opt.update(grads, state, params=small_params)
            return updates

        g_large = g_base
        g_small = {k: v * scale for k, v in g_base.items()}

        u_large = run_steps(g_large)
        u_small = run_steps(g_small)

        # Scale-invariance: both norms should be ≈ the same (sign(g) norm ≈ √d).
        # The absolute-floor bug produced norm(u_small) << norm(u_large) because
        # sqrt(v̂_small) was clamped to eps_root >> actual √v̂.
        for k in small_params:
            norm_large = u_large[k].norm().item()
            norm_small = u_small[k].norm().item()
            if norm_large < 1e-30:
                continue  # degenerate leaf — skip
            ratio = norm_small / norm_large
            # Both should be the sign-vector norm → ratio ≈ 1.0.
            assert ratio == pytest.approx(1.0, rel=0.1), (
                f"Leaf {k!r}: expected scale-invariant norm ratio ≈ 1.0, "
                f"got {ratio:.3e}; scale-aware floor may not be tracking gradient scale"
            )

    def test_bc_stable_with_scale_aware_floors(self, small_params, dp_scale_grads):
        """Noise BC advances phi and changes updates even at DP gradient scale.

        Uses sigma = 5e-5 (half of the gradient scale 1e-4) so that the noise
        variance phi (≈ sigma²) is a meaningful fraction of the second moment
        (≈ gradient²), making the bias-correction subtraction detectable.
        """
        # sigma must be < gradient_scale so phi < v and the correction is active.
        # dp_scale_grads ~ 1e-4 → v ~ 1e-8; sigma = 5e-5 → phi ~ 2.5e-9 < v.
        sigma = 5e-5
        opt_bc = adafactor(lr=1e-3, noise_bias_correction=True)
        opt_no = adafactor(lr=1e-3, noise_bias_correction=False)
        s_bc = opt_bc.init(small_params)
        s_no = opt_no.init(small_params)

        noisy_g = noised(dp_scale_grads, max_norm=1e-4, noise_stddev=sigma)
        for _ in range(10):
            _, s_bc = opt_bc.update(noisy_g, s_bc, params=small_params)
            _, s_no = opt_no.update(noisy_g, s_no, params=small_params)

        # phi should have advanced under BC.
        assert all(v > 0.0 for v in _af_state(s_bc).phi_flat), (
            "phi did not advance; BC appears inactive at DP gradient scale"
        )

        # BC and no-BC updates should differ.
        u_bc, _ = opt_bc.update(noisy_g, s_bc, params=small_params)
        u_no, _ = opt_no.update(noisy_g, s_no, params=small_params)
        any_diff = any(not torch.allclose(u_bc[k], u_no[k]) for k in small_params)
        assert any_diff, "BC had no effect on updates at DP gradient scale"
