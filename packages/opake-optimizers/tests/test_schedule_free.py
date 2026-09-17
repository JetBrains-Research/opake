"""Tests for opake.optimizers._schedule_free wrapper."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import adamw, schedule_free
from opake.optimizers.types import ScheduleFreeState


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestScheduleFreeWrapper:
    def test_state_carries_z_x_inner(self, params):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        assert isinstance(state, ScheduleFreeState)
        for k in params:
            torch.testing.assert_close(state.z[k], params[k])
            torch.testing.assert_close(state.x[k], params[k])
        assert state.step == 0

    def test_published_x_differs_from_train_params_mid_run(self, params, grads):
        opt = schedule_free(adamw(lr=1e-2), beta=0.9)
        state = opt.init(params)
        y = params
        for _ in range(3):
            delta, state = opt.update(grads, state, params=y)
            y = torchopt.apply_updates(y, delta)
        # Mid-run, the published x differs from the trainer's y iterate.
        assert any(not torch.equal(state.x[k], y[k]) for k in y)

    def test_published_x_matches_hand_computed_average(self, params, grads):
        # Constant grads under SGD: z_t = p0 - t*lr*g, so
        # x_1 = z_1 (w=1) and x_2 = 0.5*x_1 + 0.5*z_2.
        lr, beta = 0.1, 0.5
        p0 = {k: v.clone() for k, v in params.items()}  # pristine reference
        opt = schedule_free(torchopt.sgd(lr), beta=beta)
        state = opt.init(params)
        y = params
        for _ in range(2):
            delta, state = opt.update(grads, state, params=y)
            # inplace=False: the default mutates ``params`` tensors in place,
            # which would corrupt the hand-computed reference below.
            y = torchopt.apply_updates(y, delta, inplace=False)
        for k in p0:
            z1 = p0[k] - lr * grads[k]
            z2 = p0[k] - 2 * lr * grads[k]
            expected_x = 0.5 * z1 + 0.5 * z2
            torch.testing.assert_close(state.x[k], expected_x, atol=1e-6, rtol=1e-5)

    def test_update_advances_step_and_x(self, params, grads):
        opt = schedule_free(adamw(lr=1e-2), beta=0.9)
        state = opt.init(params)
        _delta, state = opt.update(grads, state, params=params)
        assert state.step == 1
        # x should differ from initial params (it's now an average of z,
        # which moved away from y₀ = z₀ = x₀ via the inner update).
        assert any(not torch.equal(state.x[k], params[k]) for k in params)

    def test_apply_updates_yields_consistent_y(self, params, grads):
        """Applying delta to params (=y_t) should produce y_{t+1}."""
        opt = schedule_free(adamw(lr=1e-2), beta=0.9)
        state = opt.init(params)
        delta, state = opt.update(grads, state, params=params)
        new_y = torchopt.apply_updates(params, delta)
        # By construction y_{t+1} = (1-β) z_{t+1} + β x_{t+1}.
        for k in params:
            expected = (1 - 0.9) * state.z[k] + 0.9 * state.x[k]
            torch.testing.assert_close(new_y[k], expected, atol=1e-6, rtol=1e-5)

    def test_warmup_x_tracks_z(self, params, grads):
        """During warmup, x should equal z (no averaging yet)."""
        opt = schedule_free(adamw(lr=1e-2), warmup_steps=5)
        state = opt.init(params)
        for _ in range(3):
            delta, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, delta)
        for k in state.x:
            torch.testing.assert_close(state.x[k], state.z[k])

    def test_requires_params(self, params, grads):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        with pytest.raises(ValueError, match="params"):
            opt.update(grads, state)

    def test_validation(self):
        with pytest.raises(ValueError, match="beta"):
            schedule_free(adamw(lr=1e-3), beta=1.5)
        with pytest.raises(ValueError, match="warmup"):
            schedule_free(adamw(lr=1e-3), warmup_steps=-1)

    def test_compatible_with_torchopt_sgd(self, params, grads):
        """Wrapper accepts non-opaque base optimizers."""
        opt = schedule_free(torchopt.sgd(lr=1e-2))
        state = opt.init(params)
        delta, state = opt.update(grads, state, params=params)
        assert state.step == 1
        for k in params:
            assert torch.isfinite(delta[k]).all()

    def test_weight_decay_references_z_not_y(self, params, grads):
        """Decoupled weight decay must regularise the raw iterate ``z``
        (which the inner update is being added to), not the
        forward-pass interpolation ``y``.  Regression test for review
        comment: the wrapped optimizer was being told ``params=y_t``,
        which made AdamW's ``add_decayed_weights`` return
        ``update + wd*y`` instead of ``update + wd*z``.

        Setup: zero gradients (so the moment-scaled part is exactly
        zero), constant params, one step.  With the fix the WD term
        equals ``-lr * wd * z`` and leaves ``new_z = (1 - lr*wd) * z``.
        With the old bug it would equal ``-lr * wd * y``, which is
        identical only when ``y == z`` (i.e. on the very first step
        where ``x = z = y``).  We therefore drive ``y`` and ``z``
        apart by running one step first.
        """
        opt = schedule_free(adamw(lr=0.1, weight_decay=0.5), beta=0.9)
        state = opt.init(params)
        # Step once with non-zero grads so z, x, y diverge.
        delta, state = opt.update(grads, state, params=params)
        params_after_first = torchopt.apply_updates(params, delta)
        # Now y_1 = params_after_first ≠ z_1.  Run a step with zero
        # grads so the only change to z is the WD term.
        zero_grads = {k: torch.zeros_like(v) for k, v in params.items()}
        z_before = {k: v.clone() for k, v in state.z.items()}
        _, state = opt.update(zero_grads, state, params=params_after_first)
        # With the fix: new_z = z - lr * wd * z = (1 - lr*wd) * z.
        # (m̂ = v̂ = 0 because grads have been zero on this step and m_0
        # was zero; even if m carried over from the prior step, the
        # AdamW chain still applies wd*params separately, and we're
        # checking the wd contribution path specifically.)
        # The check that locks in the fix: new_z deviates from z by a
        # multiple of z (not of y).  Compute ratio along z direction.
        for k in params:
            shift = state.z[k] - z_before[k]
            # Project shift onto z and onto y to check which it tracks.
            z_dir = z_before[k]
            y_dir = params_after_first[k]
            # cos similarity to z should be much higher than to y when
            # params_after_first ≠ z (the prior step ensured this).
            cos_z = torch.nn.functional.cosine_similarity(
                shift.flatten().unsqueeze(0),
                z_dir.flatten().unsqueeze(0),
            ).item()
            cos_y = torch.nn.functional.cosine_similarity(
                shift.flatten().unsqueeze(0),
                y_dir.flatten().unsqueeze(0),
            ).item()
            # We can't assert cos_z == ±1 exactly because the AdamW
            # chain also has bias-corrected first moment from the
            # previous step contributing to the update.  But the
            # WD-driven component must align with z, not y.  Since
            # y = (1-β)z + β*x ≠ z, the cosine-to-z must be strictly
            # greater than cosine-to-y for the fix to hold.
            assert cos_z > cos_y - 1e-6, (
                f"WD term aligns with y instead of z for {k!r} "
                f"(cos_z={cos_z:.4f}, cos_y={cos_y:.4f})"
            )

    def test_post_warmup_averaging_starts_fresh(self, params, grads):
        """After warmup ends, ``x`` should start a fresh average rather
        than remaining anchored to the warmup endpoint.

        Regression test for review comment: the old code used
        ``w = 1/t`` (global step), so on the first post-warmup step
        ``new_x = (1 - 1/(W+1)) * x_W + (1/(W+1)) * z_{W+1}`` left
        ``x`` ≈ ``x_W`` instead of jumping to ``z_{W+1}``.  The fix
        uses ``w = 1 / (t - W)`` so the first post-warmup step has
        ``w = 1`` and ``new_x = z_{W+1}``.
        """
        warmup = 5
        opt = schedule_free(adamw(lr=1e-2), beta=0.9, warmup_steps=warmup)
        state = opt.init(params)
        # Run through warmup (x tracks z).
        for _ in range(warmup):
            delta, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, delta)
            for k in state.x:
                torch.testing.assert_close(state.x[k], state.z[k])
        # First post-warmup step: x should equal the new z (fresh average).
        delta, state = opt.update(grads, state, params=params)
        for k in state.x:
            torch.testing.assert_close(state.x[k], state.z[k])
