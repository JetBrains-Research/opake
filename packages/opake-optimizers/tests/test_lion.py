"""Tests for opake.optimizers._lion."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.optimizers import lion
from opake.optimizers.types import LionState
from opake.types import (
    clipped,
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


def _lion_state(chain_state) -> LionState:
    return chain_state[0]


class TestLion:
    def test_returns_gradient_transformation(self):
        opt = lion(lr=1e-4)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_state_has_only_m(self, params):
        opt = lion(lr=1e-4)
        ls = _lion_state(opt.init(params))
        assert isinstance(ls, LionState)
        assert ls.step == 0
        for k in params:
            assert torch.equal(ls.m[k], torch.zeros_like(params[k]))

    def test_update_is_signed(self, params, grads):
        """Lion's update direction is the sign of (β₁m + (1-β₁)g)."""
        opt = lion(lr=1.0, weight_decay=0.0)  # lr=1 so the sign survives.
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        for k in updates:
            # update = -lr * sign(...), so all entries are ±lr.
            mags = updates[k].abs()
            torch.testing.assert_close(mags, torch.ones_like(mags))

    def test_step_increments(self, params, grads):
        opt = lion(lr=1e-4)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        assert _lion_state(state).step == 1
        _, state = opt.update(grads, state, params=params)
        assert _lion_state(state).step == 2

    def test_apply_updates_changes_params(self, params, grads):
        opt = lion(lr=1e-2)
        orig = {k: v.clone() for k, v in params.items()}
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        new = torchopt.apply_updates(params, updates)
        changed = any(not torch.equal(new[k], orig[k]) for k in params)
        assert changed

    def test_decoupled_wd_with_zero_grad(self):
        params = {"w": torch.ones(3) * 2.0}
        grads = {"w": torch.zeros(3)}
        opt = lion(lr=0.1, weight_decay=0.5, decoupled_weight_decay=True)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # sign(0) = 0 → only WD survives.  update = -lr * wd * params.
        expected = -0.1 * 0.5 * params["w"]
        torch.testing.assert_close(updates["w"], expected)

    def test_validation(self):
        with pytest.raises(ValueError, match="beta_1"):
            lion(betas=(1.0, 0.99))
        with pytest.raises(ValueError, match="non-negative"):
            lion(weight_decay=-0.1)

    def test_dp_kwargs_rejected(self, params, grads):
        """Lion has no second moment, so DP-aware kwargs aren't part of
        its update signature; passing one raises ``TypeError`` instead
        of silently ignoring it (which would mislead users into thinking
        DP-BC was active when it wasn't)."""
        opt = lion(lr=1e-4)
        state = opt.init(params)
        with pytest.raises(TypeError, match="noise_stddev"):
            opt.update(grads, state, params=params, noise_stddev=0.5)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(grads, state, params=params, noisy_squared_grads={})

    def test_noisy_updates_unwrap_silently(self, params, grads):
        """Lion has no DP-aware path, so it silently uses the wrapped
        pytree values when handed a NoisedPytree.  No warning."""
        opt = lion(lr=1e-4)
        state = opt.init(params)
        updates, _ = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.25),
            state,
            params=params,
        )
        for k in params:
            assert updates[k].shape == params[k].shape

    def test_clipped_updates_are_rejected(self, params, grads):
        opt = lion(lr=1e-4)
        state = opt.init(params)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            opt.update(clipped(grads, max_norm=1.0), state, params=params)
