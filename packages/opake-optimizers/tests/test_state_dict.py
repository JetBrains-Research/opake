"""Tests for :mod:`opake.serialization` on optimizer chain state.

Round-trip coverage for every optimizer + the schedule-free wrapper.
The contract: after serialise → fresh init → deserialise, the next
``update()`` call must produce *bit-identical* updates to the ones the
retained state would have produced.

:func:`_assert_round_trip` enforces two properties for every test: the
comparison is exact (``rtol=atol=0``), and the restored state demonstrably
influences the update, checked against a freshly-initialised one.  The second
depends on :func:`_grad_sequence` actually evolving state.
"""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opake.exceptions import CheckpointError
from opake.optimizers import (
    adafactor,
    adagrad,
    adamw,
    ademamix,
    lion,
    radam,
    rmsprop,
    schedule_free,
)
from opake.serialization import from_state_dict, state_dict
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


def _grad_sequence(params, steps: int = 6, *, seed: int = 20260819):
    """A deterministic, non-constant gradient sequence that evolves state.

    Magnitude must vary: under a constant gradient a bias-corrected optimizer
    is step-invariant (Adam has ``m̂ = g``, ``v̂ = g²`` exactly), so a fresh
    state gives the same update as an evolved one.  Direction alone is not
    enough either — Lion's ``sign(β₁m+(1−β₁)g)`` is identical from either
    state when the last gradient dominates the momentum, so magnitudes ramp
    *down* 4x → 1x.  Margins run 0.73 (Lion) to 1.1e+02 (RAdam); see
    ``_MIN_STATE_SENSITIVITY``.

    The explicit generator keeps the sequence off global RNG state.
    """
    gen = torch.Generator().manual_seed(seed)
    span = max(steps - 1, 1)
    return [
        {
            # Drawn on CPU (where ``gen`` lives) and moved onto the leaf, so
            # this keeps the device-following behavior of ``randn_like``.
            k: torch.randn(v.shape, generator=gen, dtype=v.dtype).to(v.device)
            * (1.0 + 3.0 * (span - i) / span)
            for k, v in params.items()
        }
        for i in range(steps)
    ]


@pytest.fixture
def grad_seq(params):
    return _grad_sequence(params)


# Minimum relative gap between an evolved-state update and a fresh-state one
# for a round-trip comparison to carry information.  Measured margins across
# the fixtures in this module run from 0.73 (Lion) to 1.1e+02 (RAdam), leaving
# ≥70x headroom, while a fixture that stops evolving state fails loudly.
_MIN_STATE_SENSITIVITY = 1e-2


def _rel_gap(a, b) -> float:
    """Relative L2 distance between two flat update pytrees."""
    num = sum(((a[k] - b[k]) ** 2).sum() for k in a).sqrt()
    den = sum((a[k] ** 2).sum() for k in a).sqrt()
    return (num / den).item()


def _round_trip(opt, params, seq, *, wrap=None):
    """Evolve state over ``seq[:-1]``, serialise, restore on a fresh init.

    Returns the next update from the evolved, restored, and freshly
    initialised state.  :func:`_assert_round_trip` compares the first two and
    uses the third to confirm the comparison is informative.
    """
    wrap = wrap or (lambda g: g)
    state = opt.init(params)
    for step_grads in seq[:-1]:
        _, state = opt.update(wrap(step_grads), state, params=params)

    restored = from_state_dict(opt.init(params), state_dict(state))

    final = wrap(seq[-1])
    u_evolved, _ = opt.update(final, state, params=params)
    u_restored, _ = opt.update(final, restored, params=params)
    u_fresh, _ = opt.update(final, opt.init(params), params=params)
    return u_evolved, u_restored, u_fresh


def _assert_round_trip(u_evolved, u_restored, u_fresh):
    """Assert an exact restore, and that the comparison could have failed.

    ``from_state_dict`` puts the saved values back into the template, so the
    follow-up update runs the same ops on the same bits: ``rtol=atol=0`` is
    the contract.  At ``lr=1e-3`` the updates are ~1e-3, where the default
    float32 ``atol=1e-5`` alone would absorb a genuine state difference.
    """
    for k in u_evolved:
        torch.testing.assert_close(u_restored[k], u_evolved[k], rtol=0, atol=0)

    gap = _rel_gap(u_evolved, u_fresh)
    assert gap > _MIN_STATE_SENSITIVITY, (
        f"round-trip assertion is vacuous: a freshly-initialised state produces "
        f"the same update as the evolved one (relative gap {gap:.2e} <= "
        f"{_MIN_STATE_SENSITIVITY:.0e}), so this test cannot detect a broken "
        f"restore.  The gradient sequence is not evolving optimizer state."
    )


def _noised(sigma):
    """``wrap`` for :func:`_round_trip` that attaches DP noise metadata."""
    return lambda g: noised(g, max_norm=1.0, noise_stddev=sigma)


def _second_moment(sigma=0.1):
    """``wrap`` that also supplies a privatised ``g²`` stream."""

    def wrap(g):
        sq = {k: v.pow(2) + 0.01 for k, v in g.items()}
        return SecondMomentNoiseOutput(
            noised(g, max_norm=1.0, noise_stddev=sigma),
            noised(sq, max_norm=1.0, noise_stddev=sigma),
        )

    return wrap


class TestAdamW:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(adamw(lr=1e-3, weight_decay=0.01), params, grad_seq)
        )

    def test_round_trip_bc(self, params, grad_seq):
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_noised(0.5)))

    def test_round_trip_second_moment(self, params, grad_seq):
        opt = adamw(lr=1e-3)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_second_moment()))

    def test_round_trip_l2_wd(self, params, grad_seq):
        opt = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=False)
        _assert_round_trip(*_round_trip(opt, params, grad_seq))

    def test_round_trip_with_rms_clip(self, params, grad_seq):
        opt = adamw(lr=1e-3, update_rms_clip=0.5)
        _assert_round_trip(*_round_trip(opt, params, grad_seq))

    def test_step_and_phi_preserved(self, params, grads):
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        # Adam state is at chain index 0 (decoupled WD).
        assert restored[0].step == 7
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)

    def test_per_group_phi_round_trip_nested(self):
        """Path-keyed φ survives state_dict when BC is enabled from init."""
        from opake.types import PerGroup

        nested_params = {
            "layer1": {
                "weight": torch.randn(4, 3),
                "bias": torch.randn(4),
            },
            "layer2": {"weight": torch.randn(2, 4)},
        }
        nested_grads = {
            "layer1": {
                "weight": torch.randn_like(nested_params["layer1"]["weight"]),
                "bias": torch.randn_like(nested_params["layer1"]["bias"]),
            },
            "layer2": {"weight": torch.randn_like(nested_params["layer2"]["weight"])},
        }
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
        assert isinstance(state[0].phi, dict)
        assert set(state[0].phi) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        for _ in range(3):
            _, state = opt.update(
                noised(nested_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=nested_params,
            )
        assert state[0].phi[("layer1", "weight")] != pytest.approx(0.0)
        sd = state_dict(state)
        restored = from_state_dict(opt.init(nested_params), sd)
        assert restored[0].phi == state[0].phi
        u_orig, _ = opt.update(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=nested_params,
        )
        u_rest, _ = opt.update(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            restored,
            params=nested_params,
        )
        torch.testing.assert_close(
            u_orig["layer1"]["weight"], u_rest["layer1"]["weight"]
        )
        torch.testing.assert_close(
            u_orig["layer2"]["weight"], u_rest["layer2"]["weight"]
        )

    def test_torch_save_load_round_trip(self, params, grad_seq, tmp_path):
        opt = adamw(lr=1e-3, weight_decay=0.01)
        state = opt.init(params)
        for step_grads in grad_seq[:-1]:
            _, state = opt.update(step_grads, state, params=params)
        sd = state_dict(state)
        path = tmp_path / "opt.pt"
        torch.save(sd, path)
        sd_loaded = torch.load(path, weights_only=False)
        assert set(sd_loaded.keys()) == set(sd.keys())
        restored = from_state_dict(opt.init(params), sd_loaded)

        final = grad_seq[-1]
        u_orig, _ = opt.update(final, state, params=params)
        u_rest, _ = opt.update(final, restored, params=params)
        u_fresh, _ = opt.update(final, opt.init(params), params=params)
        _assert_round_trip(u_orig, u_rest, u_fresh)


class TestLion:
    def test_round_trip(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(lion(lr=1e-4, weight_decay=0.0), params, grad_seq)
        )

    def test_step_preserved(self, params, grads):
        opt = lion(lr=1e-4)
        state = opt.init(params)
        for _ in range(4):
            _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        restored = from_state_dict(opt.init(params), sd)
        assert restored[0].step == 4


class TestAdEMAMix:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(ademamix(lr=1e-3), params, grad_seq))

    def test_round_trip_bc(self, params, grad_seq):
        opt = ademamix(lr=1e-3, noise_bias_correction=True)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_noised(0.4)))

    def test_phi_preserved(self, params, grads):
        opt = ademamix(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.4),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 7
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)


class TestAdafactor:
    @pytest.fixture
    def matrix_params(self):
        torch.manual_seed(0)
        return {"fc.weight": torch.randn(8, 4), "bias": torch.randn(4)}

    @pytest.fixture
    def matrix_grads(self, matrix_params):
        torch.manual_seed(1)
        return {k: torch.randn_like(v) for k, v in matrix_params.items()}

    @pytest.fixture
    def matrix_grad_seq(self, matrix_params):
        return _grad_sequence(matrix_params, 4)

    def test_round_trip(self, matrix_params, matrix_grad_seq):
        opt = adafactor(lr=1e-3, beta1=0.9)
        _assert_round_trip(*_round_trip(opt, matrix_params, matrix_grad_seq))

    def test_factored_v_serialised(self, matrix_params, matrix_grads):
        """v_row / v_col tensors round-trip; the optree treespec is
        skipped from the saved dict (re-derived from the template)."""
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        _, state = opt.update(matrix_grads, state, params=matrix_params)
        sd = state_dict(state)
        # treespec is opake; should not appear in the saved dict.
        assert not any("treespec" in k for k in sd)
        # v_flat tensors should be there.
        assert any("v_flat" in k for k in sd)

    def test_per_group_phi_flat_round_trip(self, matrix_params, matrix_grads):
        """Adafactor phi_flat + paths round-trip under PerGroup BC."""
        from opake.types import PerGroup

        pg = PerGroup(
            groups={
                ("fc.weight",): "attn",
                ("bias",): "mlp",
            },
            values={"attn": 0.2, "mlp": 0.8},
        )
        opt = adafactor(lr=1e-3, beta1=0.9, noise_bias_correction=True)
        state = opt.init(matrix_params)
        for _ in range(3):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=matrix_params,
            )
        sd = state_dict(state)
        restored = from_state_dict(opt.init(matrix_params), sd)
        assert restored[0].phi_flat == state[0].phi_flat
        assert restored[0].paths == state[0].paths
        u_orig, _ = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=matrix_params,
        )
        u_rest, _ = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            restored,
            params=matrix_params,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])


class TestRAdam:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(radam(lr=1e-3), params, grad_seq))

    def test_round_trip_bc(self, params, grad_seq):
        opt = radam(lr=1e-3, noise_bias_correction=True)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_noised(0.3)))

    def test_phi_preserved(self, params, grads):
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        # RAdam default uses L2 WD (decoupled_weight_decay=False), so the
        # chain is (wd, moment, clip, neg_lr) — moment state is at index 1.
        assert restored[1].step == 7
        assert isinstance(state[1].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[1].phi.values())
        assert restored[1].phi == pytest.approx(state[1].phi)


class TestRMSprop:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(rmsprop(lr=1e-2), params, grad_seq))

    def test_round_trip_bc(self, params, grad_seq):
        opt = rmsprop(lr=1e-2, noise_bias_correction=True)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_noised(0.3)))

    def test_phi_preserved(self, params, grads):
        opt = rmsprop(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 5
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)


class TestAdagrad:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(adagrad(lr=1e-2), params, grad_seq))

    def test_round_trip_bc(self, params, grad_seq):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        _assert_round_trip(*_round_trip(opt, params, grad_seq, wrap=_noised(0.3)))

    def test_phi_acc_preserved(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 5
        assert isinstance(state[0].phi_acc, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi_acc.values())
        assert restored[0].phi_acc == pytest.approx(state[0].phi_acc)


class TestScheduleFree:
    def test_round_trip_over_adamw(self, params, grad_seq):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        for step_grads in grad_seq[:-1]:
            delta, state = opt.update(step_grads, state, params=params)
            params = torchopt.apply_updates(params, delta)

        restored = from_state_dict(opt.init(params), state_dict(state))
        # x and z must come back bit-for-bit, not merely close.
        for k in state.x:
            torch.testing.assert_close(restored.x[k], state.x[k], rtol=0, atol=0)
            torch.testing.assert_close(restored.z[k], state.z[k], rtol=0, atol=0)
        assert restored.step == state.step
        assert restored.beta == state.beta

        final = grad_seq[-1]
        u_evolved, _ = opt.update(final, state, params=params)
        u_restored, _ = opt.update(final, restored, params=params)
        u_fresh, _ = opt.update(final, opt.init(params), params=params)
        _assert_round_trip(u_evolved, u_restored, u_fresh)


class TestRobustness:
    def test_missing_path_keeps_template(self, params, grads):
        """Forward-compat: a saved dict missing a path keeps the
        template's value at that path."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Drop the step entry from the saved dict.
        sd_partial = {k: v for k, v in sd.items() if not k.endswith(".step")}
        template = opt.init(params)
        restored = from_state_dict(template, sd_partial)
        # ``step`` falls back to template's 0.
        assert restored[0].step == 0

    def test_tensor_dtype_device_preserved(self, params, grads):
        """Saved tensors load back at the template's dtype/device."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Mutate the saved tensors to bf16 to simulate a saved
        # checkpoint at a different precision than the template.
        sd_bf16 = {
            k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
            for k, v in sd.items()
        }
        template = opt.init(params)
        restored = from_state_dict(template, sd_bf16)
        # Restored tensors should match the template's dtype.
        assert restored[0].mu["weight"].dtype == template[0].mu["weight"].dtype

    def test_wrong_type_raises(self, params, grads):
        """A path that should hold a tensor but holds a non-tensor in
        the dict raises ``CheckpointError`` rather than silently corrupting."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Find a tensor key and replace its value with a string.
        tensor_key = next(k for k, v in sd.items() if isinstance(v, torch.Tensor))
        sd[tensor_key] = "not a tensor"
        template = opt.init(params)
        with pytest.raises(CheckpointError, match=r"torch.Tensor"):
            from_state_dict(template, sd)
