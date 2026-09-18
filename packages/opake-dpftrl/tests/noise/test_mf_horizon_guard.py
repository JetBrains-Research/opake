"""Horizon guard: MF noise_fn must raise at step >= n_steps.

Without this guard, normalized λ-CGD emits *zero* noise at
``step == n_steps`` (the column-norm closed form collapses to 0), and
streaming / identity engines silently continue past the accountant's
horizon — releasing noise the PLD never charged for.
"""

from __future__ import annotations

import pytest
import torch

from opake.api.dpftrl.noise._engine import (
    _check_mf_horizon,
    _matrix_factorization_noise,
)
from opake.api.dpftrl.noise._lambda_cgd import _column_norm
from opake.api.dpftrl.noise._streaming_matrix import identity
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.random import key
from opake.types import clipped

_HORIZON_MATCH = r"outside the calibrated horizon"


def _zero_grads(dim: int = 32):
    return clipped({"w": torch.zeros(dim)}, max_norm=1.0)


class TestCheckMfHorizon:
    def test_accepts_in_range(self):
        for step in range(5):
            _check_mf_horizon(step, 5)

    def test_rejects_at_and_past_horizon(self):
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            _check_mf_horizon(5, 5)
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            _check_mf_horizon(6, 5)


class TestColumnNormHorizon:
    def test_nonzero_inside_horizon(self):
        for step in range(5):
            assert _column_norm(0.9, 5, step) > 0.0

    def test_rejects_at_horizon(self):
        # Pre-guard this returned 0.0, zeroing the released noise.
        with pytest.raises(ValueError, match=r"column-norm step"):
            _column_norm(0.9, 5, 5)

    def test_rejects_past_horizon(self):
        with pytest.raises(ValueError, match=r"column-norm step"):
            _column_norm(0.9, 5, 6)


class TestLambdaCgdHorizonGuard:
    @pytest.mark.parametrize("normalized", [True, False])
    def test_raises_on_nth_plus_one_call(self, normalized: bool):
        n_steps = 4
        noise_fn, state = mf_gaussian_noise(
            {"w": torch.zeros(32)},
            lambda_cgd_strategy(lambda_=0.9, normalized=normalized),
            n_steps=n_steps,
            noise_multiplier=1.0,
            key=key(0),
        )
        grads = _zero_grads()
        for _ in range(n_steps):
            _, state = noise_fn(grads, state)
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            noise_fn(grads, state)

    def test_normalized_never_emits_zero_noise_inside_horizon(self):
        n_steps = 5
        noise_fn, state = mf_gaussian_noise(
            {"w": torch.zeros(1000)},
            lambda_cgd_strategy(lambda_=0.9, normalized=True),
            n_steps=n_steps,
            noise_multiplier=1.0,
            key=key(7),
        )
        grads = clipped({"w": torch.zeros(1000)}, max_norm=1.0)
        for _ in range(n_steps):
            out, state = noise_fn(grads, state)
            assert out.pytree["w"].abs().max().item() > 0.0
            assert float(out.noise_stddev) > 0.0


class TestStreamingHorizonGuard:
    @pytest.mark.parametrize(
        "strategy_factory",
        [
            identity_strategy,
            lambda: blt_strategy(max_buffers=3),
            lambda: band_mf_strategy(bands=3),
            lambda: bisr_strategy(bandwidth=3),
        ],
        ids=["identity", "blt", "band_mf", "bisr"],
    )
    def test_raises_past_horizon(self, strategy_factory):
        n_steps = 5
        noise_fn, state = mf_gaussian_noise(
            {"w": torch.zeros(32)},
            strategy_factory(),
            n_steps=n_steps,
            noise_multiplier=1.0,
            key=key(0),
        )
        grads = _zero_grads()
        for _ in range(n_steps):
            _, state = noise_fn(grads, state)
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            noise_fn(grads, state)


class TestDenseEngineHorizonGuard:
    def test_raises_past_matrix_rows(self):
        matrix = torch.eye(3, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(
            {"w": torch.zeros(8)},
            matrix,
            key=key(0),
        )
        for _ in range(3):
            _, state = noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)

    def test_explicit_n_steps_shorter_than_matrix(self):
        matrix = torch.eye(5, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(
            {"w": torch.zeros(8)},
            matrix,
            key=key(0),
            n_steps=2,
        )
        _, state = noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)
        _, state = noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)

    def test_streaming_without_n_steps_remains_unbounded(self):
        # Direct engine callers that omit n_steps keep the previous
        # unbounded behaviour (used by low-level tests).
        noise_fn, state = _matrix_factorization_noise(
            {"w": torch.zeros(8)},
            identity(),
            key=key(0),
        )
        for _ in range(20):
            _, state = noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)
        assert state._step_counter == 20


class TestNStepsTypeValidation:
    def test_dense_rejects_float_n_steps(self):
        with pytest.raises(TypeError, match=r"n_steps must be an int"):
            _matrix_factorization_noise(
                {"w": torch.zeros(8)},
                torch.eye(5, dtype=torch.float64),
                key=key(0),
                n_steps=2.9,  # type: ignore[arg-type]
            )

    def test_streaming_rejects_float_n_steps(self):
        with pytest.raises(TypeError, match=r"n_steps must be an int"):
            _matrix_factorization_noise(
                {"w": torch.zeros(8)},
                identity(),
                key=key(0),
                n_steps=5.5,  # type: ignore[arg-type]
            )

    def test_lambda_cgd_rejects_float_n_steps(self):
        with pytest.raises(TypeError, match=r"n_steps must be an int"):
            mf_gaussian_noise(
                {"w": torch.zeros(8)},
                lambda_cgd_strategy(lambda_=0.9),
                n_steps=4.2,  # type: ignore[arg-type]
                noise_multiplier=1.0,
                key=key(0),
            )

    @pytest.mark.parametrize("n_steps", [4.2, True])
    def test_bisr_rejects_non_integer_n_steps(self, n_steps):
        with pytest.raises(TypeError, match=r"n_steps must be an int"):
            mf_gaussian_noise(
                {"w": torch.zeros(8)},
                bisr_strategy(bandwidth=3),
                n_steps=n_steps,
                noise_multiplier=1.0,
                key=key(0),
            )

    def test_rejects_bool_n_steps(self):
        with pytest.raises(TypeError, match=r"n_steps must be an int"):
            _matrix_factorization_noise(
                {"w": torch.zeros(8)},
                torch.eye(5, dtype=torch.float64),
                key=key(0),
                n_steps=True,  # type: ignore[arg-type]
            )


class TestRowL2HorizonGuard:
    def test_band_mf_row_l2_raises_value_error_past_horizon(self):
        from opake.api.dpftrl.noise._mf_gaussian_noise import _make_raw_mf_noise

        _, _, row_l2_at = _make_raw_mf_noise(
            {"w": torch.zeros(8)},
            band_mf_strategy(bands=3),
            n_steps=4,
            min_sep=1,
            max_participations=1,
            key=key(0),
        )
        assert row_l2_at(0) > 0.0
        with pytest.raises(ValueError, match=_HORIZON_MATCH):
            row_l2_at(4)
