"""Paired-stream (second-moment) support for ``gaussian_noise``."""

from __future__ import annotations

import math

import pytest
import torch

from opake.api.engine.noise_allocation import paired_noise_stddevs
from opake.dpsgd.noise import gaussian_noise
from opake.random import key
from opake.types import (
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)


def _paired_input(
    grads: torch.Tensor,
    squared: torch.Tensor,
    *,
    max_norm: float,
    squared_max_norm: float,
) -> SecondMomentClippingOutput:
    return SecondMomentClippingOutput(
        grads=clipped(grads, max_norm=max_norm),
        squared_grads=clipped(squared, max_norm=squared_max_norm),
    )


class TestPairedStreamShape:
    """``noise_fn(SecondMomentClippingOutput, state)`` returns paired noised output."""

    def test_returns_second_moment_noise_output(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads, NoisedPytree)
        assert isinstance(out.noisy_squared_grads, NoisedPytree)

    def test_streams_match_paired_noise_stddevs(self):
        """Output stddevs match ``paired_noise_stddevs`` on the same inputs."""
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        first_expected, second_expected = paired_noise_stddevs(
            1.0, first=1.0, second=1.0
        )
        assert out.noisy_grads.noise_stddev == pytest.approx(first_expected)
        assert out.noisy_squared_grads.noise_stddev == pytest.approx(second_expected)
        # With Δ₁ = Δ₂ = 1, S = 2, both streams get σ = sqrt(2).
        assert first_expected == pytest.approx(math.sqrt(2.0))
        assert second_expected == pytest.approx(math.sqrt(2.0))


class TestPairedStreamCoupling:
    """Squared-stream sensitivity enters S, so it shifts both σ's."""

    def test_squared_max_norm_shifts_both_streams(self):
        """The Mahalanobis budget couples both streams: changing Δ² shifts σ¹ and σ²."""
        noise_fn_a, state_a = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out_a, _ = noise_fn_a(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        noise_fn_b, state_b = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out_b, _ = noise_fn_b(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=2.0,
            ),
            state_b,
        )
        # Closed form: σ¹ = sqrt(Δ¹ · S), with S = Δ¹ + Δ².
        # Doubling Δ² grows S from 2 → 3, so σ¹ grows by sqrt(3/2) and σ² by
        # sqrt(2 · 3 / (1 · 2)) = sqrt(3).
        assert out_b.noisy_grads.noise_stddev == pytest.approx(
            out_a.noisy_grads.noise_stddev * math.sqrt(3 / 2)
        )
        assert out_b.noisy_squared_grads.noise_stddev == pytest.approx(
            out_a.noisy_squared_grads.noise_stddev * math.sqrt(3.0)
        )


class TestPairedStreamPerGroup:
    """Per-group max_norm on paired streams uses the joint MSE-optimal allocation."""

    def test_per_group_on_both_streams_returns_per_group_stddevs(self):
        from opake.types import PerGroup

        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        first_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        squared_norm = first_norm * first_norm  # per-group squared bounds
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)}, max_norm=first_norm
            ),
            squared_grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)},
                max_norm=squared_norm,
            ),
        )
        out, _ = noise_fn(paired, state)
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        # Per-group bound, per-group stddev keys match the input groups.
        assert out.noisy_grads.noise_stddev.groups == first_norm.groups
        assert out.noisy_squared_grads.noise_stddev.groups == squared_norm.groups

    def test_mismatched_kinds_rejected(self):
        from opake.types import PerGroup

        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        per_group_norm = PerGroup(
            groups={"weight": "g"},
            values={"g": 1.0},
        )
        paired = SecondMomentClippingOutput(
            grads=clipped({"weight": torch.zeros(4)}, max_norm=per_group_norm),
            squared_grads=clipped({"weight": torch.zeros(4)}, max_norm=1.0),
        )
        with pytest.raises(TypeError, match="same kind"):
            noise_fn(paired, state)


class TestPairedStreamReproducibility:
    """Same key → same noise; different folds → different noise streams."""

    def test_seeded_runs_match(self):
        noise_fn_a, state_a = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noise_fn_b, state_b = gaussian_noise(noise_multiplier=1.0, key=key(42))
        out_a, _ = noise_fn_a(
            _paired_input(
                torch.zeros(20),
                torch.zeros(20),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        out_b, _ = noise_fn_b(
            _paired_input(
                torch.zeros(20),
                torch.zeros(20),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_b,
        )
        assert torch.equal(out_a.noisy_grads.pytree, out_b.noisy_grads.pytree)
        assert torch.equal(
            out_a.noisy_squared_grads.pytree, out_b.noisy_squared_grads.pytree
        )

    def test_first_and_second_streams_have_different_noise(self):
        """Paired streams stay independent after namespaced string tags."""
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(50),
                torch.zeros(50),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        # Streams cover the same domain but with different noise.
        assert not torch.equal(out.noisy_grads.pytree, out.noisy_squared_grads.pytree)


class TestPairedStreamStateAdvances:
    """Step counter advances exactly once per paired call."""

    def test_state_advances(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        for expected in (1, 2, 3):
            _, state = noise_fn(
                _paired_input(
                    torch.zeros(4),
                    torch.zeros(4),
                    max_norm=1.0,
                    squared_max_norm=1.0,
                ),
                state,
            )
            assert state._step_counter == expected


class TestZeroNoiseMultiplier:
    """``noise_multiplier=0`` preserves both input streams."""

    def test_zero_noise_paired(self):
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(0))
        grads = torch.tensor([5.0, -5.0, 0.5, -0.5])
        squared = torch.tensor([10.0, 0.0, 1.0, 2.0])
        out, _ = noise_fn(
            _paired_input(
                grads.clone(),
                squared.clone(),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        torch.testing.assert_close(out.noisy_grads.pytree, grads)
        torch.testing.assert_close(out.noisy_squared_grads.pytree, squared)
