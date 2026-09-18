"""Tests for ``ClippedPytree.noise_stddev_for(noise_multiplier=, allocation=)``.

Centralises the noise-stddev computation that used to live only in
``opake.api.engine.noise_allocation.per_group_noise_stddev``.  The method handles both
scalar and :class:`PerGroup` ``max_norm`` and selects between the
MSE-optimal Mahalanobis allocation and a uniform isotropic allocation.
"""

from __future__ import annotations

import math

import pytest
import torch

from opake.types import PerGroup, clipped

# ── Scalar max_norm ──────────────────────────────────────────────────


class TestScalarMaxNorm:
    """Scalar ``max_norm`` returns ``noise_multiplier * max_norm``."""

    def test_optimal_scalar_returns_scalar(self):
        cg = clipped(torch.zeros(8), max_norm=2.0)
        assert cg.noise_stddev_for(noise_multiplier=0.8) == pytest.approx(1.6)

    def test_isotropic_scalar_returns_scalar(self):
        cg = clipped(torch.zeros(8), max_norm=2.0)
        assert cg.noise_stddev_for(
            noise_multiplier=0.8,
            allocation="isotropic",
        ) == pytest.approx(1.6)

    def test_zero_noise_multiplier(self):
        cg = clipped(torch.zeros(8), max_norm=2.0)
        assert cg.noise_stddev_for(noise_multiplier=0.0) == 0.0


# ── PerGroup max_norm ────────────────────────────────────────────────


def _per_group(values: dict[str, float]) -> PerGroup:
    """Build a one-key-per-group PerGroup for tests."""
    return PerGroup(
        groups={k: k for k in values},
        values=dict(values),
    )


class TestPerGroupOptimal:
    """Default ``allocation='optimal'`` — Mahalanobis MSE-optimal."""

    def test_returns_per_group(self):
        pg = _per_group({"a": 1.0, "b": 2.0})
        cg = clipped({"a": torch.zeros(2), "b": torch.zeros(2)}, max_norm=pg)
        out = cg.noise_stddev_for(noise_multiplier=0.8)
        assert isinstance(out, PerGroup)

    def test_mahalanobis_formula(self):
        """σᵢ = nm · √(Cᵢ · ΣⱼCⱼ)."""
        pg = _per_group({"a": 1.0, "b": 4.0})
        cg = clipped({"a": torch.zeros(2), "b": torch.zeros(2)}, max_norm=pg)
        out = cg.noise_stddev_for(noise_multiplier=0.5)
        sum_c = 5.0
        assert isinstance(out, PerGroup)
        assert out.values["a"] == pytest.approx(0.5 * math.sqrt(1.0 * sum_c))
        assert out.values["b"] == pytest.approx(0.5 * math.sqrt(4.0 * sum_c))

    def test_groups_preserved(self):
        pg = _per_group({"a": 1.0, "b": 2.0})
        cg = clipped({"a": torch.zeros(1), "b": torch.zeros(1)}, max_norm=pg)
        out = cg.noise_stddev_for(noise_multiplier=1.0)
        assert isinstance(out, PerGroup)
        assert out.groups == pg.groups


class TestPerGroupIsotropic:
    """``allocation='isotropic'`` returns scalar = nm · ‖C‖₂."""

    def test_returns_scalar(self):
        pg = _per_group({"a": 3.0, "b": 4.0})
        cg = clipped({"a": torch.zeros(1), "b": torch.zeros(1)}, max_norm=pg)
        out = cg.noise_stddev_for(noise_multiplier=1.0, allocation="isotropic")
        assert isinstance(out, float)

    def test_isotropic_matches_effective(self):
        """Isotropic stddev = nm · ‖C‖₂ (Mahalanobis, satisfied with equality)."""
        pg = _per_group({"a": 3.0, "b": 4.0})
        cg = clipped({"a": torch.zeros(1), "b": torch.zeros(1)}, max_norm=pg)
        # ‖(3,4)‖₂ = 5
        assert cg.noise_stddev_for(
            noise_multiplier=0.8,
            allocation="isotropic",
        ) == pytest.approx(0.8 * 5.0)


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Argument validation."""

    def test_negative_noise_multiplier_raises(self):
        cg = clipped(torch.zeros(4), max_norm=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            cg.noise_stddev_for(noise_multiplier=-0.1)

    def test_unknown_allocation_raises(self):
        cg = clipped(torch.zeros(4), max_norm=1.0)
        with pytest.raises(ValueError, match=r"isotropic.*optimal"):
            cg.noise_stddev_for(noise_multiplier=0.5, allocation="bogus")  # type: ignore[arg-type]

    def test_negative_per_group_bound_raises(self):
        pg = _per_group({"a": -0.1, "b": 1.0})
        cg = clipped({"a": torch.zeros(1), "b": torch.zeros(1)}, max_norm=pg)
        with pytest.raises(ValueError, match="non-negative"):
            cg.noise_stddev_for(noise_multiplier=0.5)


# ── Equivalence with the existing per_group_noise_stddev free function ──


class TestPerGroupNoiseStddevEquivalence:
    """``ClippedPytree(...).noise_stddev`` matches the free function."""

    def test_same_output_as_free_function(self):
        from opake.api.engine.noise_allocation import per_group_noise_stddev

        pg = _per_group({"a": 0.7, "b": 1.3, "c": 2.1})
        cg = clipped(
            {"a": torch.zeros(1), "b": torch.zeros(1), "c": torch.zeros(1)},
            max_norm=pg,
        )
        method_out = cg.noise_stddev_for(noise_multiplier=0.9)
        free_out = per_group_noise_stddev(pg, noise_multiplier=0.9)
        assert isinstance(method_out, PerGroup)
        for key in pg.values:
            assert method_out.values[key] == pytest.approx(free_out.values[key])
