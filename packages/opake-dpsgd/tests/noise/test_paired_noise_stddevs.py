"""Tests for ``paired_noise_stddevs`` — sensitivity-proportional joint
allocation for the paired first + second-moment Gaussian release.

The function is polymorphic in each stream:

- both ``float`` → returns ``(float, float)`` (scalar K=1 collapse).
- both :class:`PerGroup` → returns ``(PerGroup, PerGroup)``.
- mixed kinds → :class:`TypeError`.
"""

from __future__ import annotations

import math

import pytest

from opake.api.engine.noise_allocation import paired_noise_stddevs
from opake.types import PerGroup


def _make_bound(values, prefix="g", normalize_by=1.0):
    """Build a ``PerGroup`` with one parameter per group (1:1 mapping)."""
    groups = {f"p{i}": f"{prefix}{i}" for i in range(len(values))}
    vals = {f"{prefix}{i}": v / normalize_by for i, v in enumerate(values)}
    return PerGroup(groups=groups, values=vals)


# -----------------------------------------------------------------------------
# Scalar (K = 1) inputs
# -----------------------------------------------------------------------------


class TestScalarPaired:
    @pytest.mark.parametrize("nm", [0.3, 1.0, 2.5])
    @pytest.mark.parametrize(
        ("first", "second"),
        [(1.0, 1.0), (0.5, 0.25), (2.0, 4.0), (1e-3, 1e-6), (3.0, 9.0)],
    )
    def test_returns_scalars(self, nm, first, second):
        s_first, s_second = paired_noise_stddevs(nm, first=first, second=second)
        assert isinstance(s_first, float)
        assert isinstance(s_second, float)

    @pytest.mark.parametrize("nm", [0.3, 1.0, 2.5])
    @pytest.mark.parametrize(
        ("first", "second"),
        [(1.0, 1.0), (0.5, 0.25), (2.0, 4.0), (1.0, 0.01)],
    )
    def test_explicit_formula(self, nm, first, second):
        """``σ_first = nm·sqrt(Δ¹·S)``, ``σ_second = nm·sqrt(Δ²·S)`` with ``S=Δ¹+Δ²``."""
        s_first, s_second = paired_noise_stddevs(nm, first=first, second=second)
        s_total = first + second
        assert s_first == pytest.approx(nm * math.sqrt(first * s_total))
        assert s_second == pytest.approx(nm * math.sqrt(second * s_total))

    @pytest.mark.parametrize("nm", [0.3, 1.0, 2.5])
    @pytest.mark.parametrize(
        ("first", "second"),
        [(1.0, 1.0), (0.5, 0.25), (2.0, 4.0), (0.05, 0.0025)],
    )
    def test_mahalanobis_equality(self, nm, first, second):
        """Joint Mahalanobis budget evaluates to ``1/nm²`` exactly."""
        s_first, s_second = paired_noise_stddevs(nm, first=first, second=second)
        mahal = (first / s_first) ** 2 + (second / s_second) ** 2
        assert mahal == pytest.approx(1.0 / nm**2, rel=1e-12)

    def test_zero_noise_multiplier(self):
        s_first, s_second = paired_noise_stddevs(0.0, first=1.0, second=1.0)
        assert s_first == 0.0
        assert s_second == 0.0

    def test_negative_noise_multiplier_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            paired_noise_stddevs(-0.1, first=1.0, second=1.0)

    def test_negative_first_rejected(self):
        with pytest.raises(ValueError, match="first"):
            paired_noise_stddevs(1.0, first=-0.1, second=1.0)

    def test_negative_second_rejected(self):
        with pytest.raises(ValueError, match="second"):
            paired_noise_stddevs(1.0, first=1.0, second=-0.1)

    def test_zero_sensitivity_yields_zero_stddev(self):
        s_first, s_second = paired_noise_stddevs(1.0, first=0.0, second=0.0)
        assert s_first == 0.0
        assert s_second == 0.0


# -----------------------------------------------------------------------------
# PerGroup inputs
# -----------------------------------------------------------------------------


class TestPerGroupPaired:
    def test_returns_two_pergroups_with_matching_keys(self):
        first = _make_bound([1.0, 2.0, 3.0])
        squared = _make_bound([1.0, 4.0, 9.0])
        s_first, s_second = paired_noise_stddevs(1.0, first=first, second=squared)
        assert isinstance(s_first, PerGroup)
        assert isinstance(s_second, PerGroup)
        assert s_first.groups == first.groups
        assert s_second.groups == squared.groups
        assert set(s_first.values) == set(first.values)
        assert set(s_second.values) == set(squared.values)

    def test_raises_for_mismatched_groups_mapping(self):
        first = _make_bound([1.0, 2.0])
        squared = PerGroup(
            groups={"q0": "g0", "q1": "g1"},
            values={"g0": 1.0, "g1": 4.0},
        )
        with pytest.raises(ValueError, match="identical group mappings"):
            paired_noise_stddevs(1.0, first=first, second=squared)

    def test_raises_for_mismatched_group_sets(self):
        first = _make_bound([1.0, 2.0])
        squared = PerGroup(
            groups=first.groups,
            values={"g0": 1.0, "g99": 4.0},
        )
        with pytest.raises(ValueError, match="identical group sets"):
            paired_noise_stddevs(1.0, first=first, second=squared)

    def test_raises_for_negative_sensitivities(self):
        first = _make_bound([1.0, -2.0])
        squared = _make_bound([1.0, 4.0])
        with pytest.raises(ValueError, match="first"):
            paired_noise_stddevs(1.0, first=first, second=squared)
        first2 = _make_bound([1.0, 2.0])
        squared2 = _make_bound([-1.0, 4.0])
        with pytest.raises(ValueError, match="second"):
            paired_noise_stddevs(1.0, first=first2, second=squared2)

    def test_mahalanobis_constraint_holds_with_equality(self):
        cases = [
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.5, 2.0], [0.25, 4.0]),
            ([0.1, 0.4, 0.9, 1.6], [0.01, 0.16, 0.81, 2.56]),
        ]
        for first_vals, squared_vals in cases:
            for nm in [0.3, 1.0, 2.5]:
                for normalize_by in [1.0, 16.0, 128.0]:
                    first = _make_bound(first_vals, normalize_by=normalize_by)
                    squared = _make_bound(squared_vals, normalize_by=normalize_by)
                    s_first, s_second = paired_noise_stddevs(
                        nm, first=first, second=squared
                    )
                    mahal = sum(
                        first.values[g] ** 2 / s_first.values[g] ** 2
                        + squared.values[g] ** 2 / s_second.values[g] ** 2
                        for g in first.values
                    )
                    assert mahal == pytest.approx(1.0 / nm**2, rel=1e-10)

    def test_explicit_formula(self):
        first = _make_bound([1.0, 2.0])
        squared = _make_bound([1.0, 4.0])
        nm = 1.5
        s_first, s_second = paired_noise_stddevs(nm, first=first, second=squared)
        s = sum(first.values.values()) + sum(squared.values.values())  # = 8.0
        for g in first.values:
            assert s_first.values[g] == pytest.approx(
                nm * math.sqrt(first.values[g] * s)
            )
            assert s_second.values[g] == pytest.approx(
                nm * math.sqrt(squared.values[g] * s)
            )

    def test_zero_noise_multiplier_gives_zero_stddevs(self):
        first = _make_bound([1.0, 2.0])
        squared = _make_bound([1.0, 4.0])
        s_first, s_second = paired_noise_stddevs(0.0, first=first, second=squared)
        assert all(v == 0.0 for v in s_first.values.values())
        assert all(v == 0.0 for v in s_second.values.values())

    def test_zero_sensitivity_stays_zero(self):
        first = _make_bound([0.0, 1.0])
        squared = _make_bound([0.0, 1.0])
        s_first, s_second = paired_noise_stddevs(1.0, first=first, second=squared)
        keys = list(first.values)
        assert s_first.values[keys[0]] == 0.0
        assert s_second.values[keys[0]] == 0.0
        assert s_first.values[keys[1]] > 0.0
        assert s_second.values[keys[1]] > 0.0


# -----------------------------------------------------------------------------
# Mixed kinds (configuration error)
# -----------------------------------------------------------------------------


class TestMixedKinds:
    def test_first_pergroup_second_scalar_rejected(self):
        first = _make_bound([1.0, 2.0])
        with pytest.raises(TypeError, match="same kind"):
            paired_noise_stddevs(1.0, first=first, second=1.0)

    def test_first_scalar_second_pergroup_rejected(self):
        squared = _make_bound([1.0, 4.0])
        with pytest.raises(TypeError, match="same kind"):
            paired_noise_stddevs(1.0, first=1.0, second=squared)


# -----------------------------------------------------------------------------
# End-to-end: gaussian_noise + paired clipping (per-group + scalar)
# -----------------------------------------------------------------------------


class TestGaussianNoisePairedIntegration:
    def test_per_group_paired_returns_per_group_stddevs(self):
        import torch

        from opake.dpsgd.noise import gaussian_noise
        from opake.random import key
        from opake.types import (
            SecondMomentClippingOutput,
            SecondMomentNoiseOutput,
            clipped,
        )

        first_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 3.0},
        )
        squared_norm = first_norm * first_norm
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)}, max_norm=first_norm
            ),
            squared_grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)},
                max_norm=squared_norm,
            ),
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        out, _ = noise_fn(paired, state)
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        s = sum(first_norm.values.values()) + sum(squared_norm.values.values())
        for g in first_norm.values:
            assert out.noisy_grads.noise_stddev.values[g] == pytest.approx(
                math.sqrt(first_norm.values[g] * s)
            )
            assert out.noisy_squared_grads.noise_stddev.values[g] == pytest.approx(
                math.sqrt(squared_norm.values[g] * s)
            )

    def test_scalar_paired_returns_scalar_stddevs(self):
        import torch

        from opake.dpsgd.noise import gaussian_noise
        from opake.random import key
        from opake.types import (
            SecondMomentClippingOutput,
            SecondMomentNoiseOutput,
            clipped,
        )

        zeta = 0.1
        zeta_sq = zeta * zeta
        paired = SecondMomentClippingOutput(
            grads=clipped({"weight": torch.zeros(4)}, max_norm=zeta),
            squared_grads=clipped({"weight": torch.zeros(4)}, max_norm=zeta_sq),
        )
        noise_fn, state = gaussian_noise(noise_multiplier=2.0, key=key(0))
        out, _ = noise_fn(paired, state)
        assert isinstance(out, SecondMomentNoiseOutput)
        assert not isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert not isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        s_total = zeta + zeta_sq
        assert out.noisy_grads.noise_stddev == pytest.approx(
            2.0 * math.sqrt(zeta * s_total)
        )
        assert out.noisy_squared_grads.noise_stddev == pytest.approx(
            2.0 * math.sqrt(zeta_sq * s_total)
        )

    def test_per_group_paired_mismatched_kinds_rejected(self):
        import torch

        from opake.dpsgd.noise import gaussian_noise
        from opake.random import key
        from opake.types import SecondMomentClippingOutput, clipped

        per_group_norm = PerGroup(groups={"weight": "g"}, values={"g": 1.0})
        paired = SecondMomentClippingOutput(
            grads=clipped({"weight": torch.zeros(4)}, max_norm=per_group_norm),
            squared_grads=clipped({"weight": torch.zeros(4)}, max_norm=1.0),
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(TypeError, match="same kind"):
            noise_fn(paired, state)
