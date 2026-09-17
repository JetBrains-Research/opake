"""Tests for per_group_noise_stddev — MSE-optimal noise allocation."""

import math

import pytest

from opake.api.engine.noise_allocation import per_group_noise_stddev
from opake.types import PerGroup


def _make_bound(values, normalize_by=1.0):
    """Helper: create per-group contribution bounds."""
    groups = {f"p{i}": f"g{i}" for i in range(len(values))}
    vals = {f"g{i}": v / normalize_by for i, v in enumerate(values)}
    return PerGroup(groups=groups, values=vals)


class TestPerGroupNoiseStddev:
    """Tests for the per_group_noise_stddev function."""

    def test_returns_pergroup(self):
        max_norm = _make_bound([1.0, 2.0, 3.0])
        result = per_group_noise_stddev(max_norm, 1.0)
        assert isinstance(result, PerGroup)
        assert result.groups == max_norm.groups

    def test_raises_for_scalar_clipping(self):
        with pytest.raises(TypeError, match="PerGroup max_norm"):
            per_group_noise_stddev(1.0, 1.0)

    def test_mahalanobis_constraint(self):
        """Σ (C_i/n)² / σ_i² == 1/nm² for all configurations."""
        for values in [[1.0, 1.0, 1.0], [0.1, 0.5, 2.0], [0.01, 0.1, 1.0, 5.0]]:
            for nm in [0.5, 1.0, 2.0]:
                for n in [1.0, 10.0]:
                    max_norm = _make_bound(values, normalize_by=n)
                    stddev = per_group_noise_stddev(max_norm, nm)
                    # Check Σ B_i² / σ_i² = 1/nm²
                    mahal = sum(
                        c**2 / stddev.values[g] ** 2 for g, c in max_norm.values.items()
                    )
                    assert mahal == pytest.approx(1.0 / nm**2, rel=1e-10), (
                        f"Mahalanobis constraint violated: {mahal} != {1 / nm**2} "
                        f"for values={values}, nm={nm}, n={n}"
                    )

    def test_stddev_proportional_to_sqrt_clipping_norm(self):
        """σ_i / σ_j = √(C_i / C_j) for any two groups."""
        max_norm = _make_bound([0.1, 0.4, 0.9])
        stddev = per_group_noise_stddev(max_norm, 1.0)
        vals = list(stddev.values.values())
        norms = [0.1, 0.4, 0.9]
        ratio_01 = vals[0] / vals[1]
        expected_01 = math.sqrt(norms[0] / norms[1])
        assert ratio_01 == pytest.approx(expected_01, rel=1e-10)

    def test_single_group_equals_isotropic(self):
        """With K=1, per-group noise == isotropic noise."""
        max_norm = _make_bound([2.0], normalize_by=5.0)
        stddev = per_group_noise_stddev(max_norm, 1.5)
        # σ = nm * sqrt(B * B) = nm * B
        expected = 1.5 * 2.0 / 5.0
        assert next(iter(stddev.values.values())) == pytest.approx(expected)

    def test_equal_norms_gives_equal_stddev(self):
        """With equal C_i, all σ_i should be the same."""
        max_norm = _make_bound([1.0, 1.0, 1.0])
        stddev = per_group_noise_stddev(max_norm, 2.0)
        vals = list(stddev.values.values())
        assert all(v == pytest.approx(vals[0]) for v in vals)

    def test_less_noise_on_small_groups(self):
        """Small clipping norm groups get less noise than isotropic."""
        max_norm = _make_bound([0.1, 1.0])
        nm = 1.0
        stddev = per_group_noise_stddev(max_norm, nm)
        iso = nm * max_norm.effective
        small_opt = stddev.values["g0"]
        assert small_opt < iso, (
            f"Optimal noise {small_opt} should be < isotropic {iso} for small group"
        )

    def test_mse_less_than_isotropic(self):
        """Total noise MSE < isotropic when clipping norms differ."""
        max_norm = _make_bound([0.1, 0.5, 2.0])
        nm = 1.0
        stddev = per_group_noise_stddev(max_norm, nm)
        iso_sigma = nm * max_norm.effective
        mse_opt = sum(v**2 for v in stddev.values.values())
        mse_iso = len(max_norm.values) * iso_sigma**2
        assert mse_opt < mse_iso, (
            f"Optimal MSE {mse_opt} should be < isotropic MSE {mse_iso}"
        )

    def test_normalize_by_scales_correctly(self):
        """Doubling normalize_by should halve all stddevs."""
        bound_1 = _make_bound([1.0, 2.0], normalize_by=1.0)
        bound_2 = _make_bound([1.0, 2.0], normalize_by=2.0)
        nm = 1.0
        stddev_1 = per_group_noise_stddev(bound_1, nm)
        stddev_2 = per_group_noise_stddev(bound_2, nm)
        for g in stddev_1.values:
            assert stddev_2.values[g] == pytest.approx(stddev_1.values[g] / 2.0)

    def test_linear_in_noise_multiplier(self):
        """Doubling nm should double all stddevs."""
        max_norm = _make_bound([1.0, 3.0])
        stddev_1 = per_group_noise_stddev(max_norm, 1.0)
        stddev_2 = per_group_noise_stddev(max_norm, 2.0)
        for g in stddev_1.values:
            assert stddev_2.values[g] == pytest.approx(2.0 * stddev_1.values[g])
