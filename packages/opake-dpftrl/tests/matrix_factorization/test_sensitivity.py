"""Tests for sensitivity computations."""

import pytest
import torch

from opake.api.dpftrl.noise._sensitivity import (
    banded_lower_triangular_mask,
    banded_symmetric_mask,
    get_sensitivity_banded_for_X,
    max_participation_for_linear_fn,
    minsep_true_max_participations,
    single_participation_sensitivity,
)


class TestSingleParticipationSensitivity:
    def test_identity_sensitivity(self):
        C = torch.eye(4, dtype=torch.float64)
        assert single_participation_sensitivity(C) == pytest.approx(1.0)

    def test_scaled_identity(self):
        C = 3.0 * torch.eye(4, dtype=torch.float64)
        assert single_participation_sensitivity(C) == pytest.approx(3.0)

    def test_lower_triangular(self):
        C = torch.tril(torch.ones(3, 3, dtype=torch.float64))
        # Column 0 has norm sqrt(3), column 1 has norm sqrt(2), column 2 has norm 1
        expected = torch.sqrt(torch.tensor(3.0))
        assert single_participation_sensitivity(C) == pytest.approx(float(expected))


class TestMinsepTrueMaxParticipations:
    def test_basic(self):
        assert minsep_true_max_participations(10, 1) == 10
        assert minsep_true_max_participations(10, 2) == 5
        assert minsep_true_max_participations(10, 3) == 4
        assert minsep_true_max_participations(10, 10) == 1

    def test_with_max_participations(self):
        assert minsep_true_max_participations(10, 1, 3) == 3
        assert minsep_true_max_participations(10, 5, 100) == 2

    def test_ceiling_division(self):
        assert minsep_true_max_participations(7, 3) == 3  # ceil(7/3)=3


class TestMaxParticipationForLinearFn:
    def test_all_positive_single_participation(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        # min_sep=1, max_participations=1: pick max element
        result = max_participation_for_linear_fn(x, min_sep=1, max_participations=1)
        assert result == pytest.approx(5.0)

    def test_two_participations(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        # min_sep=1, max_participations=2: pick top 2
        result = max_participation_for_linear_fn(x, min_sep=1, max_participations=2)
        assert result == pytest.approx(9.0)  # 4+5

    def test_min_sep_constraint(self):
        x = torch.tensor([10.0, 1.0, 10.0], dtype=torch.float64)
        # min_sep=2, max_participations=2: can pick [0] and [2]
        result = max_participation_for_linear_fn(x, min_sep=2, max_participations=2)
        assert result == pytest.approx(20.0)

    def test_all_zeros(self):
        x = torch.zeros(5, dtype=torch.float64)
        result = max_participation_for_linear_fn(x, min_sep=1, max_participations=3)
        assert result == pytest.approx(0.0)


class TestBandedMasks:
    def test_banded_lower_triangular(self):
        mask = banded_lower_triangular_mask(4, 2)
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(mask, expected)

    def test_banded_symmetric(self):
        mask = banded_symmetric_mask(4, 2)
        expected = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(mask, expected)

    def test_invalid_bands(self):
        with pytest.raises(ValueError, match="num_bands must be >= 1"):
            banded_lower_triangular_mask(4, 0)


class TestBandedSensitivity:
    def test_identity_banded(self):
        C = torch.eye(4, dtype=torch.float64)
        X = C.T @ C
        result = get_sensitivity_banded_for_X(X, min_sep=1, max_participations=2)
        # Identity: diagonal X has x[i]=1, sum of 2 best = 2
        assert result == pytest.approx(torch.sqrt(torch.tensor(2.0)).item())
