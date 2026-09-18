"""Tests for Toeplitz matrix operations."""

import dataclasses
import warnings
from types import SimpleNamespace

import pytest
import torch

from opake.api.dpftrl.noise._band_mf import _momentum_workload_coef
from opake.api.dpftrl.noise._toeplitz import (
    _lbfgs_optimize,
    inverse_as_streaming_matrix,
    inverse_coef,
    loss,
    materialize_lower_triangular,
    max_error,
    mean_error,
    minsep_sensitivity_squared,
    minsep_sensitivity_upper_bound,
    multiply,
    optimal_max_error_strategy_coefs,
    optimize,
    pad_coefs_to_n,
    per_query_error,
    sensitivity_squared,
)


def test_lbfgs_rejects_nonfinite_result(monkeypatch):
    def fake_minimize(*_args, **_kwargs):
        return SimpleNamespace(fun=float("nan"), x=[0.5], message="converged")

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    with pytest.raises(RuntimeError, match="non-finite"):
        _lbfgs_optimize(lambda x: x.square().sum(), torch.tensor([0.5]))


class TestMaterializeLowerTriangular:
    def test_identity(self):
        coef = torch.tensor([1.0], dtype=torch.float64)
        M = materialize_lower_triangular(coef, n=3)
        torch.testing.assert_close(M, torch.eye(3, dtype=torch.float64))

    def test_basic(self):
        coef = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        M = materialize_lower_triangular(coef)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.25, 0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(M, expected)

    def test_padded(self):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        M = materialize_lower_triangular(coef, n=4)
        assert M.shape == (4, 4)
        assert M[3, 0] == pytest.approx(0.0)  # Beyond band


class TestPadCoefsToN:
    def test_pad(self):
        coef = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = pad_coefs_to_n(coef, 5)
        expected = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_truncate(self):
        coef = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = pad_coefs_to_n(coef, 2)
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected)


class TestMultiply:
    def test_identity_multiply(self):
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        result = multiply(a, b, n=3)
        torch.testing.assert_close(result, b)

    def test_convolution(self):
        a = torch.tensor([1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = multiply(a, b, n=3)
        expected = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected)


class TestInverseCoef:
    def test_identity_inverse(self):
        coef = torch.tensor([1.0], dtype=torch.float64)
        inv = inverse_coef(coef, n=3)
        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(inv, expected, atol=1e-10, rtol=1e-10)

    def test_inverse_roundtrip(self):
        coef = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        inv = inverse_coef(coef)
        # C @ C^{-1} should give identity coefs
        product = multiply(coef, inv, n=3)
        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(product, expected, atol=1e-10, rtol=1e-10)


class TestInverseAsStreamingMatrix:
    def test_vs_dense_inverse(self):
        coef = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        C = materialize_lower_triangular(coef)
        C_inv = torch.linalg.inv(C)

        streaming = inverse_as_streaming_matrix(coef)
        M = streaming.materialize(3)

        torch.testing.assert_close(M, C_inv, atol=1e-10, rtol=1e-10)


def _probing_row_norms_squared(matrix, n):
    """Reference row norms via the generic probing implementation."""
    return dataclasses.replace(matrix, row_norms_squared_fn=None).row_norms_squared(n)


class TestRowNormsClosedForm:
    """Closed-form row_norms_squared attached by inverse_as_streaming_matrix."""

    def test_banded_matches_probing_and_dense(self):
        coef = torch.tensor([1.0, 0.7, 0.3], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef)
        n = 6
        norms = streaming.row_norms_squared(n)
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(streaming, n), atol=1e-12, rtol=1e-12
        )
        dense = streaming.materialize(n)
        torch.testing.assert_close(
            norms, dense.square().sum(dim=1), atol=1e-12, rtol=1e-12
        )

    def test_column_normalized_matches_probing(self):
        coef = torch.tensor([1.0, 0.7, 0.3], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef, column_normalize_for_n=6)
        norms = streaming.row_norms_squared(6)
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(streaming, 6), atol=1e-12, rtol=1e-12
        )

    def test_n_beyond_normalization_clamps_like_probing(self):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef, column_normalize_for_n=4)
        norms = streaming.row_norms_squared(7)
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(streaming, 7), atol=1e-12, rtol=1e-12
        )

    def test_n_smaller_than_bands(self):
        coef = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef)
        norms = streaming.row_norms_squared(3)
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(streaming, 3), atol=1e-12, rtol=1e-12
        )

    def test_inverse_coefficients_hint(self):
        # C^{-1} = toeplitz([1, -0.5]) is banded, so C is the dense
        # geometric strategy toeplitz([1, 0.5, 0.25, ...]).
        n = 8
        inv_hint = torch.tensor([1.0, -0.5], dtype=torch.float64)
        coef = inverse_coef(inv_hint, n)
        with_hint = inverse_as_streaming_matrix(coef, inverse_coefficients=inv_hint)
        norms = with_hint.row_norms_squared(n)
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(with_hint, n), atol=1e-12, rtol=1e-12
        )

    def test_hint_not_used_past_validated_horizon(self):
        # coef=[1, 0.5] with inv_hint=[1, -0.5] passes validation: the full
        # convolution is [1, 0, -0.25] and only its first
        # max(len(coef), len(inv_hint)) = 2 terms are checked. The hint is
        # not the whole inverse though -- the true one continues
        # [..., 0.25, -0.125] -- so zero-padding it past that horizon
        # terminates the inverse early and under-reports the third squared
        # row norm as 1.25 instead of 1.3125.
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        inv_hint = torch.tensor([1.0, -0.5], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef, inverse_coefficients=inv_hint)
        norms = streaming.row_norms_squared(3)
        torch.testing.assert_close(
            norms,
            torch.tensor([1.0, 1.25, 1.3125], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            norms, _probing_row_norms_squared(streaming, 3), atol=1e-12, rtol=1e-12
        )

    def test_inconsistent_inverse_coefficients_raise(self):
        coef = torch.tensor([1.0, 0.7, 0.3], dtype=torch.float64)
        with pytest.raises(ValueError, match="not the Toeplitz inverse"):
            inverse_as_streaming_matrix(
                coef,
                inverse_coefficients=torch.tensor([5.0, 5.0], dtype=torch.float64),
            )

    def test_requires_grad_uses_differentiable_path(self):
        coef = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64, requires_grad=True)
        streaming = inverse_as_streaming_matrix(coef)
        norms = streaming.row_norms_squared(4)
        assert norms.requires_grad
        norms.sum().backward()
        assert coef.grad is not None
        assert torch.all(torch.isfinite(coef.grad))
        torch.testing.assert_close(
            norms.detach(),
            inverse_as_streaming_matrix(coef.detach()).row_norms_squared(4),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_n_zero(self):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        streaming = inverse_as_streaming_matrix(coef)
        assert streaming.row_norms_squared(0).shape == (0,)


class TestOptimalCoefs:
    def test_first_coef_is_one(self):
        coef = optimal_max_error_strategy_coefs(10)
        assert coef[0] == pytest.approx(1.0)

    def test_decreasing(self):
        coef = optimal_max_error_strategy_coefs(10)
        for i in range(1, len(coef)):
            assert coef[i] <= coef[i - 1]

    def test_second_coef(self):
        coef = optimal_max_error_strategy_coefs(5)
        assert coef[1] == pytest.approx(0.5)


class TestSensitivitySquared:
    def test_identity(self):
        coef = torch.tensor([1.0], dtype=torch.float64)
        result = sensitivity_squared(coef)
        assert result == pytest.approx(1.0)

    def test_two_bands(self):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        result = sensitivity_squared(coef)
        # ||[1.0, 0.5]||^2 = 1.25
        assert result == pytest.approx(1.25)


class TestMinsepSensitivitySquared:
    def test_single_participation(self):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)
        result = minsep_sensitivity_squared(coef, min_sep=1, max_participations=1)
        # For single participation, result should be positive
        assert float(result) > 0

    def test_decreasing_required(self):
        coef = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-increasing"):
            minsep_sensitivity_squared(coef, min_sep=1, max_participations=2)

    def test_nonnegative_required(self):
        coef = torch.tensor([1.0, -0.5], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-negative"):
            minsep_sensitivity_squared(coef, min_sep=1, max_participations=2)

    def test_explicit_upper_bound_uses_safe_monotone_majorant(self):
        coef = torch.tensor([1.0, -0.5, 0.75, 0.25], dtype=torch.float64)
        majorant = torch.tensor([1.0, 0.75, 0.75, 0.25], dtype=torch.float64)

        actual = minsep_sensitivity_upper_bound(coef, min_sep=1, max_participations=4)
        expected = minsep_sensitivity_squared(majorant, min_sep=1, max_participations=4)

        torch.testing.assert_close(actual, expected)

        # A concrete allowed participation pattern already exceeded the old
        # unchecked closed form for these signed coefficients.
        participation = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        lower_bound = torch.linalg.vector_norm(
            materialize_lower_triangular(coef) @ participation
        )
        assert float(actual.sqrt()) >= float(lower_bound)

    def test_upper_bound_is_exact_for_zero_and_single_participation(self):
        coef = torch.tensor([1.0, -0.5, 0.75], dtype=torch.float64)

        zero = minsep_sensitivity_upper_bound(coef, min_sep=1, max_participations=0)
        single = minsep_sensitivity_upper_bound(coef, min_sep=1, max_participations=1)

        assert zero == pytest.approx(0.0)
        assert single == pytest.approx(float(torch.dot(coef, coef)))

    @pytest.mark.parametrize(
        "sensitivity_fn",
        [minsep_sensitivity_squared, minsep_sensitivity_upper_bound],
    )
    def test_negative_max_participations_rejected(self, sensitivity_fn):
        coef = torch.tensor([1.0, 0.5], dtype=torch.float64)

        with pytest.raises(ValueError, match="max_participations must be non-negative"):
            sensitivity_fn(coef, min_sep=1, max_participations=-1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_coefficients_rejected(self, bad):
        coef = torch.tensor([1.0, bad], dtype=torch.float64)
        with pytest.raises(ValueError, match="finite"):
            minsep_sensitivity_upper_bound(coef, min_sep=1)


class TestPerQueryError:
    def test_identity_mechanism(self):
        """Identity strategy: error should grow linearly."""
        coef = torch.tensor([1.0], dtype=torch.float64)
        error = per_query_error(strategy_coef=coef, n=5)
        # Prefix sum error: cumsum of 1^2 = [1, 2, 3, 4, 5]
        expected = torch.arange(1, 6, dtype=torch.float64)
        torch.testing.assert_close(error, expected)

    def test_noising_coef(self):
        """Error from noising coefficients."""
        noising = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
        error = per_query_error(noising_coef=noising)
        assert error.shape == (3,)
        assert torch.all(error > 0)

    def test_query_weights_apply_on_training_step_axis(self):
        """Schedules scale rows of diag(eta) @ momentum, not Toeplitz lags."""
        n = 4
        momentum = 0.5
        strategy_coef = torch.tensor([1.0, 0.25], dtype=torch.float64)
        learning_rates = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        momentum_matrix = materialize_lower_triangular(
            _momentum_workload_coef(momentum, n), n
        )
        strategy = materialize_lower_triangular(strategy_coef, n)
        expected = (learning_rates[:, None] * momentum_matrix) @ torch.linalg.inv(
            strategy
        )
        expected_error = expected.square().sum(dim=1)

        error = per_query_error(
            strategy_coef=strategy_coef,
            n=n,
            workload_coef=_momentum_workload_coef(momentum, n),
            query_weights=learning_rates,
        )

        torch.testing.assert_close(error, expected_error)

    def test_max_error_uses_weighted_step_maximum(self):
        error = max_error(
            strategy_coef=torch.tensor([1.0], dtype=torch.float64),
            n=3,
            query_weights=torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64),
        )
        assert error == pytest.approx(9.0)

    def test_both_specified_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            per_query_error(
                strategy_coef=torch.ones(3),
                noising_coef=torch.ones(3),
            )


class TestMaxAndMeanError:
    def test_max_error(self):
        coef = torch.tensor([1.0], dtype=torch.float64)
        error = max_error(strategy_coef=coef, n=5)
        assert error == pytest.approx(5.0)

    def test_mean_error(self):
        coef = torch.tensor([1.0], dtype=torch.float64)
        error = mean_error(strategy_coef=coef, n=5)
        assert error == pytest.approx(3.0)  # mean of [1,2,3,4,5]

    def test_custom_error_without_optional_keywords(self):
        def custom_error(*, strategy_coef, n):
            return strategy_coef.square().sum() / n

        result = loss(
            torch.tensor([1.0], dtype=torch.float64),
            n=2,
            error_fn=custom_error,
        )

        assert result == pytest.approx(0.5)

    def test_custom_loss_without_optional_keywords(self):
        def custom_loss(strategy_coef, *, n):
            return (strategy_coef - 1).square().sum() + 1 / n

        coefs = optimize(
            n=2,
            bands=1,
            loss_fn=custom_loss,
            max_optimizer_steps=1,
        )

        torch.testing.assert_close(coefs, torch.ones(1, dtype=torch.float64))


class TestMomentumWorkloadCoef:
    """Tests for _momentum_workload_coef and workload-aware optimization."""

    def test_zero_momentum_coefficients(self):
        """β=0 produces identity workload [1, 0, 0, ...] with no NaN/Inf."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = _momentum_workload_coef(0.0, 10)
        assert coef.shape == (10,)
        assert coef[0] == pytest.approx(1.0)
        assert torch.all(coef[1:] == 0.0)
        assert torch.all(torch.isfinite(coef))

    def test_zero_momentum_emits_warning(self):
        """β=0 emits a UserWarning about identity workload."""
        with pytest.warns(UserWarning, match="identity workload"):
            _momentum_workload_coef(0.0, 5)

    def test_negative_momentum_raises(self):
        """Negative momentum is rejected."""
        with pytest.raises(ValueError, match="momentum must be >= 0"):
            _momentum_workload_coef(-0.1, 10)

    def test_typical_momentum_coefficients(self):
        """β=0.9 gives [1, 0.9, 0.81, ...]."""
        coef = _momentum_workload_coef(0.9, 4)
        expected = torch.tensor([1.0, 0.9, 0.81, 0.729], dtype=torch.float64)
        torch.testing.assert_close(coef, expected)

    def test_prefix_sum_momentum(self):
        """β=1.0 gives [1, 1, 1, ...] (prefix-sum workload)."""
        coef = _momentum_workload_coef(1.0, 5)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(coef, expected)

    def test_identity_workload_error_is_constant(self):
        """With identity workload, per-query error is constant (= 1) for identity strategy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wc = _momentum_workload_coef(0.0, 5)
        # Identity strategy: C = I, coef = [1, 0, 0, ...]
        error = per_query_error(
            strategy_coef=torch.tensor([1.0], dtype=torch.float64),
            n=5,
            workload_coef=wc,
        )
        # B_coef = solve(I, [1,0,0,0,0]) = [1,0,0,0,0]
        # cumsum([1,0,0,0,0]^2) = [1,1,1,1,1]
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(error, expected)

    def test_identity_workload_optimization_converges(self):
        """Optimizer produces finite loss with identity workload (β=0).

        With identity workload, the optimal single-band strategy is C=I,
        giving loss = sensitivity² × error = 1.0 × 1.0 = 1.0.
        The optimizer should converge to this or better (multi-band).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wc = _momentum_workload_coef(0.0, 20)

        coefs = optimize(n=20, bands=3, workload_coef=wc)

        # Basic sanity: result is finite, unit norm
        assert torch.all(torch.isfinite(coefs))
        assert torch.linalg.norm(coefs).item() == pytest.approx(1.0, abs=1e-6)

        # Loss should be finite and ≤ identity baseline (1.0)
        opt_loss = loss(coefs, n=20, workload_coef=wc)
        assert torch.isfinite(opt_loss)
        assert opt_loss.item() <= 1.0 + 1e-6

    def test_momentum_workload_vs_prefix_sum(self):
        """Momentum workload produces different (better) loss than prefix-sum
        when the optimizer actually uses momentum."""
        n = 30
        bands = 5
        wc_momentum = _momentum_workload_coef(0.95, n)
        wc_prefix = _momentum_workload_coef(1.0, n)

        coefs_mom = optimize(n=n, bands=bands, workload_coef=wc_momentum)
        coefs_pfx = optimize(n=n, bands=bands, workload_coef=wc_prefix)

        # Evaluate BOTH strategies under the momentum workload
        loss_mom = loss(coefs_mom, n=n, workload_coef=wc_momentum)
        loss_pfx = loss(coefs_pfx, n=n, workload_coef=wc_momentum)

        # Strategy optimized for momentum workload should be at least as good
        # (lower loss) as the one optimized for prefix-sum, when evaluated
        # under the momentum workload.
        assert loss_mom.item() <= loss_pfx.item() + 1e-6
