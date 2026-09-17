"""Tests for Buffered Linear Toeplitz (BLT) mechanisms."""

import dataclasses
import itertools

import pytest
import torch

from opake.api.dpftrl.noise._blt_math import (
    BufferedToeplitz,
    LossFn,
    Parameterization,
    as_streaming_matrix,
    blt_pair_from_theta_pair,
    geometric_sum,
    get_init_blt,
    get_parameterized_loss,
    inverse,
    inverse_as_streaming_matrix,
    iteration_error,
    limit_max_error,
    limit_max_loss,
    loss,
    materialize,
    max_error,
    min_buf_decay_gap,
    optimize_loss,
    penalized_loss,
    robust_max_error_Gamma_j,
    robust_max_error_Gamma_jk,
    sensitivity_squared,
    toeplitz_coefs,
)
from opake.api.dpftrl.noise._toeplitz import materialize_lower_triangular


class TestBufferedToeplitz:
    def test_build(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.9, 0.5],
            output_scale=[0.3, 0.2],
        )
        assert blt._num_buffers == 2
        assert blt.dtype == torch.float64

    def test_canonicalize_sorts_descending(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.3, 0.9, 0.5],
            output_scale=[0.1, 0.2, 0.3],
        )
        # After canonicalization, buf_decay should be descending
        for i in range(1, blt._num_buffers):
            assert float(blt.buf_decay[i]) <= float(blt.buf_decay[i - 1])

    def test_toeplitz_coefs(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        coefs = toeplitz_coefs(blt, 5)
        assert coefs[0] == pytest.approx(1.0)
        assert coefs[1] == pytest.approx(0.3)
        assert coefs[2] == pytest.approx(0.3 * 0.5)
        assert coefs[3] == pytest.approx(0.3 * 0.5**2)

    def test_materialize(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        M = materialize(blt, 3)
        expected = materialize_lower_triangular(toeplitz_coefs(blt, 3))
        torch.testing.assert_close(M, expected)

    def test_empty_blt(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        coefs = toeplitz_coefs(blt, 3)
        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(coefs, expected)

    def test_pillutla_score(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        score = blt.pillutla_score()
        expected = 0.3 / 0.8 + 0.1 / 0.4
        assert score == pytest.approx(expected)


class TestBLTInverse:
    def test_inverse_roundtrip(self):
        """C @ C^{-1} should be identity."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        inv_blt = inverse(blt)

        n = 6
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-8, rtol=1e-8
        )

    def test_single_buffer_inverse(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        inv_blt = inverse(blt)
        n = 5
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-8, rtol=1e-8
        )


class TestBLTStreamingMatrix:
    def test_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        dense = materialize(blt, n)
        streaming = as_streaming_matrix(blt)
        streaming_dense = streaming.materialize(n)
        torch.testing.assert_close(streaming_dense, dense, atol=1e-8, rtol=1e-8)

    def test_inverse_streaming_matches_dense(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.7, 0.3],
            output_scale=[0.4, 0.2],
        )
        n = 5
        C = materialize(blt, n)
        C_inv_dense = torch.linalg.inv(C)
        C_inv_streaming = inverse_as_streaming_matrix(blt)
        C_inv_streaming_dense = C_inv_streaming.materialize(n)
        torch.testing.assert_close(
            C_inv_streaming_dense, C_inv_dense, atol=1e-8, rtol=1e-8
        )

    @pytest.mark.parametrize(
        ("buf_decay", "output_scale"),
        [([0.7, 0.3], [0.4, 0.2]), ([0.5], [0.3]), ([], [])],
    )
    def test_inverse_row_norms_closed_form_matches_probing_and_dense(
        self, buf_decay, output_scale
    ):
        blt = BufferedToeplitz.build(buf_decay=buf_decay, output_scale=output_scale)
        n = 6
        C_inv = inverse_as_streaming_matrix(blt)
        norms = C_inv.row_norms_squared(n)
        probing = dataclasses.replace(
            C_inv, row_norms_squared_fn=None
        ).row_norms_squared(n)
        torch.testing.assert_close(norms, probing, atol=1e-10, rtol=1e-10)
        dense = torch.linalg.inv(materialize(blt, n))
        torch.testing.assert_close(
            norms, dense.square().sum(dim=1), atol=1e-8, rtol=1e-8
        )

    def test_inverse_row_norms_closed_form_high_buffer_near_one_decays(self):
        # The default 10-buffer init has decays within 1e-6 of 1.0 — a
        # regime where an expanded rational-transfer-function filter
        # drifts from the truth by whole percents at long horizons.
        blt = get_init_blt(10)
        n = 1024
        C_inv = inverse_as_streaming_matrix(blt)
        norms = C_inv.row_norms_squared(n)
        probing = dataclasses.replace(
            C_inv, row_norms_squared_fn=None
        ).row_norms_squared(n)
        torch.testing.assert_close(norms, probing, atol=1e-10, rtol=1e-10)


class TestGeometricSum:
    def test_finite(self):
        a = torch.tensor(1.0, dtype=torch.float64)
        r = torch.tensor(0.5, dtype=torch.float64)
        result = geometric_sum(a, r, num=4)
        expected = 1 + 0.5 + 0.25 + 0.125
        assert float(result) == pytest.approx(expected)

    def test_infinite(self):
        a = torch.tensor(1.0, dtype=torch.float64)
        r = torch.tensor(0.5, dtype=torch.float64)
        result = geometric_sum(a, r)
        assert float(result) == pytest.approx(2.0)

    def test_zero_ratio(self):
        a = torch.tensor(3.0, dtype=torch.float64)
        r = torch.tensor(0.0, dtype=torch.float64)
        result = geometric_sum(a, r, num=5)
        assert float(result) == pytest.approx(3.0)

    def test_near_one_ratio(self):
        """geometric_sum should be accurate when r is very close to 1."""
        a = torch.tensor(1.0, dtype=torch.float64)
        r = torch.tensor(1 - 1e-10, dtype=torch.float64)
        result = geometric_sum(a, r, num=100)
        # For r very close to 1, the sum should be close to n=100
        assert float(result) == pytest.approx(100.0, rel=1e-4)


class TestRobustGammaJ:
    """Tests for the numerically robust Gamma_j computation."""

    def test_matches_brute_force_small_theta(self):
        """robust_max_error_Gamma_j matches brute force for theta far from 1."""
        omega = torch.tensor(0.3, dtype=torch.float64)
        theta = torch.tensor(0.5, dtype=torch.float64)
        n = 10
        result = robust_max_error_Gamma_j(omega, theta, n)
        # Brute force: sum_{i=1}^{n-1} geometric_sum(omega, theta, i) / n
        total = sum(float(geometric_sum(omega, theta, num=i)) for i in range(1, n))
        expected = total / n
        assert float(result) == pytest.approx(expected, rel=1e-6)

    def test_near_one_theta_finite(self):
        """robust_max_error_Gamma_j returns finite values for theta near 1."""
        omega = torch.tensor(0.3, dtype=torch.float64)
        theta = torch.tensor(1 - 1e-10, dtype=torch.float64)
        result = robust_max_error_Gamma_j(omega, theta, 1000)
        assert torch.isfinite(result)
        assert float(result) > 0


class TestRobustGammaJK:
    """Tests for the numerically robust Gamma_jk computation."""

    def test_matches_brute_force_small_thetas(self):
        """robust_max_error_Gamma_jk matches brute force for small thetas."""
        omega1 = torch.tensor(0.3, dtype=torch.float64)
        theta1 = torch.tensor(0.5, dtype=torch.float64)
        omega2 = torch.tensor(0.2, dtype=torch.float64)
        theta2 = torch.tensor(0.4, dtype=torch.float64)
        n = 10
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, n)
        # Brute force
        total = sum(
            float(geometric_sum(omega1, theta1, num=i))
            * float(geometric_sum(omega2, theta2, num=i))
            for i in range(1, n)
        )
        expected = total / n
        assert float(result) == pytest.approx(expected, rel=1e-6)

    def test_both_near_one_finite(self):
        """robust_max_error_Gamma_jk handles both thetas near 1."""
        omega1 = torch.tensor(0.3, dtype=torch.float64)
        theta1 = torch.tensor(1 - 1e-10, dtype=torch.float64)
        omega2 = torch.tensor(0.2, dtype=torch.float64)
        theta2 = torch.tensor(1 - 1e-10, dtype=torch.float64)
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, 1000)
        assert torch.isfinite(result)

    def test_one_near_one_finite(self):
        """robust_max_error_Gamma_jk handles one theta near 1."""
        omega1 = torch.tensor(0.3, dtype=torch.float64)
        theta1 = torch.tensor(1 - 1e-10, dtype=torch.float64)
        omega2 = torch.tensor(0.2, dtype=torch.float64)
        theta2 = torch.tensor(0.5, dtype=torch.float64)
        result = robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, 1000)
        assert torch.isfinite(result)


class TestIterationErrorRobust:
    """Test that iteration_error is finite even for BLTs with buf_decay near 1."""

    def test_normal_blt(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5, 0.3],
            output_scale=[-0.2, -0.1],
        )
        err = iteration_error(blt, 50)
        assert torch.isfinite(err)
        assert float(err) > 0

    def test_identity_error(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        error = max_error(blt, n=5)
        assert float(error) == pytest.approx(5.0)


class TestLimitMaxError:
    def test_identity(self):
        """Identity BLT (no buffers) should have limit_max_error = 1."""
        inv_blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = limit_max_error(inv_blt)
        assert float(result) == pytest.approx(1.0)

    def test_converges_to_finite_n(self):
        """limit_max_error should approximate max_error(n)/n for large n."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        inv_blt = inverse(blt)
        limit = float(limit_max_error(inv_blt))

        # For large n, max_error(n)/n should converge to limit_max_error
        n_large = 10000
        finite = float(max_error(inv_blt, n_large)) / n_large
        assert limit == pytest.approx(finite, rel=1e-3)

    def test_single_buffer(self):
        """Sanity check for single-buffer BLT."""
        inv_blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[-0.3],
        )
        result = limit_max_error(inv_blt)
        # 1 + 2*(-0.3)/(1-0.5) + (-0.3)^2/(1-0.5)^2
        #   = 1 + 2*(-0.6) + 0.36 = 1 - 1.2 + 0.36 = 0.16
        expected = 1 + 2 * (-0.3 / 0.5) + (-0.3 / 0.5) ** 2
        assert float(result) == pytest.approx(expected, rel=1e-10)


class TestLimitMaxLoss:
    def test_identity(self):
        """Identity BLT should have limit_max_loss = 1."""
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = limit_max_loss(blt)
        assert float(result) == pytest.approx(1.0)

    def test_converges_to_finite_n(self):
        """limit_max_loss should approximate loss for large n."""
        blt = BufferedToeplitz.build(
            buf_decay=[0.8, 0.4],
            output_scale=[0.3, 0.1],
        )
        limit = float(limit_max_loss(blt))

        inv_blt = inverse(blt)
        n_large = 10000
        finite_error = float(max_error(inv_blt, n_large)) / n_large
        finite_sens = float(sensitivity_squared(blt, n_large))
        finite_loss = finite_error * finite_sens
        assert limit == pytest.approx(finite_loss, rel=1e-3)


class TestSensitivitySquared:
    def test_identity(self):
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        result = sensitivity_squared(blt, n=100)
        assert float(result) == pytest.approx(1.0)

    def test_single_buffer(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        result = sensitivity_squared(blt, n=10)
        # sens^2 = 1 + sum of geometric series
        assert float(result) > 1.0


class TestMinBufDecayGap:
    def test_basic(self):
        theta = torch.tensor([0.9, 0.5, 0.3], dtype=torch.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.2)

    def test_duplicates(self):
        theta = torch.tensor([0.5, 0.5], dtype=torch.float64)
        gap = min_buf_decay_gap(theta)
        assert float(gap) == pytest.approx(0.0)


class TestBLTPairFromThetaPair:
    def test_inverse_property(self):
        theta = torch.tensor([0.8, 0.4], dtype=torch.float64)
        theta_hat = torch.tensor([0.6, 0.2], dtype=torch.float64)
        blt, inv_blt = blt_pair_from_theta_pair(theta, theta_hat)

        n = 6
        C = materialize(blt, n)
        C_inv = materialize(inv_blt, n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-6, rtol=1e-6
        )


class TestGetInitBLT:
    def test_zero_buffers(self):
        blt = get_init_blt(num_buffers=0)
        assert blt._num_buffers == 0

    def test_default_init(self):
        blt = get_init_blt(num_buffers=3)
        assert blt._num_buffers == 3
        assert blt.dtype == torch.float64

    def test_wrong_num_buffers(self):
        blt = BufferedToeplitz.build(
            buf_decay=[0.5],
            output_scale=[0.3],
        )
        with pytest.raises(ValueError, match="num_buffers"):
            get_init_blt(num_buffers=3, init_blt=blt)


class TestLossFn:
    def test_single_participation(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt = get_init_blt(num_buffers=2)
        loss_val = loss(loss_fn, blt)
        assert float(loss_val) > 0
        assert torch.isfinite(loss_val)

    def test_min_sep(self):
        loss_fn = LossFn.build_min_sep(n=20, min_sep=2, max_participations=3)
        blt = get_init_blt(num_buffers=2)
        loss_val = loss(loss_fn, blt, skip_checks=True)
        assert float(loss_val) > 0

    def test_penalized_loss(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt = get_init_blt(num_buffers=2)
        inv_blt = inverse(blt)
        penalized = penalized_loss(loss_fn, blt, inv_blt)
        plain = loss(loss_fn, blt)
        # Penalized loss should be close to plain loss (penalty is small)
        assert torch.isfinite(penalized)
        # penalty_strength is 1e-8, so difference should be small
        assert float(penalized) == pytest.approx(float(plain), abs=1.0)


class TestParameterization:
    def test_buf_decay_pair_roundtrip(self):
        """Parameters -> BLT -> parameters should approximately round-trip."""
        param = Parameterization.buf_decay_pair()
        blt = get_init_blt(num_buffers=2)
        params = param.params_from_blt(blt)
        assert params.shape == (4,)  # 2 theta + 2 theta_hat

        blt_out, inv_blt_out = param.blt_and_inverse_from_params(params)
        n = 10
        C = materialize(blt_out, n)
        C_inv = materialize(inv_blt_out, n)
        product = C @ C_inv
        torch.testing.assert_close(
            product, torch.eye(n, dtype=torch.float64), atol=1e-6, rtol=1e-6
        )

    def test_get_loss_fn(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        param = Parameterization.buf_decay_pair()
        blt = get_init_blt(num_buffers=2)
        params = param.params_from_blt(blt)
        opt_loss_fn = get_parameterized_loss(param, loss_fn)
        result = opt_loss_fn(params)
        assert torch.isfinite(result)
        assert float(result) > 0

    def test_interlaced_decay_pair_roundtrip_and_domain(self):
        param = Parameterization.interlaced_decay_pair()
        blt = get_init_blt(num_buffers=3)
        params = param.params_from_blt(blt)

        assert torch.all((params > 0) & (params < 1))
        decays = torch.cumprod(params, dim=0)
        assert torch.all(decays[:-1] > decays[1:])

        blt_out, inv_blt_out = param.blt_and_inverse_from_params(params)
        assert torch.all(blt_out.output_scale > 0)
        assert torch.all(inv_blt_out.output_scale < 0)

        coefs = toeplitz_coefs(blt_out, 10)
        assert torch.all(coefs >= 0)
        assert torch.all(coefs[:-1] >= coefs[1:])
        torch.testing.assert_close(materialize(blt_out, 10), materialize(blt, 10))
        torch.testing.assert_close(
            materialize(blt_out, 10) @ materialize(inv_blt_out, 10),
            torch.eye(10, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_interlaced_decay_pair_rejects_invalid_initialization(self):
        invalid_blt = BufferedToeplitz.build(
            buf_decay=[0.4],
            output_scale=[-0.2],
        )

        with pytest.raises(ValueError, match="strictly interlace"):
            Parameterization.interlaced_decay_pair().params_from_blt(invalid_blt)


class TestOptimizeLoss:
    def test_zero_buffers(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=10)
        blt, loss = optimize_loss(loss_fn, num_buffers=0)
        assert blt._num_buffers == 0
        assert float(loss) == pytest.approx(10.0)

    def test_single_buffer_improves(self):
        """L-BFGS should not make the loss worse (within tolerance)."""
        loss_fn = LossFn.build_closed_form_single_participation(n=100)
        init_blt = get_init_blt(num_buffers=1)
        init_loss = float(loss(loss_fn, init_blt))

        _opt_blt, opt_loss = optimize_loss(
            loss_fn, num_buffers=1, max_optimizer_steps=50
        )
        # Allow tiny floating-point tolerance
        assert float(opt_loss) <= init_loss * (1 + 1e-10)
        assert torch.isfinite(opt_loss)

    def test_two_buffers(self):
        loss_fn = LossFn.build_closed_form_single_participation(n=50)
        opt_blt, opt_loss = optimize_loss(
            loss_fn, num_buffers=2, max_optimizer_steps=50
        )
        assert opt_blt._num_buffers == 2
        assert torch.isfinite(opt_loss)
        assert float(opt_loss) > 0

    def test_rejects_infeasible_final_iterate(self, monkeypatch):
        def return_signed_pair(*_args, **_kwargs):
            # theta < theta_hat produces a negative strategy output scale.
            return torch.tensor([0.4, 0.6], dtype=torch.float64)

        monkeypatch.setattr(
            "opake.api.dpftrl.noise._toeplitz._lbfgs_optimize",
            return_signed_pair,
        )
        loss_fn = LossFn.build_min_sep(
            n=4,
            min_sep=1,
            max_participations=4,
        )

        with pytest.raises(RuntimeError, match="outside the sensitivity domain"):
            optimize_loss(
                loss_fn,
                num_buffers=1,
                parameterization=Parameterization.buf_decay_pair(),
            )

    def test_minsep_optimization_is_nontrivial_and_exact(self):
        n, min_sep, max_participations = 12, 3, 4
        loss_fn = LossFn.build_min_sep(
            n=n,
            min_sep=min_sep,
            max_participations=max_participations,
        )

        blt, loss_value = optimize_loss(
            loss_fn,
            num_buffers=1,
            max_optimizer_steps=100,
        )
        coefs = toeplitz_coefs(blt, n)

        assert torch.isfinite(loss_value)
        assert coefs[1] > 0.1  # Exercise a real correlated strategy, not identity.
        assert torch.all(coefs >= 0)
        assert torch.all(coefs[:-1] >= coefs[1:])

        # Independently enumerate every allowed participation set. This keeps
        # the regression independent of the closed-form sensitivity helper
        # used by the optimizer objective.
        matrix = materialize(blt, n)
        exhaustive = coefs.new_zeros(())
        for count in range(max_participations + 1):
            for participation in itertools.combinations(range(n), count):
                if any(
                    right - left < min_sep
                    for left, right in itertools.pairwise(participation)
                ):
                    continue
                indicator = coefs.new_zeros(n)
                indicator[list(participation)] = 1
                exhaustive = torch.maximum(
                    exhaustive,
                    torch.linalg.vector_norm(matrix @ indicator),
                )

        reported = loss_fn.sensitivity_squared_fn(blt).sqrt()
        torch.testing.assert_close(reported, exhaustive)


class TestFromRationalApprox:
    def test_basic(self):
        blt = BufferedToeplitz.from_rational_approx_to_sqrt_x(
            num_buffers=3,
            max_buf_decay=1 - 1e-6,
            max_pillutla_score=1 - 1e-6,
        )
        assert blt._num_buffers == 3
        assert torch.all(blt.buf_decay > 0)
        assert torch.all(blt.buf_decay < 1)

    def test_invalid_num_buffers(self):
        with pytest.raises(ValueError, match="num_buffers must be >= 1"):
            BufferedToeplitz.from_rational_approx_to_sqrt_x(num_buffers=0)
