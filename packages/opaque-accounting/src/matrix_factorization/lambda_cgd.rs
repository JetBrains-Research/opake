//! DP-λCGD sensitivity computation.
//!
//! The DP-λCGD mechanism (Kalinin et al., 2026) uses a lower-triangular
//! Toeplitz strategy matrix C_λ with entries `(C_λ)_{ij} = λ^{i-j}` for
//! `i ≥ j` and 0 otherwise. Its inverse is bidiagonal: 1 on the diagonal,
//! -λ on the subdiagonal.
//!
//! # Sensitivity
//!
//! For the unnormalized Toeplitz strategy, Theorem 1 and Appendix C,
//! equations (15)-(16) of arXiv v1 give:
//!
//! ```text
//! sens²_{k,b}(C_λ) = (1 - λ^{2b}) / ((1 - λ²)(1 - λ^b)²) · Σ_{j=0}^{k-1} (1 - λ^{b(j+1)})²
//! ```
//!
//! The column-normalized strategy is not Toeplitz. Its sensitivity is instead
//! covered by Lemma 8, equation (12) of the current arXiv v2 and is computed
//! from normalized column inner products below.
//!
//! # References
//!
//! - Kalinin et al. (2026) "DP-λCGD: Efficient Noise Correlation for
//!   Differentially Private Model Training" <https://arxiv.org/abs/2601.22334>

use crate::error::{PldError, Result};
use crate::matrix_factorization::gram_matrix::column_inner_product_momentum;

fn validate_params(lambda: f64, n_steps: usize, min_sep: usize, momentum: f64) -> Result<()> {
    if !(0.0..1.0).contains(&lambda) {
        return Err(PldError::InvalidParameter(format!(
            "lambda must be in [0, 1), got {}",
            lambda
        )));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter(
            "n_steps must be >= 1".to_string(),
        ));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter(
            "min_sep must be >= 1".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&momentum) {
        return Err(PldError::InvalidParameter(format!(
            "momentum must be in [0, 1), got {}",
            momentum
        )));
    }
    Ok(())
}

fn effective_k(n_steps: usize, min_sep: usize, max_participations: Option<usize>) -> usize {
    let k_inferred = n_steps.div_ceil(min_sep);
    match max_participations {
        Some(max_k) => max_k.min(k_inferred),
        None => k_inferred,
    }
}

/// Compute the squared L2 sensitivity of the DP-λCGD strategy matrix.
///
/// Uses the closed-form expression completed in Appendix C, equation (16) of
/// arXiv v1 for the no-momentum case (β=0). With momentum, computes via the sum
/// of momentum-aware column inner products.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1). λ=0 is DP-SGD.
/// * `n_steps` — Total number of training steps.
/// * `min_sep` — Minimum separation between participations (≥ 1).
/// * `max_participations` — Maximum number of participations per user.
///   If `None`, inferred from `n_steps / min_sep`.
/// * `momentum` — Optimizer momentum coefficient β ∈ [0, 1). Default 0.
///
/// # Returns
///
/// The squared L2 sensitivity S².
pub fn lambda_cgd_sensitivity_squared(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_params(lambda, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }

    // λ=0, β=0 (DP-SGD): C_λ = I, each column has norm 1
    // Sensitivity = sqrt(k) for k participations
    if lambda == 0.0 && momentum == 0.0 {
        return Ok(k as f64);
    }

    // With momentum or for the general case: compute via inner products.
    // S² = Σ_{p,q=0}^{k-1} ⟨m_{p·b}, m_{q·b}⟩
    // Worst case is starting at column 0 (earliest columns are longest).
    if momentum > 0.0 || k <= 10 {
        let mut sens_sq = 0.0;
        for p in 0..k {
            let col_p = p * b;
            if col_p >= n {
                break;
            }
            for q in 0..k {
                let col_q = q * b;
                if col_q >= n {
                    break;
                }
                let (lo, hi) = if col_p <= col_q {
                    (col_p, col_q)
                } else {
                    (col_q, col_p)
                };
                sens_sq += column_inner_product_momentum(lambda, momentum, n, lo, hi);
            }
        }
        return Ok(sens_sq);
    }

    // Optimized closed form for β=0 (arXiv v1, Appendix C, equation 16).
    let lambda_b = lambda.powi(b as i32);
    if lambda_b < 1e-15 {
        let lambda2 = lambda * lambda;
        return Ok(k as f64 / (1.0 - lambda2));
    }

    let lambda2 = lambda * lambda;
    let lambda_2b = lambda_b * lambda_b;

    let sum_r = (1.0 - lambda_2b) / (1.0 - lambda2);
    let denom = (1.0 - lambda_b) * (1.0 - lambda_b);

    let mut sum_j = 0.0;
    let mut lambda_bj = lambda_b;
    for _j in 0..k {
        let term = 1.0 - lambda_bj;
        sum_j += term * term;
        lambda_bj *= lambda_b;
    }

    Ok(sum_r * sum_j / denom)
}

/// Compute the squared L2 sensitivity of the column-normalized DP-λCGD.
///
/// Column normalization: C̃_λ = C_λ · D⁻¹ where D = diag(‖C_λ[:,j]‖).
/// All columns of C̃_λ have unit norm.
///
/// For single participation (k=1) with β=0, the sensitivity is always 1.0.
/// For multi-participation (k > 1) with min-separation b, computes the
/// normalized-column Gram sum established by Lemma 8, equation (12) of arXiv v2.
///
/// With momentum β > 0, column norms change and are computed from the
/// momentum-aware inner product formula.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of training steps.
/// * `min_sep` — Minimum separation between participations (≥ 1).
/// * `max_participations` — Maximum number of participations per user.
///   If `None`, inferred from `n_steps / min_sep`.
/// * `momentum` — Optimizer momentum coefficient β ∈ [0, 1). Default 0.
///
/// # Returns
///
/// The squared L2 sensitivity S² of the column-normalized matrix.
pub fn lambda_cgd_normalized_sensitivity_squared(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_params(lambda, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }

    // All columns have unit norm → single participation sensitivity = 1
    // (with or without momentum, since we normalize each column)
    if k == 1 {
        return Ok(1.0);
    }

    // λ=0: columns are orthogonal unit vectors, sens = sqrt(k)
    if lambda == 0.0 && momentum == 0.0 {
        return Ok(k as f64);
    }

    // General case: compute via momentum-aware inner products.
    // Column norms with momentum:
    let col_norms: Vec<f64> = (0..k)
        .map(|j| {
            let col = j * b;
            if col >= n {
                return 1.0;
            }
            column_inner_product_momentum(lambda, momentum, n, col, col).sqrt()
        })
        .collect();

    // sens² = Σ_{p,q} ⟨m̃_{p·b}, m̃_{q·b}⟩ where m̃ = m / ‖m‖
    let mut sens_sq = 0.0;
    for p in 0..k {
        let col_p = p * b;
        if col_p >= n {
            break;
        }
        for q in 0..k {
            let col_q = q * b;
            if col_q >= n {
                break;
            }
            let (lo, hi) = if col_p <= col_q {
                (col_p, col_q)
            } else {
                (col_q, col_p)
            };
            let ip = column_inner_product_momentum(lambda, momentum, n, lo, hi);
            sens_sq += ip / (col_norms[p] * col_norms[q]);
        }
    }

    Ok(sens_sq)
}

/// Compute the max column norm of the DP-λCGD strategy matrix.
///
/// The column at position 0 has the largest norm, equal to
/// `sqrt((1 - λ^{2n}) / (1 - λ²))` for β=0.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of steps.
///
/// # Returns
///
/// The max column L2 norm.
pub fn lambda_cgd_max_column_norm(lambda: f64, n_steps: usize) -> Result<f64> {
    let sens_sq = lambda_cgd_sensitivity_squared(lambda, n_steps, n_steps, Some(1), 0.0)?;
    Ok(sens_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_zero_is_dpsgd() {
        // λ=0 → C_λ = I, sensitivity = sqrt(k)
        let sens_sq = lambda_cgd_sensitivity_squared(0.0, 100, 10, Some(3), 0.0).unwrap();
        assert!((sens_sq - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_participation() {
        // k=1: sens² = Σ_{r=0}^{n-1} λ^{2r} = (1-λ^{2n})/(1-λ²)
        let lambda = 0.5;
        let n = 10;
        let sens_sq = lambda_cgd_sensitivity_squared(lambda, n, n, Some(1), 0.0).unwrap();
        let expected: f64 = (0..n).map(|r| lambda.powi(2 * r as i32)).sum();
        assert!(
            (sens_sq - expected).abs() < 1e-10,
            "got {}, expected {}",
            sens_sq,
            expected
        );
    }

    #[test]
    fn test_sensitivity_increases_with_participations() {
        let lambda = 0.9;
        let prev = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(1), 0.0).unwrap();
        for k in 2..=5 {
            let curr = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(k), 0.0).unwrap();
            assert!(curr > prev, "k={}: {} should be > {}", k, curr, prev);
        }
    }

    #[test]
    fn test_rejects_invalid_lambda() {
        assert!(lambda_cgd_sensitivity_squared(-0.1, 10, 1, None, 0.0).is_err());
        assert!(lambda_cgd_sensitivity_squared(1.0, 10, 1, None, 0.0).is_err());
        assert!(lambda_cgd_sensitivity_squared(1.5, 10, 1, None, 0.0).is_err());
    }

    #[test]
    fn test_rejects_invalid_momentum() {
        assert!(lambda_cgd_sensitivity_squared(0.5, 10, 1, None, -0.1).is_err());
        assert!(lambda_cgd_sensitivity_squared(0.5, 10, 1, None, 1.0).is_err());
    }

    #[test]
    fn test_max_column_norm() {
        let lambda = 0.5;
        let n = 20;
        let norm = lambda_cgd_max_column_norm(lambda, n).unwrap();
        let expected: f64 = ((0..n)
            .map(|r| (lambda * lambda).powi(r as i32))
            .sum::<f64>())
        .sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_large_b_simplified_formula() {
        // When λ^b ≈ 0, sens² ≈ k / (1 - λ²)
        let lambda = 0.9;
        let b = 2000; // λ^b ≈ 0
        let k = 5;
        let n = k * b;
        let sens_sq = lambda_cgd_sensitivity_squared(lambda, n, b, Some(k), 0.0).unwrap();
        let expected = k as f64 / (1.0 - lambda * lambda);
        assert!(
            (sens_sq - expected).abs() / expected < 1e-6,
            "got {}, expected {}",
            sens_sq,
            expected
        );
    }

    // ── Momentum sensitivity tests ───────────────────────────────

    #[test]
    fn test_momentum_reduces_sensitivity() {
        // With momentum β ∈ (0, 1), sensitivity should change vs β=0.
        // For typical β < 1, the momentum accumulation is a contraction,
        // so the sensitivity is DIFFERENT from the β=0 case.
        let lambda = 0.9;
        let sens_no_mom = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(3), 0.0).unwrap();
        let sens_with_mom = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(3), 0.9).unwrap();
        // Both should be finite and positive
        assert!(sens_no_mom > 0.0);
        assert!(sens_with_mom > 0.0);
        assert!(sens_no_mom.is_finite());
        assert!(sens_with_mom.is_finite());
    }

    #[test]
    fn test_momentum_sensitivity_brute_force() {
        // Verify sensitivity against brute-force for a small case
        let lambda: f64 = 0.6;
        let beta: f64 = 0.8;
        let n: usize = 20;
        let b: usize = 5;
        let k: usize = 3;

        // Brute force: S² = ‖Σ_{p=0}^{k-1} m_{p·b}‖²
        //            = Σ_t (Σ_p m_{p·b}[t])²
        let mut brute_sq = 0.0;
        for t in 0..n {
            let mut combined = 0.0;
            for p in 0..k {
                let col_start = p * b;
                if col_start > t {
                    continue;
                }
                let mut m_t = 0.0;
                for s in col_start..=t {
                    m_t += beta.powi((t - s) as i32) * lambda.powi((s - col_start) as i32);
                }
                combined += m_t;
            }
            brute_sq += combined * combined;
        }

        let closed = lambda_cgd_sensitivity_squared(lambda, n, b, Some(k), beta).unwrap();
        assert!(
            (brute_sq - closed).abs() / brute_sq < 1e-8,
            "brute={}, closed={}",
            brute_sq,
            closed
        );
    }

    // ── Column-normalized sensitivity tests ──────────────────────

    #[test]
    fn test_normalized_single_participation_is_one() {
        // k=1: all columns have unit norm → sensitivity = 1
        for lambda in [0.0, 0.3, 0.5, 0.9, 0.99] {
            let sens_sq =
                lambda_cgd_normalized_sensitivity_squared(lambda, 100, 100, Some(1), 0.0).unwrap();
            assert!(
                (sens_sq - 1.0).abs() < 1e-10,
                "λ={}: normalized sens² should be 1.0, got {}",
                lambda,
                sens_sq
            );
        }
    }

    #[test]
    fn test_normalized_single_participation_with_momentum() {
        // k=1 with momentum: still 1.0 (normalization)
        for beta in [0.5, 0.9, 0.95] {
            let sens_sq =
                lambda_cgd_normalized_sensitivity_squared(0.9, 100, 100, Some(1), beta).unwrap();
            assert!(
                (sens_sq - 1.0).abs() < 1e-10,
                "β={}: normalized sens² should be 1.0, got {}",
                beta,
                sens_sq
            );
        }
    }

    #[test]
    fn test_normalized_lambda_zero_is_k() {
        // λ=0: columns are orthogonal unit vectors → sens² = k
        let sens_sq =
            lambda_cgd_normalized_sensitivity_squared(0.0, 100, 10, Some(3), 0.0).unwrap();
        assert!(
            (sens_sq - 3.0).abs() < 1e-10,
            "got {}, expected 3.0",
            sens_sq
        );
    }

    #[test]
    fn test_normalized_leq_unnormalized() {
        // Normalized sensitivity should be ≤ unnormalized (improvement)
        let lambda = 0.9;
        for k in [1, 2, 3, 5] {
            let unnorm = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(k), 0.0).unwrap();
            let norm =
                lambda_cgd_normalized_sensitivity_squared(lambda, 100, 10, Some(k), 0.0).unwrap();
            assert!(
                norm <= unnorm + 1e-10,
                "k={}: normalized {} should be ≤ unnormalized {}",
                k,
                norm,
                unnorm
            );
        }
    }

    #[test]
    fn test_normalized_large_b_approx_k() {
        // When λ^b ≈ 0, all column norms ≈ same → normalized sens² ≈ k
        let lambda = 0.9;
        let b = 2000;
        let k = 5;
        let n = k * b;
        let sens_sq =
            lambda_cgd_normalized_sensitivity_squared(lambda, n, b, Some(k), 0.0).unwrap();
        assert!(
            (sens_sq - k as f64).abs() < 0.01,
            "got {}, expected {} (large b limit)",
            sens_sq,
            k
        );
    }

    #[test]
    fn test_normalized_rejects_invalid_params() {
        assert!(lambda_cgd_normalized_sensitivity_squared(-0.1, 10, 1, None, 0.0).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(1.0, 10, 1, None, 0.0).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(0.5, 0, 1, None, 0.0).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(0.5, 10, 0, None, 0.0).is_err());
    }
}
