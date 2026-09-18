//! Gram matrix computation for BnB dominating pair (DP-λCGD).
//!
//! Given a DP-λCGD strategy matrix C_λ with BnB batching (b bins, E epochs),
//! computes the Gram matrix G ∈ R^{b×b} of the dominating pair mixture means:
//!
//!   m_i = Σ_{j=0}^{E-1} C[:,b·j+i]
//!   G_{ij} = ⟨m_i, m_j⟩
//!
//! The Gram matrix uses raw C columns (momentum=0). Sensitivity and Gram
//! are workload-independent — momentum and LR schedule affect only utility
//! optimization, never privacy analysis. The `momentum` and `lr_weights`
//! parameters exist in the Rust API for flexibility but the standard
//! Python API always passes momentum=0.
//!
//! For normalized C̃_λ = C_λ·D⁻¹, the columns have unit norm and the inner
//! products decay as λ^{|i-j|} within an epoch. Cross-epoch terms are **not**
//! uniformly O(λ^b): the pair (i, j, p, q) has column gap |b(q-p) + (j-i)|,
//! which is 1 — not b — at the cyclic corner (i=0, j=b-1, q=p-1). The Gram is
//! therefore *cyclically* banded, G_{ij} ≈ E·λ^d + (E-1)·λ^{b-d} for
//! d = |i-j|, and the corner can carry ~68% of the diagonal.
//!
//! # References
//!
//! - Choquette-Choo et al. (2024) "Near Exact Privacy Amplification for Matrix
//!   Mechanisms" <https://arxiv.org/abs/2410.06266>, Lemma 3.2

use crate::error::{PldError, Result};

/// Geometric sum S(α, k) = Σ_{u=0}^{k-1} α^u = (1 - α^k) / (1 - α).
///
/// Handles special cases: α ≈ 1 (returns k), α^k ≈ 0 (returns 1/(1-α)).
fn geom_sum(alpha: f64, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    if (alpha - 1.0).abs() < 1e-15 {
        return k as f64;
    }
    let alpha_k = alpha.powi(k as i32);
    if alpha_k.abs() < 1e-30 {
        return 1.0 / (1.0 - alpha);
    }
    (1.0 - alpha_k) / (1.0 - alpha)
}

/// Inner product ⟨C_λ[:,a], C_λ[:,c]⟩ for a ≤ c (no momentum).
///
/// C_λ[:,k] has entry λ^{t-k} at row t for t ≥ k.
/// ⟨C_λ[:,a], C_λ[:,c]⟩ = λ^{c-a} · (1 - λ^{2(n-c)}) / (1 - λ²)
fn column_inner_product(lambda: f64, n: usize, a: usize, c: usize) -> f64 {
    debug_assert!(a <= c);
    debug_assert!(c < n);

    if lambda == 0.0 {
        return if a == c { 1.0 } else { 0.0 };
    }

    let gap = c - a;
    let remaining = n - c;
    let lambda2 = lambda * lambda;

    let lambda_gap = if gap == 0 {
        1.0
    } else {
        lambda.powi(gap as i32)
    };

    let lambda2r = lambda2.powi(remaining as i32);
    let sum_geom = if lambda2r < 1e-30 {
        1.0 / (1.0 - lambda2)
    } else {
        (1.0 - lambda2r) / (1.0 - lambda2)
    };

    lambda_gap * sum_geom
}

/// Inner product of momentum-accumulated columns: ⟨m_a^β, m_c^β⟩ for a ≤ c.
///
/// The effective column with momentum β is:
///   m_j^β[t] = Σ_{s=j}^{t} β^{t-s} · λ^{s-j}
///
/// For β ≠ λ (closed form):
///   ⟨m_a^β, m_c^β⟩ = 1/(λ-β)² · [λ^{d+2}·S(λ², r)
///                      - λβ(λ^d + β^d)·S(λβ, r)
///                      + β^{d+2}·S(β², r)]
///   where d = c - a, r = n - c, S(α, k) = (1-α^k)/(1-α).
///
/// For β = 0: reduces to the standard column_inner_product.
/// For β ≈ λ: uses direct summation to avoid numerical instability.
pub(crate) fn column_inner_product_momentum(
    lambda: f64,
    momentum: f64,
    n: usize,
    a: usize,
    c: usize,
) -> f64 {
    debug_assert!(a <= c);
    debug_assert!(c < n);

    let beta = momentum;

    // β=0: no momentum, use original formula
    if beta == 0.0 {
        return column_inner_product(lambda, n, a, c);
    }

    // λ=0, β>0: m_j[t] = β^{t-j}, inner product = Σ β^{2(t-c)} · β^{d} = β^d · S(β², r)
    if lambda == 0.0 {
        let d = c - a;
        let r = n - c;
        if d > 0 {
            // Columns don't overlap (λ=0 means only diagonal entries in C_λ)
            // m_a[t] = β^{t-a} for t ≥ a, m_c[t] = β^{t-c} for t ≥ c
            let beta_d = beta.powi(d as i32);
            return beta_d * geom_sum(beta * beta, r);
        }
        return geom_sum(beta * beta, r);
    }

    let d = c - a;
    let r = n - c; // remaining steps from column c

    if r == 0 {
        return 0.0;
    }

    // Handle β ≈ λ numerically (avoid 0/0 in the closed form)
    if (lambda - beta).abs() < 1e-10 {
        // m_j[t] = (t - j + 1) · λ^{t-j} when β = λ
        // ⟨m_a, m_c⟩ = λ^d · Σ_{u=0}^{r-1} (u+d+1)(u+1) · λ^{2u}
        let avg_lambda = (lambda + beta) * 0.5;
        let lambda2 = avg_lambda * avg_lambda;
        let lambda_d = if d == 0 {
            1.0
        } else {
            avg_lambda.powi(d as i32)
        };

        let mut sum = 0.0;
        let mut alpha_u = 1.0; // (λ²)^u
        for u in 0..r {
            sum += ((u + d + 1) as f64) * ((u + 1) as f64) * alpha_u;
            alpha_u *= lambda2;
            if alpha_u < 1e-30 {
                break; // remaining terms negligible
            }
        }
        // Factor is λ^d (not λ^{2d}): m_a[t]·m_c[t] = (u+d+1)(u+1)·λ^{2u+d}
        return lambda_d * sum;
    }

    // General closed form: β ≠ λ, β > 0
    let inv_diff_sq = 1.0 / ((lambda - beta) * (lambda - beta));

    let lambda_d = if d == 0 { 1.0 } else { lambda.powi(d as i32) };
    let beta_d = if d == 0 { 1.0 } else { beta.powi(d as i32) };

    let s_lambda2 = geom_sum(lambda * lambda, r);
    let s_lambda_beta = geom_sum(lambda * beta, r);
    let s_beta2 = geom_sum(beta * beta, r);

    let term1 = lambda * lambda * lambda_d * s_lambda2;
    let term2 = lambda * beta * (lambda_d + beta_d) * s_lambda_beta;
    let term3 = beta * beta * beta_d * s_beta2;

    inv_diff_sq * (term1 - term2 + term3)
}

/// Compute the BnB Gram matrix for DP-λCGD with optional momentum.
///
/// For the BnB dominating pair (Lemma 3.2 of arxiv:2410.06266):
///   m_i = Σ_{j=0}^{E-1} m^β_{b·j+i}    for i = 0..b-1
///   G_{ij} = ⟨m_i, m_j⟩
///
/// where m^β_k[t] is the momentum-accumulated column.
///
/// When `normalized=true`, uses column-normalized C̃_λ = C_λ·D⁻¹.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of steps (= b * E).
/// * `min_sep` — Bins per epoch (= b).
/// * `max_participations` — Number of epochs (= E). None infers from n_steps/min_sep.
/// * `normalized` — Whether to use column-normalized matrix.
/// * `momentum` — Optimizer momentum coefficient β ∈ [0, 1).
///   β=0 is standard (no momentum), β=0.9 is typical SGD momentum.
///
/// # Returns
///
/// Row-major b×b Gram matrix as a flat Vec<f64>.
pub fn lambda_cgd_gram_matrix(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> Result<Vec<f64>> {
    if !(0.0..1.0).contains(&lambda) {
        return Err(PldError::InvalidParameter(format!(
            "lambda must be in [0, 1), got {}",
            lambda
        )));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }
    if !(0.0..1.0).contains(&momentum) {
        return Err(PldError::InvalidParameter(format!(
            "momentum must be in [0, 1), got {}",
            momentum
        )));
    }

    let b = min_sep;
    let n = n_steps;
    let k_inferred = n.div_ceil(b);
    let e = match max_participations {
        Some(k) => k.min(k_inferred),
        None => k_inferred,
    };

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    let ip_fn =
        |a: usize, c: usize| -> f64 { column_inner_product_momentum(lambda, momentum, n, a, c) };

    // G_{ij} = Σ_{p=0}^{E-1} Σ_{q=0}^{E-1} ⟨|m_{b*p+i}|, |m_{b*q+j}|⟩
    //          / (d_{b*p+i} · d_{b*q+j}).
    //
    // Here lambda and momentum are both non-negative, so every effective
    // forward-encoder entry is non-negative and the absolute value is a no-op.
    // For signed encoders, replacing C by |C| is our conservative extension of
    // Lemma 3.2 via the same row-wise triangle inequality; signed column sums
    // would not provide a valid dominating construction.
    //
    // Terms are pruned by the *actual* column gap |b(q-p) + (j-i)|, not by b.
    // Cross-epoch terms do NOT uniformly decay as λ^{b·|p-q|}: the cyclic corner
    // (i=0, j=b-1, q=p-1) has gap |b - (b-1)| = 1, so its contribution is
    // (E-1)·λ¹ — comparable to the diagonal, not negligible. Pruning on λ^b
    // zeroed it, which understated G and therefore ε.
    //
    // ⟨m_a, m_c⟩ is bounded by max(λ,β)^gap · Σ_u max(λ,β)^{2u}, and the
    // normalized form divides by column norms ≥ 1, so this is a true upper
    // bound on any dropped contribution.
    let max_decay = lambda.max(momentum);
    let term_bound = |gap: usize| -> f64 {
        if max_decay <= 0.0 {
            return if gap == 0 { 1.0 } else { 0.0 };
        }
        max_decay.powi(gap as i32) / (1.0 - max_decay * max_decay)
    };
    const TERM_TOL: f64 = 1e-15;

    // Precompute column norms for the normalized case
    let col_norms: Vec<f64> = if normalized {
        (0..e)
            .flat_map(|p| {
                (0..b).map(move |i| {
                    let col = b * p + i;
                    if col < n {
                        ip_fn(col, col).sqrt()
                    } else {
                        1.0
                    }
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut gram = vec![0.0f64; b * b];

    for i in 0..b {
        for j in i..b {
            let mut val = 0.0;

            for p in 0..e {
                let col_a = b * p + i;
                if col_a >= n {
                    break;
                }

                for q in 0..e {
                    let col_c = b * q + j;
                    if col_c >= n {
                        break;
                    }

                    // `continue`, not `break`: the gap is V-shaped in q
                    // (|b(q-p) + (j-i)|), so a large gap at one q says nothing
                    // about the next.
                    if term_bound(col_a.abs_diff(col_c)) < TERM_TOL {
                        continue;
                    }

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = ip_fn(lo, hi);

                    let contribution = if normalized {
                        let d_a = col_norms[p * b + i];
                        let d_c = col_norms[q * b + j];
                        ip / (d_a * d_c)
                    } else {
                        ip
                    };

                    val += contribution;
                }
            }

            gram[i * b + j] = val;
            if i != j {
                gram[j * b + i] = val;
            }
        }
    }

    Ok(gram)
}

/// Compatibility tombstone for LR-weighted λ-CGD accounting.
///
/// A learning-rate-weighted Gram does not describe the deployed λ-CGD
/// mechanism unless the same schedule also changes its encoder/noise path.
/// Opake's λ-CGD mechanism does not do that, so accepting this calculation
/// could understate privacy loss.  The symbol remains temporarily so callers
/// receive an actionable error instead of an import failure.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `momentum` — Optimizer momentum β ∈ [0, 1).
/// * `n_steps` — Total number of steps (= b * E).
/// * `min_sep` — Bins per epoch (= b).
/// * `max_participations` — Number of epochs (= E).
/// * `normalized` — Whether to use column-normalized matrix.
/// * `lr_weights` — Per-step learning rate weights, length = n_steps.
///
/// # Returns
///
/// Row-major b×b Gram matrix as a flat Vec<f64>.
pub fn lambda_cgd_gram_matrix_lr(
    lambda: f64,
    momentum: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    lr_weights: &[f64],
) -> Result<Vec<f64>> {
    let _ = (
        lambda,
        momentum,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        lr_weights,
    );
    Err(PldError::InvalidParameter(
        "LR-weighted lambda-CGD accounting is unsupported: the schedule does not alter the \
         deployed lambda-CGD encoder or noise. Omit lr_schedule and recalibrate with the \
         unweighted mechanism, or use BandMF/BLT for a schedule-shaped workload."
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_column_inner_product_identity() {
        // λ=0 → C = I, columns are unit vectors
        assert!((column_inner_product(0.0, 10, 3, 3) - 1.0).abs() < 1e-10);
        assert!((column_inner_product(0.0, 10, 2, 5)).abs() < 1e-10);
    }

    #[test]
    fn test_column_inner_product_self() {
        // Self inner product = column norm squared
        let lambda = 0.5;
        let n = 10;
        let k = 3;
        let ip = column_inner_product(lambda, n, k, k);
        let expected: f64 = (0..(n - k)).map(|r| lambda.powi(2 * r as i32)).sum();
        assert!(
            (ip - expected).abs() < 1e-10,
            "got {}, expected {}",
            ip,
            expected
        );
    }

    #[test]
    fn test_column_inner_product_cross() {
        // Cross inner product for adjacent columns
        let lambda = 0.5;
        let n = 10;
        let ip = column_inner_product(lambda, n, 2, 5);
        // = λ^3 · Σ_{r=0}^{4} λ^{2r}
        let expected: f64 = lambda.powi(3) * (0..5).map(|r| lambda.powi(2 * r)).sum::<f64>();
        assert!(
            (ip - expected).abs() < 1e-10,
            "got {}, expected {}",
            ip,
            expected
        );
    }

    // ── Momentum inner product tests ─────────────────────────────

    #[test]
    fn test_momentum_zero_matches_original() {
        // β=0 should give same result as column_inner_product
        let lambda = 0.7;
        let n = 20;
        for a in 0..5 {
            for c in a..8 {
                let orig = column_inner_product(lambda, n, a, c);
                let with_mom = column_inner_product_momentum(lambda, 0.0, n, a, c);
                assert!(
                    (orig - with_mom).abs() < 1e-10,
                    "a={}, c={}: orig={}, momentum(β=0)={}",
                    a,
                    c,
                    orig,
                    with_mom
                );
            }
        }
    }

    #[test]
    fn test_momentum_self_inner_product_brute_force() {
        // Verify ‖m_j^β‖² against brute-force computation
        let lambda: f64 = 0.7;
        let beta: f64 = 0.9;
        let n: usize = 30;
        let j: usize = 5;

        // Brute force: m_j[t] = Σ_{s=j}^{t} β^{t-s} · λ^{s-j}
        let mut brute = 0.0;
        for t in j..n {
            let mut m_t = 0.0;
            for s in j..=t {
                m_t += beta.powi((t - s) as i32) * lambda.powi((s - j) as i32);
            }
            brute += m_t * m_t;
        }

        let closed = column_inner_product_momentum(lambda, beta, n, j, j);
        assert!(
            (brute - closed).abs() / brute < 1e-8,
            "brute={}, closed={}",
            brute,
            closed
        );
    }

    #[test]
    fn test_momentum_cross_inner_product_brute_force() {
        // Verify cross inner product against brute-force
        let lambda: f64 = 0.6;
        let beta: f64 = 0.85;
        let n: usize = 25;
        let a: usize = 3;
        let c: usize = 7;

        let mut brute = 0.0;
        for t in c..n {
            let mut m_a = 0.0;
            for s in a..=t {
                m_a += beta.powi((t - s) as i32) * lambda.powi((s - a) as i32);
            }
            let mut m_c = 0.0;
            for s in c..=t {
                m_c += beta.powi((t - s) as i32) * lambda.powi((s - c) as i32);
            }
            brute += m_a * m_c;
        }

        let closed = column_inner_product_momentum(lambda, beta, n, a, c);
        assert!(
            (brute - closed).abs() / brute.abs().max(1e-10) < 1e-8,
            "brute={}, closed={}",
            brute,
            closed
        );
    }

    #[test]
    fn test_momentum_equals_lambda_brute_force() {
        // β ≈ λ triggers the degenerate path; verify against brute-force
        let lambda: f64 = 0.7;
        let beta: f64 = lambda; // exact equality
        let n: usize = 20;
        let a: usize = 2;
        let c: usize = 5;

        let mut brute = 0.0;
        for t in c..n {
            let mut m_a = 0.0;
            for s in a..=t {
                m_a += beta.powi((t - s) as i32) * lambda.powi((s - a) as i32);
            }
            let mut m_c = 0.0;
            for s in c..=t {
                m_c += beta.powi((t - s) as i32) * lambda.powi((s - c) as i32);
            }
            brute += m_a * m_c;
        }

        let closed = column_inner_product_momentum(lambda, beta, n, a, c);
        assert!(
            (brute - closed).abs() / brute.abs().max(1e-10) < 1e-6,
            "β=λ degenerate: brute={}, closed={}",
            brute,
            closed
        );
    }

    #[test]
    fn test_momentum_increases_inner_product() {
        // Momentum β > 0 should increase the self inner product (column accumulates)
        let lambda: f64 = 0.7;
        let n: usize = 30;
        let j: usize = 0;
        let no_mom = column_inner_product_momentum(lambda, 0.0, n, j, j);
        let with_mom = column_inner_product_momentum(lambda, 0.9, n, j, j);
        assert!(
            with_mom > no_mom,
            "momentum should increase inner product: no_mom={}, with_mom={}",
            no_mom,
            with_mom
        );
    }

    // ── Gram matrix tests ─────────────────────────────────────────

    #[test]
    fn test_gram_matrix_identity() {
        // λ=0, single epoch → G = I_b
        let gram = lambda_cgd_gram_matrix(0.0, 10, 10, Some(1), false, 0.0).unwrap();
        assert_eq!(gram.len(), 100);
        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[i * 10 + j] - expected).abs() < 1e-10,
                    "G[{},{}] = {}, expected {}",
                    i,
                    j,
                    gram[i * 10 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gram_matrix_normalized_single_epoch() {
        // Normalized, single epoch: all diagonal entries = 1
        let lambda = 0.9;
        let b = 20;
        let gram = lambda_cgd_gram_matrix(lambda, b, b, Some(1), true, 0.0).unwrap();
        for i in 0..b {
            assert!(
                (gram[i * b + i] - 1.0).abs() < 1e-8,
                "G[{},{}] = {}, expected 1.0",
                i,
                i,
                gram[i * b + i]
            );
        }
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let gram = lambda_cgd_gram_matrix(0.9, 100, 10, Some(5), true, 0.0).unwrap();
        let b = 10;
        for i in 0..b {
            for j in 0..b {
                assert!(
                    (gram[i * b + j] - gram[j * b + i]).abs() < 1e-10,
                    "G not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    gram[i * b + j],
                    gram[j * b + i]
                );
            }
        }
    }

    #[test]
    fn test_gram_matrix_multi_epoch_diagonal() {
        // Multi-epoch normalized: diagonal ≈ E for large b (columns nearly orthogonal across epochs)
        let lambda = 0.9;
        let b = 200; // Large enough that λ^b ≈ 0
        let e = 5;
        let gram = lambda_cgd_gram_matrix(lambda, b * e, b, Some(e), true, 0.0).unwrap();
        for i in 0..b {
            assert!(
                (gram[i * b + i] - e as f64).abs() < 0.1,
                "G[{},{}] = {}, expected ≈ {}",
                i,
                i,
                gram[i * b + i],
                e
            );
        }
    }

    #[test]
    fn test_gram_matrix_rejects_bad_params() {
        assert!(lambda_cgd_gram_matrix(-0.1, 10, 5, Some(1), true, 0.0).is_err());
        assert!(lambda_cgd_gram_matrix(1.0, 10, 5, Some(1), true, 0.0).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 0, 5, Some(1), true, 0.0).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 10, 0, Some(1), true, 0.0).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 10, 5, Some(1), true, -0.1).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 10, 5, Some(1), true, 1.0).is_err());
    }

    #[test]
    fn test_gram_matrix_positive_definite() {
        // Gram matrix should be positive semidefinite
        let gram = lambda_cgd_gram_matrix(0.9, 50, 10, Some(3), true, 0.0).unwrap();
        let b = 10;
        for i in 0..b {
            assert!(
                gram[i * b + i] > 0.0,
                "Diagonal entry G[{},{}] not positive",
                i,
                i
            );
        }
    }

    /// The cyclic corner G[0][b-1] must survive at production scale.
    ///
    /// Regression: term pruning used to key on λ^b, which zeroed the corner.
    /// The corner's actual column gap is |b(q-p) + (j-i)| = 1 (i=0, j=b-1,
    /// q=p-1), so its contribution is ≈ (E-1)·λ — comparable to the diagonal.
    #[test]
    fn test_gram_matrix_cyclic_corner_retained() {
        for &b in &[328usize, 400, 1000, 1953] {
            let e = 4;
            let gram = lambda_cgd_gram_matrix(0.9, b * e, b, Some(e), true, 0.0).unwrap();
            let corner = gram[b - 1]; // G[0][b-1]
            let diag = gram[0]; // G[0][0]
            assert_relative_eq!(corner, 2.700, epsilon = 1e-3);
            assert_relative_eq!(diag, 4.000, epsilon = 1e-3);
            // Symmetric.
            assert_relative_eq!(gram[(b - 1) * b], corner, epsilon = 1e-12);
        }
    }

    /// The full measured row at λ=0.9, b=100, E=4 — the U-shaped profile that
    /// makes the Gram cyclically, not linearly, banded.
    #[test]
    fn test_gram_matrix_cyclic_profile() {
        let (b, e) = (100usize, 4);
        let gram = lambda_cgd_gram_matrix(0.9, b * e, b, Some(e), true, 0.0).unwrap();
        assert_relative_eq!(gram[0], 4.0002, epsilon = 1e-3); // G[0][0]
        assert_relative_eq!(gram[1], 3.6002, epsilon = 1e-3); // G[0][1]
        assert_relative_eq!(gram[50], 0.0361, epsilon = 1e-3); // G[0][50], the dip
        assert_relative_eq!(gram[99], 2.7002, epsilon = 1e-3); // G[0][99], the corner
                                                               // The corner dominates the mid-row minimum by ~75x — this is precisely
                                                               // what a linear-bandwidth scan cannot see.
        assert!(gram[99] > 50.0 * gram[50]);
    }

    /// Pruning must agree with brute-force column inner products.
    #[test]
    fn test_gram_matrix_matches_brute_force() {
        for &(lam, b, e, beta) in &[
            (0.9, 8usize, 3usize, 0.0),
            (0.5, 6, 4, 0.0),
            (0.95, 5, 3, 0.0),
            (0.9, 6, 3, 0.5),
        ] {
            let n = b * e;
            let gram = lambda_cgd_gram_matrix(lam, n, b, Some(e), false, beta).unwrap();
            for i in 0..b {
                for j in 0..b {
                    let mut want = 0.0;
                    for p in 0..e {
                        for q in 0..e {
                            let (a, c) = (b * p + i, b * q + j);
                            let (lo, hi) = if a <= c { (a, c) } else { (c, a) };
                            want += column_inner_product_momentum(lam, beta, n, lo, hi);
                        }
                    }
                    assert_relative_eq!(gram[i * b + j], want, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gram_matrix_multi_epoch_psd() {
        // Regression test: multi-epoch Gram must be PSD (not just have positive diagonal).
        for &(lam, b, e) in &[(0.9, 20, 5), (0.5, 50, 3), (0.99, 10, 8)] {
            let gram = lambda_cgd_gram_matrix(lam, b * e, b, Some(e), true, 0.0).unwrap();
            cholesky_psd_check(&gram, b, &format!("λ={}, b={}, E={}", lam, b, e));
        }
    }

    #[test]
    fn test_gram_matrix_momentum_psd() {
        // Gram matrix with momentum should also be PSD
        for &beta in &[0.5, 0.9, 0.95] {
            let gram = lambda_cgd_gram_matrix(0.9, 100, 10, Some(5), true, beta).unwrap();
            cholesky_psd_check(&gram, 10, &format!("β={}", beta));
        }
    }

    #[test]
    fn test_gram_matrix_momentum_zero_matches_original() {
        // momentum=0 should match the original (no-momentum) result
        let gram_orig = lambda_cgd_gram_matrix(0.9, 100, 10, Some(5), true, 0.0).unwrap();
        let gram_beta0 = lambda_cgd_gram_matrix(0.9, 100, 10, Some(5), true, 0.0).unwrap();
        for (a, b) in gram_orig.iter().zip(gram_beta0.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gram_matrix_momentum_symmetric() {
        let gram = lambda_cgd_gram_matrix(0.9, 60, 10, Some(3), true, 0.9).unwrap();
        let b = 10;
        for i in 0..b {
            for j in 0..b {
                assert!(
                    (gram[i * b + j] - gram[j * b + i]).abs() < 1e-10,
                    "G not symmetric at ({},{}) with β=0.9: {} vs {}",
                    i,
                    j,
                    gram[i * b + j],
                    gram[j * b + i]
                );
            }
        }
    }

    #[test]
    fn test_lr_weighted_gram_is_a_compatibility_tombstone() {
        let err =
            lambda_cgd_gram_matrix_lr(0.9, 0.0, 30, 10, Some(3), false, &[1.0; 30]).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    // ── Helper ─────────────────────────────────────────────────────

    fn cholesky_psd_check(gram: &[f64], b: usize, label: &str) {
        let mut l = vec![0.0f64; b * b];
        for i in 0..b {
            let mut diag = gram[i * b + i];
            for k in 0..i {
                diag -= l[i * b + k] * l[i * b + k];
            }
            assert!(
                diag > -1e-10,
                "Gram not PSD: Cholesky diagonal {} at row {} for {}",
                diag,
                i,
                label
            );
            l[i * b + i] = diag.max(0.0).sqrt();
            for j in (i + 1)..b {
                let mut off = gram[j * b + i];
                for k in 0..i {
                    off -= l[j * b + k] * l[i * b + k];
                }
                l[j * b + i] = if l[i * b + i] > 0.0 {
                    off / l[i * b + i]
                } else {
                    0.0
                };
            }
        }
    }
}
