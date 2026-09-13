//! BISR (Banded Inverse Square Root) sensitivity and Gram matrix computation.
//!
//! BISR (Kalinin et al., ICLR 2026) generalises DP-λCGD to arbitrary
//! bandwidth p ≥ 2. The strategy matrix C = A^{1/2} (square root of the
//! workload), and C^{-1} = A^{-1/2} is a lower-triangular Toeplitz matrix
//! truncated to bandwidth p.
//!
//! Given the banded C^{-1} coefficients `[c̃_0, c̃_1, ..., c̃_{p-1}]`,
//! the columns of C satisfy the recurrence:
//!
//! ```text
//! C[j, j] = 1 / c̃_0
//! C[t, j] = -Σ_{k=1}^{min(t-j, p-1)} c̃_k · C[t-k, j] / c̃_0   for t > j
//! ```
//!
//! For p=2 with c̃ = [1, -λ], this gives C[t,j] = λ^{t-j} (standard λCGD).
//!
//! # References
//!
//! - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots:
//!   An Optimal Bound on the Matrix Factorization Error for Multi-Epoch
//!   Differentially Private SGD"
//!   <https://arxiv.org/abs/2505.12128>

use crate::error::{PldError, Result};

/// Build column 0 of the strategy matrix C from banded C^{-1} coefficients.
///
/// Due to the Toeplitz structure, column j has entries identical to column 0
/// shifted by j positions (and truncated to n-j entries). So we only ever
/// need column 0 of length n.
///
/// Returns a Vec of length `len` containing C[0..len, 0].
///
/// This preserves the original unchecked Rust API. New callers, including the
/// Python boundary, should prefer [`try_bisr_column_zero`].
pub fn bisr_column_zero_pub(coefficients: &[f64], len: usize) -> Vec<f64> {
    bisr_column_zero(coefficients, len)
}

/// Checked variant of [`bisr_column_zero_pub`].
pub fn try_bisr_column_zero(coefficients: &[f64], len: usize) -> Result<Vec<f64>> {
    validate_common(coefficients, len, 1, 0.0)?;
    bisr_effective_column_zero(coefficients, len, 0.0)
}

fn bisr_column_zero(coefficients: &[f64], len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let p = coefficients.len();
    let alpha0 = coefficients[0];
    debug_assert!(alpha0.abs() >= 1e-30, "c̃_0 must have magnitude >= 1e-30");

    let mut col = vec![0.0f64; len];
    col[0] = 1.0 / alpha0;

    for t in 1..len {
        let mut s = 0.0;
        let k_max = t.min(p - 1);
        for k in 1..=k_max {
            s += coefficients[k] * col[t - k];
        }
        col[t] = -s / alpha0;
    }
    col
}

/// Inner product of two BISR columns with momentum accumulation.
///
/// Column `a` has entries `col0[0..n-a]` (shifted from column 0).
/// Column `c` has entries `col0[0..n-c]` (with c ≥ a).
///
/// With momentum β:
///   m_j^β[t] = Σ_{s=j}^{t} β^{t-s} · C[s, j]
///
/// The inner product is `⟨m_a^β, m_c^β⟩ = Σ_{t=c}^{n-1} m_a[t] · m_c[t]`.
///
/// Due to Toeplitz structure, this depends only on `gap = c - a` and
/// `remaining = n - c`. We exploit this in the Gram matrix computation.
fn bisr_column_inner_product_momentum_impl(
    col0: &[f64],
    momentum: f64,
    n: usize,
    a: usize,
    c: usize,
    absolute: bool,
) -> f64 {
    debug_assert!(a <= c);
    debug_assert!(c < n);

    let remaining = n - c;
    let gap = c - a;

    if momentum == 0.0 {
        // No momentum: direct column inner product
        // ⟨C[:,a], C[:,c]⟩ = Σ_{t=c}^{n-1} col0[t-a] · col0[t-c]
        //                   = Σ_{u=0}^{remaining-1} col0[u+gap] · col0[u]
        let mut dot = 0.0;
        for u in 0..remaining {
            let idx_a = u + gap;
            if idx_a >= col0.len() || u >= col0.len() {
                break;
            }
            dot += if absolute {
                col0[idx_a].abs() * col0[u].abs()
            } else {
                col0[idx_a] * col0[u]
            };
        }
        return dot;
    }

    // With momentum: accumulate m_a[t] and m_c[t] incrementally.
    // m_j[t] = β · m_j[t-1] + C[t, j]  for t ≥ j, else 0
    //
    // We need m_a[t] for t = c..n-1 and m_c[t] for t = c..n-1.
    // m_a starts accumulating at t = a (gap steps before m_c).
    // m_c starts accumulating at t = c.

    // First, accumulate m_a from t=a to t=c-1 (gap steps, before the dot starts)
    let mut acc_a = 0.0;
    for t in 0..gap {
        // C[t+a, a] = col0[t]
        if t < col0.len() {
            acc_a = momentum * acc_a + col0[t];
        } else {
            acc_a *= momentum;
        }
    }

    // Now accumulate both and dot from t=c to t=n-1
    let mut acc_c = 0.0;
    let mut dot = 0.0;
    for u in 0..remaining {
        let idx_a = u + gap;
        let ca = if idx_a < col0.len() { col0[idx_a] } else { 0.0 };
        let cc = if u < col0.len() { col0[u] } else { 0.0 };

        acc_a = momentum * acc_a + ca;
        acc_c = momentum * acc_c + cc;

        dot += if absolute {
            acc_a.abs() * acc_c.abs()
        } else {
            acc_a * acc_c
        };
    }

    dot
}

fn bisr_column_inner_product_momentum(
    col0: &[f64],
    momentum: f64,
    n: usize,
    a: usize,
    c: usize,
) -> f64 {
    bisr_column_inner_product_momentum_impl(col0, momentum, n, a, c, false)
}

/// Inner product of elementwise-absolute effective forward columns.
///
/// Balls-in-Bins accounting uses `|C|` in its dominating pair. Keep this
/// separate from the signed helper used by sensitivity calculations.
fn bisr_abs_column_inner_product_momentum(
    col0: &[f64],
    momentum: f64,
    n: usize,
    a: usize,
    c: usize,
) -> f64 {
    bisr_column_inner_product_momentum_impl(col0, momentum, n, a, c, true)
}

fn validate_coefficients(coefficients: &[f64]) -> Result<()> {
    if coefficients.len() < 2 {
        return Err(PldError::InvalidParameter(format!(
            "BISR bandwidth must be >= 2 (coefficients length), got {}",
            coefficients.len()
        )));
    }
    if let Some((index, coefficient)) = coefficients
        .iter()
        .enumerate()
        .find(|(_, coefficient)| !coefficient.is_finite())
    {
        return Err(PldError::InvalidParameter(format!(
            "BISR coefficients must be finite, found {coefficient} at index {index}"
        )));
    }
    if coefficients[0].abs() < 1e-30 {
        return Err(PldError::InvalidParameter(
            "c̃_0 (first coefficient) must have magnitude >= 1e-30".into(),
        ));
    }
    Ok(())
}

fn validate_common(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    momentum: f64,
) -> Result<()> {
    validate_coefficients(coefficients)?;
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }
    if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
        return Err(PldError::InvalidParameter(format!(
            "momentum must be in [0, 1), got {}",
            momentum
        )));
    }
    Ok(())
}

/// Build the first column after the optional optimizer-momentum transform.
///
/// The transformed encoder remains lower-triangular Toeplitz, with first
/// column `effective[t] = momentum * effective[t - 1] + col0[t]`.
fn bisr_effective_column_zero(
    coefficients: &[f64],
    n_steps: usize,
    momentum: f64,
) -> Result<Vec<f64>> {
    let col0 = bisr_column_zero(coefficients, n_steps);
    let mut effective = Vec::with_capacity(n_steps);
    let mut accumulator = 0.0;

    for (index, value) in col0.into_iter().enumerate() {
        accumulator = momentum * accumulator + value;
        if !accumulator.is_finite() {
            return Err(PldError::NumericalError(format!(
                "BISR forward recurrence produced a non-finite value at index {index}"
            )));
        }
        effective.push(accumulator);
    }

    Ok(effective)
}

/// Smallest non-negative, non-increasing pointwise majorant of `|values|`.
fn absolute_nonincreasing_majorant(values: &[f64]) -> Vec<f64> {
    let mut majorant = vec![0.0; values.len()];
    let mut suffix_max = 0.0_f64;
    for index in (0..values.len()).rev() {
        suffix_max = suffix_max.max(values[index].abs());
        majorant[index] = suffix_max;
    }
    majorant
}

/// Safe min-sep sensitivity bound for a (possibly signed) Toeplitz encoder.
///
/// Pointwise domination by a non-negative encoder is valid for arbitrary
/// record-update directions by the triangle inequality. The reverse suffix
/// maximum is the smallest pointwise majorant satisfying the monotonicity
/// assumption of the specialized Toeplitz sensitivity theorem.
fn toeplitz_majorant_sensitivity_squared(
    values: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    let majorant = absolute_nonincreasing_majorant(values);
    let result = crate::matrix_factorization::sensitivity::toeplitz_minsep_sensitivity_squared(
        &majorant,
        n_steps,
        min_sep,
        max_participations,
    )?;
    if !result.is_finite() {
        return Err(PldError::NumericalError(
            "BISR sensitivity bound is non-finite".into(),
        ));
    }
    Ok(result)
}

fn effective_k(n_steps: usize, min_sep: usize, max_participations: Option<usize>) -> usize {
    debug_assert!(n_steps >= 1);
    debug_assert!(min_sep >= 1);
    let k_inferred = n_steps.div_ceil(min_sep);
    match max_participations {
        Some(max_k) => max_k.min(k_inferred),
        None => k_inferred,
    }
}

/// Squared sensitivity upper bound for BISR under min-sep participation.
///
/// The result is exact when the effective forward Toeplitz coefficients are
/// non-negative and non-increasing. For arbitrary custom coefficients, their
/// absolute reverse-suffix-max majorant is used, giving a safe upper bound.
pub fn bisr_sensitivity_squared(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }
    debug_assert!(k >= 1);

    let effective = bisr_effective_column_zero(coefficients, n, momentum)?;
    if k == 1 {
        let result: f64 = effective.iter().map(|value| value * value).sum();
        if !result.is_finite() {
            return Err(PldError::NumericalError(
                "BISR single-participation sensitivity is non-finite".into(),
            ));
        }
        return Ok(result);
    }

    toeplitz_majorant_sensitivity_squared(&effective, n, b, Some(k))
}

/// Squared sensitivity upper bound for column-normalized BISR.
///
/// At lag `r`, every normalized column entry is pointwise bounded by
/// `q_r = |c_r| / sqrt(sum_{u=0}^r c_u^2)`, where `c` is the effective
/// forward Toeplitz sequence. Applying the non-negative, non-increasing
/// majorant theorem to `q` yields a safe bound for arbitrary coefficients.
/// The result is capped by `k²`, since normalized columns have unit norm.
pub fn bisr_normalized_sensitivity_squared(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }
    debug_assert!(k >= 1);
    let effective = bisr_effective_column_zero(coefficients, n, momentum)?;
    if k == 1 {
        return Ok(1.0);
    }

    let mut prefix_norm = 0.0_f64;
    let mut normalized_envelope = Vec::with_capacity(n);
    for (index, value) in effective.into_iter().enumerate() {
        // `hypot` avoids avoidable overflow/underflow in the prefix norm.
        prefix_norm = prefix_norm.hypot(value);
        if !prefix_norm.is_finite() || prefix_norm == 0.0 {
            return Err(PldError::NumericalError(format!(
                "BISR normalized prefix norm is invalid at index {index}"
            )));
        }
        normalized_envelope.push(value.abs() / prefix_norm);
    }

    let theorem_bound = toeplitz_majorant_sensitivity_squared(&normalized_envelope, n, b, Some(k))?;
    let column_norm_bound = (k as f64) * (k as f64);
    let result = theorem_bound.min(column_norm_bound);
    if !result.is_finite() {
        return Err(PldError::NumericalError(
            "BISR normalized sensitivity bound is non-finite".into(),
        ));
    }
    Ok(result)
}

/// BnB Gram matrix for BISR with optional momentum.
///
/// G_{ij} = ⟨m_i, m_j⟩ where
/// `m_i = Σ_e |C̃[:, i + e·b]|` and `C̃` is independently
/// column-normalized when requested.
pub fn bisr_gram_matrix(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> Result<Vec<f64>> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let e = effective_k(n, b, max_participations);

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    // Materialize the post-momentum Toeplitz column once. Besides avoiding
    // repeated recurrence work, this rejects a non-finite recurrence before
    // it can become a NaN/Inf Gram matrix.
    let col0 = bisr_effective_column_zero(coefficients, n, momentum)?;

    // Precompute column norms for normalization
    let col_norms: Vec<f64> = if normalized {
        let mut norms = Vec::with_capacity(e * b);
        for p in 0..e {
            for i in 0..b {
                let col = b * p + i;
                if col < n {
                    let squared_norm = bisr_column_inner_product_momentum(&col0, 0.0, n, col, col);
                    if !squared_norm.is_finite() || squared_norm <= 0.0 {
                        return Err(PldError::NumericalError(format!(
                            "BISR column norm is invalid at column {col}"
                        )));
                    }
                    norms.push(squared_norm.sqrt());
                } else {
                    norms.push(1.0);
                }
            }
        }
        norms
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

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = bisr_abs_column_inner_product_momentum(&col0, 0.0, n, lo, hi);

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

    if gram.iter().any(|value| !value.is_finite()) {
        return Err(PldError::NumericalError(
            "BISR Gram matrix is non-finite".into(),
        ));
    }
    Ok(gram)
}

/// Compatibility tombstone for LR-weighted BISR accounting.
///
/// A learning-rate-weighted Gram does not describe the deployed BISR
/// mechanism unless the same schedule changes its encoder/noise path. Opaque's
/// BISR mechanism does not do that, so the symbol remains only to return an
/// actionable migration error to old callers.
pub fn bisr_gram_matrix_lr(
    coefficients: &[f64],
    momentum: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    lr_weights: &[f64],
) -> Result<Vec<f64>> {
    let _ = (
        coefficients,
        momentum,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        lr_weights,
    );
    Err(PldError::InvalidParameter(
        "LR-weighted BISR accounting is unsupported: the schedule does not alter the \
         deployed BISR encoder or noise. Omit lr_schedule and recalibrate with the \
         unweighted mechanism, or use BandMF/BLT for a schedule-shaped workload."
            .into(),
    ))
}

/// BnB Gram matrix for a banded Toeplitz strategy with known forward coefficients.
///
/// For a Toeplitz strategy with coefficients `[c_0, c_1, ..., c_{p-1}]`,
/// column j has entries `|C[t,j]| = |c_{t-j}|` for `j ≤ t < j+p`, else 0.
///
/// The inner product of columns a and c (a ≤ c, gap d = c-a) is:
///   ⟨|C[:,a]|, |C[:,c]|⟩ = Σ_{k=0}^{p-1-d} |c_{k+d}| · |c_k|
///   (if d < p, else 0)
///
/// This is used for BnB accounting of BandMF/BLT mechanisms where the
/// strategy coefficients are known from the Toeplitz optimization.
///
/// Note: the columns are NOT generated via a recurrence (unlike BISR).
/// The strategy coefficients ARE the column entries directly.
pub fn toeplitz_gram_matrix(
    strategy_coef: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
) -> Result<Vec<f64>> {
    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if strategy_coef
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return Err(PldError::InvalidParameter(
            "strategy_coef must contain only finite values".into(),
        ));
    }
    if strategy_coef[0] == 0.0 {
        return Err(PldError::InvalidParameter(
            "strategy_coef[0] must be non-zero".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    // Lemma 3.2 is stated for a non-negative forward encoder. Replacing a
    // signed encoder by its elementwise absolute value is a conservative
    // extension via the row-wise triangle inequality and prevents cancellation
    // when epoch columns are aggregated. Column norms are unchanged, so
    // normalization still applies to each original column before aggregation.
    let strategy_coef: Vec<f64> = strategy_coef.iter().map(|coef| coef.abs()).collect();
    let strategy_coef = strategy_coef.as_slice();

    let p = strategy_coef.len(); // bandwidth
    let b = min_sep;
    let n = n_steps;
    let e = effective_k(n, b, max_participations);

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    // Boundary columns have truncated entries, so compute each norm from its
    // actually present prefix. `hypot` avoids avoidable overflow/underflow.
    let col_norm = |col: usize| -> f64 {
        let remaining = n - col; // entries available
        let effective_len = remaining.min(p);
        strategy_coef[..effective_len]
            .iter()
            .fold(0.0_f64, |norm, coefficient| norm.hypot(*coefficient))
    };

    let col_ip = |a: usize, c: usize| -> f64 {
        // Inner product of columns a and c (a ≤ c)
        let d = c - a;
        if d >= p {
            return 0.0;
        }
        let remaining = n - c; // both columns exist from c to min(c+p, n)
        let effective_len = remaining.min(p - d);
        let mut s = 0.0;
        for k in 0..effective_len {
            s += strategy_coef[k + d] * strategy_coef[k];
        }
        s
    };

    // Build BnB Gram matrix: G[i,j] = Σ_{p,q epochs} ⟨C[:,b*p+i], C[:,b*q+j]⟩ / (norm * norm)
    let mut gram = vec![0.0f64; b * b];

    for i in 0..b {
        for j in i..b {
            let mut val = 0.0;

            for ep_p in 0..e {
                let col_a = b * ep_p + i;
                if col_a >= n {
                    break;
                }

                for ep_q in 0..e {
                    let col_c = b * ep_q + j;
                    if col_c >= n {
                        break;
                    }

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = col_ip(lo, hi);

                    let contribution = if normalized {
                        let d_a = col_norm(col_a);
                        let d_c = col_norm(col_c);
                        if d_a > 0.0 && d_c > 0.0 {
                            ip / (d_a * d_c)
                        } else {
                            0.0
                        }
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

    if gram.iter().any(|value| !value.is_finite()) {
        return Err(PldError::NumericalError(
            "Toeplitz Gram matrix is non-finite".into(),
        ));
    }
    Ok(gram)
}

#[cfg(test)]
mod tests {
    use super::*;

    // BISR inverse coefficients for prefix-sum workload:
    // r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}
    // All negative after r̃_0: [1, -1/2, -1/8, -1/16, -5/128, ...]
    const BISR_P2: [f64; 2] = [1.0, -0.5];
    const BISR_P3: [f64; 3] = [1.0, -0.5, -0.125];
    const BISR_P4: [f64; 4] = [1.0, -0.5, -0.125, -0.0625];
    const BISR_P5: [f64; 5] = [1.0, -0.5, -0.125, -0.0625, -0.0390625];

    // λCGD coefficients for comparison
    fn lambda_cgd_coefs(lambda: f64) -> [f64; 2] {
        [1.0, -lambda]
    }

    fn inverse_coefficients_from_forward(forward: &[f64]) -> Vec<f64> {
        assert!(!forward.is_empty());
        assert_ne!(forward[0], 0.0);

        let mut inverse = vec![0.0; forward.len()];
        inverse[0] = 1.0 / forward[0];
        for index in 1..forward.len() {
            let convolution: f64 = (0..index).map(|j| inverse[j] * forward[index - j]).sum();
            inverse[index] = -convolution / forward[0];
        }
        inverse
    }

    /// Exhaustive feasible scalar-update sensitivity for a small Toeplitz
    /// encoder. This is a lower bound on the vector-valued sensitivity and is
    /// useful for checking that an upper bound never relies on cancellation.
    fn exhaustive_scalar_sensitivity_squared(
        effective: &[f64],
        min_sep: usize,
        max_participations: usize,
        normalized: bool,
    ) -> f64 {
        let n = effective.len();
        assert!(n < usize::BITS as usize);

        let column_norms: Vec<f64> = (0..n)
            .map(|column| {
                effective[..n - column]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let mut best = 0.0_f64;
        for support_mask in 1_usize..(1_usize << n) {
            let participation_count = support_mask.count_ones() as usize;
            if participation_count > max_participations {
                continue;
            }

            let support: Vec<usize> = (0..n)
                .filter(|index| support_mask & (1_usize << index) != 0)
                .collect();
            if support.windows(2).any(|pair| pair[1] - pair[0] < min_sep) {
                continue;
            }

            for sign_mask in 0_usize..(1_usize << participation_count) {
                let mut squared_norm = 0.0;
                for row in 0..n {
                    let mut output = 0.0;
                    for (support_index, &column) in support.iter().enumerate() {
                        if column > row {
                            continue;
                        }
                        let sign = if sign_mask & (1_usize << support_index) == 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let scale = if normalized {
                            column_norms[column]
                        } else {
                            1.0
                        };
                        output += sign * effective[row - column] / scale;
                    }
                    squared_norm += output * output;
                }
                best = best.max(squared_norm);
            }
        }
        best
    }

    fn next_random_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn next_random_signed_unit(state: &mut u64) -> f64 {
        let fraction = (next_random_u64(state) >> 11) as f64 / ((1_u64 << 53) as f64);
        2.0 * fraction - 1.0
    }

    #[test]
    fn test_column_zero_p2_is_lambda_cgd() {
        // p=2 with [1, -0.5] should give col0[t] = 0.5^t (λCGD with λ=0.5)
        let col = bisr_column_zero(&BISR_P2, 10);
        for t in 0..10 {
            let expected = 0.5_f64.powi(t as i32);
            assert!(
                (col[t] - expected).abs() < 1e-10,
                "t={}: got {}, expected {}",
                t,
                col[t],
                expected
            );
        }
    }

    #[test]
    fn test_checked_column_zero_accepts_minimum_leading_magnitude() {
        let col = try_bisr_column_zero(&[1e-30, 0.0], 1).unwrap();
        assert_eq!(col.len(), 1);
        assert!(col[0].is_finite());
        assert!((col[0] / 1e30 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_column_zero_p2_general_lambda() {
        // [1, -λ] should give col0[t] = λ^t
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);
        let col = bisr_column_zero(&coefs, 20);
        for t in 0..20 {
            let expected = lambda.powi(t as i32);
            assert!(
                (col[t] - expected).abs() < 1e-10,
                "t={}: got {}, expected {}",
                t,
                col[t],
                expected
            );
        }
    }

    #[test]
    fn test_column_zero_p3_entries_decay() {
        // Higher bandwidth columns should still decay
        let col = bisr_column_zero(&BISR_P3, 30);
        assert!((col[0] - 1.0).abs() < 1e-10);
        // Entries should decay towards 0
        let max_entry = col.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_entry < 10.0, "column entries should be bounded");
        // Late entries should be small
        assert!(col[29].abs() < 0.1, "col[29] = {} should be small", col[29]);
    }

    #[test]
    fn test_sensitivity_p2_matches_lambda_cgd() {
        // BISR p=2 with [1, -λ] should match λCGD sensitivity
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);

        let bisr_sens = bisr_sensitivity_squared(&coefs, 100, 10, Some(3), 0.0).unwrap();

        // Compare with λCGD
        let lcgd_sens = crate::matrix_factorization::lambda_cgd_sensitivity_squared(
            lambda,
            100,
            10,
            Some(3),
            0.0,
        )
        .unwrap();

        assert!(
            (bisr_sens - lcgd_sens).abs() / lcgd_sens < 1e-6,
            "BISR p=2: {}, λCGD: {}",
            bisr_sens,
            lcgd_sens
        );
    }

    #[test]
    fn test_sensitivity_p2_normalized_majorant_bounds_lambda_cgd() {
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);
        let k = 3;

        let bisr_bound =
            bisr_normalized_sensitivity_squared(&coefs, 100, 10, Some(k), 0.0).unwrap();
        let lcgd_sens = crate::matrix_factorization::lambda_cgd_normalized_sensitivity_squared(
            lambda,
            100,
            10,
            Some(k),
            0.0,
        )
        .unwrap();

        assert!(
            bisr_bound + 1e-12 >= lcgd_sens,
            "normalized BISR bound: {}, λCGD sensitivity: {}",
            bisr_bound,
            lcgd_sens
        );
        assert!(bisr_bound <= (k * k) as f64);
    }

    #[test]
    fn test_normalized_k1_is_one() {
        // Normalized sensitivity with k=1 should always be 1.0
        for coefs in [&BISR_P2[..], &BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let sens = bisr_normalized_sensitivity_squared(coefs, 100, 100, Some(1), 0.0).unwrap();
            assert!(
                (sens - 1.0).abs() < 1e-10,
                "p={}: normalized k=1 sens = {}, expected 1.0",
                coefs.len(),
                sens
            );
        }
    }

    #[test]
    fn test_zero_participations_has_zero_sensitivity() {
        assert_eq!(
            bisr_sensitivity_squared(&BISR_P3, 10, 1, Some(0), 0.0).unwrap(),
            0.0
        );
        assert_eq!(
            bisr_normalized_sensitivity_squared(&BISR_P3, 10, 1, Some(0), 0.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_sensitivity_positive_finite() {
        for coefs in [&BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let sens = bisr_sensitivity_squared(coefs, 200, 20, Some(5), 0.0).unwrap();
            assert!(
                sens > 0.0 && sens.is_finite(),
                "p={}: sens = {}",
                coefs.len(),
                sens
            );
        }
    }

    #[test]
    fn test_sensitivity_brute_force_small() {
        // Brute-force: build C, compute ‖Σ columns at participation steps‖²
        let coefs = &BISR_P3;
        let n: usize = 15;
        let b: usize = 5;
        let k: usize = 3;

        let col0 = bisr_column_zero(coefs, n);

        // S² = Σ_t (Σ_{p=0}^{k-1} C[t, p*b])²
        let mut brute_sq = 0.0;
        for t in 0..n {
            let mut combined = 0.0;
            for p in 0..k {
                let col_start = p * b;
                if col_start > t {
                    continue;
                }
                let entry_idx = t - col_start;
                if entry_idx < col0.len() {
                    combined += col0[entry_idx];
                }
            }
            brute_sq += combined * combined;
        }

        let computed = bisr_sensitivity_squared(coefs, n, b, Some(k), 0.0).unwrap();
        assert!(
            (brute_sq - computed).abs() / brute_sq < 1e-8,
            "brute={}, computed={}",
            brute_sq,
            computed
        );
    }

    #[test]
    fn test_signed_custom_sensitivity_uses_non_cancelling_majorant() {
        // C^{-1} = [1, 1] gives forward coefficients [1, -1, 1, -1].
        // Feasible scalar updates [1, -1, 1, -1] make the released vector
        // [1, -2, 3, -4], whose squared norm is exactly 30.
        let sensitivity_sq = bisr_sensitivity_squared(&[1.0, 1.0], 4, 1, Some(4), 0.0).unwrap();
        assert!((sensitivity_sq - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalized_majorant_covers_later_placement_counterexample() {
        // These inverse coefficients recover the non-negative,
        // non-increasing forward sequence [1, 1, 1, 0] at n=4. Column
        // normalization invalidates the old earliest-placement shortcut:
        // columns (1, 3) are more correlated than columns (0, 2).
        let inverse = [1.0, -1.0, 0.0, 1.0];
        let old_earliest_squared = 2.0 + 2.0 / 6.0_f64.sqrt();
        let feasible_later_squared = 2.0 + 2.0 / 3.0_f64.sqrt();
        assert!(old_earliest_squared < feasible_later_squared);

        let bound = bisr_normalized_sensitivity_squared(&inverse, 4, 2, Some(2), 0.0).unwrap();
        assert!(bound + 1e-12 >= feasible_later_squared);
        assert!(bound <= 4.0); // universal k^2 cap
    }

    #[test]
    fn test_nonnegative_nonincreasing_forward_sequence_remains_exact() {
        let forward = [1.0, 0.8, 0.55, 0.3, 0.1, 0.05];
        let inverse = inverse_coefficients_from_forward(&forward);
        let expected = crate::matrix_factorization::toeplitz_minsep_sensitivity_squared(
            &forward,
            forward.len(),
            2,
            Some(3),
        )
        .unwrap();
        let actual = bisr_sensitivity_squared(&inverse, forward.len(), 2, Some(3), 0.0).unwrap();

        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_is_applied_before_majorization() {
        let inverse = [1.0, 1.0];
        let effective = bisr_effective_column_zero(&inverse, 4, 0.5).unwrap();
        assert_eq!(effective, vec![1.0, -0.5, 0.75, -0.625]);

        let expected = toeplitz_majorant_sensitivity_squared(&effective, 4, 1, Some(4)).unwrap();
        let actual = bisr_sensitivity_squared(&inverse, 4, 1, Some(4), 0.5).unwrap();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_random_small_majorants_dominate_all_scalar_witnesses() {
        let mut random_state = 0x5eed_cafe_f00d_beef_u64;

        for case_index in 0..96 {
            let n = 2 + (next_random_u64(&mut random_state) as usize % 6);
            let bandwidth = 2 + (next_random_u64(&mut random_state) as usize % 3).min(n - 2);
            let mut inverse = Vec::with_capacity(bandwidth);
            inverse.push(1.0);
            for _ in 1..bandwidth {
                inverse.push(0.9 * next_random_signed_unit(&mut random_state));
            }
            let min_sep = 1 + next_random_u64(&mut random_state) as usize % n;
            let feasible_k = (n + min_sep - 1) / min_sep;
            let k = 1 + next_random_u64(&mut random_state) as usize % feasible_k;
            let momentum =
                0.8 * ((next_random_u64(&mut random_state) >> 11) as f64 / ((1_u64 << 53) as f64));
            let effective = bisr_effective_column_zero(&inverse, n, momentum).unwrap();

            for normalized in [false, true] {
                let bound = if normalized {
                    bisr_normalized_sensitivity_squared(&inverse, n, min_sep, Some(k), momentum)
                        .unwrap()
                } else {
                    bisr_sensitivity_squared(&inverse, n, min_sep, Some(k), momentum).unwrap()
                };
                let witness =
                    exhaustive_scalar_sensitivity_squared(&effective, min_sep, k, normalized);
                let tolerance = 1e-10 * bound.max(1.0);
                assert!(
                    witness <= bound + tolerance,
                    "case={case_index}, normalized={normalized}, n={n}, b={min_sep}, \
                     k={k}, momentum={momentum}, inverse={inverse:?}, \
                     witness={witness}, bound={bound}"
                );
            }
        }
    }

    #[test]
    fn test_sensitivity_with_momentum() {
        let coefs = &BISR_P3;
        let sens_no_mom = bisr_sensitivity_squared(coefs, 60, 10, Some(3), 0.0).unwrap();
        let sens_with_mom = bisr_sensitivity_squared(coefs, 60, 10, Some(3), 0.9).unwrap();
        assert!(sens_no_mom > 0.0);
        assert!(sens_with_mom > 0.0);
        // Both should be finite
        assert!(sens_no_mom.is_finite());
        assert!(sens_with_mom.is_finite());
    }

    #[test]
    fn test_gram_matrix_p2_matches_lambda_cgd() {
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);
        let b: usize = 10;
        let e: usize = 3;
        let n = b * e;

        let bisr_gram = bisr_gram_matrix(&coefs, n, b, Some(e), true, 0.0).unwrap();
        let lcgd_gram =
            crate::matrix_factorization::lambda_cgd_gram_matrix(lambda, n, b, Some(e), true, 0.0)
                .unwrap();

        for idx in 0..(b * b) {
            assert!(
                (bisr_gram[idx] - lcgd_gram[idx]).abs() / lcgd_gram[idx].abs().max(1e-10) < 1e-4,
                "entry {}: bisr={}, lcgd={}",
                idx,
                bisr_gram[idx],
                lcgd_gram[idx]
            );
        }
    }

    #[test]
    fn test_gram_matrix_signed_forward_uses_absolute_encoder() {
        // C^{-1} = [1, 1] gives forward coefficients [1, -1, 1, -1].
        // Signed epoch sums would produce G[0,1] = -7. The conservative
        // signed extension uses |C| and gives the non-cancelling Gram below.
        let gram = bisr_gram_matrix(&[1.0, 1.0], 4, 2, Some(2), false, 0.0).unwrap();
        let expected = [10.0, 7.0, 7.0, 6.0];
        for (actual, expected) in gram.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let gram = bisr_gram_matrix(&BISR_P4, 80, 10, Some(4), true, 0.0).unwrap();
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
    fn test_gram_matrix_psd() {
        // Cholesky PSD check
        for coefs in [&BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let b = 8;
            let e = 3;
            let gram = bisr_gram_matrix(coefs, b * e, b, Some(e), true, 0.0).unwrap();
            // Naive Cholesky
            let mut l = vec![0.0f64; b * b];
            for i in 0..b {
                let mut diag = gram[i * b + i];
                for k in 0..i {
                    diag -= l[i * b + k] * l[i * b + k];
                }
                assert!(
                    diag > -1e-10,
                    "Gram not PSD at row {} for p={}",
                    i,
                    coefs.len()
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

    #[test]
    fn test_lr_weighted_gram_is_a_compatibility_tombstone() {
        let err =
            bisr_gram_matrix_lr(&BISR_P3, 0.0, 24, 8, Some(3), false, &[1.0; 24]).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn test_toeplitz_gram_signed_coefficients_do_not_cancel() {
        let gram = toeplitz_gram_matrix(&[1.0, -1.0, 1.0, -1.0], 4, 2, Some(2), false).unwrap();
        assert_eq!(gram, vec![10.0, 7.0, 7.0, 6.0]);
    }

    #[test]
    fn test_toeplitz_gram_normalizes_columns_before_epoch_sum() {
        let coefs = [1.0, -1.0, 1.0, -1.0];
        let gram = toeplitz_gram_matrix(&coefs, 4, 2, Some(2), true).unwrap();

        let sqrt2 = 2.0_f64.sqrt();
        let sqrt3 = 3.0_f64.sqrt();
        let means = [
            [0.5, 0.5, 0.5 + 1.0 / sqrt2, 0.5 + 1.0 / sqrt2],
            [0.0, 1.0 / sqrt3, 1.0 / sqrt3, 1.0 + 1.0 / sqrt3],
        ];
        let mut expected = vec![0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                expected[i * 2 + j] = means[i]
                    .iter()
                    .zip(means[j])
                    .map(|(left, right)| left * right)
                    .sum();
            }
        }

        for (actual, expected) in gram.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rejects_bad_params() {
        assert!(bisr_sensitivity_squared(&[1.0], 10, 1, None, 0.0).is_err()); // bandwidth < 2
        assert!(bisr_sensitivity_squared(&BISR_P3, 0, 1, None, 0.0).is_err()); // n=0
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 0, None, 0.0).is_err()); // min_sep=0
        assert!(bisr_sensitivity_squared(&[0.0, -0.5], 10, 1, None, 0.0).is_err());
        assert!(bisr_sensitivity_squared(&[1e-31, -0.5], 10, 1, None, 0.0).is_err());
        assert!(bisr_sensitivity_squared(&[f64::NAN, -0.5], 10, 1, None, 0.0).is_err());
        assert!(bisr_sensitivity_squared(&[1.0, f64::INFINITY], 10, 1, None, 0.0).is_err());
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 1, None, f64::NAN).is_err());
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 1, None, -0.1).is_err());
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 1, None, 1.0).is_err()); // momentum=1

        // Finite inverse inputs can still yield an overflowing forward
        // recurrence; expose that as an error rather than an infinite bound.
        assert!(bisr_sensitivity_squared(&[1e-29, f64::MAX], 2, 1, None, 0.0).is_err());
        assert!(
            bisr_normalized_sensitivity_squared(&[1e-29, f64::MAX], 2, 2, Some(1), 0.0,).is_err()
        );
        assert!(try_bisr_column_zero(&BISR_P3, 0).is_err());
        assert!(try_bisr_column_zero(&[1.0, f64::INFINITY], 2).is_err());
        assert!(try_bisr_column_zero(&[1e-29, f64::MAX], 2).is_err());
        assert!(bisr_gram_matrix(&[1e-29, f64::MAX], 2, 1, None, true, 0.0).is_err());
        assert!(toeplitz_gram_matrix(&[f64::NAN], 2, 1, None, false).is_err());
        assert!(toeplitz_gram_matrix(&[0.0], 2, 1, None, true).is_err());
    }

    #[test]
    fn test_public_column_helper_retains_vec_return_type() {
        assert_eq!(bisr_column_zero_pub(&BISR_P2, 3), vec![1.0, 0.5, 0.25]);
    }
}
