//! Sensitivity computation for matrix factorization mechanisms.
//!
//! Implements sensitivity algorithms for MF-DP under various participation
//! patterns: single participation, min-sep (banded), and fixed-epoch.
//!
//! # References
//!
//! - Algorithm 3 (VecSens): Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
//! - Algorithm 4 (Sensitivity UB): Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
//! - Fixed-epoch: Choquette-Choo et al. (2022) <https://arxiv.org/abs/2211.06530>

use crate::error::{PldError, Result};

/// Maximum number of participations under a min-sep constraint.
///
/// Returns the largest number of participations possible in `n` rounds
/// with at least `min_sep` separation between consecutive participations.
///
/// # Arguments
///
/// * `n` — Number of rounds
/// * `min_sep` — Minimum separation between participations (≥ 1)
/// * `max_participations` — Optional upper bound on participations
///
/// # Returns
///
/// The effective maximum number of participations.
pub fn minsep_true_max_participations(
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> usize {
    let max_part_ub = n.div_ceil(min_sep);
    match max_participations {
        Some(max_p) => max_part_ub.min(max_p),
        None => max_part_ub,
    }
}

/// Solve max_u <x, u> where u is a {0,1}-vector respecting min-sep participation.
///
/// Uses dynamic programming with runtime O(len(x) * max_participations).
/// This is Algorithm 3 (VecSens) from Choquette-Choo et al. (2023).
///
/// # Arguments
///
/// * `x` — Vector of values to optimize over
/// * `min_sep` — Minimum separation between selected indices (≥ 1)
/// * `max_participations` — Maximum number of selections (None = unlimited by min_sep)
///
/// # Returns
///
/// The optimal inner product value.
///
/// # References
///
/// Algorithm 3 from <https://arxiv.org/abs/2306.08153>
pub fn max_participation_for_linear_fn(
    x: &[f64],
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let max_part = minsep_true_max_participations(n, min_sep, max_participations);
    if max_part == 0 {
        return 0.0;
    }

    // f[i] represents the best value achievable starting from index i
    // with remaining participations, including padding for boundary
    let padded_len = n + min_sep;
    let mut f = vec![0.0_f64; padded_len];

    for _ in 0..max_part {
        // f[i] = x[i] + f[i + min_sep] (selecting x[i])
        for i in 0..n {
            f[i] = x[i] + f[i + min_sep];
        }
        // Accumulate max right-to-left (reverse cumulative max)
        for i in (0..padded_len - 1).rev() {
            if f[i + 1] > f[i] {
                f[i] = f[i + 1];
            }
        }
    }

    f[0]
}

/// L2 sensitivity under single participation (each user contributes once).
///
/// Returns the maximum L2 column norm of the encoder matrix, provided
/// as a vector of pre-computed column norms.
///
/// # Arguments
///
/// * `column_norms` — L2 norms of each column of the encoder matrix C
///
/// # Errors
///
/// Returns `InvalidParameter` if `column_norms` is empty or contains non-finite values.
pub fn single_participation_sensitivity(column_norms: &[f64]) -> Result<f64> {
    if column_norms.is_empty() {
        return Err(PldError::InvalidParameter(
            "column_norms must be non-empty".into(),
        ));
    }
    let max_norm = column_norms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_norm.is_finite() {
        return Err(PldError::InvalidParameter(
            "column_norms contains non-finite values".into(),
        ));
    }
    Ok(max_norm)
}

/// Exact L2 sensitivity for banded Gram matrices under min-sep participation.
///
/// When the Gram matrix X = C^T C is min_sep-banded (entries with |i-j| >= min_sep
/// are zero), the sensitivity can be computed exactly from the diagonal of X
/// using the VecSens algorithm.
///
/// # Arguments
///
/// * `gram_diag` — Diagonal of the Gram matrix X = C^T C
/// * `min_sep` — Minimum separation between participations (≥ 1)
/// * `max_participations` — Optional upper bound on participations
///
/// # Returns
///
/// The exact L2 sensitivity.
///
/// # Errors
///
/// Returns `InvalidParameter` if `gram_diag` is empty or `min_sep` is 0.
///
/// # References
///
/// Section 4.2 of Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
pub fn banded_sensitivity(
    gram_diag: &[f64],
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    if gram_diag.is_empty() {
        return Err(PldError::InvalidParameter(
            "gram_diag must be non-empty".into(),
        ));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    let value = max_participation_for_linear_fn(gram_diag, min_sep, max_participations);
    if value < 0.0 {
        return Err(PldError::NumericalError(
            "Negative squared sensitivity encountered".into(),
        ));
    }
    Ok(value.sqrt())
}

/// Upper bound on L2 sensitivity for general (non-banded) Gram matrices.
///
/// Computes a two-stage upper bound: first optimizes per-row of the Gram
/// matrix, then optimizes over the resulting row maxima. This is valid
/// for any symmetric Gram matrix, not just banded ones.
///
/// # Arguments
///
/// * `gram_matrix` — Flattened row-major Gram matrix X = C^T C of size n×n
/// * `n` — Matrix dimension
/// * `min_sep` — Minimum separation between participations (≥ 1)
/// * `max_participations` — Optional upper bound on participations
///
/// # Returns
///
/// An upper bound on the L2 sensitivity.
///
/// # Errors
///
/// Returns `InvalidParameter` if dimensions are inconsistent or `min_sep` is 0.
///
/// # References
///
/// Algorithm 4 of Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
pub fn general_sensitivity_upper_bound(
    gram_matrix: &[f64],
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    if n == 0 {
        return Err(PldError::InvalidParameter(
            "matrix dimension n must be > 0".into(),
        ));
    }
    if gram_matrix.len() != n * n {
        return Err(PldError::InvalidParameter(format!(
            "gram_matrix length {} doesn't match n*n = {}",
            gram_matrix.len(),
            n * n
        )));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    // Stage 1: For each row, find max participation over |X[i, :]|
    let mut row_max = vec![0.0_f64; n];
    for (i, rm) in row_max.iter_mut().enumerate() {
        let row_start = i * n;
        let abs_row: Vec<f64> = gram_matrix[row_start..row_start + n]
            .iter()
            .map(|&v| v.abs())
            .collect();
        *rm = max_participation_for_linear_fn(&abs_row, min_sep, max_participations);
    }

    // Stage 2: Find max participation over the row maxima
    let result = max_participation_for_linear_fn(&row_max, min_sep, max_participations);
    if result < 0.0 {
        return Err(PldError::NumericalError(
            "Negative squared sensitivity encountered".into(),
        ));
    }
    Ok(result.sqrt())
}

/// L2 sensitivity under (k,b)-fixed-epoch participation.
///
/// Divides n rounds into epochs and computes the worst-case sensitivity
/// over all epoch-aligned participation patterns.
///
/// # Arguments
///
/// * `gram_matrix` — Flattened row-major Gram matrix X = C^T C of size n×n
/// * `n` — Matrix dimension (total number of rounds)
/// * `epochs` — Number of epochs (must divide n)
///
/// # Returns
///
/// The L2 sensitivity under fixed-epoch participation.
///
/// # Errors
///
/// Returns `InvalidParameter` if epochs doesn't divide n or parameters are invalid.
///
/// # References
///
/// Choquette-Choo et al. (2022) <https://arxiv.org/abs/2211.06530>
pub fn fixed_epoch_sensitivity(gram_matrix: &[f64], n: usize, epochs: usize) -> Result<f64> {
    if n == 0 {
        return Err(PldError::InvalidParameter(
            "matrix dimension n must be > 0".into(),
        ));
    }
    if epochs == 0 {
        return Err(PldError::InvalidParameter("epochs must be > 0".into()));
    }
    if n % epochs != 0 {
        return Err(PldError::InvalidParameter(format!(
            "epochs {} must divide n {}",
            epochs, n
        )));
    }
    if gram_matrix.len() != n * n {
        return Err(PldError::InvalidParameter(format!(
            "gram_matrix length {} doesn't match n*n = {}",
            gram_matrix.len(),
            n * n
        )));
    }

    let rounds_per_epoch = n / epochs;
    let submatrix_size = epochs; // n / rounds_per_epoch = epochs

    // Build index groups: indices[g][k] = g + k * rounds_per_epoch
    // Each group contains the rounds that a single user participates in
    let mut max_sq_sens = 0.0_f64;

    for col_idx in 0..rounds_per_epoch {
        // Collect indices for this group
        let indices: Vec<usize> = (0..submatrix_size)
            .map(|k| k * rounds_per_epoch + col_idx)
            .collect();

        // Sum absolute values of the submatrix X[indices, indices]
        let mut sq_sens = 0.0_f64;
        for &i in &indices {
            for &j in &indices {
                sq_sens += gram_matrix[i * n + j].abs();
            }
        }

        if sq_sens > max_sq_sens {
            max_sq_sens = sq_sens;
        }
    }

    Ok(max_sq_sens.sqrt())
}

/// Sensitivity squared for a Buffered Linear Toeplitz (BLT) strategy matrix.
///
/// Implements Lemma 5.3 of the BLT paper. The sensitivity is:
///   sensitivity^2(C) = 1 + sum_{i,j} omega_i * omega_j * geometric_sum(1, theta_i * theta_j, n-1)
///
/// # Arguments
///
/// * `buf_decay` — Decay factors θ for each buffer, each in (0, 1).
/// * `output_scale` — Scale factors ω for each buffer. Must be same length as buf_decay.
/// * `n` — Number of iterations (can be f64::INFINITY for asymptotic limit).
///
/// # Returns
///
/// The squared sensitivity as f64.
///
/// # Errors
///
/// Returns `InvalidParameter` if buf_decay and output_scale have different lengths.
///
/// # References
///
/// Lemma 5.3 of <https://arxiv.org/abs/2404.16706>
pub fn blt_sensitivity_squared(buf_decay: &[f64], output_scale: &[f64], n: f64) -> Result<f64> {
    if buf_decay.len() != output_scale.len() {
        return Err(PldError::InvalidParameter(
            "buf_decay and output_scale must have the same length".into(),
        ));
    }

    if buf_decay.is_empty() {
        return Ok(1.0);
    }

    for &d in buf_decay {
        if d > 1.0 {
            return Ok(f64::INFINITY);
        }
    }

    let num = if n.is_infinite() {
        f64::INFINITY
    } else {
        n - 1.0
    };

    let mut total = 0.0;
    for i in 0..buf_decay.len() {
        for j in 0..buf_decay.len() {
            let omega_pair = output_scale[i] * output_scale[j];
            let theta_pair = buf_decay[i] * buf_decay[j];
            total += crate::numerics::special::geometric_sum(omega_pair, theta_pair, num);
        }
    }

    Ok(1.0 + total)
}

/// Sensitivity squared for a Toeplitz strategy matrix under min-sep participation.
///
/// Implements the closed-form from Theorem 2 of the BSR paper. Requires
/// non-negative, non-increasing Toeplitz coefficients.
///
/// # Arguments
///
/// * `strategy_coef` — Toeplitz coefficients of C (non-negative, non-increasing).
/// * `n` — Matrix dimension (number of rounds).
/// * `min_sep` — Minimum separation between participations (>= 1).
/// * `max_participations` — Optional upper bound on participations.
///
/// # Returns
///
/// The squared sensitivity as f64.
///
/// # Errors
///
/// Returns `InvalidParameter` if coefficients are not non-negative non-increasing,
/// or if min_sep is 0, or if n is 0.
///
/// # References
///
/// Theorem 2 of <https://arxiv.org/abs/2405.13763>
pub fn toeplitz_minsep_sensitivity_squared(
    strategy_coef: &[f64],
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    if n == 0 {
        return Err(PldError::InvalidParameter("n must be > 0".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    // Validate non-negative
    for &c in strategy_coef {
        if c < 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "coef must be non-negative, found {}",
                c
            )));
        }
    }

    // Validate non-increasing
    for i in 1..strategy_coef.len() {
        if strategy_coef[i] > strategy_coef[i - 1] {
            return Err(PldError::InvalidParameter(format!(
                "coef must be non-increasing, found increase at index {}",
                i
            )));
        }
    }

    let k = minsep_true_max_participations(n, min_sep, max_participations);

    // Build full coefficients: zero-pad strategy_coef to length n
    let mut coef = vec![0.0; n];
    let copy_len = strategy_coef.len().min(n);
    coef[..copy_len].copy_from_slice(&strategy_coef[..copy_len]);

    // Pad to next multiple of min_sep
    let padding = (min_sep - n % min_sep) % min_sep;
    let padded_len = n + padding;
    let mut vector = vec![0.0; padded_len];
    vector[..n].copy_from_slice(&coef);

    // Cumulative sum across blocks of size min_sep
    // Equivalent to: reshape(-1, min_sep).cumsum(dim=0).flatten()
    let num_blocks = padded_len / min_sep;
    for block in 1..num_blocks {
        for offset in 0..min_sep {
            vector[block * min_sep + offset] += vector[(block - 1) * min_sep + offset];
        }
    }

    // Sliding-window subtraction: truncate at k participations
    let k_start = k * min_sep;
    if k_start < padded_len {
        // We need the original (pre-subtraction) values, so clone first
        let saved = vector.clone();
        for i in k_start..padded_len {
            vector[i] = saved[i] - saved[i - k_start];
        }
    }

    // Return dot product of first n elements
    let result: f64 = vector[..n].iter().map(|&v| v * v).sum();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- minsep_true_max_participations ----

    #[test]
    fn test_max_participations_basic() {
        assert_eq!(minsep_true_max_participations(10, 1, None), 10);
        assert_eq!(minsep_true_max_participations(10, 2, None), 5);
        assert_eq!(minsep_true_max_participations(10, 3, None), 4); // ceil(10/3)
        assert_eq!(minsep_true_max_participations(10, 10, None), 1);
        assert_eq!(minsep_true_max_participations(10, 11, None), 1); // ceil(10/11)
    }

    #[test]
    fn test_max_participations_with_cap() {
        assert_eq!(minsep_true_max_participations(10, 1, Some(3)), 3);
        assert_eq!(minsep_true_max_participations(10, 5, Some(10)), 2);
    }

    // ---- max_participation_for_linear_fn ----

    #[test]
    fn test_vecsens_empty() {
        assert_eq!(max_participation_for_linear_fn(&[], 1, None), 0.0);
    }

    #[test]
    fn test_vecsens_single_element() {
        assert_eq!(max_participation_for_linear_fn(&[5.0], 1, None), 5.0);
        assert_eq!(max_participation_for_linear_fn(&[5.0], 1, Some(1)), 5.0);
    }

    #[test]
    fn test_vecsens_all_ones_minsep_1() {
        // With min_sep=1, all elements can be selected
        let x = vec![1.0; 5];
        assert!((max_participation_for_linear_fn(&x, 1, None) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vecsens_alternating_minsep_2() {
        // x = [1, 2, 3, 4, 5], min_sep=2, max_part=ceil(5/2)=3
        // Best 3-pick with sep≥2: indices {0, 2, 4} → 1+3+5=9
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = max_participation_for_linear_fn(&x, 2, None);
        assert!((result - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_vecsens_with_max_participations() {
        let x = vec![10.0, 1.0, 1.0, 1.0, 1.0];
        // min_sep=1, max_part=1: just pick 10
        let result = max_participation_for_linear_fn(&x, 1, Some(1));
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_vecsens_uniform_minsep_3() {
        // n=9, min_sep=3, all values = 1.0
        // Can pick at most ceil(9/3) = 3 elements
        let x = vec![1.0; 9];
        let result = max_participation_for_linear_fn(&x, 3, None);
        assert!((result - 3.0).abs() < 1e-10);
    }

    // ---- single_participation_sensitivity ----

    #[test]
    fn test_single_sens_basic() {
        let norms = vec![1.0, 2.0, 3.0, 2.5];
        assert!((single_participation_sensitivity(&norms).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_sens_empty() {
        assert!(single_participation_sensitivity(&[]).is_err());
    }

    // ---- banded_sensitivity ----

    #[test]
    fn test_banded_sens_identity() {
        // Identity Gram: diag = [1, 1, 1], min_sep=1
        // All can be selected: sens = sqrt(3)
        let diag = vec![1.0, 1.0, 1.0];
        let sens = banded_sensitivity(&diag, 1, None).unwrap();
        assert!((sens - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_banded_sens_minsep_2() {
        // diag = [1, 1, 1, 1], min_sep=2
        // Can pick at most 2 elements: pick any 2 → sens = sqrt(2)
        let diag = vec![1.0, 1.0, 1.0, 1.0];
        let sens = banded_sensitivity(&diag, 2, None).unwrap();
        assert!((sens - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_banded_sens_minsep_invalid() {
        assert!(banded_sensitivity(&[1.0], 0, None).is_err());
    }

    // ---- general_sensitivity_upper_bound ----

    #[test]
    fn test_general_sens_identity() {
        // 3x3 identity Gram matrix, min_sep=1
        let gram = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let sens = general_sensitivity_upper_bound(&gram, 3, 1, None).unwrap();
        // Each row max = 1.0, then total max = 3.0, sqrt = sqrt(3)
        assert!((sens - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_general_sens_dimension_mismatch() {
        let gram = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        assert!(general_sensitivity_upper_bound(&gram, 3, 1, None).is_err());
    }

    // ---- fixed_epoch_sensitivity ----

    #[test]
    fn test_fixed_epoch_identity() {
        // 4x4 identity, 2 epochs → rounds_per_epoch=2
        // Group 0: indices [0, 2], Group 1: indices [1, 3]
        // Submatrix for group 0: X[{0,2},{0,2}] = [[1,0],[0,1]], sum_abs = 2
        // Submatrix for group 1: X[{1,3},{1,3}] = [[1,0],[0,1]], sum_abs = 2
        // max_sq_sens = 2, sens = sqrt(2)
        let gram = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let sens = fixed_epoch_sensitivity(&gram, 4, 2).unwrap();
        assert!((sens - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_epoch_epochs_not_dividing() {
        let gram = vec![1.0; 9]; // 3x3
        assert!(fixed_epoch_sensitivity(&gram, 3, 2).is_err());
    }

    #[test]
    fn test_fixed_epoch_single_epoch() {
        // 3x3 all-ones Gram, 1 epoch → rounds_per_epoch=3, submatrix_size=1
        // Each group has 1 element: [0], [1], [2]
        // Each submatrix is 1x1 with value 1.0
        // max_sq_sens = 1, sens = 1
        let gram = vec![1.0; 9]; // 3x3 all ones
        let sens = fixed_epoch_sensitivity(&gram, 3, 1).unwrap();
        assert!((sens - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_epoch_full_epochs() {
        // 4x4 identity, 4 epochs → rounds_per_epoch=1, submatrix_size=4
        // Only group 0: indices [0, 1, 2, 3], full 4x4 identity submatrix
        // sum_abs = 4 (diagonal only), sens = sqrt(4) = 2
        let gram = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let sens = fixed_epoch_sensitivity(&gram, 4, 4).unwrap();
        assert!((sens - 2.0).abs() < 1e-10);
    }

    // ---- blt_sensitivity_squared ----

    #[test]
    fn test_blt_empty_buffers() {
        // No buffers: sensitivity^2 = 1.0
        assert!((blt_sensitivity_squared(&[], &[], 100.0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blt_decay_exceeds_one() {
        // buf_decay > 1 → infinity
        assert!(blt_sensitivity_squared(&[1.5], &[1.0], 10.0)
            .unwrap()
            .is_infinite());
    }

    #[test]
    fn test_blt_length_mismatch() {
        assert!(blt_sensitivity_squared(&[0.5, 0.3], &[1.0], 10.0).is_err());
    }

    #[test]
    fn test_blt_single_buffer() {
        // Single buffer: 1 + omega^2 * geometric_sum(1, theta^2, n-1)
        let theta = 0.5;
        let omega = 0.3;
        let n = 50.0;
        let result = blt_sensitivity_squared(&[theta], &[omega], n).unwrap();
        // omega^2 * (1 + theta^2 + theta^4 + ... + theta^(2*(n-2)))
        let geo = omega * omega * (1.0 - (theta * theta).powf(n - 1.0)) / (1.0 - theta * theta);
        let expected = 1.0 + geo;
        assert!(
            (result - expected).abs() / expected < 1e-10,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn test_blt_infinite_n() {
        // Infinite n: 1 + sum(omega_i*omega_j / (1 - theta_i*theta_j))
        let buf_decay = vec![0.5, 0.3];
        let output_scale = vec![0.2, 0.1];
        let result = blt_sensitivity_squared(&buf_decay, &output_scale, f64::INFINITY).unwrap();
        let mut expected = 1.0;
        for i in 0..2 {
            for j in 0..2 {
                expected += output_scale[i] * output_scale[j] / (1.0 - buf_decay[i] * buf_decay[j]);
            }
        }
        assert!(
            (result - expected).abs() / expected < 1e-10,
            "result={result}, expected={expected}"
        );
    }

    // ---- toeplitz_minsep_sensitivity_squared ----

    #[test]
    fn test_toeplitz_minsep_identity() {
        // coef=[1.0] (identity Toeplitz), n=5, min_sep=1
        // Full coef: [1, 0, 0, 0, 0]
        // After cumsum with min_sep=1 blocks: [1, 1, 1, 1, 1]
        // k=5, k_start=5=padded_len, no subtraction
        // dot = 1+1+1+1+1 = 5.0
        let result = toeplitz_minsep_sensitivity_squared(&[1.0], 5, 1, None).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_minsep_single_participation() {
        // min_sep >= n means k=1, so only one participation
        // coef=[1.0, 0.5, 0.25], n=3, min_sep=3
        // k = ceil(3/3) = 1
        // Zero-pad to n=3: coef=[1.0, 0.5, 0.25]
        // padding = (3-3%3)%3 = 0, padded_len=3
        // cumsum: 1 block only, no cumsum
        // k_start = 1*3 = 3 = padded_len, no subtraction
        // dot = 1.0^2 + 0.5^2 + 0.25^2 = 1.3125
        let result = toeplitz_minsep_sensitivity_squared(&[1.0, 0.5, 0.25], 3, 3, None).unwrap();
        assert!((result - 1.3125).abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_minsep_invalid_n() {
        assert!(toeplitz_minsep_sensitivity_squared(&[1.0], 0, 1, None).is_err());
    }

    #[test]
    fn test_toeplitz_minsep_invalid_minsep() {
        assert!(toeplitz_minsep_sensitivity_squared(&[1.0], 5, 0, None).is_err());
    }

    #[test]
    fn test_toeplitz_minsep_negative_coef() {
        assert!(toeplitz_minsep_sensitivity_squared(&[1.0, -0.5], 5, 1, None).is_err());
    }

    #[test]
    fn test_toeplitz_minsep_increasing_coef() {
        assert!(toeplitz_minsep_sensitivity_squared(&[0.5, 1.0], 5, 1, None).is_err());
    }

    #[test]
    fn test_toeplitz_minsep_with_max_participations() {
        // coef=[1.0], n=10, min_sep=1, max_participations=Some(5)
        // Full coef: [1,0,0,0,0,0,0,0,0,0]
        // After cumsum with min_sep=1: [1,1,1,1,1,1,1,1,1,1]
        // k=5, k_start=5
        // subtraction: vector[5..10] = saved[5..10] - saved[0..5] = 0
        // vector = [1,1,1,1,1,0,0,0,0,0]
        // dot = 5.0
        let result = toeplitz_minsep_sensitivity_squared(&[1.0], 10, 1, Some(5)).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_toeplitz_minsep_empty_coef() {
        // Empty coef → all zeros → sensitivity^2 = 0
        let result = toeplitz_minsep_sensitivity_squared(&[], 5, 1, None).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }
}
