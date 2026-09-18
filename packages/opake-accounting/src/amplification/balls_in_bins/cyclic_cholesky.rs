//! Cholesky factorisation for *cyclically* banded Gram matrices.
//!
//! The BnB Gram over bins is cyclically banded, not linearly banded:
//! `G_{ij} ≈ E·λ^d + (E-1)·λ^{b-d}` for `d = |i-j|`, so the corner
//! `G[0][b-1]` can carry ~68% of the diagonal while the mid-row minimum is
//! under 1% of it (see `matrix_factorization::gram_matrix`).
//!
//! A *linearly* banded factorisation keeps `|i-j| ≤ p` and therefore discards
//! that corner no matter how the bandwidth is measured. Sampling
//! `u ~ N(m_i, σ²·LLᵀ)` with `LLᵀ ≠ G` is not conservative in either
//! direction — it is simply the wrong distribution — so the factorisation
//! itself has to respect the cyclic structure.
//!
//! # Fill-in structure
//!
//! For `G` cyclically `p`-banded, eliminating column 0 couples every row that
//! is nonzero in it — the band rows `1..=p` *and* the corner rows `b-p..b`.
//! Inductively the trailing `p` rows fill in densely, while rows `0..b-p`
//! keep the band. So `L` is stored as:
//!
//! - `band`: rows `0..b-p`, half-width `p` — `(b-p)·(p+1)` entries.
//! - `tail`: rows `b-p..b`, dense — `p·b` entries.
//!
//! Factorisation is `O(b·p²)`; a sample is `O(b·p)`, the same order as the
//! linear band it replaces.

use crate::error::{PldError, Result};

/// Relative tolerance for declaring a pivot non-positive.
const PSD_TOL: f64 = 1e-12;

/// Cholesky factor of a cyclically banded symmetric positive-definite matrix.
#[derive(Debug)]
pub(crate) struct CyclicBandedCholesky {
    /// Banded rows `0..n_band`, row-major, stride `p + 1`.
    band: Vec<f64>,
    /// Dense trailing rows `n_band..b`, row-major, stride `b`.
    tail: Vec<f64>,
    b: usize,
    p: usize,
    /// Number of banded rows: `b - p` (0 when the matrix is effectively dense).
    n_band: usize,
}

impl CyclicBandedCholesky {
    /// Lowest column index that can be nonzero in row `i`.
    #[inline]
    fn lo(&self, i: usize) -> usize {
        if i < self.n_band {
            i.saturating_sub(self.p)
        } else {
            0
        }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        if j > i {
            return 0.0;
        }
        if i < self.n_band {
            let lo = i.saturating_sub(self.p);
            if j < lo {
                return 0.0;
            }
            self.band[i * (self.p + 1) + (j - lo)]
        } else {
            self.tail[(i - self.n_band) * self.b + j]
        }
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.n_band {
            let lo = i.saturating_sub(self.p);
            self.band[i * (self.p + 1) + (j - lo)] = v;
        } else {
            self.tail[(i - self.n_band) * self.b + j] = v;
        }
    }

    /// Detect the cyclic bandwidth: the largest `min(|i-j|, b-|i-j|)` at which
    /// any entry exceeds `threshold · max_diag`.
    ///
    /// Scans every entry — unlike the linear scan this replaces, which sampled
    /// ~20 entries per diagonal (`step_by((b/20).max(1))`) and could
    /// under-detect on its own, independently of the cyclic issue.
    fn detect_bandwidth(gram: &[f64], b: usize, threshold: f64) -> usize {
        let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
        let abs_thresh = threshold * max_diag;

        let mut p = 0usize;
        for i in 0..b {
            for j in (i + 1)..b {
                if gram[i * b + j].abs() > abs_thresh {
                    let d = j - i;
                    p = p.max(d.min(b - d));
                }
            }
        }
        p.min(b.saturating_sub(1))
    }

    /// Factorise a cyclically banded PSD Gram matrix.
    ///
    /// # Errors
    ///
    /// Returns `NumericalError` if a pivot is negative beyond `PSD_TOL`
    /// relative to the largest diagonal — i.e. the matrix handed in is not
    /// positive semi-definite. This is deliberately *not* regularised: the
    /// previous code set such a pivot to `sqrt(1e-30) = 1e-15` and then
    /// divided by it, producing `|L|` up to 1e219 and `inf`, which the sampler
    /// then discarded silently.
    pub(crate) fn compute(gram: &[f64], b: usize, threshold: f64) -> Result<Self> {
        if b == 0 {
            return Err(PldError::InvalidParameter("b must be >= 1".into()));
        }
        if gram.len() != b * b {
            return Err(PldError::InvalidParameter(format!(
                "gram length {} does not match b²={}",
                gram.len(),
                b * b
            )));
        }

        let p = Self::detect_bandwidth(gram, b, threshold);
        let n_band = b.saturating_sub(p);
        let tail_rows = b - n_band;

        let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
        let neg_tol = PSD_TOL * max_diag.max(1.0);

        let mut chol = CyclicBandedCholesky {
            band: vec![0.0; n_band * (p + 1)],
            tail: vec![0.0; tail_rows * b],
            b,
            p,
            n_band,
        };

        for i in 0..b {
            let j_lo = chol.lo(i);
            for j in j_lo..=i {
                // L[i][k]·L[j][k] is nonzero only where both rows are.
                let k_lo = chol.lo(i).max(chol.lo(j));
                let mut sum = 0.0;
                for k in k_lo..j {
                    sum += chol.get(i, k) * chol.get(j, k);
                }

                if i == j {
                    let diag = gram[i * b + i] - sum;
                    if diag < -neg_tol {
                        return Err(PldError::NumericalError(format!(
                            "Gram matrix is not positive semi-definite: pivot {} \
                             at row {} (b={}, cyclic bandwidth={}, max_diag={}). \
                             Refusing to regularise — the sampled covariance \
                             would not be the Gram matrix.",
                            diag, i, b, p, max_diag
                        )));
                    }
                    // A pivot in [-tol, 0] is a rank deficiency, not an error:
                    // zero the row's diagonal and let the `l_jj > 0` guard
                    // below leave its off-diagonals at zero.
                    chol.set(i, j, if diag > 0.0 { diag.sqrt() } else { 0.0 });
                } else {
                    let l_jj = chol.get(j, j);
                    if l_jj > 0.0 {
                        chol.set(i, j, (gram[i * b + j] - sum) / l_jj);
                    }
                }
            }
        }

        Ok(chol)
    }

    /// Largest absolute deviation between `L·Lᵀ` and `gram` over the retained
    /// sparsity pattern. Used by tests and by the caller's sanity check.
    pub(crate) fn max_residual(&self, gram: &[f64]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..self.b {
            for j in self.lo(i)..=i {
                let k_lo = self.lo(i).max(self.lo(j));
                let mut acc = 0.0;
                for k in k_lo..=j {
                    acc += self.get(i, k) * self.get(j, k);
                }
                worst = worst.max((acc - gram[i * self.b + j]).abs());
            }
        }
        worst
    }

    /// `out = mean + σ · L · z`.
    pub(crate) fn sample_gaussian(&self, mean: &[f64], sigma: f64, z: &[f64], out: &mut [f64]) {
        for i in 0..self.b {
            let lo = self.lo(i);
            let mut lz = 0.0;
            if i < self.n_band {
                let base = i * (self.p + 1);
                for (idx, zj) in z[lo..=i].iter().enumerate() {
                    lz += self.band[base + idx] * zj;
                }
            } else {
                let base = (i - self.n_band) * self.b;
                for (j, zj) in z[..=i].iter().enumerate() {
                    lz += self.tail[base + j] * zj;
                }
            }
            out[i] = mean[i] + sigma * lz;
        }
    }

    #[cfg(test)]
    pub(crate) fn bandwidth(&self) -> usize {
        self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Dense reference Cholesky (lower triangular), for validation only.
    fn dense_cholesky(gram: &[f64], b: usize) -> Vec<f64> {
        let mut l = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * b + k] * l[j * b + k];
                }
                if i == j {
                    l[i * b + j] = (gram[i * b + i] - sum).max(0.0).sqrt();
                } else if l[j * b + j] > 0.0 {
                    l[i * b + j] = (gram[i * b + j] - sum) / l[j * b + j];
                }
            }
        }
        l
    }

    /// The true cyclic λ-CGD Gram: G_ij = Σ_{p,q} λ^{|b(p-q)+(i-j)|}.
    fn cyclic_gram(b: usize, e: usize, lambda: f64) -> Vec<f64> {
        let mut g = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..b {
                let mut s = 0.0;
                for p in 0..e {
                    for q in 0..e {
                        let gap = (b as i64 * (p as i64 - q as i64) + (i as i64 - j as i64))
                            .unsigned_abs() as i32;
                        s += lambda.powi(gap);
                    }
                }
                g[i * b + j] = s;
            }
        }
        g
    }

    fn reconstruct(chol: &CyclicBandedCholesky, b: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..b {
                let mut acc = 0.0;
                for k in 0..=i.min(j) {
                    acc += chol.get(i, k) * chol.get(j, k);
                }
                out[i * b + j] = acc;
            }
        }
        out
    }

    #[test]
    fn test_identity() {
        let b = 4;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let chol = CyclicBandedCholesky::compute(&gram, b, 1e-6).unwrap();
        let mut out = vec![0.0; b];
        chol.sample_gaussian(&vec![0.0; b], 1.0, &[1.0, 2.0, 3.0, 4.0], &mut out);
        assert_relative_eq!(out[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[3], 4.0, epsilon = 1e-12);
    }

    /// The corner must survive: this is the whole point of the module.
    #[test]
    fn test_reconstructs_cyclic_corner() {
        for &(b, e, lam) in &[(100usize, 4usize, 0.9), (64, 4, 0.9), (128, 2, 0.95)] {
            let gram = cyclic_gram(b, e, lam);
            // Structural check on the fixture: the corner is ~(E-1)λ/E of the
            // diagonal — never negligible, and far above the mid-row minimum.
            // (Exact values against the real builder are pinned in
            // `matrix_factorization::gram_matrix`.)
            let (corner, diag, mid) = (gram[b - 1], gram[0], gram[b / 2]);
            assert!(
                corner > 0.4 * diag,
                "b={} E={} λ={}: corner {} vs diagonal {}",
                b,
                e,
                lam,
                corner,
                diag
            );
            assert!(corner > 5.0 * mid, "corner {} vs mid-row {}", corner, mid);

            let chol = CyclicBandedCholesky::compute(&gram, b, 1e-6).unwrap();
            let llt = reconstruct(&chol, b);
            assert_relative_eq!(llt[b - 1], gram[b - 1], epsilon = 1e-8);
            assert_relative_eq!(llt[0], gram[0], epsilon = 1e-8);
        }
    }

    /// Agree with a dense Cholesky across the regimes the plan names.
    #[test]
    fn test_matches_dense_cholesky() {
        for &lam in &[0.5, 0.7, 0.9, 0.95] {
            for &e in &[2usize, 4, 8] {
                for &b in &[32usize, 64, 128] {
                    let gram = cyclic_gram(b, e, lam);
                    let chol = CyclicBandedCholesky::compute(&gram, b, 1e-14).unwrap();
                    let llt = reconstruct(&chol, b);
                    let dense = dense_cholesky(&gram, b);
                    let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
                    let mut worst: f64 = 0.0;
                    for i in 0..b {
                        for j in 0..b {
                            worst = worst.max((llt[i * b + j] - gram[i * b + j]).abs());
                        }
                    }
                    assert!(
                        worst < 1e-10 * max_diag,
                        "λ={} E={} b={}: max|G - LLᵀ| = {} (tol {})",
                        lam,
                        e,
                        b,
                        worst,
                        1e-10 * max_diag
                    );
                    // Same factor as dense, up to sign-free lower-triangular uniqueness.
                    for i in 0..b {
                        assert_relative_eq!(chol.get(i, i), dense[i * b + i], epsilon = 1e-8);
                    }
                }
            }
        }
    }

    /// A non-PSD input must be rejected, not regularised into 1e219 / inf.
    #[test]
    fn test_rejects_non_psd() {
        let b = 3;
        // Symmetric but indefinite.
        let gram = vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let err = CyclicBandedCholesky::compute(&gram, b, 1e-6);
        assert!(err.is_err(), "indefinite matrix must be rejected");
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("positive semi-definite"), "got: {}", msg);
    }

    /// A rank-deficient but PSD matrix is fine — zero pivot, no error.
    #[test]
    fn test_accepts_rank_deficient_psd() {
        let b = 2;
        // [[1,1],[1,1]] is PSD with rank 1.
        let gram = vec![1.0, 1.0, 1.0, 1.0];
        let chol = CyclicBandedCholesky::compute(&gram, b, 1e-6).unwrap();
        assert!(chol.max_residual(&gram) < 1e-12);
        // No blow-up.
        for i in 0..b {
            for j in 0..=i {
                assert!(chol.get(i, j).abs() < 10.0);
            }
        }
    }

    #[test]
    fn test_detects_cyclic_not_linear_bandwidth() {
        // λ=0.9, b=100, E=4: the corner is ~68% of the diagonal, so the cyclic
        // bandwidth must come out large enough to retain it.
        let gram = cyclic_gram(100, 4, 0.9);
        let chol = CyclicBandedCholesky::compute(&gram, 100, 1e-6).unwrap();
        let llt = reconstruct(&chol, 100);
        assert_relative_eq!(llt[99], gram[99], epsilon = 1e-8);
        assert!(chol.bandwidth() >= 1);
    }

    #[test]
    fn test_rejects_bad_shape() {
        assert!(CyclicBandedCholesky::compute(&[1.0], 2, 1e-6).is_err());
        assert!(CyclicBandedCholesky::compute(&[], 0, 1e-6).is_err());
    }
}
