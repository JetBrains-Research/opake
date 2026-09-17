//! Matrix factorization (MF) privacy accounting.
//!
//! This module provides privacy accounting for matrix factorization DP
//! mechanisms: BandMF, BLT, DP-λCGD, and BISR. Unlike standard DP-SGD
//! which uses i.i.d. Gaussian noise at each step, MF mechanisms inject
//! *correlated* noise across all training steps via a matrix factorization.
//!
//! # Privacy Analysis
//!
//! The privacy of the entire MF training run reduces to a single Gaussian
//! mechanism with effective noise multiplier σ/S, where:
//! - σ is the raw noise standard deviation
//! - S is the L2 sensitivity of the encoder matrix C under the given
//!   participation pattern
//!
//! Sensitivity depends only on C, not on the optimizer workload (momentum,
//! LR schedule). The workload affects utility optimization, never privacy.
//!
//! | Strategy | Participation | Sensitivity function |
//! |----------|--------------|---------------------|
//! | BandMF   | Single       | [`single_participation_sensitivity`] |
//! | BandMF   | Min-sep      | [`banded_sensitivity`] |
//! | BLT      | Min-sep      | [`banded_sensitivity`] or [`general_sensitivity_upper_bound`] |
//! | λCGD     | Min-sep      | [`lambda_cgd_sensitivity_squared`] |
//! | λCGD     | Normalized   | [`lambda_cgd_normalized_sensitivity_squared`] |
//! | BISR     | Min-sep      | [`bisr_sensitivity_squared`] |
//! | BISR     | Normalized   | [`bisr_normalized_sensitivity_squared`] |
//! | Any      | General      | [`general_sensitivity_upper_bound`] |
//!
//! For BnB amplification, Gram matrices are available via:
//! - [`lambda_cgd_gram_matrix`] — closed-form for λCGD
//! - [`bisr_gram_matrix`] — numerical for BISR (general bandwidth)
//! - [`toeplitz_gram_matrix`] — for BandMF/BLT with known strategy coefs
//!
//! # References
//!
//! - BandMF: Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
//! - BLT: Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>
//! - DP-λCGD: Kalinin et al. (2026) <https://arxiv.org/abs/2601.22334>
//! - BISR: Kalinin et al. (2026) <https://arxiv.org/abs/2505.12128>
//! - MC BnB: Choquette-Choo et al. (2024) <https://arxiv.org/abs/2410.06266>

pub mod bisr;
pub mod gram_matrix;
pub mod lambda_cgd;
mod mf_gaussian;
pub mod sensitivity;

pub use bisr::{
    bisr_gram_matrix, bisr_gram_matrix_lr, bisr_normalized_sensitivity_squared,
    bisr_sensitivity_squared, toeplitz_gram_matrix,
};
pub use gram_matrix::{lambda_cgd_gram_matrix, lambda_cgd_gram_matrix_lr};
pub use lambda_cgd::{
    lambda_cgd_max_column_norm, lambda_cgd_normalized_sensitivity_squared,
    lambda_cgd_sensitivity_squared,
};
pub use mf_gaussian::mf_gaussian_pld;
pub use sensitivity::{
    banded_sensitivity, blt_sensitivity_squared, fixed_epoch_sensitivity,
    general_sensitivity_upper_bound, max_participation_for_linear_fn,
    minsep_true_max_participations, single_participation_sensitivity,
    toeplitz_minsep_sensitivity_squared,
};
