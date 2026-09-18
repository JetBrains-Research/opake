//! Adaptive clipping helpers.

use pyo3::prelude::*;

/// Compute the combined sensitivity for adaptive clipping.
///
/// When using both gradient clipping (noise multiplier ``z``) and ``K``
/// independent noisy quantile estimators (each with std ``σ_b``), the
/// combined L2 sensitivity is:
///
/// ``z̃ = sqrt(1/z² + K/(4·σ_b²))``
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier z.
///     quantile_noise_std (float): Std of noise added to the quantile estimator σ_b.
///     num_groups (int): Number of independent quantile queries (default 1).
///
/// Returns:
///     float: The combined L2 sensitivity z̃.
#[pyfunction]
#[pyo3(name = "adaclip_sensitivity", signature = (noise_multiplier, quantile_noise_std, num_groups=1))]
pub fn py_adaclip_sensitivity(
    noise_multiplier: f64,
    quantile_noise_std: f64,
    num_groups: u32,
) -> f64 {
    crate::transformations::adaclip::adaclip_sensitivity(
        noise_multiplier,
        quantile_noise_std,
        num_groups,
    )
}
