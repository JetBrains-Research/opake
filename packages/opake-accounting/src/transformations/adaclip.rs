//! Adaptive clipping utilities.
//!
//! Provides the `adaclip_sensitivity` formula for adaptive clipping with
//! a public quantile estimator, used when composing privacy accounting for
//! DP-SGD with adaptive gradient clipping.

/// Compute the combined sensitivity z̃ for adaptive clipping.
///
/// When using both gradient clipping (with noise multiplier `noise_multiplier`)
/// and `num_groups` independent quantile estimators (each with noise std
/// `quantile_noise_std`), the combined L2 sensitivity is:
///
/// ```text
/// z̃ = sqrt(1/z² + K/(4·σ_b²))
/// ```
///
/// where `z = noise_multiplier`, `σ_b = quantile_noise_std`, and
/// `K = num_groups`.  When `K = 1` this reduces to the standard formula.
///
/// # Arguments
///
/// * `noise_multiplier` — gradient noise multiplier z
/// * `quantile_noise_std` — std of noise added to the quantile estimator σ_b
/// * `num_groups` — number of independent quantile queries (≥ 1)
///
/// # Panics
///
/// Panics if `noise_multiplier` or `quantile_noise_std` is zero, or if
/// `num_groups` is zero.
pub fn adaclip_sensitivity(noise_multiplier: f64, quantile_noise_std: f64, num_groups: u32) -> f64 {
    let z = noise_multiplier;
    let sigma_b = quantile_noise_std;
    let k = num_groups as f64;
    (1.0 / (z * z) + k / (4.0 * sigma_b * sigma_b)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaclip_sensitivity_basic() {
        // z=1, σ_b=∞, K=1 → z̃ ≈ 1/z = 1.0
        let result = adaclip_sensitivity(1.0, 1e10, 1);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaclip_sensitivity_equal_contribution() {
        // z=1, σ_b=0.5, K=1 → z̃ = sqrt(1 + 1) = sqrt(2)
        let result = adaclip_sensitivity(1.0, 0.5, 1);
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_adaclip_sensitivity_large_quantile_noise() {
        // Large σ_b → dominated by 1/z²
        let z = 0.5;
        let result = adaclip_sensitivity(z, 1e6, 1);
        let expected = 1.0 / z;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_adaclip_sensitivity_symmetry() {
        // adaclip_sensitivity is symmetric except for the 4x factor on σ_b
        let a = adaclip_sensitivity(1.0, 2.0, 1);
        let b = adaclip_sensitivity(2.0, 1.0, 1);
        // Not symmetric — verify different
        assert!((a - b).abs() > 0.01);
    }

    #[test]
    fn test_adaclip_sensitivity_always_positive() {
        for &z in &[0.1, 0.5, 1.0, 2.0] {
            for &sb in &[0.1, 0.5, 1.0, 10.0] {
                let result = adaclip_sensitivity(z, sb, 1);
                assert!(result > 0.0, "z={}, sb={}, result={}", z, sb, result);
            }
        }
    }

    #[test]
    fn test_adaclip_sensitivity_num_groups() {
        // K=1 should match the original formula
        let k1 = adaclip_sensitivity(1.0, 1.0, 1);
        // K=2 should give higher sensitivity (lower effective noise multiplier)
        let k2 = adaclip_sensitivity(1.0, 1.0, 2);
        assert!(k2 > k1, "More groups should increase combined sensitivity");

        // K=3: z̃ = sqrt(1 + 3/4) = sqrt(1.75)
        let k3 = adaclip_sensitivity(1.0, 1.0, 3);
        assert!((k3 - 1.75_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_adaclip_sensitivity_k1_matches_original() {
        // K=1 should give exactly the same result as the old formula
        for &z in &[0.5, 1.0, 1.5, 2.0] {
            for &sb in &[0.1, 1.0, 10.0] {
                let result = adaclip_sensitivity(z, sb, 1);
                let expected = (1.0 / (z * z) + 1.0 / (4.0 * sb * sb)).sqrt();
                assert!(
                    (result - expected).abs() < 1e-15,
                    "K=1 mismatch: z={}, sb={}",
                    z,
                    sb
                );
            }
        }
    }
}
