//! Analytic Gaussian mechanism functions
//!
//! Closed-form and numerical routines for the Gaussian mechanism's
//! privacy loss properties, used by both the functional and legacy APIs.

use statrs::distribution::{ContinuousCDF, Normal};

/// Analytic delta for the Gaussian mechanism at a given epsilon.
///
/// Computes the ε-hockey stick divergence:
/// ```text
/// δ(ε) = Φ(0.5Δ̃ − ε/Δ̃) − eᵉ · Φ(0.5Δ̃ − ε/Δ̃ − Δ̃)
/// ```
/// where Δ̃ = sensitivity / sigma is the standardized sensitivity.
///
/// Returns 0.0 when the result would be negative (numerical noise).
pub(crate) fn gaussian_delta_at(delta_tilde: f64, epsilon: f64) -> f64 {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let x_upper = 0.5 * delta_tilde - epsilon / delta_tilde;
    (standard_normal.cdf(x_upper) - epsilon.exp() * standard_normal.cdf(x_upper - delta_tilde))
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_delta_at_zero_epsilon() {
        // delta(0) = Phi(0.5*dt) - Phi(0.5*dt - dt) = Phi(0.5*dt) - Phi(-0.5*dt)
        // For dt=1: Phi(0.5) - Phi(-0.5) ≈ 0.383
        let d = gaussian_delta_at(1.0, 0.0);
        assert!((d - 0.3829).abs() < 0.001);
    }

    #[test]
    fn test_gaussian_delta_at_large_epsilon() {
        // For large epsilon, delta should be near zero
        let d = gaussian_delta_at(1.0, 10.0);
        assert!(d < 1e-10);
    }
}
