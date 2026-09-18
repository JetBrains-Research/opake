//! Special mathematical functions for privacy accounting
//!
//! Provides numerically stable implementations of special functions
//! needed for subsampled differential privacy mechanisms, particularly
//! for Poisson subsampling with REPLACE adjacency.
//!
//! # Functions
//!
//! - [`log_sinh`]: Numerically stable log(sinh(x))
//! - [`arcsinh`]: Standard inverse hyperbolic sine
//! - [`arcsinh_exp`]: Numerically stable arcsinh(b * e^a)
//!
//! # Numerical Stability
//!
//! These functions are critical for computing privacy loss in subsampled
//! mechanisms. Direct computation can lead to numerical overflow or
//! underflow, so we use the stable formulations cited below.
//!
//! # References
//!
//! - Lebeda, Regehr, Kamath, and Steinke, "Avoiding Pitfalls for Privacy
//!   Accounting of Subsampled Mechanisms under Composition"
//!   <https://arxiv.org/abs/2405.20769>

use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::LN_2;

/// Numerically stable computation of log(sinh(x))
///
/// For large x, direct computation of sinh(x) = (e^x - e^(-x))/2 can overflow.
/// We use the stable formulation:
///
/// ```text
/// log(sinh(x)) = x + log(1 - e^(-2x)) - log(2)
///              = x + log1p(-e^(-2x)) - log(2)
/// ```
///
/// This avoids computing e^x directly for large x.
///
/// # Arguments
///
/// * `x` - Input value (must be > 0)
///
/// # Returns
///
/// log(sinh(x)) computed in a numerically stable way
///
/// # Panics
///
/// Panics if x ≤ 0 (sinh is not positive for non-positive values)
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::special::log_sinh;
///
/// let result = log_sinh(1.0);
/// assert!((result - 1.0_f64.sinh().ln()).abs() < 1e-10);
///
/// // Works for large values without overflow
/// let large = log_sinh(100.0);
/// assert!(large.is_finite());
/// ```
pub fn log_sinh(x: f64) -> f64 {
    assert!(x > 0.0, "log_sinh requires x > 0, got {}", x);

    // For very small x, use Taylor expansion: sinh(x) ≈ x
    if x < 1e-8 {
        return x.ln();
    }

    // For moderate to large x, use stable formulation:
    // log(sinh(x)) = log((e^x - e^(-x))/2)
    //              = x + log(1 - e^(-2x)) - log(2)
    // Note: ln_1p(y) computes log(1 + y), so we pass -e^(-2x)
    x + (-(-2.0 * x).exp()).ln_1p() - LN_2
}

/// Standard inverse hyperbolic sine (arcsinh)
///
/// Computes arcsinh(x) = log(x + sqrt(x^2 + 1)) in a numerically stable way.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// arcsinh(x)
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::special::arcsinh;
///
/// assert!((arcsinh(0.0)).abs() < 1e-10);
/// assert!((arcsinh(1.0) - 0.881373587).abs() < 1e-6);
/// ```
pub fn arcsinh(x: f64) -> f64 {
    // Standard formula: arcsinh(x) = log(x + sqrt(x^2 + 1))
    // For large |x|, use asymptotic form to avoid overflow: arcsinh(x) ≈ sign(x) * (ln(2|x|))
    let abs_x = x.abs();

    let result = if abs_x > 1e150 {
        // For very large |x|, x^2 would overflow, use asymptotic form
        abs_x.ln() + LN_2
    } else {
        (abs_x + (abs_x * abs_x + 1.0).sqrt()).ln()
    };

    if x < 0.0 {
        -result
    } else {
        result
    }
}

/// Numerically stable computation of arcsinh(b * e^a)
///
/// Directly computing b * e^a can overflow for large a. This function
/// computes arcsinh(b * e^a) without materializing the intermediate value.
///
/// Uses the formulation:
/// ```text
/// arcsinh(b * e^a) = sign(b) * log(|b| * e^a + sqrt((b * e^a)^2 + 1))
///                  = sign(b) * log(|b| * e^a + sqrt(b^2 * e^(2a) + 1))
/// ```
///
/// # Arguments
///
/// * `a` - Exponent term
/// * `b` - Coefficient term
///
/// # Returns
///
/// arcsinh(b * e^a) computed in a numerically stable way
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::special::arcsinh_exp;
///
/// // For small values, should match direct computation
/// let result = arcsinh_exp(0.0, 1.0);
/// assert!((result - 0.881373587).abs() < 1e-6);
///
/// // Works for large exponents without overflow
/// let large = arcsinh_exp(100.0, 1.0);
/// assert!(large.is_finite());
/// ```
pub fn arcsinh_exp(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return 0.0;
    }

    let sign = b.signum();
    let abs_b = b.abs();

    // Compute log(|b| * e^a + sqrt(b^2 * e^(2a) + 1))
    //        = log(e^a * (|b| + sqrt(b^2 + e^(-2a))))
    //        = a + log(|b| + sqrt(b^2 + e^(-2a)))

    // For large a, e^(-2a) is negligible, so:
    //   log(|b| + sqrt(b^2 + e^(-2a))) ≈ log(|b| + |b|) = log(2|b|)
    if a > 50.0 {
        return sign * (a + (2.0 * abs_b).ln());
    }

    // For moderate a, compute carefully
    let exp_minus_2a = (-2.0 * a).exp();
    let sqrt_term = (abs_b * abs_b + exp_minus_2a).sqrt();

    sign * (a + (abs_b + sqrt_term).ln())
}

/// Numerically stable computation of log(Φ(z)) for standard normal CDF
///
/// Computes the natural logarithm of the cumulative distribution function (CDF)
/// of the standard normal distribution N(0,1) evaluated at z.
///
/// # Arguments
///
/// * `z` - Point at which to evaluate log(Φ(z))
///
/// # Returns
///
/// Natural logarithm of Φ(z) where Φ is the standard normal CDF
///
/// # Numerical Stability
///
/// For very small or large z values, computing Φ(z).ln() directly can lose precision.
/// This function provides a more stable computation by:
/// - For z far in the left tail (z << 0), using asymptotic expansions
/// - For z near 0 or in the right tail, using log1p when possible
///
/// Currently uses a simple implementation via statrs. More sophisticated
/// implementations (e.g., using erfcx) could be added if needed.
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::special::gaussian_log_cdf;
///
/// // At z=0, Φ(0) = 0.5, so log(Φ(0)) = -ln(2)
/// assert!((gaussian_log_cdf(0.0) - (-0.693)).abs() < 0.001);
///
/// // For large positive z, log(CDF) approaches 0
/// assert!(gaussian_log_cdf(10.0) > -1e-10);
///
/// // For large negative z, log(CDF) is very negative
/// assert!(gaussian_log_cdf(-10.0) < -50.0);
/// ```
#[inline]
pub fn gaussian_log_cdf(z: f64) -> f64 {
    // For numerical stability, we could use more sophisticated methods like:
    // - For z << 0: use asymptotic expansion of erfc
    // - For z > 0: use log1p(-Φ(-z))
    //
    // For now, use statrs which handles most cases reasonably well.
    // The standard normal has mean=0, std=1.
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    standard_normal.cdf(z).ln()
}

/// Numerically stable geometric sum: a * (1 + r + r^2 + ... + r^(num-1)).
///
/// Uses a quadratic Taylor approximation near r=1 for numerical stability.
/// Threshold calibrated to minimize gradient error, following JAX-Privacy.
///
/// # Arguments
///
/// * `a` — Scale factor
/// * `r` — Common ratio, requires |r| < 1
/// * `num` — Number of terms (must be > 0, may be f64::INFINITY)
///
/// # References
///
/// BLT paper, Lemma 5.3: <https://arxiv.org/abs/2404.16706>
pub fn geometric_sum(a: f64, r: f64, num: f64) -> f64 {
    if num.is_infinite() {
        return a / (1.0 - r);
    }

    // Adaptive threshold: calibrated to minimise gradient error.
    // Constants from regression on numerical experiments (see JAX-Privacy).
    const SLOPE: f64 = 0.53018965;
    const INTERCEPT: f64 = 3.33503185;
    let pow_threshold = INTERCEPT + SLOPE * num.ln();
    let threshold = 1.0 - 10.0_f64.powf(-pow_threshold);

    if r < threshold {
        // Direct computation (safe when r is not near 1)
        a * (1.0 - r.powf(num)) / (1.0 - r)
    } else {
        // Quadratic Taylor polynomial at r = 1 (from sympy)
        let x0 = num - 1.0;
        let x1 = r - 1.0;
        (1.0 / 6.0) * a * num * (x0 * x1 * x1 * (num - 2.0) + 3.0 * x0 * x1 + 6.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sinh_small() {
        let x: f64 = 0.1;
        let expected = x.sinh().ln();
        let result = log_sinh(x);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sinh_moderate() {
        let x: f64 = 1.0;
        let expected = x.sinh().ln();
        let result = log_sinh(x);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sinh_large() {
        let x = 100.0;
        // For large x, sinh(x) ≈ e^x / 2, so log(sinh(x)) ≈ x - log(2)
        let expected = x - LN_2;
        let result = log_sinh(x);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_arcsinh_zero() {
        assert!((arcsinh(0.0)).abs() < 1e-15);
    }

    #[test]
    fn test_arcsinh_positive() {
        // arcsinh(1) = log(1 + sqrt(2)) ≈ 0.881373587
        assert!((arcsinh(1.0) - 0.881373587).abs() < 1e-6);
    }

    #[test]
    fn test_arcsinh_negative() {
        // arcsinh is odd function: arcsinh(-x) = -arcsinh(x)
        let x = 1.0;
        assert!((arcsinh(-x) + arcsinh(x)).abs() < 1e-10);
    }

    #[test]
    fn test_arcsinh_exp_zero_b() {
        assert_eq!(arcsinh_exp(1.0, 0.0), 0.0);
        assert_eq!(arcsinh_exp(100.0, 0.0), 0.0);
    }

    #[test]
    fn test_arcsinh_exp_zero_a() {
        // arcsinh_exp(0, b) = arcsinh(b * e^0) = arcsinh(b)
        let b = 1.0;
        let result = arcsinh_exp(0.0, b);
        let expected = arcsinh(b);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_arcsinh_exp_moderate() {
        // For moderate a, compare with direct computation
        let a = 1.0;
        let b = 2.0;
        let result = arcsinh_exp(a, b);
        let expected = arcsinh(b * a.exp());
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_arcsinh_exp_large_positive() {
        // For large a and b=1: arcsinh(e^a) ≈ a + log(2)
        let a = 100.0;
        let b = 1.0;
        let result = arcsinh_exp(a, b);
        let expected = a + LN_2;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_arcsinh_exp_large_negative_b() {
        // Should handle negative b correctly
        let a = 50.0;
        let b = -1.0;
        let result = arcsinh_exp(a, b);
        // Should be negative of positive case
        let positive_result = arcsinh_exp(a, 1.0);
        assert!((result + positive_result).abs() < 1e-10);
    }

    #[test]
    fn test_arcsinh_exp_numerical_stability() {
        // Test that large a doesn't cause overflow
        let a = 200.0;
        let b = 1.0;
        let result = arcsinh_exp(a, b);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    // ---- geometric_sum ----

    #[test]
    fn test_geometric_sum_basic() {
        // 1 + 0.5 + 0.25 + ... + 0.5^9 = (1 - 0.5^10)/(1 - 0.5)
        let result = geometric_sum(1.0, 0.5, 10.0);
        let expected = (1.0 - 0.5_f64.powi(10)) / (1.0 - 0.5);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_sum_single_term() {
        // a * r^0 = a
        assert!((geometric_sum(2.0, 0.5, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_sum_r_zero() {
        // a + 0 + 0 + ... = a
        assert!((geometric_sum(2.0, 0.0, 5.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_sum_near_one() {
        // r very close to 1 should use Taylor series and still be accurate
        // For r=1 exactly, the sum is a*num
        let result = geometric_sum(1.0, 0.9999, 100.0);
        // Direct: (1 - 0.9999^100)/(1 - 0.9999) ≈ 99.5...
        let expected = (1.0 - 0.9999_f64.powf(100.0)) / (1.0 - 0.9999);
        assert!((result - expected).abs() / expected < 1e-4);
    }

    #[test]
    fn test_geometric_sum_infinite() {
        // a / (1 - r)
        let result = geometric_sum(1.0, 0.5, f64::INFINITY);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_sum_scaled() {
        // a * (1 + r + r^2 + ... + r^4) = a * (1 - r^5)/(1 - r)
        let a = 3.0;
        let r = 0.7;
        let result = geometric_sum(a, r, 5.0);
        let expected = a * (1.0 - r.powi(5)) / (1.0 - r);
        assert!((result - expected).abs() < 1e-10);
    }
}
