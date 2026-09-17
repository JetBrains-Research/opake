//! Log-space arithmetic operations
//!
//! Numerically stable operations in log space.
//! These functions avoid overflow/underflow by working with logarithms.

#[cfg(test)]
use std::f64::consts;

/// Compute log(e^x + e^y) in a numerically stable way
///
/// # Arguments
///
/// * `log_x` - The logarithm of the first value
/// * `log_y` - The logarithm of the second value
///
/// # Returns
///
/// log(exp(log_x) + exp(log_y)) computed without overflow
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::logspace::log_add;
///
/// let result = log_add(100.0, 101.0);
/// // exp(100) + exp(101) = exp(100) * (1 + e)
/// // log(exp(100) * (1 + e)) = 100 + log(1 + e) ≈ 101.31
/// assert!((result - 101.313).abs() < 0.01);
/// ```
///
/// # Algorithm
///
/// Uses the identity: log(e^x + e^y) = max(x,y) + log(1 + e^(min(x,y) - max(x,y)))
/// This avoids overflow since the exponential term is always ≤ 1.
///
/// Reference: SciPy's `_log_sumexp.py`
pub fn log_add(log_x: f64, log_y: f64) -> f64 {
    // Handle special cases
    if log_x == f64::NEG_INFINITY {
        return log_y;
    }
    if log_y == f64::NEG_INFINITY {
        return log_x;
    }
    if log_x.is_nan() || log_y.is_nan() {
        return f64::NAN;
    }

    // Ensure log_x >= log_y for numerical stability
    let (max, min) = if log_x >= log_y {
        (log_x, log_y)
    } else {
        (log_y, log_x)
    };

    // log(e^max + e^min) = max + log(1 + e^(min - max))
    // Since min <= max, (min - max) <= 0, so e^(min - max) <= 1
    max + (min - max).exp().ln_1p()
}

/// Compute log(e^x - e^y) in a numerically stable way (requires x > y)
///
/// # Arguments
///
/// * `log_x` - The logarithm of the first value (must be >= log_y)
/// * `log_y` - The logarithm of the second value
///
/// # Returns
///
/// log(exp(log_x) - exp(log_y)) if log_x > log_y, or an error if log_x < log_y
///
/// # Errors
///
/// Returns `Err` if log_x < log_y (would result in log of negative number)
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::logspace::log_sub;
///
/// let result = log_sub(101.0, 100.0).unwrap();
/// // exp(101) - exp(100) = exp(100) * (e - 1)
/// // log(exp(100) * (e - 1)) = 100 + log(e - 1) ≈ 100.54
/// assert!((result - 100.541).abs() < 0.01);
/// ```
///
/// # Algorithm
///
/// Uses the identity: log(e^x - e^y) = x + log(1 - e^(y - x))
/// For x > y, we have y - x < 0, so e^(y - x) < 1, and 1 - e^(y - x) > 0.
pub fn log_sub(log_x: f64, log_y: f64) -> Result<f64, &'static str> {
    if log_x < log_y {
        return Err("log_sub requires log_x >= log_y (cannot take log of negative number)");
    }

    // Handle special cases
    if log_y == f64::NEG_INFINITY {
        return Ok(log_x);
    }
    if log_x.is_nan() || log_y.is_nan() {
        return Ok(f64::NAN);
    }
    if log_x == log_y {
        return Ok(f64::NEG_INFINITY); // log(0)
    }

    // log(e^x - e^y) = x + log(1 - e^(y - x))
    // Since y < x, we have (y - x) < 0, so e^(y - x) < 1
    let diff = log_y - log_x;

    // Use exp_m1 for better accuracy: e^x - 1
    // log(1 - e^(y - x)) = log(-(e^(y - x) - 1)) = log(-exp_m1(y - x))
    Ok(log_x + (-diff.exp_m1()).ln())
}

/// Compute log(a * exp(b) + c) in a numerically stable way
///
/// # Arguments
///
/// * `a` - Coefficient (must be non-negative for positive result)
/// * `b` - Exponent
/// * `c` - Additive constant
///
/// # Returns
///
/// log(a * exp(b) + c)
///
/// # Panics
///
/// Panics if a * exp(b) + c <= 0
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::logspace::log_a_times_exp_b_plus_c;
///
/// let result = log_a_times_exp_b_plus_c(2.0, 1.0, 3.0);
/// // 2 * e^1 + 3 = 2 * 2.718 + 3 ≈ 8.436
/// // log(8.436) ≈ 2.132
/// assert!((result - 2.132).abs() < 0.01);
/// ```
///
/// # Algorithm
///
/// Uses log_add and log_sub for numerical stability, following Google's dp_accounting
/// implementation in common.py lines 339-354.
pub fn log_a_times_exp_b_plus_c(a: f64, b: f64, c: f64) -> f64 {
    // Handle edge cases
    if a == 0.0 {
        return c.ln();
    }
    if a < 0.0 {
        assert!(
            c > 0.0,
            "a * exp(b) + c must be positive: a={}, b={}, c={}",
            a,
            b,
            c
        );
        return log_sub(c.ln(), (-a).ln() + b).unwrap();
    }
    if b == 0.0 {
        return (a + c).ln();
    }

    let d = b + a.ln();
    if c == 0.0 {
        d
    } else if c < 0.0 {
        log_sub(d, (-c).ln()).unwrap()
    } else {
        log_add(d, c.ln())
    }
}

/// Compute log(sum(exp(values))) in a numerically stable way
///
/// # Arguments
///
/// * `values` - Slice of log-space values
///
/// # Returns
///
/// log(sum(exp(v) for v in values))
///
/// # Examples
///
/// ```
/// use opake_accounting::numerics::logspace::log_sumexp;
///
/// let values = vec![0.0, 1.0, 2.0];
/// let result = log_sumexp(&values);
/// // exp(0) + exp(1) + exp(2) = 1 + e + e^2 ≈ 11.39
/// // log(11.39) ≈ 2.43
/// assert!((result - 2.407).abs() < 0.01);
/// ```
///
/// # Algorithm
///
/// Uses the identity: log(Σ e^xᵢ) = max(xᵢ) + log(Σ e^(xᵢ - max(xᵢ)))
/// This prevents overflow since all exponential terms are ≤ 1.
///
/// Reference: SciPy's `_log_sumexp.py`, numpy's `logaddexp.reduce`
pub fn log_sumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    if values.len() == 1 {
        return values[0];
    }

    // Find maximum value (for numerical stability)
    let max = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::NEG_INFINITY);

    if !max.is_finite() {
        return max; // All values are -inf or max is inf/nan
    }

    // Compute sum of exp(x - max) for all x
    let sum: f64 = values.iter().map(|&x| (x - max).exp()).sum();

    // log(sum(exp(x))) = max + log(sum(exp(x - max)))
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_log_add_basic() {
        // log(e^0 + e^0) = log(2)
        let result = log_add(0.0, 0.0);
        assert_relative_eq!(result, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log_add_large_values() {
        // log(e^100 + e^101) = log(e^100 * (1 + e)) = 100 + log(1 + e)
        let result = log_add(100.0, 101.0);
        let expected = 101.0 + (1.0 + consts::E.recip()).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_add_negative_infinity() {
        assert_eq!(log_add(f64::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_add(5.0, f64::NEG_INFINITY), 5.0);
        assert_eq!(
            log_add(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_log_add_commutative() {
        let a = 3.5;
        let b = 7.2;
        assert_relative_eq!(log_add(a, b), log_add(b, a), epsilon = 1e-10);
    }

    #[test]
    fn test_log_sub_basic() {
        // log(e^1 - e^0) = log(e - 1)
        let result = log_sub(1.0, 0.0).unwrap();
        assert_relative_eq!(result, (consts::E - 1.0).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log_sub_large_values() {
        // log(e^101 - e^100) = log(e^100 * (e - 1)) = 100 + log(e - 1)
        let result = log_sub(101.0, 100.0).unwrap();
        let expected = 100.0 + (consts::E - 1.0).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_sub_equal_values() {
        let result = log_sub(5.0, 5.0).unwrap();
        assert_eq!(result, f64::NEG_INFINITY); // log(0) = -inf
    }

    #[test]
    fn test_log_sub_error() {
        let result = log_sub(5.0, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_sub_negative_infinity() {
        let result = log_sub(5.0, f64::NEG_INFINITY).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_log_sumexp_basic() {
        // log(e^0 + e^1 + e^2) = log(1 + e + e^2)
        let values = vec![0.0, 1.0, 2.0];
        let result = log_sumexp(&values);
        let expected = (1.0 + consts::E + consts::E.powi(2)).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_sumexp_large_values() {
        // Should not overflow even with large values
        let values = vec![100.0, 101.0, 102.0];
        let result = log_sumexp(&values);
        // e^100 + e^101 + e^102 = e^100 * (1 + e + e^2)
        let expected = 100.0 + (1.0 + consts::E + consts::E.powi(2)).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_sumexp_empty() {
        assert_eq!(log_sumexp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sumexp_single() {
        assert_eq!(log_sumexp(&[5.0]), 5.0);
    }

    #[test]
    fn test_log_sumexp_with_neg_inf() {
        let values = vec![f64::NEG_INFINITY, 1.0, 2.0];
        let result = log_sumexp(&values);
        // -inf contributes nothing, so result = log(e^1 + e^2)
        let expected = log_sumexp(&[1.0, 2.0]);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_add_matches_log_sumexp() {
        // log_add(a, b) should equal log_sumexp([a, b])
        let a = 3.5;
        let b = 7.2;
        assert_relative_eq!(log_add(a, b), log_sumexp(&[a, b]), epsilon = 1e-10);
    }
}
