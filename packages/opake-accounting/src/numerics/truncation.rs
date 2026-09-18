//! Chernoff bound truncation for self-convolution
//!
//! Implements optimal truncation bounds for repeated convolution using
//! Chernoff bounds on moment generating functions. This reduces
//! the number of elements we need to compute while maintaining accuracy.
//!
//! # References
//!
//! - Google dp_accounting: `pld/common.py:compute_self_convolve_bounds()`
//! - Chernoff bounds: <https://en.wikipedia.org/wiki/Chernoff_bound>

use crate::numerics::logspace::log_sumexp;

/// Compute optimal truncation bounds for self-convolution using Chernoff bounds
///
/// When convolving a PMF with itself N times, the result can have
/// N × size elements. Most of these have negligible probability mass.
/// This function computes tight bounds on which elements to keep.
///
/// The `tail_mass_truncation` budget is split equally between both tails.
/// For independent control of each tail, use [`compute_self_convolve_bounds_asymmetric`].
///
/// # Arguments
///
/// * `probs` - Probability mass function to self-convolve
/// * `num_times` - Number of self-convolutions
/// * `tail_mass_truncation` - Maximum probability mass to truncate (split equally between tails)
///
/// # Returns
///
/// Tuple of (lower_bound_index, upper_bound_index) for the truncated result
///
/// # Algorithm
///
/// Uses Chernoff bounds on the moment generating function (MGF):
/// ```text
/// P(S >= k) <= exp(num_times * log_MGF(order) - order * k)
/// ```
/// where S is the sum of N independent samples from the PMF.
///
/// By testing multiple orders, we find the tightest bound.
///
/// # Example
///
/// ```
/// use opake_accounting::numerics::truncation::compute_self_convolve_bounds;
///
/// let probs = vec![0.1, 0.3, 0.4, 0.2];
/// let (lower, upper) = compute_self_convolve_bounds(&probs, 100, 1e-10);
///
/// // Without truncation: result would have 301 elements
/// // With Chernoff bounds: result has (upper - lower + 1) ~= 50 elements
/// ```
pub fn compute_self_convolve_bounds(
    probs: &[f64],
    num_times: usize,
    tail_mass_truncation: f64,
) -> (usize, usize) {
    let half = tail_mass_truncation / 2.0;
    compute_self_convolve_bounds_asymmetric(probs, num_times, half, half)
}

/// Compute optimal truncation bounds with independent per-tail budgets
///
/// Like [`compute_self_convolve_bounds`], but allows different budgets for each tail.
/// Mass truncated from the right tail is added to `infinity_mass` (affects delta),
/// while mass truncated from the left tail is discarded (affects beta).
///
/// # Arguments
///
/// * `probs` - Probability mass function to self-convolve
/// * `num_times` - Number of self-convolutions
/// * `right_tail_budget` - Maximum mass to truncate from the right (upper) tail.
///   A value of `0.0` means no truncation on this tail (use full range).
/// * `left_tail_budget` - Maximum mass to truncate from the left (lower) tail.
///   A value of `0.0` means no truncation on this tail (use full range).
///
/// # Returns
///
/// Tuple of (lower_bound_index, upper_bound_index) for the truncated result
pub fn compute_self_convolve_bounds_asymmetric(
    probs: &[f64],
    num_times: usize,
    right_tail_budget: f64,
    left_tail_budget: f64,
) -> (usize, usize) {
    if probs.is_empty() {
        return (0, 0);
    }

    if num_times == 0 {
        return (0, 0);
    }

    if num_times == 1 {
        return (0, probs.len() - 1);
    }

    // Start with full range (no truncation)
    let full_lower = 0;
    let full_upper = probs.len() * num_times - num_times;

    // If both budgets are zero, no Chernoff truncation possible
    if right_tail_budget <= 0.0 && left_tail_budget <= 0.0 {
        return (full_lower, full_upper);
    }

    // Use multiple orders to get tight bounds
    // Orders range from -20/n to +20/n where n = len(probs)
    let n = probs.len() as f64;
    let num_orders = 40;
    let orders: Vec<f64> = (0..num_orders)
        .map(|i| {
            let offset = i as f64 - (num_orders as f64 / 2.0);
            offset / n
        })
        .filter(|&o| o != 0.0) // Skip order=0 (no bound)
        .collect();

    let mut best_lower = full_lower;
    let mut best_upper = full_upper;

    // Compute Chernoff bound for each order
    for order in orders {
        // Compute log MGF: log E[exp(order * X)]
        // = log(sum_i exp(order * i) * probs[i])
        let log_mgf = {
            let terms: Vec<f64> = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    if p > 0.0 {
                        order * i as f64 + p.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();
            log_sumexp(&terms)
        };

        if order > 0.0 && right_tail_budget > 0.0 {
            // Upper tail: P(S >= k) <= exp(n * log_mgf - order * k) <= right_tail_budget
            // Solve: k >= (n * log_mgf - ln(right_tail_budget)) / order
            let log_threshold = right_tail_budget.ln();
            let bound = (num_times as f64 * log_mgf - log_threshold) / order;
            let upper_bound = bound.ceil() as usize;
            best_upper = best_upper.min(upper_bound);
        } else if order < 0.0 && left_tail_budget > 0.0 {
            // Lower tail: P(S <= k) <= exp(n * log_mgf - order * k) <= left_tail_budget
            // Solve: k <= (n * log_mgf - ln(left_tail_budget)) / order
            // (order < 0, so dividing flips the inequality)
            let log_threshold = left_tail_budget.ln();
            let bound = (num_times as f64 * log_mgf - log_threshold) / order;
            let lower_bound = bound.floor() as usize;
            best_lower = best_lower.max(lower_bound);
        }
    }

    // Ensure bounds are valid
    best_lower = best_lower.min(full_upper);
    best_upper = best_upper.max(best_lower);

    (best_lower, best_upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bounds_simple() {
        // Uniform distribution — symmetric budgets
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        let (lower, upper) = compute_self_convolve_bounds(&probs, 10, 2e-10);

        // Result should be much smaller than full size (40 elements)
        assert!(upper - lower < 40);
        // But should be > 0
        assert!(upper > lower);
    }

    #[test]
    fn test_compute_bounds_concentrated() {
        // Concentrated distribution (most mass at one point)
        let probs = vec![0.01, 0.98, 0.01];

        let (lower, upper) = compute_self_convolve_bounds(&probs, 100, 2e-10);

        // Should be very tight (most samples will be near index 1)
        assert!(upper - lower < 30);
    }

    #[test]
    fn test_bounds_grow_with_num_times() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        let (lower10, upper10) = compute_self_convolve_bounds(&probs, 10, 2e-10);
        let (lower100, upper100) = compute_self_convolve_bounds(&probs, 100, 2e-10);

        // Bounds should grow with num_times (but slower than linear)
        let size10 = upper10 - lower10;
        let size100 = upper100 - lower100;

        assert!(size100 > size10);
        // But not 10x (due to Chernoff bound)
        assert!(size100 < size10 * 10);
    }

    #[test]
    fn test_edge_cases() {
        // Empty
        let (l, u) = compute_self_convolve_bounds(&[], 10, 2e-10);
        assert_eq!(l, 0);
        assert_eq!(u, 0);

        // Single element
        let (l, u) = compute_self_convolve_bounds(&[1.0], 10, 2e-10);
        assert_eq!(l, 0);
        assert_eq!(u, 0);

        // num_times = 1 (no composition)
        let probs = vec![0.5, 0.5];
        let (l, u) = compute_self_convolve_bounds(&probs, 1, 2e-10);
        assert_eq!(l, 0);
        assert_eq!(u, 1);
    }

    #[test]
    fn test_asymmetric_budgets() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        // Tight right, loose left
        let (lower_tight_r, upper_tight_r) =
            compute_self_convolve_bounds_asymmetric(&probs, 100, 1e-15, 1e-5);

        // Loose right, tight left
        let (lower_tight_l, upper_tight_l) =
            compute_self_convolve_bounds_asymmetric(&probs, 100, 1e-5, 1e-15);

        // Tight right budget → larger upper bound (keeps more of the right tail)
        assert!(upper_tight_r > upper_tight_l);

        // Tight left budget → smaller lower bound (keeps more of the left tail)
        assert!(lower_tight_l < lower_tight_r);
    }

    #[test]
    fn test_zero_budget_skips_truncation() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let num_times = 100;
        let full_upper = probs.len() * num_times - num_times;

        // Both zero → full range (no Chernoff)
        let (lower, upper) = compute_self_convolve_bounds_asymmetric(&probs, num_times, 0.0, 0.0);
        assert_eq!(lower, 0);
        assert_eq!(upper, full_upper);

        // Only right budget → lower stays at full range, upper is truncated
        let (lower_r, upper_r) =
            compute_self_convolve_bounds_asymmetric(&probs, num_times, 1e-10, 0.0);
        assert_eq!(lower_r, 0); // no left truncation
        assert!(upper_r < full_upper); // right is truncated

        // Only left budget → upper stays at full range, lower is truncated
        let (lower_l, upper_l) =
            compute_self_convolve_bounds_asymmetric(&probs, num_times, 0.0, 1e-10);
        assert!(lower_l > 0); // left is truncated
        assert_eq!(upper_l, full_upper); // no right truncation
    }
}
