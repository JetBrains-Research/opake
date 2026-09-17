//! Probability mass function for Privacy Loss Distributions
//!
//! Array-based PMF representation, efficient for large support where most
//! grid points have non-zero probability mass (typical for Gaussian mechanisms).

pub mod dense;
pub(crate) mod geom;

pub use dense::Pmf;

use std::collections::BTreeMap;

/// Floor-division toward negative infinity (not truncation toward zero).
///
/// Rust's `/` operator truncates toward zero, but for grid alignment we need
/// consistent downward rounding so that negative indices align correctly.
pub(crate) fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r ^ b) < 0) {
        d - 1
    } else {
        d
    }
}

/// Describes which side to coarsen before composition.
///
/// When two PMFs have different (but power-of-2 compatible) discretizations,
/// the finer one must be coarsened to match the coarser before convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoarsenAction {
    /// `self` needs coarsening by this factor
    CoarsenSelf(usize),
    /// `other` needs coarsening by this factor
    CoarsenOther(usize),
}

/// Truncate probability array from both tails, removing low-mass elements
///
/// Removes the maximum prefix and suffix from `probs`, each having
/// cumulative mass ≤ `tail_mass_truncation/2`.
///
/// # Returns
///
/// Tuple of (offset, truncated_probs, right_tail_mass) where:
/// - offset: number of elements removed from the left
/// - truncated_probs: the remaining probability array
/// - right_tail_mass: mass removed from the right (to be added to infinity_mass)
pub(crate) fn truncate_tails(probs: &[f64], tail_mass_truncation: f64) -> (usize, Vec<f64>, f64) {
    if probs.is_empty() || tail_mass_truncation <= 0.0 {
        return (0, probs.to_vec(), 0.0);
    }

    let half_truncation = tail_mass_truncation / 2.0;

    // Find maximum prefix with cumulative mass ≤ half_truncation
    let mut left_mass = 0.0;
    let mut left_truncate = 0;
    for (i, &p) in probs.iter().enumerate() {
        if left_mass + p > half_truncation {
            break;
        }
        left_mass += p;
        left_truncate = i + 1;
    }

    // Find maximum suffix with cumulative mass ≤ half_truncation
    let mut right_mass = 0.0;
    let mut right_truncate = 0;
    for (i, &p) in probs.iter().rev().enumerate() {
        if right_mass + p > half_truncation {
            break;
        }
        right_mass += p;
        right_truncate = i + 1;
    }

    // Ensure we don't truncate everything
    if left_truncate + right_truncate >= probs.len() {
        return (0, probs.to_vec(), 0.0);
    }

    // Extract the middle portion
    let end = probs.len() - right_truncate;
    let mut truncated = probs[left_truncate..end].to_vec();

    if !truncated.is_empty() {
        // Add left truncated mass to the first remaining element
        truncated[0] += left_mass;
        // Right truncated mass goes to infinity_mass (returned separately)
    }

    (left_truncate, truncated, right_mass)
}

impl Pmf {
    /// Create a PMF from a sparse map of bucket indices to probabilities.
    ///
    /// Converts the map into a contiguous dense array representation.
    /// This is used by the discretization algorithm to build PMFs from
    /// Connect-the-Dots output and by degenerate mechanisms (eps_delta, identity).
    pub fn from_sparse(
        discretization: f64,
        loss_probs: BTreeMap<i64, f64>,
        infinity_mass: f64,
        max_grid_size: usize,
    ) -> Self {
        if loss_probs.is_empty() {
            return Pmf::new(discretization, 0, vec![], infinity_mass, max_grid_size);
        }

        let min_key = *loss_probs.keys().min().unwrap();
        let max_key = *loss_probs.keys().max().unwrap();
        let size = (max_key - min_key + 1) as usize;

        let mut probs = vec![0.0; size];
        for (&key, &prob) in &loss_probs {
            probs[(key - min_key) as usize] = prob;
        }

        Pmf::new(discretization, min_key, probs, infinity_mass, max_grid_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pmf_compose_dense_dense() {
        let pmf1 = Pmf::new(
            0.1,
            0,
            (0..1001).map(|_| 1.0 / 1001.0).collect(),
            0.0,
            usize::MAX,
        );
        let pmf2 = Pmf::new(
            0.1,
            0,
            (0..1001).map(|_| 1.0 / 1001.0).collect(),
            0.0,
            usize::MAX,
        );

        let composed = pmf1.compose(pmf2, 0.0).unwrap();
        assert!(composed.size() > 0);
    }

    #[test]
    fn test_from_sparse_small() {
        let mut masses = BTreeMap::new();
        masses.insert(0, 0.5);
        masses.insert(5, 0.5);
        let pmf = Pmf::from_sparse(0.1, masses, 0.0, usize::MAX);

        assert_eq!(pmf.size(), 6); // indices 0..5 inclusive
        assert_relative_eq!(pmf.probs[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(pmf.probs[5], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_from_sparse_empty() {
        let masses = BTreeMap::new();
        let pmf = Pmf::from_sparse(0.1, masses, 0.0, usize::MAX);
        assert_eq!(pmf.size(), 0);
    }

    #[test]
    fn test_from_sparse_large() {
        let mut masses = BTreeMap::new();
        for i in 0..100 {
            masses.insert(i, 0.01);
        }
        let pmf = Pmf::from_sparse(0.1, masses, 0.0, usize::MAX);
        assert_eq!(pmf.size(), 100);
        let total: f64 = pmf.probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }
}
