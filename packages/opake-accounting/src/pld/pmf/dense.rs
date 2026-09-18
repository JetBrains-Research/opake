//! PMF representation for Privacy Loss Distributions
//!
//! Array-based representation, efficient for large support where most
//! grid points have non-zero probability mass (typical for Gaussian mechanisms).

use super::{floor_div, truncate_tails, CoarsenAction};
use crate::error::{PldError, Result};
use crate::numerics::{fft, truncation};

/// Combine two tail budgets during heterogeneous composition.
///
/// Tail budgets control how much mass Chernoff truncation may discard during
/// `self_compose`. They originate from `config.tail_mass_truncation`, split
/// equally between right and left tails.
///
/// When composing two mechanisms with different budgets, we take the **minimum**
/// (tightest) budget. This is conservative: the composed PLD will never
/// truncate more mass than the stricter component intended.
///
/// - Both positive → `min(a, b)` (respect the tightest constraint)
/// - One is `0.0` ("no budget set") → inherit the other's budget
/// - Both `0.0` → `0.0` (no truncation at all)
///
/// Note: in practice, `self_compose` is the dominant composition path (same
/// mechanism repeated k times), where the budget is simply inherited unchanged.
/// This function only matters for `compose(A, B)` with different mechanisms.
fn combine_budgets(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        a.min(b)
    }
}

/// Probability Mass Function for Privacy Loss Distributions
///
/// Stores probabilities as a contiguous array on a discretized grid.
/// The grid starts at `lower_loss_index * discretization` and contains
/// `probs.len()` buckets, each of width `discretization`.
///
/// # Memory Layout
///
/// ```text
/// Privacy Loss:  [-∞, lower_loss_index*Δ, ..., (lower_loss_index+n-1)*Δ, +∞]
/// Probability:   [negative_infinity_mass,  probs[0], ..., probs[n-1],  infinity_mass]
/// ```
///
/// where Δ = discretization, n = probs.len()
#[derive(Debug, Clone)]
pub struct Pmf {
    /// Discretization interval (grid spacing)
    pub discretization: f64,

    /// Lower bound bucket index (privacy loss = lower_loss_index * discretization)
    pub lower_loss_index: i64,

    /// Probability mass for each bucket
    pub probs: Vec<f64>,

    /// Probability mass at +∞ (right-tail mass)
    ///
    /// Represents outputs where the privacy loss is beyond the grid's upper bound.
    /// Contributes directly to delta (conservative: treated as worst-case loss).
    pub infinity_mass: f64,

    /// Probability mass at −∞ (left-tail mass)
    ///
    /// Represents outputs where the privacy loss is below the grid's lower bound
    /// (mechanism gave "perfect" privacy for those outputs). Accumulated during
    /// Chernoff truncation in `self_compose` to keep beta estimates conservative.
    ///
    /// Contributes to beta (added to CDF from the left) but contributes zero to
    /// delta (at loss = −∞, the hockey-stick term is always 0).
    pub negative_infinity_mass: f64,

    /// Maximum grid size before post-composition coarsening triggers.
    /// `usize::MAX` disables coarsening (default).
    pub max_grid_size: usize,

    /// Mass budget for right-tail Chernoff truncation during `self_compose`.
    ///
    /// Truncated right-tail mass is added to `infinity_mass`, setting a floor
    /// on the smallest delta that can be accurately measured. Derived from
    /// `config.tail_mass_truncation / 2`.
    ///
    /// `0.0` means no truncation (exact mechanism or tiny grid).
    pub right_tail_budget: f64,

    /// Mass budget for left-tail Chernoff truncation during `self_compose`.
    ///
    /// Truncated left-tail mass is added to `negative_infinity_mass`, keeping
    /// beta estimates conservative. Derived from `config.tail_mass_truncation / 2`.
    ///
    /// `0.0` means no truncation (exact mechanism or tiny grid).
    pub left_tail_budget: f64,
}

impl Pmf {
    /// Create a new dense PMF
    ///
    /// Tail budgets default to `0.0` (no Chernoff truncation). Use
    /// [`with_tail_budgets`](Self::with_tail_budgets) to set them.
    ///
    /// # Arguments
    ///
    /// * `discretization` - Grid spacing for privacy loss buckets
    /// * `lower_loss_index` - Index of the first bucket
    /// * `probs` - Probability masses for each bucket
    /// * `infinity_mass` - Tail probability at +∞
    /// * `max_grid_size` - Maximum grid size before coarsening
    pub fn new(
        discretization: f64,
        lower_loss_index: i64,
        probs: Vec<f64>,
        infinity_mass: f64,
        max_grid_size: usize,
    ) -> Self {
        Self {
            discretization,
            lower_loss_index,
            probs,
            infinity_mass,
            negative_infinity_mass: 0.0,
            max_grid_size,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        }
    }

    /// Set the Chernoff tail budgets for composition truncation.
    ///
    /// * `right` — max mass to truncate from the right tail (added to `infinity_mass`)
    /// * `left` — max mass to truncate from the left tail (added to `negative_infinity_mass`)
    pub fn with_tail_budgets(mut self, right: f64, left: f64) -> Self {
        self.right_tail_budget = right;
        self.left_tail_budget = left;
        self
    }

    /// Number of buckets in the PMF
    pub fn size(&self) -> usize {
        self.probs.len()
    }

    /// Get the privacy loss value for a bucket index
    pub(crate) fn loss_at_index(&self, index: i64) -> f64 {
        (self.lower_loss_index + index) as f64 * self.discretization
    }

    /// Validate that two PMFs can be composed, detecting any coarsening needed.
    ///
    /// Returns `Ok(None)` if discretizations match exactly. Returns
    /// `Ok(Some(CoarsenAction))` if one side must be coarsened (power-of-2 ratio).
    /// Returns `Err` for incompatible parameters.
    pub(crate) fn validate_composable(&self, other: &Pmf) -> Result<Option<CoarsenAction>> {
        let (larger, smaller) = if self.discretization >= other.discretization {
            (self.discretization, other.discretization)
        } else {
            (other.discretization, self.discretization)
        };

        // Exact match (within tolerance)
        if (larger - smaller).abs() <= 1e-10 * smaller {
            return Ok(None);
        }

        // Check if larger is a power-of-2 multiple of smaller
        let ratio = larger / smaller;
        let rounded = ratio.round();
        if (ratio - rounded).abs() > 1e-9 * ratio {
            return Err(PldError::DiscretizationMismatch(
                self.discretization,
                other.discretization,
            ));
        }
        let factor = rounded as usize;
        if !factor.is_power_of_two() {
            return Err(PldError::DiscretizationMismatch(
                self.discretization,
                other.discretization,
            ));
        }

        // The side with smaller discretization (finer grid) needs coarsening
        if self.discretization < other.discretization {
            Ok(Some(CoarsenAction::CoarsenSelf(factor)))
        } else {
            Ok(Some(CoarsenAction::CoarsenOther(factor)))
        }
    }

    /// Compose two PMFs using FFT convolution
    ///
    /// Computes the PMF of the sum of two independent privacy losses.
    /// This is the key operation for composing differential privacy guarantees.
    ///
    /// If the two PMFs have different but power-of-2-compatible discretizations,
    /// the finer one is automatically coarsened to match before convolution.
    ///
    /// # Errors
    ///
    /// * `PldError::DiscretizationMismatch` - If discretizations are incompatible
    pub fn compose(self, other: Pmf, tail_mass_truncation: f64) -> Result<Pmf> {
        let action = self.validate_composable(&other)?;

        // Apply coarsening if needed to align discretizations
        let (lhs, rhs) = match action {
            None => (self, other),
            Some(CoarsenAction::CoarsenSelf(f)) => (self.coarsen(f), other),
            Some(CoarsenAction::CoarsenOther(f)) => (self, other.coarsen(f)),
        };

        // Use FFT convolution from math
        let conv_result = fft::convolve(&lhs.probs, &rhs.probs);

        // Apply tail truncation if requested
        let (offset, result_probs, right_tail_mass) = if tail_mass_truncation > 0.0 {
            truncate_tails(&conv_result, tail_mass_truncation)
        } else {
            (0, conv_result, 0.0)
        };

        // The lower_loss_index of the result is the sum of the lower_loss_indexes, plus truncation offset
        let result_lower_loss_index = lhs.lower_loss_index + rhs.lower_loss_index + offset as i64;

        // Compose infinity masses: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        // For independent events: P(A ∩ B) = P(A) * P(B)
        // Add right tail mass from truncation
        let result_infinity_mass = lhs.infinity_mass + rhs.infinity_mass
            - lhs.infinity_mass * rhs.infinity_mass
            + right_tail_mass;

        // Compose negative infinity masses: P(Λ₁ + Λ₂ = -∞).
        // Since -∞ + finite = -∞, the composed loss is -∞ when AT LEAST ONE
        // step has -∞ loss → union formula (same as infinity_mass).
        let result_negative_infinity_mass = lhs.negative_infinity_mass + rhs.negative_infinity_mass
            - lhs.negative_infinity_mass * rhs.negative_infinity_mass;

        let result = Pmf {
            discretization: lhs.discretization,
            lower_loss_index: result_lower_loss_index,
            probs: result_probs,
            infinity_mass: result_infinity_mass,
            negative_infinity_mass: result_negative_infinity_mass,
            max_grid_size: lhs.max_grid_size.max(rhs.max_grid_size),
            right_tail_budget: combine_budgets(lhs.right_tail_budget, rhs.right_tail_budget),
            left_tail_budget: combine_budgets(lhs.left_tail_budget, rhs.left_tail_budget),
        };
        Ok(result.maybe_coarsen())
    }

    /// Self-compose a PMF multiple times using FFT power method
    ///
    /// Uses `IFFT(FFT(pmf)^count)` — O(n log n) with 2 FFTs total,
    /// vs O(n * count * log n) for repeated composition.
    ///
    /// Chernoff bounds truncate the composed grid using the PMF's tail budgets:
    /// - `right_tail_budget` mass is truncated from the right tail and
    ///   **conservatively added to `infinity_mass`** (affects delta)
    /// - `left_tail_budget` mass is truncated from the left tail and
    ///   **conservatively added to `negative_infinity_mass`** (affects beta)
    ///
    /// The effective left budget is reduced by already-accumulated
    /// `negative_infinity_mass`, so the total never exceeds `left_tail_budget`.
    /// This prevents repeated composition (e.g., via caching) from exceeding
    /// the promised accuracy contract.
    ///
    /// If both effective budgets are `0.0`, no Chernoff truncation is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if exact composition would exceed the bounded FFT size.
    ///
    /// # Panics
    ///
    /// Panics if `count` is 0
    pub fn self_compose(self, count: usize) -> Result<Pmf> {
        if count == 0 {
            panic!("count must be >= 1");
        }

        if count == 1 {
            return Ok(self);
        }

        if self.probs.is_empty() {
            let composed_infinity_mass = if self.infinity_mass == 0.0 {
                0.0
            } else {
                -(count as f64 * (-self.infinity_mass).ln_1p()).exp_m1()
            };
            let composed_neg_infinity_mass = if self.negative_infinity_mass == 0.0 {
                0.0
            } else {
                -(count as f64 * (-self.negative_infinity_mass).ln_1p()).exp_m1()
            };

            return Ok(Pmf {
                probs: vec![],
                infinity_mass: composed_infinity_mass,
                negative_infinity_mass: composed_neg_infinity_mass,
                ..self
            });
        }

        // Budget-aware Chernoff truncation:
        // - Right budget is used directly (budget is tiny, accumulation negligible)
        // - Left budget is reduced by already-consumed negative_infinity_mass,
        //   ensuring total never exceeds the promised tail_mass_truncation / 2.
        let effective_left_budget = (self.left_tail_budget - self.negative_infinity_mass).max(0.0);
        let effective_right_budget = self.right_tail_budget;

        let use_chernoff = effective_right_budget > 0.0 || effective_left_budget > 0.0;

        let (lower_bound, upper_bound) = if use_chernoff {
            truncation::compute_self_convolve_bounds_asymmetric(
                &self.probs,
                count,
                effective_right_budget,
                effective_left_budget,
            )
        } else {
            (
                0,
                fft::self_convolution_full_result_len(self.probs.len(), count)? - 1,
            )
        };

        let result_probs = fft::self_convolve_with_bounds(
            &self.probs,
            count,
            Some((lower_bound, upper_bound)),
            use_chernoff,
        )?;

        // The lower_loss_index scales by count AND shifts by the truncation offset
        let result_lower_loss_index = self.lower_loss_index * count as i64 + lower_bound as i64;

        // Infinity mass after k-fold composition:
        // P(at least one infinite) = 1 - P(all finite) = 1 - (1 - p)^k
        // Using numerically stable formula: -expm1(k * ln1p(-p))
        // ln1p(-p) avoids catastrophic cancellation when p is tiny (e.g. 1e-15).
        let composed_infinity_mass = if self.infinity_mass == 0.0 {
            0.0
        } else {
            -(count as f64 * (-self.infinity_mass).ln_1p()).exp_m1()
        };

        // Conservatively add right_tail_budget to infinity_mass:
        // the Chernoff bound guarantees at most this much mass was
        // truncated from the right tail. Adding it ensures delta
        // estimates remain conservative (upper bounds).
        let result_infinity_mass = composed_infinity_mass + effective_right_budget;

        // Negative infinity mass: P(at least one of k steps has -∞ loss) = 1-(1-p)^k.
        // Same union formula as infinity_mass (since -∞ + finite = -∞).
        // Then add left-tail Chernoff budget (effective budget was already reduced
        // by existing negative_infinity_mass, so the total stays ≤ left_tail_budget).
        let composed_neg_infinity_mass = if self.negative_infinity_mass == 0.0 {
            0.0
        } else {
            -(count as f64 * (-self.negative_infinity_mass).ln_1p()).exp_m1()
        };
        let result_negative_infinity_mass = composed_neg_infinity_mass + effective_left_budget;

        Ok(Pmf {
            discretization: self.discretization,
            lower_loss_index: result_lower_loss_index,
            probs: result_probs,
            infinity_mass: result_infinity_mass,
            negative_infinity_mass: result_negative_infinity_mass,
            max_grid_size: self.max_grid_size,
            right_tail_budget: self.right_tail_budget,
            left_tail_budget: self.left_tail_budget,
        }
        .maybe_coarsen())
    }

    /// Override the max grid size on this PMF.
    pub fn with_max_grid_size(&self, max_grid_size: usize) -> Self {
        Pmf {
            max_grid_size,
            ..self.clone()
        }
    }

    /// Self-compose with an explicit `max_grid_size` override.
    pub fn self_compose_with_max_grid_size(
        mut self,
        count: usize,
        max_grid_size: usize,
    ) -> Result<Pmf> {
        self.max_grid_size = max_grid_size;
        self.self_compose(count)
    }

    /// Coarsen this PMF by merging every `factor` adjacent bins.
    ///
    /// The resulting PMF has `discretization * factor` grid spacing and
    /// approximately `probs.len() / factor` elements. Total probability
    /// mass (including infinity_mass) is conserved.
    ///
    /// Each group's coarse index is shifted up by 1 so that the coarse bin epsilon
    /// overestimates the maximum original epsilon in the group.
    pub(crate) fn coarsen(&self, factor: usize) -> Pmf {
        if factor <= 1 {
            return self.clone();
        }

        let f = factor as i64;

        // Always group using floor_div alignment — consistent blocks of `factor`.
        let aligned_lower = floor_div(self.lower_loss_index, f) * f;
        let left_padding = (self.lower_loss_index - aligned_lower) as usize;

        // Build padded array: [0..left_padding | self.probs | 0..right_padding]
        let total_len = left_padding + self.probs.len();
        let right_padding = (factor - total_len % factor) % factor;
        let padded_len = total_len + right_padding;

        let mut padded = vec![0.0; padded_len];
        for (i, &p) in self.probs.iter().enumerate() {
            padded[left_padding + i] = p;
        }

        // Sum groups of `factor` bins
        let coarse_probs: Vec<f64> = padded
            .chunks(factor)
            .map(|chunk| chunk.iter().sum())
            .collect();

        // Coarse index of the first group.
        let base_coarse_lower = aligned_lower / f; // aligned_lower is already a multiple of f
        let coarse_lower = base_coarse_lower + 1;

        Pmf {
            discretization: self.discretization * factor as f64,
            lower_loss_index: coarse_lower,
            probs: coarse_probs,
            infinity_mass: self.infinity_mass,
            negative_infinity_mass: self.negative_infinity_mass,
            max_grid_size: self.max_grid_size,
            right_tail_budget: self.right_tail_budget,
            left_tail_budget: self.left_tail_budget,
        }
    }

    /// Coarsen this PMF if it exceeds `max_grid_size`.
    ///
    /// Uses the smallest power-of-2 factor that brings the grid within max_grid_size.
    pub(super) fn maybe_coarsen(self) -> Self {
        if self.probs.len() <= self.max_grid_size {
            return self;
        }
        let raw_ratio = (self.probs.len() as f64 / self.max_grid_size as f64).ceil();
        let k = (raw_ratio.log2().ceil() as u32).max(1);
        let factor = 1usize << k;
        let mut coarsened = self.coarsen(factor);
        coarsened.max_grid_size = self.max_grid_size;
        coarsened
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dense_compose_basic() {
        let pmf1 = Pmf::new(0.1, 0, vec![0.3, 0.7], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.1, 0, vec![0.4, 0.6], 0.0, usize::MAX);

        let composed = pmf1.compose(pmf2, 0.0).unwrap();

        // Result size = sum of sizes - 1 (convolution)
        assert_eq!(composed.size(), 3);
        assert_eq!(composed.lower_loss_index, 0);

        // Verify probabilities (manual convolution)
        assert_relative_eq!(composed.probs[0], 0.12, epsilon = 1e-10); // 0.3 * 0.4
        assert_relative_eq!(composed.probs[1], 0.46, epsilon = 1e-10); // 0.3*0.6 + 0.7*0.4
        assert_relative_eq!(composed.probs[2], 0.42, epsilon = 1e-10); // 0.7 * 0.6
    }

    #[test]
    fn test_dense_compose_with_infinity_mass() {
        let pmf1 = Pmf::new(0.1, 0, vec![0.9], 0.1, usize::MAX);
        let pmf2 = Pmf::new(0.1, 0, vec![0.9], 0.1, usize::MAX);

        let composed = pmf1.compose(pmf2, 0.0).unwrap();

        // infinity_mass = 0.1 + 0.1 - 0.1*0.1 = 0.19
        assert_relative_eq!(composed.infinity_mass, 0.19, epsilon = 1e-10);
    }

    #[test]
    fn test_dense_compose_offset() {
        let pmf1 = Pmf::new(0.1, -5, vec![0.3, 0.7], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.1, 3, vec![0.4, 0.6], 0.0, usize::MAX);

        let composed = pmf1.compose(pmf2, 0.0).unwrap();

        // Result lower_loss_index = -5 + 3 = -2
        assert_eq!(composed.lower_loss_index, -2);
    }

    #[test]
    fn test_dense_self_compose() {
        let pmf = Pmf::new(0.1, 0, vec![0.3, 0.5, 0.2], 0.0, usize::MAX);

        let composed = pmf.self_compose(3).unwrap();

        assert_eq!(composed.size(), 7);
        assert_eq!(composed.lower_loss_index, 0);

        let total: f64 = composed.probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dense_self_compose_conservatively_adds_right_tail_budget() {
        let pmf = Pmf::new(0.1, 0, vec![0.8, 0.1], 0.1, usize::MAX).with_tail_budgets(0.01, 0.0);

        let composed = pmf.self_compose(2).unwrap();

        // infinity_mass = 1 - (1 - 0.1)^2 + 0.01 = 0.20
        assert_relative_eq!(composed.infinity_mass, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_pmf_self_compose_preserves_empty_finite_support() {
        let pmf = Pmf::new(0.1, 0, vec![], 0.25, usize::MAX);

        let composed = pmf.self_compose(3).unwrap();

        assert!(composed.probs.is_empty());
        assert_relative_eq!(
            composed.infinity_mass,
            1.0 - 0.75_f64.powi(3),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_compose_discretization_mismatch() {
        let pmf1 = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.3, 0, vec![1.0], 0.0, usize::MAX);

        let result = pmf1.compose(pmf2, 0.0);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PldError::DiscretizationMismatch(_, _)
        ));
    }

    #[test]
    fn test_composition_commutativity() {
        let pmf1 = Pmf::new(0.1, 0, vec![0.3, 0.5, 0.2], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.1, 0, vec![0.4, 0.4, 0.2], 0.0, usize::MAX);

        let composed1 = pmf1.clone().compose(pmf2.clone(), 0.0).unwrap();
        let composed2 = pmf2.compose(pmf1, 0.0).unwrap();

        assert_eq!(composed1.size(), composed2.size());
        for (p1, p2) in composed1.probs.iter().zip(composed2.probs.iter()) {
            assert_relative_eq!(p1, p2, epsilon = 1e-10);
        }
        assert_eq!(composed1.lower_loss_index, composed2.lower_loss_index);
    }

    #[test]
    fn test_composition_associativity() {
        let pmf1 = Pmf::new(0.1, 0, vec![0.4, 0.6], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.1, 0, vec![0.3, 0.7], 0.0, usize::MAX);
        let pmf3 = Pmf::new(0.1, 0, vec![0.5, 0.5], 0.0, usize::MAX);

        let left = pmf1
            .clone()
            .compose(pmf2.clone(), 0.0)
            .unwrap()
            .compose(pmf3.clone(), 0.0)
            .unwrap();

        let right = pmf1.compose(pmf2.compose(pmf3, 0.0).unwrap(), 0.0).unwrap();

        assert_eq!(left.size(), right.size());
        assert_eq!(left.lower_loss_index, right.lower_loss_index);
        for (p1, p2) in left.probs.iter().zip(right.probs.iter()) {
            assert_relative_eq!(p1, p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_self_compose_vs_repeated_composition() {
        let pmf = Pmf::new(0.1, 0, vec![0.4, 0.6], 0.0, usize::MAX);

        let composed_fast = pmf.clone().self_compose(3).unwrap();
        let composed_slow = pmf
            .clone()
            .compose(pmf.clone(), 0.0)
            .unwrap()
            .compose(pmf, 0.0)
            .unwrap();

        assert_eq!(composed_fast.size(), composed_slow.size());
        assert_eq!(
            composed_fast.lower_loss_index,
            composed_slow.lower_loss_index
        );
        for (p1, p2) in composed_fast.probs.iter().zip(composed_slow.probs.iter()) {
            assert_relative_eq!(p1, p2, epsilon = 1e-8);
        }
    }

    // --- Coarsening tests ---

    #[test]
    fn test_coarsen_factor_1_identity() {
        let pmf = Pmf::new(0.1, -5, vec![0.2, 0.3, 0.4, 0.1], 0.0, usize::MAX);
        let coarsened = pmf.coarsen(1);
        assert_eq!(coarsened.discretization, 0.1);
        assert_eq!(coarsened.lower_loss_index, -5);
        assert_eq!(coarsened.probs, vec![0.2, 0.3, 0.4, 0.1]);
    }

    #[test]
    fn test_coarsen_factor_2() {
        let pmf = Pmf::new(0.1, -4, vec![0.1, 0.2, 0.3, 0.4], 0.0, usize::MAX);
        let coarsened = pmf.coarsen(2);
        assert_relative_eq!(coarsened.discretization, 0.2);
        assert_eq!(coarsened.lower_loss_index, -1);
        assert_eq!(coarsened.probs.len(), 2);
        assert_relative_eq!(coarsened.probs[0], 0.3, epsilon = 1e-12);
        assert_relative_eq!(coarsened.probs[1], 0.7, epsilon = 1e-12);
    }

    #[test]
    fn test_coarsen_mass_conservation() {
        let pmf = Pmf::new(
            0.1,
            -10,
            vec![0.1, 0.15, 0.25, 0.2, 0.1, 0.05, 0.1, 0.05],
            0.0,
            usize::MAX,
        );
        let original_mass: f64 = pmf.probs.iter().sum::<f64>() + pmf.infinity_mass;
        for factor in [2, 4, 8] {
            let coarsened = pmf.coarsen(factor);
            let coarsened_mass: f64 = coarsened.probs.iter().sum::<f64>() + coarsened.infinity_mass;
            assert_relative_eq!(original_mass, coarsened_mass, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_coarsen_negative_index_alignment() {
        let pmf = Pmf::new(0.1, -7, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);
        let coarsened = pmf.coarsen(4);
        assert_relative_eq!(coarsened.discretization, 0.4);
        assert_eq!(coarsened.lower_loss_index, -1);
        assert_eq!(coarsened.probs.len(), 1);
        assert_relative_eq!(coarsened.probs[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_validate_composable_equal_disc() {
        let pmf1 = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);
        let result = pmf1.validate_composable(&pmf2).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_composable_power_of_2_multiple() {
        let pmf1 = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.4, 0, vec![1.0], 0.0, usize::MAX);
        let result = pmf1.validate_composable(&pmf2).unwrap();
        assert_eq!(result, Some(CoarsenAction::CoarsenSelf(4)));

        let result = pmf2.validate_composable(&pmf1).unwrap();
        assert_eq!(result, Some(CoarsenAction::CoarsenOther(4)));
    }

    #[test]
    fn test_validate_composable_non_power_of_2_rejects() {
        let pmf1 = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.3, 0, vec![1.0], 0.0, usize::MAX);
        assert!(pmf1.validate_composable(&pmf2).is_err());
    }

    #[test]
    fn test_compose_auto_coarsens() {
        let pmf1 = Pmf::new(0.1, -2, vec![0.1, 0.3, 0.4, 0.2], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.2, -1, vec![0.6, 0.4], 0.0, usize::MAX);
        let composed = pmf1.compose(pmf2, 0.0).unwrap();
        assert_relative_eq!(composed.discretization, 0.2);
        let total: f64 = composed.probs.iter().sum::<f64>() + composed.infinity_mass;
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compose_auto_coarsens_matches_manual() {
        let pmf1 = Pmf::new(0.1, -4, vec![0.1, 0.2, 0.3, 0.4], 0.0, usize::MAX);
        let pmf2 = Pmf::new(0.2, -2, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);

        let pmf1_coarsened = pmf1.coarsen(2);
        let manual = pmf1_coarsened.compose(pmf2.clone(), 0.0).unwrap();
        let auto = pmf1.compose(pmf2, 0.0).unwrap();

        assert_eq!(manual.probs.len(), auto.probs.len());
        assert_eq!(manual.lower_loss_index, auto.lower_loss_index);
        for (m, a) in manual.probs.iter().zip(auto.probs.iter()) {
            assert_relative_eq!(m, a, epsilon = 1e-12);
        }
        assert_relative_eq!(manual.infinity_mass, auto.infinity_mass, epsilon = 1e-12);
    }
}
