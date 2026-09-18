//! High-level Privacy Loss Distribution with adjacency support
//!
//! This module provides the `PrivacyLossDistribution` wrapper around `Pmf`
//! that supports different adjacency types (ADD/REMOVE) and handles both
//! symmetric and asymmetric mechanisms.

pub(crate) mod metrics;
pub mod pmf;
pub(crate) mod realization;

pub use pmf::Pmf;

use crate::error::Result;

/// High-level privacy loss distribution with adjacency support
///
/// Wraps one or two `Pmf` objects to support differential privacy mechanisms
/// that may have different privacy loss distributions depending on the adjacency
/// type (whether a dataset has one more or one fewer element).
///
/// # Adjacency Types
///
/// - **REMOVE adjacency**: Dataset D has one fewer element than D'
/// - **ADD adjacency**: Dataset D has one more element than D'
///
/// # Symmetric vs Asymmetric Mechanisms
///
/// - **Symmetric mechanisms** (e.g., Gaussian): Have the same PLD for
///   both adjacency types. Created with `new_symmetric()`.
/// - **Asymmetric mechanisms** (e.g., Poisson subsampling, shuffling): Have
///   different PLDs for each adjacency type. Created with `new_asymmetric()`.
///
/// # Privacy Guarantee
///
/// The (ε, δ)-DP guarantee is computed by taking the worst case (maximum)
/// over both adjacency types:
///
/// - `delta_at(ε) = max(δ_remove(ε), δ_add(ε))`
/// - `epsilon_at(δ) = max(ε_remove(δ), ε_add(δ))`
///
/// # Examples
///
/// ## From a Gaussian mechanism
///
/// ```rust,ignore
/// use opake_accounting::{mechanisms::gaussian_pld, DiscretizationConfig};
///
/// let pld = gaussian_pld(1.1, &DiscretizationConfig::default())?;
///
/// let delta = pld.delta_at(1.0);
/// let epsilon = pld.epsilon_at(1e-5);
/// let advantage = pld.advantage();
/// ```
///
#[derive(Debug, Clone)]
pub struct PrivacyLossDistribution {
    /// PLD for REMOVE adjacency (D has fewer elements than D')
    pub(crate) pmf_remove: Pmf,

    /// PLD for ADD adjacency (D has more elements than D')
    ///
    /// If `None`, this is a symmetric mechanism and `pmf_remove` is used for both.
    pub(crate) pmf_add: Option<Pmf>,

    /// Failure probability of the Monte Carlo confidence event.
    ///
    /// Zero for analytic PLDs.
    estimation_failure_probability: f64,

    /// Unresolved Monte Carlo mass represented by the confidence construction.
    ///
    /// Zero for analytic PLDs. This is tracked separately from numerical tail
    /// truncation even though both contribute to the operational infinity mass.
    mc_resolution: f64,

    /// Gaussian source parameters used to construct a transform-safe
    /// realization when this PLD becomes a pointwise amplification input.
    gaussian_source: Option<(f64, f64)>,
}

impl PrivacyLossDistribution {
    /// Create a symmetric privacy loss distribution
    ///
    /// For symmetric mechanisms like Gaussian noise,
    /// the privacy loss distribution is the same regardless of adjacency type.
    ///
    /// # Arguments
    ///
    /// * `pmf` - The privacy loss PMF (same for both adjacencies)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::BTreeMap;
    /// use opake_accounting::PrivacyLossDistribution;
    /// use opake_accounting::pld::pmf::Pmf;
    ///
    /// let mut masses = BTreeMap::new();
    /// masses.insert(0, 0.5);
    /// masses.insert(5, 0.5);
    ///
    /// let pmf = Pmf::from_sparse(0.1, masses, 0.0, usize::MAX);
    /// let pld = PrivacyLossDistribution::new_symmetric(pmf);
    ///
    /// assert!(pld.is_symmetric());
    /// ```
    pub(crate) fn new_symmetric(pmf: Pmf) -> Self {
        Self {
            pmf_remove: pmf,
            pmf_add: None,
            estimation_failure_probability: 0.0,
            mc_resolution: 0.0,
            gaussian_source: None,
        }
    }

    /// Create an asymmetric privacy loss distribution
    ///
    /// For asymmetric mechanisms like Poisson subsampling or shuffling,
    /// the privacy loss distribution depends on the adjacency type.
    ///
    /// # Arguments
    ///
    /// * `pmf_remove` - PLD for REMOVE adjacency (D has fewer elements than D')
    /// * `pmf_add` - PLD for ADD adjacency (D has more elements than D')
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::BTreeMap;
    /// use opake_accounting::PrivacyLossDistribution;
    /// use opake_accounting::pld::pmf::Pmf;
    ///
    /// let mut masses_remove = BTreeMap::new();
    /// masses_remove.insert(0, 0.3);
    /// masses_remove.insert(5, 0.7);
    ///
    /// let mut masses_add = BTreeMap::new();
    /// masses_add.insert(0, 0.4);
    /// masses_add.insert(5, 0.6);
    ///
    /// let pmf_remove = Pmf::from_sparse(0.1, masses_remove, 0.0, usize::MAX);
    /// let pmf_add = Pmf::from_sparse(0.1, masses_add, 0.0, usize::MAX);
    ///
    /// let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);
    ///
    /// assert!(!pld.is_symmetric());
    /// ```
    pub(crate) fn new_asymmetric(pmf_remove: Pmf, pmf_add: Pmf) -> Self {
        Self {
            pmf_remove,
            pmf_add: Some(pmf_add),
            estimation_failure_probability: 0.0,
            mc_resolution: 0.0,
            gaussian_source: None,
        }
    }

    /// Attach the statistical guarantee for a Monte Carlo PLD.
    pub(crate) fn with_monte_carlo_guarantee(
        mut self,
        failure_probability: f64,
        resolution: f64,
    ) -> Self {
        self.estimation_failure_probability = failure_probability;
        self.mc_resolution = resolution;
        self
    }

    /// Record an analytic Gaussian source for pointwise amplification.
    pub(crate) fn with_gaussian_source(mut self, noise_multiplier: f64, tail_mass: f64) -> Self {
        self.gaussian_source = Some((noise_multiplier, tail_mass));
        self
    }

    pub(crate) fn gaussian_source(&self) -> Option<(f64, f64)> {
        self.gaussian_source
    }

    /// Failure probability of the Monte Carlo confidence event.
    pub fn estimation_failure_probability(&self) -> f64 {
        self.estimation_failure_probability
    }

    /// Confidence level of the Monte Carlo PLD bound.
    pub fn estimation_confidence(&self) -> f64 {
        1.0 - self.estimation_failure_probability
    }

    /// Unresolved Monte Carlo mass before numerical composition truncation.
    pub fn mc_resolution(&self) -> f64 {
        self.mc_resolution
    }

    /// Check if this PLD is symmetric
    ///
    /// Returns `true` if the mechanism has the same privacy loss distribution
    /// for both ADD and REMOVE adjacency types.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::pld::*;
    ///
    /// // Gaussian is symmetric (same PLD for ADD and REMOVE)
    /// let pld = gaussian(1.1).pld()?;
    /// assert!(pld.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        self.pmf_add.is_none()
    }

    /// Set Chernoff tail budgets on all contained PMFs.
    ///
    /// Propagates the budgets to both the REMOVE and ADD (if present) PMFs.
    /// This controls how much mass may be truncated during `self_compose`:
    ///
    /// * `right` — right-tail budget (added to `infinity_mass`)
    /// * `left` — left-tail budget (added to `negative_infinity_mass`)
    pub fn with_tail_budgets(mut self, right: f64, left: f64) -> Self {
        self.pmf_remove.right_tail_budget = right;
        self.pmf_remove.left_tail_budget = left;
        if let Some(ref mut pmf_add) = self.pmf_add {
            pmf_add.right_tail_budget = right;
            pmf_add.left_tail_budget = left;
        }
        self
    }

    /// Override the max grid size on all contained PMFs.
    ///
    /// After `compose()` or `self_compose()`, the result PMF is automatically
    /// coarsened if it exceeds this limit. Use `usize::MAX` to disable
    /// post-composition coarsening.
    pub fn with_max_grid_size(&self, max_grid_size: usize) -> Self {
        Self {
            pmf_remove: self.pmf_remove.with_max_grid_size(max_grid_size),
            pmf_add: self
                .pmf_add
                .as_ref()
                .map(|p| p.with_max_grid_size(max_grid_size)),
            estimation_failure_probability: self.estimation_failure_probability,
            mc_resolution: self.mc_resolution,
            gaussian_source: self.gaussian_source,
        }
    }

    /// Get delta for a given epsilon
    ///
    /// Computes the minimum δ such that the mechanism is (ε, δ)-DP.
    /// For asymmetric mechanisms, this is the maximum over both adjacency types.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The privacy parameter ε
    ///
    /// # Returns
    ///
    /// The corresponding δ value (worst case over adjacencies)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    /// let delta = pld.delta_at(1.0);
    /// assert!(delta >= 0.0 && delta <= 1.0);
    /// ```
    pub fn delta_at(&self, epsilon: f64) -> f64 {
        metrics::delta(self, epsilon).clamp(0.0, 1.0)
    }

    /// Worst-case infinity mass over both adjacency types.
    ///
    /// `delta_at(ε)` is always ≥ `infinity_mass`; if the two values are equal,
    /// the configured tail truncation budget has been exhausted and the true δ
    /// may be smaller than what is returned.
    pub fn infinity_mass(&self) -> f64 {
        metrics::infinity_mass(self)
    }

    /// Get epsilon for a given delta
    ///
    /// Computes the minimum ε such that the mechanism is (ε, δ)-DP.
    /// For asymmetric mechanisms, this is the maximum over both adjacency types.
    ///
    /// # Arguments
    ///
    /// * `delta` - The privacy parameter δ
    ///
    /// # Returns
    ///
    /// The corresponding ε value (worst case over adjacencies)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    /// let epsilon = pld.epsilon_at(1e-5);
    /// assert!(epsilon >= 0.0);
    /// ```
    pub fn epsilon_at(&self, delta: f64) -> f64 {
        if self.estimation_failure_probability > 0.0 && self.infinity_mass() >= delta {
            return f64::INFINITY;
        }
        metrics::epsilon(self, delta)
    }

    /// Compute the advantage (TV privacy) directly from the PLD
    ///
    /// The advantage represents the maximum discriminative power of any attacker,
    /// defined as max(TPR - FPR) over all possible thresholds.
    ///
    /// This computes directly from the privacy loss distribution without
    /// going through (ε,δ) approximation, giving more accurate results.
    ///
    /// For asymmetric mechanisms, returns the worst-case (maximum) advantage
    /// over both adjacency types.
    ///
    /// # Returns
    ///
    /// The advantage value in [0, 1], where:
    /// - 0 means perfect privacy (attacker does no better than random guessing)
    /// - 1 means no privacy (attacker can perfectly distinguish)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    /// let advantage = pld.advantage();
    /// assert!(advantage >= 0.0 && advantage <= 1.0);
    /// ```
    pub fn advantage(&self) -> f64 {
        metrics::advantage(self).clamp(0.0, 1.0)
    }

    /// Compute the trade-off function β(α) directly from the PLD
    ///
    /// Maps false positive rate (α) to false negative rate (β) for the
    /// optimal hypothesis test distinguishing neighboring datasets.
    ///
    /// This computes directly from the privacy loss distribution without
    /// going through (ε,δ) approximation, giving more accurate results.
    ///
    /// For asymmetric mechanisms, returns the worst-case (minimum) β
    /// over both adjacency types (lower β means attacker is more powerful).
    ///
    /// # Arguments
    ///
    /// * `alpha` - False positive rate (Type I error rate) in [0, 1]
    ///
    /// # Returns
    ///
    /// The false negative rate β in [0, 1]
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    ///
    /// // At 1% false positive rate, what's the false negative rate?
    /// let beta = pld.beta_at(0.01);
    /// assert!(beta >= 0.0 && beta <= 1.0);
    /// ```
    pub fn beta_at(&self, target_alpha: f64) -> f64 {
        if self.estimation_failure_probability > 0.0 {
            return 0.0;
        }
        metrics::beta(self, target_alpha).clamp(0.0, 1.0)
    }

    /// Compute the Bayes risk for a given prior probability
    ///
    /// Bayes risk measures the maximum accuracy of an attack against privacy
    /// of a single record under a binary prior (e.g., accuracy of attribute
    /// inference attacks).
    ///
    /// # Algorithm
    ///
    /// Bayes risk is computed as:
    /// ```text
    /// Bayes risk = min over α: prior * α + (1 - prior) * β(α)
    /// ```
    ///
    /// This uses Brent's method for bounded scalar optimization over α ∈ [0, 1].
    ///
    /// # Arguments
    ///
    /// * `prior` - Prior probability that the sensitive attribute takes value 1,
    ///   must be in [0, 1]
    ///
    /// # Returns
    ///
    /// The Bayes risk value in [0, 0.5], where:
    /// - 0 means no privacy (attacker achieves perfect accuracy)
    /// - 0.5 means perfect privacy (attacker does no better than random guessing)
    ///
    /// # References
    ///
    /// Kulynych et al. (2025), Proposition D.1. <https://arxiv.org/abs/2507.06969>
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    ///
    /// // With uniform prior (50% probability for each value)
    /// let risk = pld.risk_at(0.5);
    /// assert!(risk >= 0.0 && risk <= 0.5);
    /// ```
    pub fn risk_at(&self, prior: f64) -> f64 {
        if self.estimation_failure_probability > 0.0 {
            return 0.0;
        }
        let max_risk = prior.min(1.0 - prior);
        metrics::bayes_risk(self, prior).clamp(0.0, max_risk)
    }

    /// Compose two privacy loss distributions
    ///
    /// Computes the PLD of the composition of two mechanisms.
    /// Handles all combinations of symmetric and asymmetric mechanisms:
    ///
    /// - Symmetric + Symmetric → Symmetric result
    /// - Symmetric + Asymmetric → Asymmetric result
    /// - Asymmetric + Asymmetric → Asymmetric result
    ///
    /// # Arguments
    ///
    /// * `other` - The PLD to compose with
    ///
    /// # Returns
    ///
    /// A new `PrivacyLossDistribution` representing the composition
    ///
    /// # Errors
    ///
    /// Returns an error if PMFs have incompatible discretizations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld1 = gaussian(1.1).pld()?;
    /// let pld2 = gaussian(0.8).pld()?;
    ///
    /// let composed = pld1.compose(&pld2)?;
    /// // Composition increases privacy loss
    /// assert!(composed.delta_at(1.0) >= pld1.delta_at(1.0));
    /// ```
    pub fn compose(&self, other: &Self) -> Result<Self> {
        // Compose REMOVE adjacency PMFs
        let pmf_remove = self
            .pmf_remove
            .clone()
            .compose(other.pmf_remove.clone(), 0.0)?;

        // Handle ADD adjacency based on symmetry
        let pmf_add = match (&self.pmf_add, &other.pmf_add) {
            // Both symmetric → result is symmetric
            (None, None) => None,

            // Self symmetric, other asymmetric → use self.pmf_remove (symmetric)
            (None, Some(other_add)) => {
                let composed = self.pmf_remove.clone().compose(other_add.clone(), 0.0)?;
                Some(composed)
            }

            // Self asymmetric, other symmetric → use other.pmf_remove (symmetric)
            (Some(self_add), None) => {
                let composed = self_add.clone().compose(other.pmf_remove.clone(), 0.0)?;
                Some(composed)
            }

            // Both asymmetric → compose both ADD PMFs
            (Some(self_add), Some(other_add)) => {
                let composed = self_add.clone().compose(other_add.clone(), 0.0)?;
                Some(composed)
            }
        };

        Ok(Self {
            pmf_remove,
            pmf_add,
            estimation_failure_probability: (self.estimation_failure_probability
                + other.estimation_failure_probability)
                .min(1.0),
            mc_resolution: self.mc_resolution + other.mc_resolution
                - self.mc_resolution * other.mc_resolution,
            gaussian_source: None,
        })
    }

    /// Self-compose this PLD multiple times
    ///
    /// Efficiently computes the result of composing this PLD with itself `count` times.
    /// Uses the efficient FFT power method for dense PMFs.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of times to self-compose (must be >= 1)
    ///
    /// # Returns
    ///
    /// A new `PrivacyLossDistribution` representing the `count`-fold composition
    ///
    /// # Errors
    ///
    /// Returns an error if exact composition would exceed the bounded FFT size.
    /// Set a positive tail-mass truncation budget to permit a bounded circular
    /// fallback, or use a coarser grid.
    ///
    /// # Panics
    ///
    /// Panics if `count` is 0.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use opake_accounting::mechanisms::gaussian_pld;
    ///
    /// let pld = gaussian(1.1).pld()?;
    ///
    /// // Compose with itself 100 times (e.g., 100 training steps)
    /// let composed = pld.self_compose(100)?;
    /// let epsilon = composed.epsilon_at(1e-5);
    /// ```
    pub fn self_compose(&self, count: usize) -> Result<Self> {
        let pmf_remove = self.pmf_remove.clone().self_compose(count)?;

        let pmf_add = self
            .pmf_add
            .as_ref()
            .map(|pmf| pmf.clone().self_compose(count))
            .transpose()?;

        Ok(Self {
            pmf_remove,
            pmf_add,
            // The same estimated PLD is reused, so self-composition does not
            // spend the statistical confidence event repeatedly.
            estimation_failure_probability: self.estimation_failure_probability,
            mc_resolution: if self.mc_resolution == 0.0 {
                0.0
            } else {
                -(count as f64 * (-self.mc_resolution).ln_1p()).exp_m1()
            },
            gaussian_source: None,
        })
    }

    /// Compose two PLDs with an explicit `max_grid_size` override.
    ///
    /// Same as `compose()` but overrides the max grid size used for
    /// post-composition coarsening.
    pub fn compose_with_max_grid_size(&self, other: &Self, max_grid_size: usize) -> Result<Self> {
        let lhs = Self {
            pmf_remove: self.pmf_remove.with_max_grid_size(max_grid_size),
            pmf_add: self
                .pmf_add
                .as_ref()
                .map(|p| p.with_max_grid_size(max_grid_size)),
            estimation_failure_probability: self.estimation_failure_probability,
            mc_resolution: self.mc_resolution,
            gaussian_source: self.gaussian_source,
        };
        let rhs = Self {
            pmf_remove: other.pmf_remove.with_max_grid_size(max_grid_size),
            pmf_add: other
                .pmf_add
                .as_ref()
                .map(|p| p.with_max_grid_size(max_grid_size)),
            estimation_failure_probability: other.estimation_failure_probability,
            mc_resolution: other.mc_resolution,
            gaussian_source: other.gaussian_source,
        };
        lhs.compose(&rhs)
    }

    /// Self-compose with an explicit `max_grid_size` override.
    ///
    /// Same as `self_compose()` but overrides the max grid size used for
    /// post-composition coarsening.
    pub fn self_compose_with_max_grid_size(
        &self,
        count: usize,
        max_grid_size: usize,
    ) -> Result<Self> {
        let pmf_remove = self
            .pmf_remove
            .clone()
            .self_compose_with_max_grid_size(count, max_grid_size)?;

        let pmf_add = self
            .pmf_add
            .as_ref()
            .map(|pmf| {
                pmf.clone()
                    .self_compose_with_max_grid_size(count, max_grid_size)
            })
            .transpose()?;

        Ok(Self {
            pmf_remove,
            pmf_add,
            estimation_failure_probability: self.estimation_failure_probability,
            mc_resolution: if self.mc_resolution == 0.0 {
                0.0
            } else {
                -(count as f64 * (-self.mc_resolution).ln_1p()).exp_m1()
            },
            gaussian_source: None,
        })
    }
}

// Note: In the functional API, PrivacyLossDistribution is a pure data structure.
// It does not implement the old Mechanism trait, which is specific to the legacy API.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn create_test_pmf(offset: i64, prob1: f64, prob2: f64) -> Pmf {
        let mut masses = BTreeMap::new();
        masses.insert(offset, prob1);
        masses.insert(offset + 5, prob2);
        Pmf::from_sparse(0.1, masses, 0.0, usize::MAX)
    }

    #[test]
    fn test_symmetric_pld_creation() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        assert!(pld.is_symmetric());
    }

    #[test]
    fn test_asymmetric_pld_creation() {
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        assert!(!pld.is_symmetric());
    }

    #[test]
    fn test_symmetric_same_pmf_both_adjacencies() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // For symmetric PLD, pmf_add should be None (pmf_remove used for both)
        assert!(pld.pmf_add.is_none());

        // Wrapping the single PMF in two PLDs should give identical deltas
        let epsilon = 1.0;
        let pld_r = PrivacyLossDistribution::new_symmetric(pld.pmf_remove.clone());
        let delta = pld_r.delta_at(epsilon);
        assert!(delta >= 0.0 && delta <= 1.0);
    }

    #[test]
    fn test_asymmetric_has_both_pmfs() {
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        // Asymmetric PLD should have both PMFs
        assert!(pld.pmf_add.is_some());

        // Verify both produce valid deltas
        let epsilon = 1.0;
        let pld_r = PrivacyLossDistribution::new_symmetric(pld.pmf_remove.clone());
        let pld_a = PrivacyLossDistribution::new_symmetric(pld.pmf_add.unwrap().clone());
        let delta_remove = pld_r.delta_at(epsilon);
        let delta_add = pld_a.delta_at(epsilon);

        assert!(delta_remove >= 0.0 && delta_remove <= 1.0);
        assert!(delta_add >= 0.0 && delta_add <= 1.0);
    }

    #[test]
    fn test_symmetric_delta_query() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf.clone());
        let pld_pmf = PrivacyLossDistribution::new_symmetric(pmf);

        let epsilon = 1.0;
        let delta_pld = pld.delta_at(epsilon);
        let delta_pmf = pld_pmf.delta_at(epsilon);

        // For symmetric PLD, result should match underlying PMF
        assert!((delta_pld - delta_pmf).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_delta_query_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());
        let pld_remove = PrivacyLossDistribution::new_symmetric(pmf_remove);
        let pld_add = PrivacyLossDistribution::new_symmetric(pmf_add);

        let epsilon = 0.5;
        let delta_pld = pld.delta_at(epsilon);
        let delta_remove = pld_remove.delta_at(epsilon);
        let delta_add = pld_add.delta_at(epsilon);

        // PLD should return the maximum
        let expected_max = delta_remove.max(delta_add);
        assert!((delta_pld - expected_max).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_epsilon_query() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf.clone());
        let pld_pmf = PrivacyLossDistribution::new_symmetric(pmf);

        let delta = 0.1;
        let epsilon_pld = pld.epsilon_at(delta);
        let epsilon_pmf = pld_pmf.epsilon_at(delta);

        // For symmetric PLD, result should match underlying PMF
        assert!((epsilon_pld - epsilon_pmf).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_epsilon_query_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());
        let pld_remove = PrivacyLossDistribution::new_symmetric(pmf_remove);
        let pld_add = PrivacyLossDistribution::new_symmetric(pmf_add);

        let delta = 0.1;
        let epsilon_pld = pld.epsilon_at(delta);
        let epsilon_remove = pld_remove.epsilon_at(delta);
        let epsilon_add = pld_add.epsilon_at(delta);

        // PLD should return the maximum
        let expected_max = epsilon_remove.max(epsilon_add);
        assert!((epsilon_pld - expected_max).abs() < 1e-10);
    }

    #[test]
    fn test_compose_symmetric_symmetric() {
        let pmf1 = create_test_pmf(0, 0.5, 0.5);
        let pmf2 = create_test_pmf(0, 0.6, 0.4);

        let pld1 = PrivacyLossDistribution::new_symmetric(pmf1);
        let pld2 = PrivacyLossDistribution::new_symmetric(pmf2);

        let composed = pld1.compose(&pld2).unwrap();

        // Symmetric + Symmetric = Symmetric
        assert!(composed.is_symmetric());
    }

    #[test]
    fn test_compose_symmetric_asymmetric() {
        let pmf_sym = create_test_pmf(0, 0.5, 0.5);
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);

        let pld_sym = PrivacyLossDistribution::new_symmetric(pmf_sym);
        let pld_asym = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        let composed = pld_sym.compose(&pld_asym).unwrap();

        // Symmetric + Asymmetric = Asymmetric
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_compose_asymmetric_asymmetric() {
        let pmf_remove1 = create_test_pmf(0, 0.2, 0.8);
        let pmf_add1 = create_test_pmf(0, 0.3, 0.7);
        let pmf_remove2 = create_test_pmf(0, 0.4, 0.6);
        let pmf_add2 = create_test_pmf(0, 0.5, 0.5);

        let pld1 = PrivacyLossDistribution::new_asymmetric(pmf_remove1, pmf_add1);
        let pld2 = PrivacyLossDistribution::new_asymmetric(pmf_remove2, pmf_add2);

        let composed = pld1.compose(&pld2).unwrap();

        // Asymmetric + Asymmetric = Asymmetric
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_symmetric() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let composed = pld.self_compose(5).unwrap();

        // Self-compose preserves symmetry
        assert!(composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_asymmetric() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        let composed = pld.self_compose(3).unwrap();

        // Self-compose preserves asymmetry
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_count_one() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let composed = pld.self_compose(1).unwrap();

        // Self-compose with count=1 should be equivalent
        let epsilon = 1.0;
        let delta_original = pld.delta_at(epsilon);
        let delta_composed = composed.delta_at(epsilon);

        assert!((delta_original - delta_composed).abs() < 1e-10);
    }

    #[test]
    fn test_compose_commutativity() {
        let pmf1 = create_test_pmf(0, 0.3, 0.7);
        let pmf2 = create_test_pmf(0, 0.4, 0.6);

        let pld1 = PrivacyLossDistribution::new_symmetric(pmf1);
        let pld2 = PrivacyLossDistribution::new_symmetric(pmf2);

        let composed_12 = pld1.compose(&pld2).unwrap();
        let composed_21 = pld2.compose(&pld1).unwrap();

        // Composition should be commutative
        let epsilon = 1.0;
        let delta_12 = composed_12.delta_at(epsilon);
        let delta_21 = composed_21.delta_at(epsilon);

        assert!((delta_12 - delta_21).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_risk_symmetric() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // Bayes risk with uniform prior
        let risk = pld.risk_at(0.5);

        // Should be in valid range [0, 0.5]
        assert!(risk >= 0.0);
        assert!(risk <= 0.5);

        // Higher privacy loss means lower Bayes risk (worse privacy)
        // With significant privacy loss, risk should be noticeably below 0.5
        assert!(risk < 0.5);
    }

    #[test]
    fn test_bayes_risk_edge_cases() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // Prior = 0: attacker knows with certainty, risk = 0
        let risk_0 = pld.risk_at(0.0);
        assert!((risk_0 - 0.0).abs() < 1e-10);

        // Prior = 1: attacker knows with certainty, risk = 0
        let risk_1 = pld.risk_at(1.0);
        assert!((risk_1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_risk_symmetry_in_prior() {
        let pmf = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // For symmetric PLDs, Bayes risk should be symmetric around prior = 0.5
        let risk_03 = pld.risk_at(0.3);
        let risk_07 = pld.risk_at(0.7);

        // Should be equal (within tolerance)
        assert!((risk_03 - risk_07).abs() < 1e-6);
    }

    #[test]
    fn test_bayes_risk_monotonicity() {
        // As privacy improves (lower epsilon), Bayes risk should increase toward 0.5
        let pmf_low_privacy = create_test_pmf(0, 0.1, 0.9); // High privacy loss
        let pmf_high_privacy = create_test_pmf(0, 0.5, 0.5); // Lower privacy loss

        let pld_low = PrivacyLossDistribution::new_symmetric(pmf_low_privacy);
        let pld_high = PrivacyLossDistribution::new_symmetric(pmf_high_privacy);

        let risk_low = pld_low.risk_at(0.5);
        let risk_high = pld_high.risk_at(0.5);

        // Lower privacy (higher epsilon) means lower Bayes risk
        assert!(risk_low < risk_high);
    }

    // -----------------------------------------------------------------
    // Identity PLD (all mass at loss=0)
    // -----------------------------------------------------------------

    fn make_identity_pld() -> PrivacyLossDistribution {
        let mut masses = BTreeMap::new();
        masses.insert(0, 1.0);
        let pmf = Pmf::from_sparse(1e-4, masses, 0.0, usize::MAX);
        PrivacyLossDistribution::new_symmetric(pmf)
    }

    #[test]
    fn test_monte_carlo_metadata_composes_conservatively() {
        let pld = make_identity_pld().with_monte_carlo_guarantee(0.01, 0.1);

        let repeated = pld.self_compose(2).unwrap();
        assert!((repeated.estimation_failure_probability() - 0.01).abs() < 1e-15);
        assert!((repeated.mc_resolution() - 0.19).abs() < 1e-15);

        let composed = pld.compose(&pld).unwrap();
        assert!((composed.estimation_failure_probability() - 0.02).abs() < 1e-15);
        assert!((composed.mc_resolution() - 0.19).abs() < 1e-15);
    }

    #[test]
    fn test_identity_epsilon_is_zero() {
        let pld = make_identity_pld();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_is_zero() {
        let pld = make_identity_pld();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_is_zero() {
        let pld = make_identity_pld();
        assert_eq!(pld.advantage(), 0.0);
    }

    #[test]
    fn test_identity_beta_is_one_minus_alpha() {
        let pld = make_identity_pld();
        for &alpha in &[0.0, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let beta = pld.beta_at(alpha);
            assert!(
                (beta - (1.0 - alpha)).abs() < 1e-9,
                "alpha={}, beta={}, expected={}",
                alpha,
                beta,
                1.0 - alpha
            );
        }
    }

    #[test]
    fn test_identity_risk_is_min_prior() {
        let pld = make_identity_pld();
        for &prior in &[0.1, 0.3, 0.5] {
            let risk = pld.risk_at(prior);
            let expected = prior.min(1.0 - prior);
            assert!(
                (risk - expected).abs() < 1e-6,
                "prior={}, risk={}, expected={}",
                prior,
                risk,
                expected
            );
        }
    }

    #[test]
    fn test_identity_is_symmetric() {
        let pld = make_identity_pld();
        assert!(pld.is_symmetric());
    }

    // -----------------------------------------------------------------
    // Advantage
    // -----------------------------------------------------------------

    #[test]
    fn test_advantage_equals_delta_at_zero() {
        // advantage() should equal delta_at(0.0) by definition
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        assert!((pld.advantage() - pld.delta_at(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_advantage_asymmetric_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());

        let adv_remove = PrivacyLossDistribution::new_symmetric(pmf_remove).advantage();
        let adv_add = PrivacyLossDistribution::new_symmetric(pmf_add).advantage();

        assert!((pld.advantage() - adv_remove.max(adv_add)).abs() < 1e-10);
    }

    #[test]
    fn test_advantage_in_valid_range() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        let adv = pld.advantage();
        assert!(adv >= 0.0 && adv <= 1.0, "advantage={}", adv);
    }

    // -----------------------------------------------------------------
    // Beta (trade-off function)
    // -----------------------------------------------------------------

    #[test]
    fn test_beta_at_boundary_alpha_zero() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        // At alpha=0 (no false positives), beta should be 1-infinity_mass
        // For our test PMF with infinity_mass=0, beta(0) = 1.0
        let beta = pld.beta_at(0.0);
        assert!((beta - 1.0).abs() < 1e-9, "beta(0)={}", beta);
    }

    #[test]
    fn test_beta_at_boundary_alpha_one() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        // At alpha=1, beta should be 0
        let beta = pld.beta_at(1.0);
        assert!((beta - 0.0).abs() < 1e-9, "beta(1)={}", beta);
    }

    #[test]
    fn test_beta_decreases_with_alpha() {
        // Trade-off function: as false positive rate increases,
        // false negative rate should decrease (or stay equal)
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let alphas = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
        let mut prev_beta = f64::INFINITY;
        for &alpha in &alphas {
            let beta = pld.beta_at(alpha);
            assert!(
                beta <= prev_beta + 1e-10,
                "beta should decrease: alpha={}, beta={}, prev={}",
                alpha,
                beta,
                prev_beta
            );
            prev_beta = beta;
        }
    }

    // -----------------------------------------------------------------
    // Composition increases privacy cost
    // -----------------------------------------------------------------

    #[test]
    fn test_composition_increases_delta() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let epsilon = 0.5;
        let delta_1 = pld.delta_at(epsilon);
        let delta_2 = pld.compose(&pld).unwrap().delta_at(epsilon);
        let delta_5 = pld.self_compose(5).unwrap().delta_at(epsilon);

        assert!(
            delta_2 >= delta_1,
            "delta_2={} < delta_1={}",
            delta_2,
            delta_1
        );
        assert!(
            delta_5 >= delta_2,
            "delta_5={} < delta_2={}",
            delta_5,
            delta_2
        );
    }

    #[test]
    fn test_composition_decreases_beta() {
        // More compositions = weaker privacy = lower beta for same alpha
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let alpha = 0.1;
        let beta_1 = pld.beta_at(alpha);
        let beta_5 = pld.self_compose(5).unwrap().beta_at(alpha);

        assert!(beta_5 <= beta_1, "beta_5={} > beta_1={}", beta_5, beta_1);
    }
}
