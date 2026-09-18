//! Discretization configuration and epsilon bounds
//!
//! Controls the accuracy-performance tradeoff when approximating continuous privacy
//! loss distributions with discrete probability mass functions.

use crate::error::{PldError, Result};

/// Configuration for Connect-the-Dots PLD discretization
///
/// Controls the accuracy-performance tradeoff when approximating continuous privacy
/// loss distributions with discrete probability mass functions.
///
/// # Defaults
///
/// The default configuration (`DiscretizationConfig::default()`) uses:
/// discretization = 1e-4, log_mass_truncation_bound = -50.0.
///
/// The truncation bound -50 matches Google dp_accounting's default, ensuring
/// Poisson-subsampled mechanisms produce identical epsilon bounds and grid
/// sizes for accurate beta computation after FFT-based composition.
///
/// # References
///
/// See Doroshenko et al. (2022) for algorithm details and truncation analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscretizationConfig {
    /// Grid spacing Δε between consecutive epsilon values
    ///
    /// Smaller values give more accurate approximations but require more computation.
    /// Typical range: 1e-5 (very accurate) to 1e-3 (coarse). Default: 1e-4.
    ///
    /// The grid will have approximately (ε_upper - ε_lower) / discretization points.
    /// If this exceeds `max_grid_size`, the effective discretization is automatically
    /// coarsened to `discretization * 2^k` for the smallest k that fits.
    pub discretization: f64,

    /// Log of probability mass to truncate from distribution tails
    ///
    /// The PLD is computed only for outcomes with tail probability ≥ exp(log_mass_truncation_bound).
    /// More negative values give higher accuracy but larger grids.
    ///
    /// Example: -50 means truncate mass < e^{-50} ≈ 1.9×10^{-22}. Default: -50.0.
    pub log_mass_truncation_bound: f64,

    /// Maximum number of grid points allowed before adaptive coarsening kicks in
    ///
    /// When the epsilon range would produce more than `max_grid_size` points at the
    /// base `discretization`, the effective discretization is automatically coarsened
    /// to `discretization * 2^k` (smallest power-of-2 multiplier that fits).
    ///
    /// Default: 10,000,000. Set to `usize::MAX` to disable adaptive coarsening.
    pub max_grid_size: usize,

    /// Total tail mass budget for Chernoff truncation during composition.
    ///
    /// During `self_compose()`, the composed PLD is truncated using Chernoff bounds
    /// with this total budget split equally between left and right tails.
    /// Smaller values preserve more tail precision at the cost of larger composed grids.
    ///
    /// Default: 1e-15, matching Google dp_accounting's `tail_mass_truncation`.
    pub tail_mass_truncation: f64,

    /// Maximum interior grid for the random-allocation convolution transform.
    ///
    /// The random-allocation PLD transform (`random_allocation_gaussian_pld` and
    /// related functions) operates on a geometric grid. Its convolution step is
    /// O(G²) rather than O(G log G), so it uses a separate budget from the
    /// FFT-based composition grid (`max_grid_size`).
    ///
    /// The ε interval width follows W ≈ c/G; the cost grows as G².
    /// Doubling this cap halves the interval at ~4× the convolution cost for
    /// configurations where the cap binds. A finer grid can only tighten (or
    /// leave unchanged) the upper bound — it never loosens it.
    ///
    /// Default: 32 768. Set higher (e.g. 65 536) for tighter bounds at
    /// production depth (large t), or lower for faster exploratory runs.
    pub max_conv_grid: usize,

    /// RNG seed for reproducible Monte Carlo PLDs across Rayon worker counts
    /// within a fixed Opake and dependency build. Exact streams may change
    /// across releases.
    ///
    /// Default: 42.
    pub seed: u64,

    /// Maximum unresolved Monte Carlo probability mass.
    ///
    /// This has the same units as DP delta. The confidence-bounded empirical
    /// PLD moves at most this mass to +infinity. Default: 1e-5.
    pub mc_resolution: f64,

    /// Failure probability of the simultaneous Monte Carlo confidence band.
    ///
    /// This is statistical confidence metadata, not mechanism delta. Default:
    /// 1e-6.
    pub mc_failure_probability: f64,
}

impl DiscretizationConfig {
    /// Create a new conservative discretization configuration.
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If discretization ≤ 0 or log_mass_truncation_bound ≥ 0
    pub fn new(discretization: f64, log_mass_truncation_bound: f64) -> Result<Self> {
        if discretization <= 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "Discretization must be positive, got {}",
                discretization
            )));
        }
        if log_mass_truncation_bound >= 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "Log mass truncation bound must be negative, got {}",
                log_mass_truncation_bound
            )));
        }

        Ok(Self {
            discretization,
            log_mass_truncation_bound,
            max_grid_size: 10_000_000,
            tail_mass_truncation: 1e-15,
            max_conv_grid: 32_768,
            seed: 42,
            mc_resolution: 1e-5,
            mc_failure_probability: 1e-6,
        })
    }

    /// Builder method to override the maximum grid size
    pub fn with_max_grid_size(mut self, max_grid_size: usize) -> Self {
        self.max_grid_size = max_grid_size;
        self
    }

    /// Builder method to override the random-allocation convolution grid cap.
    pub fn with_max_conv_grid(mut self, max_conv_grid: usize) -> Self {
        self.max_conv_grid = max_conv_grid;
        self
    }

    /// Validate Monte Carlo confidence parameters and resolve the sample count.
    ///
    /// `num_directions` is two for Opake's asymmetric add/remove accountants.
    /// The simultaneous band allocates failure probability across every order
    /// statistic and direction. The last order statistic has lower CDF bound
    /// `(failure / (num_directions * n))^(1/n)`, so its unresolved tail is
    /// `1 - (...)`; binary search finds the smallest `n` meeting the requested
    /// resolution.
    pub fn resolved_num_mc_samples(&self, num_directions: usize) -> Result<usize> {
        if num_directions == 0 {
            return Err(PldError::InvalidParameter(
                "num_directions must be > 0".into(),
            ));
        }
        self.validate_mc_parameters()?;

        let residual = |n: usize| {
            let per_rank = self.mc_failure_probability / (num_directions as f64 * n as f64);
            -(per_rank.ln() / n as f64).exp_m1()
        };

        let mut upper = 1usize;
        while residual(upper) > self.mc_resolution {
            upper = upper.checked_mul(2).ok_or_else(|| {
                PldError::InvalidParameter(format!(
                    "requested mc_resolution={} requires too many samples",
                    self.mc_resolution
                ))
            })?;
        }
        let mut lower = 1usize;
        while lower < upper {
            let midpoint = lower + (upper - lower) / 2;
            if residual(midpoint) <= self.mc_resolution {
                upper = midpoint;
            } else {
                lower = midpoint + 1;
            }
        }

        Ok(lower)
    }

    /// Validate statistical confidence parameters without resolving work.
    pub fn validate_mc_parameters(&self) -> Result<()> {
        if !self.mc_resolution.is_finite() || self.mc_resolution <= 0.0 || self.mc_resolution >= 1.0
        {
            return Err(PldError::InvalidParameter(format!(
                "mc_resolution must be finite and in (0, 1), got {}",
                self.mc_resolution
            )));
        }
        if !self.mc_failure_probability.is_finite()
            || self.mc_failure_probability <= 0.0
            || self.mc_failure_probability >= 1.0
        {
            return Err(PldError::InvalidParameter(format!(
                "mc_failure_probability must be finite and in (0, 1), got {}",
                self.mc_failure_probability
            )));
        }
        Ok(())
    }

    /// Compute the effective discretization adapted to the given epsilon bounds.
    ///
    /// If the grid at `self.discretization` would exceed `max_grid_size`, coarsen
    /// to `self.discretization * 2^k` for the smallest k that fits.
    pub(crate) fn effective_discretization(&self, bounds: &EpsilonBounds) -> f64 {
        let range = bounds.epsilon_upper - bounds.epsilon_lower;
        let grid_at_base = (range / self.discretization).ceil() as usize;
        if grid_at_base <= self.max_grid_size {
            return self.discretization;
        }
        let raw_ratio = (grid_at_base as f64 / self.max_grid_size as f64).ceil();
        let k = (raw_ratio.log2().ceil() as u32).max(1);
        self.discretization * 2_f64.powi(k as i32)
    }
}

impl Default for DiscretizationConfig {
    fn default() -> Self {
        Self {
            discretization: 1e-4,
            log_mass_truncation_bound: -50.0,
            max_grid_size: 10_000_000,
            tail_mass_truncation: 1e-15,
            max_conv_grid: 32_768,
            seed: 42,
            mc_resolution: 1e-5,
            mc_failure_probability: 1e-6,
        }
    }
}

/// Privacy loss bounds for PLD discretization
///
/// Specifies the range [ε_lower, ε_upper] over which the Privacy Loss Distribution
/// will be discretized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EpsilonBounds {
    /// Lower bound of the privacy loss range (minimum epsilon)
    pub epsilon_lower: f64,

    /// Upper bound of the privacy loss range (maximum epsilon)
    pub epsilon_upper: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretization_config_valid() {
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        assert_eq!(config.discretization, 0.01);
        assert_eq!(config.log_mass_truncation_bound, -50.0);
    }

    #[test]
    fn test_discretization_config_default() {
        let config = DiscretizationConfig::default();
        assert_eq!(config.discretization, 1e-4);
        assert_eq!(config.log_mass_truncation_bound, -50.0);
        assert_eq!(config.max_conv_grid, 32_768);
        assert_eq!(config.mc_resolution, 1e-5);
        assert_eq!(config.mc_failure_probability, 1e-6);
    }

    #[test]
    fn test_resolved_mc_samples_meets_resolution() {
        let config = DiscretizationConfig {
            mc_resolution: 1e-3,
            mc_failure_probability: 1e-4,
            ..DiscretizationConfig::default()
        };
        let n = config.resolved_num_mc_samples(2).unwrap();
        let per_rank = config.mc_failure_probability / (2.0 * n as f64);
        let residual = -(per_rank.ln() / n as f64).exp_m1();
        assert!(residual <= config.mc_resolution);
        if n > 1 {
            let previous = n - 1;
            let per_rank = config.mc_failure_probability / (2.0 * previous as f64);
            let residual = -(per_rank.ln() / previous as f64).exp_m1();
            assert!(residual > config.mc_resolution);
        }
    }

    #[test]
    fn test_discretization_config_invalid_discretization() {
        assert!(DiscretizationConfig::new(0.0, -50.0).is_err());
        assert!(DiscretizationConfig::new(-0.01, -50.0).is_err());
    }

    #[test]
    fn test_discretization_config_invalid_log_mass() {
        assert!(DiscretizationConfig::new(0.01, 0.0).is_err());
        assert!(DiscretizationConfig::new(0.01, 1.0).is_err());
    }

    #[test]
    fn test_effective_disc_no_coarsening_when_grid_fits() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -50.0,
            epsilon_upper: 50.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!((eff - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_effective_disc_coarsens_when_grid_exceeds_max() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -1600.0,
            epsilon_upper: 1200.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!(eff > 1e-4);
        let grid = ((bounds.epsilon_upper - bounds.epsilon_lower) / eff).ceil() as usize;
        assert!(grid <= 10_000_000);
    }

    #[test]
    fn test_effective_disc_is_power_of_2_multiple() {
        let config = DiscretizationConfig::default();
        let test_ranges = [
            (-1600.0, 1200.0),
            (-650.0, 550.0),
            (-281.0, 256.0),
            (-45.0, 41.0),
        ];
        for (lo, hi) in test_ranges {
            let bounds = EpsilonBounds {
                epsilon_lower: lo,
                epsilon_upper: hi,
            };
            let eff = config.effective_discretization(&bounds);
            let ratio = eff / config.discretization;
            let rounded = ratio.round();
            assert!((ratio - rounded).abs() < 1e-9);
            assert!((rounded as usize).is_power_of_two());
        }
    }

    #[test]
    fn test_effective_disc_unlimited_grid_never_coarsens() {
        let config = DiscretizationConfig::default().with_max_grid_size(usize::MAX);
        let bounds = EpsilonBounds {
            epsilon_lower: -20000.0,
            epsilon_upper: 10000.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!((eff - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_effective_disc_smallest_sufficient_power_of_2() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -1600.0,
            epsilon_upper: 1200.0,
        };
        let eff = config.effective_discretization(&bounds);
        let factor = (eff / config.discretization).round() as usize;
        assert_eq!(factor, 4);
    }
}
