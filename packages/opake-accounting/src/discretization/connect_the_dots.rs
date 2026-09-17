//! Connect-the-Dots discretization algorithm
//!
//! Implements the Connect-the-Dots algorithm for building discrete Privacy Loss
//! Distributions from continuous mechanisms.
//!
//! # References
//!
//! > Doroshenko, Ghazi, Kamath, Kumar, Manurangsi. "Connect the Dots: Tighter
//! > Discrete Approximations of Privacy Loss Distributions." PETS 2022.
//! > <https://arxiv.org/abs/2207.04380>

use super::config::{DiscretizationConfig, EpsilonBounds};
use crate::adjacency::Adjacency;
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global parallel threshold - can be changed for benchmarking
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(10_000);

/// Set the parallel threshold for discretization
///
/// Controls when to parallelize delta computation. Default: 10,000.
#[allow(dead_code)] // used in tests
pub fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// Get the current parallel threshold
pub fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

/// Build a PMF from epsilon bounds and delta values using Connect-the-Dots
///
/// This implements the core Connect-the-Dots algorithm from dp_accounting.
///
/// # Errors
///
/// * `PldError::InfiniteBounds` - If bounds are infinite
/// * `PldError::NumericalError` - If delta array size mismatches
pub(crate) fn discretize_from_deltas(
    bounds: EpsilonBounds,
    deltas: &[f64],
    config: &DiscretizationConfig,
    _adjacency: Adjacency,
) -> Result<Pmf> {
    if !bounds.epsilon_lower.is_finite() || !bounds.epsilon_upper.is_finite() {
        return Err(PldError::InfiniteBounds(format!(
            "Infinite epsilon bounds: [{}, {}]",
            bounds.epsilon_lower, bounds.epsilon_upper
        )));
    }

    let rounded_epsilon_upper = (bounds.epsilon_upper / config.discretization).ceil() as i64;
    let rounded_epsilon_lower = (bounds.epsilon_lower / config.discretization).floor() as i64;

    let expected_size = (rounded_epsilon_upper - rounded_epsilon_lower + 1) as usize;
    if deltas.len() != expected_size {
        return Err(PldError::NumericalError(format!(
            "Delta array size mismatch: got {}, expected {}",
            deltas.len(),
            expected_size
        )));
    }

    create_pmf_connect_the_dots_uniform(
        deltas,
        rounded_epsilon_lower,
        rounded_epsilon_upper,
        config.discretization,
        config.max_grid_size,
    )
}

/// Create PMF using Connect-the-Dots formula (fixed-gap variant)
///
/// Implements Algorithm 1 (PLD Discretization) for **uniform** epsilon grids.
fn create_pmf_connect_the_dots_uniform(
    deltas: &[f64],
    rounded_epsilon_lower: i64,
    rounded_epsilon_upper: i64,
    discretization: f64,
    max_grid_size: usize,
) -> Result<Pmf> {
    let n = deltas.len();

    debug_assert_eq!(
        n,
        (rounded_epsilon_upper - rounded_epsilon_lower + 1) as usize,
    );

    // Enforce monotonicity by taking cumulative minimum
    let mut deltas = deltas.to_vec();
    for i in 1..n {
        if deltas[i] > deltas[i - 1] {
            deltas[i] = deltas[i - 1];
        }
    }

    // Case 1: Single epsilon in support
    if n == 1 {
        let mut masses = BTreeMap::new();
        let prob = 1.0 - deltas[0];
        if prob > 0.0 {
            masses.insert(rounded_epsilon_lower, prob);
        }
        return Ok(Pmf::from_sparse(
            discretization,
            masses,
            deltas[0],
            max_grid_size,
        ));
    }

    // Case 2: Multiple epsilons (n >= 2)
    let d = discretization;
    let exp_d = d.exp();
    let expm1_d = d.exp_m1();
    let expm1_neg_d = (-d).exp_m1();

    let delta_diffs: Vec<f64> = (1..n).map(|i| deltas[i] - deltas[i - 1]).collect();

    let mut probs = vec![0.0; n];

    probs[0] = (1.0 - deltas[0]) + delta_diffs[0] / expm1_d;

    for i in 1..n - 1 {
        probs[i] = (delta_diffs[i] - exp_d * delta_diffs[i - 1]) / expm1_d;
    }

    probs[n - 1] = delta_diffs[n - 2] / expm1_neg_d;

    // Enforce non-negativity
    for p in &mut probs {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

    let infinity_mass = deltas[n - 1];

    // Renormalize after non-negativity clamping.
    //
    // In exact arithmetic, sum(probs) + infinity_mass == 1.0 by construction.
    // Clamping tiny negative probabilities to zero inflates total mass by
    // |sum of clamped negatives|. We scale the remaining probabilities down
    // to restore the invariant. The correction is typically O(1e-10) and
    // does not meaningfully affect any privacy computation.
    let target_prob_mass = 1.0 - infinity_mass;
    let actual_prob_mass: f64 = probs.iter().sum();
    if actual_prob_mass > 0.0 && actual_prob_mass != target_prob_mass {
        let scale = target_prob_mass / actual_prob_mass;
        for p in &mut probs {
            *p *= scale;
        }
    }

    let mut masses = BTreeMap::new();
    for (i, &prob) in probs.iter().enumerate() {
        if prob > 0.0 {
            let rounded_epsilon = rounded_epsilon_lower + i as i64;
            masses.insert(rounded_epsilon, prob);
        }
    }

    Ok(Pmf::from_sparse(
        discretization,
        masses,
        infinity_mass,
        max_grid_size,
    ))
}

/// Discretize a symmetric mechanism into a PLD
///
/// For symmetric mechanisms like Gaussian, the privacy loss distribution is identical
/// for ADD and REMOVE adjacencies.
#[allow(dead_code)]
pub(crate) fn discretize_symmetric_mechanism<F>(
    config: &DiscretizationConfig,
    bounds: EpsilonBounds,
    get_delta: F,
) -> Result<PrivacyLossDistribution>
where
    F: Fn(f64) -> f64 + Sync,
{
    let parallel_threshold = get_parallel_threshold();

    let effective_disc = config.effective_discretization(&bounds);
    let effective_config = DiscretizationConfig {
        discretization: effective_disc,
        ..config.clone()
    };

    let rounded_upper = (bounds.epsilon_upper / effective_disc).ceil() as i64;
    let rounded_lower = (bounds.epsilon_lower / effective_disc).floor() as i64;

    let epsilons: Vec<f64> = (rounded_lower..=rounded_upper)
        .map(|i| i as f64 * effective_disc)
        .collect();

    let deltas: Vec<f64> = if epsilons.len() >= parallel_threshold {
        epsilons.par_iter().map(|&eps| get_delta(eps)).collect()
    } else {
        epsilons.iter().map(|&eps| get_delta(eps)).collect()
    };

    let pmf = discretize_from_deltas(bounds, &deltas, &effective_config, Adjacency::Remove)?;

    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

/// Discretize an asymmetric mechanism into a PLD
///
/// For asymmetric mechanisms, the privacy loss distribution differs between
/// ADD and REMOVE adjacencies.
pub(crate) fn discretize_asymmetric_mechanism<F>(
    config: &DiscretizationConfig,
    bounds_remove: EpsilonBounds,
    bounds_add: EpsilonBounds,
    get_delta: F,
) -> Result<PrivacyLossDistribution>
where
    F: Fn(f64, Adjacency) -> Result<f64> + Sync,
{
    let parallel_threshold = get_parallel_threshold();

    let union_bounds = EpsilonBounds {
        epsilon_lower: bounds_remove.epsilon_lower.min(bounds_add.epsilon_lower),
        epsilon_upper: bounds_remove.epsilon_upper.max(bounds_add.epsilon_upper),
    };
    let effective_disc = config.effective_discretization(&union_bounds);
    let effective_config = DiscretizationConfig {
        discretization: effective_disc,
        ..config.clone()
    };

    let compute_deltas =
        |bounds: EpsilonBounds, adjacency: Adjacency| -> Result<(Vec<f64>, EpsilonBounds)> {
            let rounded_upper = (bounds.epsilon_upper / effective_disc).ceil() as i64;
            let rounded_lower = (bounds.epsilon_lower / effective_disc).floor() as i64;

            let epsilons: Vec<f64> = (rounded_lower..=rounded_upper)
                .map(|i| i as f64 * effective_disc)
                .collect();

            let deltas: Vec<f64> = if epsilons.len() >= parallel_threshold {
                epsilons
                    .par_iter()
                    .map(|&eps| get_delta(eps, adjacency))
                    .collect::<Result<Vec<_>>>()?
            } else {
                epsilons
                    .iter()
                    .map(|&eps| get_delta(eps, adjacency))
                    .collect::<Result<Vec<_>>>()?
            };

            Ok((deltas, bounds))
        };

    let (deltas_remove, bounds_remove) = compute_deltas(bounds_remove, Adjacency::Remove)?;
    let (deltas_add, bounds_add) = compute_deltas(bounds_add, Adjacency::Add)?;

    let pmf_remove = discretize_from_deltas(
        bounds_remove,
        &deltas_remove,
        &effective_config,
        Adjacency::Remove,
    )?;
    let pmf_add =
        discretize_from_deltas(bounds_add, &deltas_add, &effective_config, Adjacency::Add)?;

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_threshold_get_set() {
        let original = get_parallel_threshold();
        set_parallel_threshold(5000);
        assert_eq!(get_parallel_threshold(), 5000);
        set_parallel_threshold(original);
        assert_eq!(get_parallel_threshold(), original);
    }

    #[test]
    fn test_discretize_from_deltas_single_point() {
        let bounds = EpsilonBounds {
            epsilon_lower: 0.0,
            epsilon_upper: 0.0,
        };
        let deltas = vec![0.1];
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();

        let pmf = discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).unwrap();

        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        assert!(pld.delta_at(0.0) > 0.0);
    }

    #[test]
    fn test_discretize_from_deltas_invalid_bounds() {
        let bounds = EpsilonBounds {
            epsilon_lower: f64::NEG_INFINITY,
            epsilon_upper: 1.0,
        };
        let deltas = vec![0.1, 0.05];
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();

        assert!(discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).is_err());
    }

    #[test]
    fn test_discretize_from_deltas_size_mismatch() {
        let bounds = EpsilonBounds {
            epsilon_lower: 0.0,
            epsilon_upper: 0.1,
        };
        let deltas = vec![0.1]; // Wrong size
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();

        assert!(discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).is_err());
    }

    #[test]
    fn test_discretize_from_deltas_monotonic_input() {
        let bounds = EpsilonBounds {
            epsilon_lower: 0.0,
            epsilon_upper: 0.04,
        };
        let deltas = vec![0.5, 0.6, 0.4, 0.3, 0.2];
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();

        let pmf = discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).unwrap();
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        let delta = pld.delta_at(0.0);
        assert!(delta >= 0.0 && delta <= 1.0);
    }

    #[test]
    fn test_discretize_from_deltas_both_infinite_bounds() {
        let bounds = EpsilonBounds {
            epsilon_lower: f64::NEG_INFINITY,
            epsilon_upper: f64::INFINITY,
        };
        let deltas = vec![0.1];
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        assert!(discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).is_err());
    }

    /// Gaussian hockey-stick divergence for testing
    fn gaussian_delta(sigma: f64, eps: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();
        let d = normal.cdf(-eps / sigma + sigma / 2.0)
            - (eps).exp() * normal.cdf(-eps / sigma - sigma / 2.0);
        d.max(0.0)
    }

    fn gaussian_bounds(sigma: f64) -> EpsilonBounds {
        EpsilonBounds {
            epsilon_lower: -1.0 / (2.0 * sigma * sigma) - 5.0 * sigma,
            epsilon_upper: 1.0 / (2.0 * sigma * sigma) + 5.0 * sigma,
        }
    }

    #[test]
    fn test_connect_the_dots_probabilities_sum_to_one() {
        let sigma = 0.5;
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        let bounds = gaussian_bounds(sigma);

        let rounded_upper = (bounds.epsilon_upper / config.discretization).ceil() as i64;
        let rounded_lower = (bounds.epsilon_lower / config.discretization).floor() as i64;
        let n = (rounded_upper - rounded_lower + 1) as usize;

        let deltas: Vec<f64> = (0..n)
            .map(|i| {
                let eps = (rounded_lower + i as i64) as f64 * config.discretization;
                gaussian_delta(sigma, eps)
            })
            .collect();

        let pmf = discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).unwrap();

        let total: f64 = pmf.probs.iter().sum::<f64>() + pmf.infinity_mass;
        assert!((total - 1.0).abs() < 1e-6, "Total mass = {}", total);
    }

    #[test]
    fn test_connect_the_dots_all_probs_nonnegative() {
        let sigma = 0.3;
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        let bounds = gaussian_bounds(sigma);

        let rounded_upper = (bounds.epsilon_upper / config.discretization).ceil() as i64;
        let rounded_lower = (bounds.epsilon_lower / config.discretization).floor() as i64;
        let n = (rounded_upper - rounded_lower + 1) as usize;

        let deltas: Vec<f64> = (0..n)
            .map(|i| {
                let eps = (rounded_lower + i as i64) as f64 * config.discretization;
                gaussian_delta(sigma, eps)
            })
            .collect();

        let pmf = discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove).unwrap();

        for (i, &p) in pmf.probs.iter().enumerate() {
            assert!(p >= 0.0, "Negative prob at index {}: {}", i, p);
        }
        assert!(pmf.infinity_mass >= 0.0);
    }

    #[test]
    fn test_symmetric_mechanism_produces_symmetric_pld() {
        let config = DiscretizationConfig::new(0.01, -20.0).unwrap();
        let bounds = EpsilonBounds {
            epsilon_lower: -2.0,
            epsilon_upper: 3.0,
        };

        let pld =
            discretize_symmetric_mechanism(&config, bounds, |eps| (1.0 - eps).max(0.0).min(1.0))
                .unwrap();

        assert!(pld.is_symmetric());
        assert!(pld.pmf_add.is_none());
    }

    #[test]
    fn test_asymmetric_mechanism_produces_asymmetric_pld() {
        let config = DiscretizationConfig::new(0.01, -20.0).unwrap();
        let bounds_remove = EpsilonBounds {
            epsilon_lower: -2.0,
            epsilon_upper: 3.0,
        };
        let bounds_add = EpsilonBounds {
            epsilon_lower: -3.0,
            epsilon_upper: 2.0,
        };

        let pld = discretize_asymmetric_mechanism(&config, bounds_remove, bounds_add, |eps, _| {
            Ok((1.0 - eps).max(0.0).min(1.0))
        })
        .unwrap();

        assert!(!pld.is_symmetric());
        assert!(pld.pmf_add.is_some());
    }

    #[test]
    fn test_finer_discretization_gives_tighter_bounds() {
        let sigma = 0.8;
        let bounds = gaussian_bounds(sigma);

        let get_delta = |eps: f64| -> f64 { gaussian_delta(sigma, eps) };

        let config_coarse = DiscretizationConfig::new(0.1, -20.0).unwrap();
        let pld_coarse =
            discretize_symmetric_mechanism(&config_coarse, bounds, &get_delta).unwrap();

        let config_fine = DiscretizationConfig::new(0.001, -20.0).unwrap();
        let pld_fine = discretize_symmetric_mechanism(&config_fine, bounds, &get_delta).unwrap();

        let eps = 1.0 / (2.0 * sigma * sigma);
        let delta_continuous = get_delta(eps);
        let delta_coarse = pld_coarse.delta_at(eps);
        let delta_fine = pld_fine.delta_at(eps);

        let error_coarse = (delta_coarse - delta_continuous).abs();
        let error_fine = (delta_fine - delta_continuous).abs();
        assert!(error_fine <= error_coarse + 1e-10);
    }
}
