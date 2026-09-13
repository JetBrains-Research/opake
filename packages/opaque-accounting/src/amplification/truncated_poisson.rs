//! Truncated Poisson-subsampled Gaussian mechanism PLD.

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{Binomial, DiscreteCDF};

use super::poisson::{poisson_gaussian_epsilon_bounds, poisson_gaussian_get_delta, poisson_pld};
use super::{validate_noise_multiplier, validate_rate};

/// Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.
///
/// This is the variant actually used in production DP-SGD systems. Unlike
/// standard Poisson (variable batch size), truncated sampling caps the batch
/// at `batch_size_max` for predictable memory/compute.
///
/// Uses Theorem 3.1 and the stable conditional-rate formula from section 3.4
/// of Ganesh (2025), "Tighter Privacy Analysis for Truncated Poisson Sampling"
/// <https://arxiv.org/abs/2508.15089>:
/// - Component 1 (prob `1 − p_trunc`): standard Poisson PLD
/// - Component 2 (prob `p_trunc`): Poisson with doubled sensitivity at conditional rate
///
/// The theorem uses directed, dataset-size-indexed add/remove adjacency:
/// the larger dataset has exactly `dataset_size` records and the smaller one
/// has `dataset_size - 1`. The cap must be applied by selecting a uniform
/// size-`batch_size_max` subset whenever the provisional Poisson sample exceeds
/// the cap.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0
/// * `rate` — Poisson sampling probability q ∈ (0, 1)
/// * `batch_size_max` — maximum batch size B_max > 0
/// * `dataset_size` — public larger-neighbor size n > 0
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn truncated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if batch_size_max == 0 {
        return Err(PldError::InvalidParameter(
            "batch_size_max must be > 0".into(),
        ));
    }
    if dataset_size == 0 {
        return Err(PldError::InvalidParameter(
            "dataset_size must be > 0".into(),
        ));
    }

    let sigma = noise_multiplier;
    let sensitivity = 1.0;
    let p_trunc = truncation_probability(dataset_size, rate, batch_size_max);

    // No truncation → fall back to plain Poisson.
    if p_trunc == 0.0 {
        let base = crate::mechanisms::gaussian_pld(noise_multiplier, config)?;
        return poisson_pld(&base, rate);
    }

    let q_cond = conditional_sampling_probability(dataset_size, rate, batch_size_max, p_trunc);
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    let bounds_remove = truncated_epsilon_bounds(
        sigma,
        sensitivity,
        rate,
        Adjacency::Remove,
        log_mass,
        q_cond,
    );
    let bounds_add =
        truncated_epsilon_bounds(sigma, sensitivity, rate, Adjacency::Add, log_mass, q_cond);

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(truncated_get_delta(
            epsilon,
            adj,
            sigma,
            sensitivity,
            rate,
            p_trunc,
            q_cond,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

// ===========================================================================
// Truncated Poisson math
// ===========================================================================

/// Probability that truncation occurs: Pr[Binom(n−1, q) ≥ B_max].
fn truncation_probability(dataset_size: usize, rate: f64, batch_size_max: usize) -> f64 {
    if batch_size_max >= dataset_size {
        return 0.0;
    }

    let binom = Binomial::new(rate, (dataset_size - 1) as u64).unwrap();
    1.0 - binom.cdf((batch_size_max - 1) as u64)
}

/// Conditional sampling probability for the truncated component.
fn conditional_sampling_probability(
    dataset_size: usize,
    rate: f64,
    batch_size_max: usize,
    p_trunc: f64,
) -> f64 {
    if p_trunc == 0.0 {
        return 0.0;
    }

    let n = dataset_size;
    let binom = Binomial::new(rate, n as u64).unwrap();
    let pr_exceed = 1.0 - binom.cdf(batch_size_max as u64);

    pr_exceed * (batch_size_max as f64) / (n as f64) / p_trunc
}

/// Hockey-stick divergence for the Ganesh (2025), Theorem 3.1 mixture.
fn truncated_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    p_trunc: f64,
    q_cond: f64,
) -> f64 {
    if p_trunc == 0.0 {
        return poisson_gaussian_get_delta(epsilon, adjacency, sigma, sensitivity, rate);
    }

    // Component 1: standard Poisson, pessimistic bound for ADD/REMOVE
    let delta_comp1 = match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let d_add =
                poisson_gaussian_get_delta(epsilon, Adjacency::Add, sigma, sensitivity, rate);
            let d_rem =
                poisson_gaussian_get_delta(epsilon, Adjacency::Remove, sigma, sensitivity, rate);
            d_add.max(d_rem)
        }
        Adjacency::Replace => {
            poisson_gaussian_get_delta(epsilon, Adjacency::Replace, sigma, sensitivity, rate)
        }
    };

    // Component 2: doubled sensitivity (σ/2), REPLACE adjacency, conditional rate
    let delta_comp2 = poisson_gaussian_get_delta(
        epsilon,
        Adjacency::Replace,
        sigma / 2.0,
        sensitivity,
        q_cond,
    );

    (1.0 - p_trunc) * delta_comp1 + p_trunc * delta_comp2
}

/// Epsilon bounds for truncated Poisson (union of component bounds).
fn truncated_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass_truncation_bound: f64,
    q_cond: f64,
) -> EpsilonBounds {
    // Component 1: standard Poisson, pessimistic max(ADD, REMOVE) for ADD/REMOVE
    let bounds1 = match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let b_add = poisson_gaussian_epsilon_bounds(
                sigma,
                sensitivity,
                rate,
                Adjacency::Add,
                log_mass_truncation_bound,
            );
            let b_rem = poisson_gaussian_epsilon_bounds(
                sigma,
                sensitivity,
                rate,
                Adjacency::Remove,
                log_mass_truncation_bound,
            );
            EpsilonBounds {
                epsilon_lower: b_add.epsilon_lower.min(b_rem.epsilon_lower),
                epsilon_upper: b_add.epsilon_upper.max(b_rem.epsilon_upper),
            }
        }
        Adjacency::Replace => poisson_gaussian_epsilon_bounds(
            sigma,
            sensitivity,
            rate,
            Adjacency::Replace,
            log_mass_truncation_bound,
        ),
    };

    // Component 2: Poisson REPLACE, σ/2 (doubled sensitivity), rate q_cond
    let bounds2 = poisson_gaussian_epsilon_bounds(
        sigma / 2.0,
        sensitivity,
        q_cond,
        Adjacency::Replace,
        log_mass_truncation_bound,
    );

    EpsilonBounds {
        epsilon_lower: bounds1.epsilon_lower.min(bounds2.epsilon_lower),
        epsilon_upper: bounds1.epsilon_upper.max(bounds2.epsilon_upper),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::Discrete;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_truncated_rejects_zero_batch_size() {
        assert!(truncated_poisson_gaussian_pld(0.5, 0.01, 0, 1000, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_rejects_zero_dataset_size() {
        assert!(truncated_poisson_gaussian_pld(0.5, 0.01, 100, 0, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_no_truncation_matches_poisson() {
        // batch_size_max ≥ dataset_size → no truncation
        let cfg = default_config();
        let base = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
        let pld_poisson = poisson_pld(&base, 0.01).unwrap();
        let pld_trunc = truncated_poisson_gaussian_pld(0.5, 0.01, 200, 100, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_t = pld_trunc.epsilon_at(1e-5);
        assert!(
            (eps_p - eps_t).abs() < 1e-6,
            "no-truncation should match Poisson: {} vs {}",
            eps_p,
            eps_t
        );
    }

    #[test]
    fn test_truncated_gives_higher_epsilon_than_poisson() {
        // Truncation is a pessimistic bound → more privacy loss
        let cfg = default_config();
        let base = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
        let pld_poisson = poisson_pld(&base, 0.01).unwrap();
        let pld_trunc = truncated_poisson_gaussian_pld(0.5, 0.01, 50, 100000, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_t = pld_trunc.epsilon_at(1e-5);
        assert!(
            eps_t >= eps_p - 1e-9,
            "truncated should have >= epsilon: {} vs {}",
            eps_t,
            eps_p
        );
    }

    // ---- Truncated math ----

    #[test]
    fn test_truncation_probability_no_truncation() {
        assert_eq!(truncation_probability(100, 0.1, 100), 0.0);
        assert_eq!(truncation_probability(100, 0.1, 200), 0.0);
    }

    #[test]
    fn test_truncation_probability_realistic() {
        let p = truncation_probability(1_000_000, 0.001, 1024);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_conditional_sampling_probability_zero_truncation() {
        assert_eq!(conditional_sampling_probability(100, 0.1, 200, 0.0), 0.0);
    }

    #[test]
    fn test_conditional_rate_matches_theorem_three_one() {
        let n = 12;
        let rate = 0.3;
        let cap = 4;
        let truncation = truncation_probability(n, rate, cap);
        let binom = Binomial::new(rate, (n - 1) as u64).unwrap();
        let direct = rate
            * (cap..n)
                .map(|s| binom.pmf(s as u64) * cap as f64 / (s + 1) as f64)
                .sum::<f64>()
            / truncation;

        let stable = conditional_sampling_probability(n, rate, cap, truncation);
        assert!((stable - direct).abs() < 1e-12);
    }
}
