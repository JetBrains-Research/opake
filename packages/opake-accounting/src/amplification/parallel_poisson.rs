//! Parallel Poisson-subsampled Gaussian mechanism PLD.
//!
//! Models scenarios where multiple independent Poisson samples are summed before
//! adding noise once. Use cases: gradient accumulation, parallel workers.

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::numerics::logspace::log_sumexp;
use crate::numerics::special::gaussian_log_cdf;
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{ContinuousCDF, Normal};

use super::discrete_mixture::{binomial_log_probs, truncate_upper_tail};
use super::poisson::poisson_pld;
use super::{validate_noise_multiplier, validate_rate};

/// Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism.
///
/// Models summing `microbatches` independent Poisson-sampled batches with noise
/// added **once** after summation. Examples appear K ~ Binomial(m, q) times,
/// creating a Mixture of Gaussians.
///
/// **Use cases**:
/// - **Gradient accumulation**: Split batch, sum clipped gradients, add noise
/// - **Parallel workers**: Independent Poisson sampling on K workers, aggregate
///
/// This is mathematically different from composing `m` independent Poisson
/// steps: parallel composition adds noise once (less total noise, worse privacy)
/// but produces larger effective batch sizes for better ML utility.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0
/// * `rate` — Poisson sampling probability q ∈ (0, 1)
/// * `microbatches` — number of independent samples m > 0
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn parallel_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if microbatches == 0 {
        return Err(PldError::InvalidParameter(
            "microbatches must be > 0".into(),
        ));
    }
    // m=1 reduces to plain Poisson.
    if microbatches == 1 {
        let base = crate::mechanisms::gaussian_pld(noise_multiplier, config)?;
        return poisson_pld(&base, rate);
    }

    let sigma = noise_multiplier;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;
    let one_sided_x_tail_mass = 0.5 * log_mass.exp();

    let full_log_probs = binomial_log_probs(microbatches, rate);
    let (normalized_log_probs, head_mass, tail_mass) =
        truncate_upper_tail(&full_log_probs, one_sided_x_tail_mass);

    let c = MixtureConstants::from_log_probs(sigma, normalized_log_probs);

    let bounds_remove = mixture_gaussian_epsilon_bounds(Adjacency::Remove, &c, log_mass)?;
    let bounds_add = mixture_gaussian_epsilon_bounds(Adjacency::Add, &c, log_mass)?;

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        let delta_head = mixture_gaussian_get_delta(epsilon, adj, &c)?;
        Ok((head_mass * delta_head + tail_mass).clamp(0.0, 1.0))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

// ===========================================================================
// Accumulated (Mixture of Gaussians) math
// ===========================================================================

/// Precomputed constants for mixture Gaussian privacy loss computation.
struct MixtureConstants {
    log_probs: Vec<f64>,
    sensitivities: Vec<f64>,
    precomputed_remove: Vec<f64>,
    precomputed_add: Vec<f64>,
    sigma: f64,
    variance: f64,
    sampling_prob: f64,
}

impl MixtureConstants {
    fn from_log_probs(sigma: f64, log_probs: Vec<f64>) -> Self {
        let m = log_probs.len() - 1;
        let sensitivities: Vec<f64> = (0..=m).map(|k| k as f64).collect();
        Self::new(sigma, log_probs, sensitivities)
    }

    fn new(sigma: f64, log_probs: Vec<f64>, sensitivities: Vec<f64>) -> Self {
        assert_eq!(log_probs.len(), sensitivities.len());
        let variance = sigma * sigma;

        let precomputed_remove: Vec<f64> = log_probs
            .iter()
            .zip(&sensitivities)
            .map(|(&log_probability, &sensitivity)| {
                log_probability - 0.5 * sensitivity * sensitivity / variance
            })
            .collect();

        let precomputed_add: Vec<f64> = log_probs
            .iter()
            .zip(&sensitivities)
            .map(|(&log_probability, &sensitivity)| {
                log_probability - 0.5 * sensitivity * sensitivity / variance
            })
            .collect();

        let sampling_prob = log_probs
            .iter()
            .zip(&sensitivities)
            .filter(|(_, sensitivity)| **sensitivity > 0.0)
            .map(|(log_probability, _)| log_probability.exp())
            .sum::<f64>()
            .min(1.0);

        MixtureConstants {
            log_probs,
            sensitivities,
            precomputed_remove,
            precomputed_add,
            sigma,
            variance,
            sampling_prob,
        }
    }

    fn max_sensitivity(&self) -> f64 {
        self.sensitivities.iter().copied().fold(0.0, f64::max)
    }

    fn min_positive_sensitivity(&self) -> f64 {
        self.sensitivities
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Privacy loss at point x for mixture Gaussian.
fn mixture_privacy_loss(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let summands: Vec<f64> = match adj {
        Adjacency::Remove => c
            .sensitivities
            .iter()
            .zip(c.precomputed_remove.iter())
            .map(|(&s, &pre)| pre - s * x / c.variance)
            .collect(),
        Adjacency::Add => c
            .sensitivities
            .iter()
            .zip(c.precomputed_add.iter())
            .map(|(&s, &pre)| pre + s * x / c.variance)
            .collect(),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    };

    let lse = log_sumexp(&summands);
    Ok(match adj {
        Adjacency::Remove => lse,
        Adjacency::Add => -lse,
        Adjacency::Replace => unreachable!(),
    })
}

/// Inverse privacy loss for single Gaussian (closed form).
fn inverse_privacy_loss_single_gaussian(epsilon: f64, sigma: f64, sensitivity: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - epsilon * sigma_sq / sensitivity
}

/// Inverse privacy loss for mixture Gaussian (bisection).
fn mixture_inverse_privacy_loss(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let sens_min = c.min_positive_sensitivity();
    let sens_max = c.max_sensitivity();

    let candidates = [
        inverse_privacy_loss_single_gaussian(epsilon, c.sigma, sens_min),
        inverse_privacy_loss_single_gaussian(epsilon, c.sigma, sens_max),
    ];

    let mut lo = candidates
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);
    let mut hi = candidates
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let margin = (hi - lo).abs().max(1.0) * 0.5;
    lo -= margin;
    hi += margin;

    for _ in 0..20 {
        let pl_lo = mixture_privacy_loss(lo, adj, c)?;
        let pl_hi = mixture_privacy_loss(hi, adj, c)?;
        if pl_lo >= epsilon && pl_hi <= epsilon {
            break;
        }
        if pl_lo < epsilon {
            lo -= (hi - lo).abs().max(1.0);
        }
        if pl_hi > epsilon {
            hi += (hi - lo).abs().max(1.0);
        }
    }

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo).abs() < 1e-15 * hi.abs().max(1.0) {
            break;
        }
        let pl = mixture_privacy_loss(mid, adj, c)?;
        if pl > epsilon {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((lo + hi) / 2.0)
}

/// CDF of mu_upper at x.
fn mixture_mu_upper_cdf(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    Ok(match adj {
        Adjacency::Remove => c
            .log_probs
            .iter()
            .zip(c.sensitivities.iter())
            .map(|(&log_p, &s)| log_p.exp() * standard_normal.cdf((x + s) / c.sigma))
            .sum(),
        Adjacency::Add => standard_normal.cdf(x / c.sigma),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    })
}

/// Log CDF of mu_lower at x.
fn mixture_mu_lower_log_cdf(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    Ok(match adj {
        Adjacency::Remove => gaussian_log_cdf(x / c.sigma),
        Adjacency::Add => {
            let summands: Vec<f64> = c
                .log_probs
                .iter()
                .zip(c.sensitivities.iter())
                .map(|(&log_p, &s)| log_p + gaussian_log_cdf((x - s) / c.sigma))
                .collect();
            log_sumexp(&summands)
        }
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    })
}

/// Hockey-stick divergence for mixture Gaussian.
fn mixture_gaussian_get_delta(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    if c.sampling_prob < 1.0 {
        match adj {
            Adjacency::Add => {
                let upper_bound = -(1.0 - c.sampling_prob).ln();
                if epsilon >= upper_bound - 1e-10 {
                    return Ok(0.0);
                }
            }
            Adjacency::Remove => {
                let lower_bound = (1.0 - c.sampling_prob).ln();
                if epsilon <= lower_bound {
                    return Ok((-epsilon.exp_m1()).max(0.0));
                }
            }
            Adjacency::Replace => {
                return Err(PldError::InvalidParameter(
                    "REPLACE adjacency not supported for mixture Gaussian".into(),
                ));
            }
        }
    }

    let x_cutoff = mixture_inverse_privacy_loss(epsilon, adj, c)?;
    let mu_upper = mixture_mu_upper_cdf(x_cutoff, adj, c)?;
    let log_mu_lower = mixture_mu_lower_log_cdf(x_cutoff, adj, c)?;
    let delta = mu_upper - (epsilon + log_mu_lower).exp();

    Ok(delta.clamp(0.0, 1.0))
}

/// Epsilon bounds for mixture Gaussian via x-space truncation.
fn mixture_gaussian_epsilon_bounds(
    adj: Adjacency,
    c: &MixtureConstants,
    log_mass_truncation_bound: f64,
) -> Result<EpsilonBounds> {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let z = standard_normal.inverse_cdf(half_mass);
    let m = c.max_sensitivity();

    let (lower_x, upper_x) = match adj {
        Adjacency::Remove => (c.sigma * z - m, -c.sigma * z),
        Adjacency::Add => (c.sigma * z, -c.sigma * z + m),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    };

    let epsilon_upper = mixture_privacy_loss(lower_x, adj, c)?;
    let epsilon_lower = mixture_privacy_loss(upper_x, adj, c)?;

    Ok(EpsilonBounds {
        epsilon_lower,
        epsilon_upper,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_accumulated_rejects_rate_one() {
        assert!(parallel_poisson_gaussian_pld(0.5, 1.0, 4, &default_config()).is_err());
    }

    #[test]
    fn test_accumulated_rejects_zero_microbatches() {
        assert!(parallel_poisson_gaussian_pld(0.5, 0.01, 0, &default_config()).is_err());
    }

    #[test]
    fn test_accumulated_m1_matches_poisson() {
        let cfg = default_config();
        let base = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
        let pld_poisson = poisson_pld(&base, 0.01).unwrap();
        let pld_acc = parallel_poisson_gaussian_pld(0.5, 0.01, 1, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_a = pld_acc.epsilon_at(1e-5);
        assert!(
            (eps_p - eps_a).abs() < 1e-6,
            "m=1 should match Poisson: {} vs {}",
            eps_p,
            eps_a
        );
    }

    #[test]
    fn test_accumulated_more_microbatches_higher_epsilon() {
        let cfg = default_config();
        let eps2 = parallel_poisson_gaussian_pld(0.5, 0.01, 2, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps4 = parallel_poisson_gaussian_pld(0.5, 0.01, 4, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps8 = parallel_poisson_gaussian_pld(0.5, 0.01, 8, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            eps2 < eps4 && eps4 < eps8,
            "more microbatches → higher ε: {} < {} < {}",
            eps2,
            eps4,
            eps8
        );
    }

    #[test]
    fn test_accumulated_looser_log_mass_is_not_tighter() {
        let cfg = default_config();
        let strict = parallel_poisson_gaussian_pld(0.8, 0.02, 16, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        let mut loose_cfg = default_config();
        loose_cfg.log_mass_truncation_bound = -20.0;
        let loose = parallel_poisson_gaussian_pld(0.8, 0.02, 16, &loose_cfg)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            loose >= strict,
            "looser log mass truncation should not tighten ε: {} >= {}",
            loose,
            strict
        );
    }

    // ---- Mixture math ----

    #[test]
    fn test_binomial_log_probs_m1() {
        let log_probs = binomial_log_probs(1, 0.01);
        assert_eq!(log_probs.len(), 2);
        assert_relative_eq!(log_probs[0].exp(), 0.99, epsilon = 1e-12);
        assert_relative_eq!(log_probs[1].exp(), 0.01, epsilon = 1e-12);
    }

    #[test]
    fn test_binomial_log_probs_m4() {
        let log_probs = binomial_log_probs(4, 0.01);
        let total: f64 = log_probs.iter().map(|lp| lp.exp()).sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
    }

    /// Validate against Google dp_accounting test vectors.
    #[test]
    fn test_mixture_privacy_loss_matches_google() {
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        let pl_add = mixture_privacy_loss(0.5, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(pl_add, 0.1351602748368097, epsilon = 1e-10);

        let pl_rem = mixture_privacy_loss(0.5, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(pl_rem, -0.8423781325734492, epsilon = 1e-10);
    }

    #[test]
    fn test_mixture_get_delta_matches_google() {
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        let delta_add = mixture_gaussian_get_delta(1.0, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(delta_add, 0.036691263832032806, epsilon = 1e-6);

        let delta_rem = mixture_gaussian_get_delta(1.0, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(delta_rem, 0.15768284088654105, epsilon = 1e-6);
    }

    #[test]
    fn test_mixture_delta_matches_quadrature_for_unsorted_nonunit_sensitivities() {
        let mixture = make_test_constants(1.0, &[2.0, 0.0, 3.0, 0.5], &[0.4, 0.1, 0.2, 0.3]);

        assert_eq!(mixture.min_positive_sensitivity(), 0.5);
        assert_eq!(mixture.max_sensitivity(), 3.0);
        assert_relative_eq!(mixture.sampling_prob, 0.9, epsilon = 1e-12);

        // Direct integration of max(mu_upper - exp(epsilon) * mu_lower, 0) over R.
        let delta_add = mixture_gaussian_get_delta(1.0, Adjacency::Add, &mixture).unwrap();
        assert_relative_eq!(delta_add, 0.09379446981341562, epsilon = 1e-10);

        let delta_remove = mixture_gaussian_get_delta(1.0, Adjacency::Remove, &mixture).unwrap();
        assert_relative_eq!(delta_remove, 0.3456419720394463, epsilon = 1e-10);
    }

    /// Helper: construct MixtureConstants from arbitrary sensitivities and probs.
    fn make_test_constants(sigma: f64, sensitivities: &[f64], probs: &[f64]) -> MixtureConstants {
        let log_probs: Vec<f64> = probs.iter().map(|&p| p.ln()).collect();
        MixtureConstants::new(sigma, log_probs, sensitivities.to_vec())
    }
}
