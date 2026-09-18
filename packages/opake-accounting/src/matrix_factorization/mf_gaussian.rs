//! Matrix factorization Gaussian mechanism PLD constructor.
//!
//! Computes the PLD for the entire MF training run as a single Gaussian
//! mechanism with effective noise multiplier σ/S, where σ is the raw noise
//! standard deviation and S is the L2 sensitivity of the encoder matrix.
//!
//! # How MF privacy differs from DP-SGD
//!
//! In standard DP-SGD, each step uses i.i.d. Gaussian noise and the total
//! privacy is computed by composing per-step PLDs. In MF-DP, the noise
//! is *correlated* across steps via the matrix factorization A = C⁻¹.
//! The sensitivity S captures this correlation structure, and the privacy
//! of the entire training run reduces to a *single* Gaussian mechanism
//! with effective noise multiplier σ/S.
//!
//! # References
//!
//! - BandMF: Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
//! - BLT: Dvijotham et al. (2024) <https://arxiv.org/abs/2404.16706>

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

/// Minimum effective noise multiplier for MF mechanisms.
///
/// Below this threshold, the privacy loss is too large for accurate
/// PLD discretization. At σ_eff=0.01 the epsilon range exceeds 5000,
/// which even with adaptive coarsening may be impractical.
const MF_MIN_EFFECTIVE_NOISE_MULTIPLIER: f64 = 0.01;

/// Maximum effective noise multiplier for MF mechanisms.
///
/// Above this threshold, the mechanism provides essentially perfect
/// privacy and the PLD computation is wasteful. The epsilon range
/// shrinks to near zero.
const MF_MAX_EFFECTIVE_NOISE_MULTIPLIER: f64 = 1000.0;

/// Compute the PLD for a matrix factorization Gaussian mechanism.
///
/// This computes the privacy guarantee for the entire MF training run
/// as a single Gaussian mechanism with effective noise multiplier σ/S.
///
/// The sensitivity S should be pre-computed based on the MF strategy
/// (BandMF, BLT) and participation pattern (single, min-sep,
/// fixed-epoch) using the functions in [`super::sensitivity`].
///
/// # Arguments
///
/// * `noise_multiplier` — Raw noise standard deviation σ (before matrix
///   factorization). Must be positive.
/// * `sensitivity` — L2 sensitivity S of the encoder matrix under the
///   given participation pattern. Must be positive.
/// * `config` — Discretization configuration for PLD grid.
///
/// # Returns
///
/// A symmetric PLD representing the total privacy cost of the MF
/// mechanism across all training steps.
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range or if the
/// effective noise multiplier σ/S falls outside practical bounds.
///
/// # Examples
///
/// ```rust,ignore
/// use opake_accounting::matrix_factorization::mf_gaussian_pld;
/// use opake_accounting::DiscretizationConfig;
///
/// let config = DiscretizationConfig::default();
/// // noise_multiplier=1.0, sensitivity=2.0 → effective σ/S = 0.5
/// let pld = mf_gaussian_pld(1.0, 2.0, &config).unwrap();
/// let epsilon = pld.epsilon_at(1e-5);
/// ```
pub fn mf_gaussian_pld(
    noise_multiplier: f64,
    sensitivity: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if !noise_multiplier.is_finite() || noise_multiplier <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be a positive finite number, got {}",
            noise_multiplier
        )));
    }
    if !sensitivity.is_finite() || sensitivity <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sensitivity must be a positive finite number, got {}",
            sensitivity
        )));
    }

    let effective_nm = noise_multiplier / sensitivity;

    if effective_nm < MF_MIN_EFFECTIVE_NOISE_MULTIPLIER {
        return Err(PldError::InvalidParameter(format!(
            "effective noise multiplier σ/S = {:.6e} is below minimum {:.6e} \
             (noise_multiplier={}, sensitivity={}). \
             The mechanism provides too little privacy for accurate accounting.",
            effective_nm, MF_MIN_EFFECTIVE_NOISE_MULTIPLIER, noise_multiplier, sensitivity
        )));
    }
    if effective_nm > MF_MAX_EFFECTIVE_NOISE_MULTIPLIER {
        return Err(PldError::InvalidParameter(format!(
            "effective noise multiplier σ/S = {:.6e} exceeds maximum {:.6e} \
             (noise_multiplier={}, sensitivity={}). \
             Consider using identity_pld() for near-perfect privacy.",
            effective_nm, MF_MAX_EFFECTIVE_NOISE_MULTIPLIER, noise_multiplier, sensitivity
        )));
    }

    let bounds = mf_gaussian_epsilon_bounds(effective_nm, config.log_mass_truncation_bound);
    let delta_tilde = 1.0 / effective_nm;
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        crate::numerics::gaussian::gaussian_delta_at(delta_tilde, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// X-space truncation → epsilon bounds for a Gaussian mechanism with
/// arbitrary effective noise multiplier.
fn mf_gaussian_epsilon_bounds(
    effective_noise_multiplier: f64,
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    use statrs::distribution::{ContinuousCDF, Normal};

    let sigma = effective_noise_multiplier;
    let sensitivity = 1.0;

    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let z = standard_normal.inverse_cdf(half_mass);

    let epsilon_upper = sensitivity * (0.5 * sensitivity - sigma * z) / (sigma * sigma);

    EpsilonBounds {
        epsilon_lower: -epsilon_upper,
        epsilon_upper,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    // ---- Parameter validation ----

    #[test]
    fn test_mf_gaussian_rejects_zero_noise() {
        assert!(mf_gaussian_pld(0.0, 1.0, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_negative_noise() {
        assert!(mf_gaussian_pld(-1.0, 1.0, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_zero_sensitivity() {
        assert!(mf_gaussian_pld(1.0, 0.0, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_negative_sensitivity() {
        assert!(mf_gaussian_pld(1.0, -1.0, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_nan() {
        assert!(mf_gaussian_pld(f64::NAN, 1.0, &default_config()).is_err());
        assert!(mf_gaussian_pld(1.0, f64::NAN, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_infinite() {
        assert!(mf_gaussian_pld(f64::INFINITY, 1.0, &default_config()).is_err());
        assert!(mf_gaussian_pld(1.0, f64::INFINITY, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_too_small_effective_nm() {
        // noise_multiplier=0.001, sensitivity=1.0 → effective_nm=0.001 < 0.01
        assert!(mf_gaussian_pld(0.001, 1.0, &default_config()).is_err());
    }

    #[test]
    fn test_mf_gaussian_rejects_too_large_effective_nm() {
        // noise_multiplier=2000.0, sensitivity=1.0 → effective_nm=2000 > 1000
        assert!(mf_gaussian_pld(2000.0, 1.0, &default_config()).is_err());
    }

    // ---- Consistency with standard Gaussian ----

    #[test]
    fn test_mf_gaussian_matches_standard_gaussian_sensitivity_one() {
        // With sensitivity=1.0, MF Gaussian should match standard Gaussian
        let cfg = default_config();
        for &sigma in &[0.25, 0.5, 0.8, 1.0, 1.2] {
            let mf_pld = mf_gaussian_pld(sigma, 1.0, &cfg).unwrap();
            let std_pld = crate::mechanisms::gaussian_pld(sigma, &cfg).unwrap();

            let mf_eps = mf_pld.epsilon_at(1e-5);
            let std_eps = std_pld.epsilon_at(1e-5);

            assert!(
                (mf_eps - std_eps).abs() < 1e-6,
                "σ={}: mf_eps={}, std_eps={}, diff={}",
                sigma,
                mf_eps,
                std_eps,
                (mf_eps - std_eps).abs()
            );
        }
    }

    // ---- Sensitivity monotonicity ----

    #[test]
    fn test_mf_gaussian_higher_sensitivity_higher_epsilon() {
        let cfg = default_config();
        let sigma = 0.5;
        let sensitivities = [0.5, 1.0, 1.5, 2.0, 3.0];

        let epsilons: Vec<f64> = sensitivities
            .iter()
            .map(|&s| mf_gaussian_pld(sigma, s, &cfg).unwrap().epsilon_at(1e-5))
            .collect();

        for w in epsilons.windows(2) {
            assert!(
                w[0] < w[1],
                "Higher sensitivity should give higher epsilon: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_mf_gaussian_higher_noise_lower_epsilon() {
        let cfg = default_config();
        let sensitivity = 1.5;
        let sigmas = [0.2, 0.5, 1.0, 2.0, 5.0];

        let epsilons: Vec<f64> = sigmas
            .iter()
            .map(|&s| {
                mf_gaussian_pld(s, sensitivity, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5)
            })
            .collect();

        for w in epsilons.windows(2) {
            assert!(
                w[0] > w[1],
                "Higher noise should give lower epsilon: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    // ---- Analytical validation ----

    #[test]
    fn test_mf_gaussian_delta_at_zero_epsilon() {
        // With effective_nm = 0.5, δ(0) ≈ 0.6827 (same as Gaussian)
        let pld = mf_gaussian_pld(0.5, 1.0, &default_config()).unwrap();
        let delta = pld.delta_at(0.0);
        assert!(
            (delta - 0.6827).abs() < 1e-3,
            "delta_at(0) = {}, expected ≈ 0.6827",
            delta
        );
    }

    #[test]
    fn test_mf_gaussian_vs_analytical_delta() {
        use statrs::distribution::{ContinuousCDF, Normal};
        let n = Normal::new(0.0, 1.0).unwrap();

        // Test with various (noise_multiplier, sensitivity) pairs
        let cases = [
            (0.5, 1.0), // effective_nm = 0.5
            (1.0, 2.0), // effective_nm = 0.5
            (0.8, 1.0), // effective_nm = 0.8
            (2.0, 2.5), // effective_nm = 0.8
            (1.0, 1.0), // effective_nm = 1.0
            (3.0, 3.0), // effective_nm = 1.0
        ];

        for &(sigma, sens) in &cases {
            let effective_nm = sigma / sens;
            let pld = mf_gaussian_pld(sigma, sens, &default_config()).unwrap();
            let dt = 1.0 / effective_nm;

            for &eps in &[0.1, 0.5, 1.0, 3.0] {
                let analytical =
                    (n.cdf(dt / 2.0 - eps / dt) - eps.exp() * n.cdf(-dt / 2.0 - eps / dt)).max(0.0);
                let numerical = pld.delta_at(eps);
                let err = (numerical - analytical).abs();
                assert!(
                    err < 1e-3,
                    "σ={}, S={}, ε={}: numerical={:.6e}, analytical={:.6e}, err={:.6e}",
                    sigma,
                    sens,
                    eps,
                    numerical,
                    analytical,
                    err
                );
            }
        }
    }

    // ---- Wide range of effective noise multipliers ----

    #[test]
    fn test_mf_gaussian_very_private() {
        // Large effective noise_multiplier → very small epsilon
        // At σ_eff=100, the Gaussian mechanism gives ε < 0.1 for δ=1e-5
        let pld = mf_gaussian_pld(100.0, 1.0, &default_config()).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(
            eps < 0.1,
            "Very private mechanism should have small epsilon, got {}",
            eps
        );
    }

    #[test]
    fn test_mf_gaussian_low_privacy() {
        // Small effective noise_multiplier → large epsilon
        let pld = mf_gaussian_pld(0.1, 1.0, &default_config()).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(
            eps > 1.0,
            "Low privacy mechanism should have large epsilon, got {}",
            eps
        );
    }

    // ---- Equivalence: same effective noise_multiplier ----

    #[test]
    fn test_mf_gaussian_same_effective_nm_same_pld() {
        // Different (σ, S) pairs with same σ/S should give same PLD
        let cfg = default_config();
        let pairs = [(1.0, 2.0), (2.0, 4.0), (0.5, 1.0)]; // all have σ/S = 0.5

        let epsilons: Vec<f64> = pairs
            .iter()
            .map(|&(sigma, sens)| mf_gaussian_pld(sigma, sens, &cfg).unwrap().epsilon_at(1e-5))
            .collect();

        for w in epsilons.windows(2) {
            assert!(
                (w[0] - w[1]).abs() < 1e-6,
                "Same effective nm should give same epsilon: {} vs {}",
                w[0],
                w[1]
            );
        }
    }
}
