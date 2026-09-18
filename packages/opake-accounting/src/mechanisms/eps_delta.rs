//! (ε, δ)-DP mechanism PLD constructor.

use std::collections::BTreeMap;

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

/// Compute the PLD for a mechanism with a known (ε, δ) guarantee.
///
/// Constructs a worst-case PLD: mass `(1 - δ)` at privacy loss `ε`,
/// infinity mass `δ`.
///
/// # Arguments
///
/// * `epsilon` — privacy loss, must be ≥ 0
/// * `delta` — failure probability, must be in \[0, 1\]
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if `epsilon < 0` or `delta ∉ [0, 1]`.
pub fn eps_delta_pld(
    epsilon: f64,
    delta: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if epsilon < 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "epsilon must be non-negative, got {}",
            epsilon
        )));
    }
    if !(0.0..=1.0).contains(&delta) {
        return Err(PldError::InvalidParameter(format!(
            "delta must be in [0, 1], got {}",
            delta
        )));
    }

    let disc = config.discretization;
    let index = (epsilon / disc).ceil() as i64;
    let mut masses = BTreeMap::new();
    masses.insert(index, 1.0 - delta);

    let pmf = Pmf::from_sparse(disc, masses, delta, config.max_grid_size);
    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_eps_delta_rejects_negative_epsilon() {
        assert!(eps_delta_pld(-0.1, 1e-5, &default_config()).is_err());
    }

    #[test]
    fn test_eps_delta_rejects_invalid_delta() {
        assert!(eps_delta_pld(1.0, 1.1, &default_config()).is_err());
        assert!(eps_delta_pld(1.0, -0.1, &default_config()).is_err());
    }

    #[test]
    fn test_eps_delta_round_trip() {
        let pld = eps_delta_pld(1.0, 1e-5, &default_config()).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(
            (eps - 1.0).abs() < 0.001,
            "epsilon_at(1e-5) = {}, expected ~1.0",
            eps
        );
    }

    #[test]
    fn test_eps_delta_pure_dp() {
        let pld = eps_delta_pld(1.0, 0.0, &default_config()).unwrap();
        assert_eq!(pld.delta_at(1.0), 0.0);
    }

    #[test]
    fn test_eps_delta_rounds_off_grid_atoms_upward() {
        let mut config = default_config();
        config.discretization = 0.1;

        let pld = eps_delta_pld(0.11, 0.0, &config).unwrap();
        assert_eq!(pld.pmf_remove.lower_loss_index, 2);
        assert!(pld.delta_at(0.11) > 0.0);
        assert_eq!(pld.delta_at(0.2), 0.0);

        let composed = pld.compose(&pld).unwrap();
        assert_eq!(composed.pmf_remove.lower_loss_index, 4);
        assert!(composed.delta_at(0.22) > 0.0);
        assert_eq!(composed.delta_at(0.4), 0.0);
    }

    #[test]
    fn test_eps_delta_zero_is_identity() {
        let pld = eps_delta_pld(0.0, 0.0, &default_config()).unwrap();
        assert_eq!(pld.epsilon_at(1e-5), 0.0);
        assert_eq!(pld.delta_at(0.0), 0.0);
    }
}
