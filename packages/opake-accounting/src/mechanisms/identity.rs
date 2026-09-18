//! Identity (zero privacy loss) mechanism PLD constructor.

use std::collections::BTreeMap;

use crate::discretization::DiscretizationConfig;
use crate::error::Result;
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

/// Compute the PLD for the identity (zero privacy loss) mechanism.
///
/// This is the neutral element for composition: composing any PLD with
/// the identity PLD yields the same privacy guarantee.
pub fn identity_pld(config: &DiscretizationConfig) -> Result<PrivacyLossDistribution> {
    let mut masses = BTreeMap::new();
    masses.insert(0, 1.0);

    let pmf = Pmf::from_sparse(config.discretization, masses, 0.0, config.max_grid_size);
    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_identity_epsilon_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        assert_eq!(pld.advantage(), 0.0);
    }

    #[test]
    fn test_identity_composed_with_gaussian() {
        let config = default_config();
        let id = identity_pld(&config).unwrap();
        let g = crate::mechanisms::gaussian_pld(1.0, &config).unwrap();

        // Identity is the neutral element: composing with Gaussian gives the same result.
        let composed = id
            .self_compose(1)
            .unwrap()
            .compose(&g.self_compose(1).unwrap())
            .unwrap();
        for &delta in &[1e-10, 1e-5, 0.1] {
            let eps_g = g.epsilon_at(delta);
            let eps_c = composed.epsilon_at(delta);
            assert!(
                (eps_c - eps_g).abs() < 1e-6,
                "delta={}: gaussian eps={}, composed eps={}",
                delta,
                eps_g,
                eps_c,
            );
        }
    }
}
