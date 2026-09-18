//! Non-private mechanism PLD constructor — infinite privacy loss.

use crate::discretization::DiscretizationConfig;
use crate::error::Result;
use crate::pld::PrivacyLossDistribution;

/// Compute the PLD for a non-private mechanism (ε = ∞, δ = 1).
///
/// All mass sits at +∞ (`infinity_mass = 1`), representing a mechanism
/// that provides no privacy guarantee whatsoever. This is the
/// annihilator for composition: composing any PLD with a non-private
/// PLD yields a non-private result.
///
/// Equivalent to `eps_delta_pld(0.0, 1.0, config)`.
pub fn non_private_pld(config: &DiscretizationConfig) -> Result<PrivacyLossDistribution> {
    // Delegate to eps_delta with ε=0, δ=1: places zero finite mass at
    // index 0 and infinity_mass=1.
    crate::mechanisms::eps_delta_pld(0.0, 1.0, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_non_private_epsilon_is_inf() {
        let pld = non_private_pld(&default_config()).unwrap();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert!(
                pld.epsilon_at(delta).is_infinite(),
                "delta={}: expected inf, got {}",
                delta,
                pld.epsilon_at(delta),
            );
        }
    }

    #[test]
    fn test_non_private_delta_is_one() {
        let pld = non_private_pld(&default_config()).unwrap();
        let d = pld.delta_at(0.0);
        assert!(
            (d - 1.0).abs() < 1e-12,
            "expected delta≈1.0 at ε=0, got {}",
            d
        );
    }

    #[test]
    fn test_non_private_matches_eps_delta() {
        let config = default_config();
        let np = non_private_pld(&config).unwrap();
        let ed = crate::mechanisms::eps_delta_pld(0.0, 1.0, &config).unwrap();
        // Both should give the same epsilon at any delta.
        for &delta in &[1e-10, 1e-5, 0.1] {
            assert_eq!(np.epsilon_at(delta), ed.epsilon_at(delta));
        }
    }

    #[test]
    fn test_non_private_composed_with_gaussian() {
        let config = default_config();
        let np = non_private_pld(&config).unwrap();
        let g = crate::mechanisms::gaussian_pld(1.0, &config).unwrap();

        // non_private is the annihilator: composing with anything stays non-private.
        let composed = np
            .self_compose(1)
            .unwrap()
            .compose(&g.self_compose(1).unwrap())
            .unwrap();
        for &delta in &[1e-10, 1e-5, 0.1] {
            assert!(
                composed.epsilon_at(delta).is_infinite(),
                "delta={}: expected inf after composition, got {}",
                delta,
                composed.epsilon_at(delta),
            );
        }
        let d = composed.delta_at(0.0);
        assert!(
            (d - 1.0).abs() < 1e-12,
            "expected delta≈1.0 at ε=0 after composition, got {}",
            d,
        );
    }
}
