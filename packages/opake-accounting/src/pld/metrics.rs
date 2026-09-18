//! Privacy metric computations for Privacy Loss Distributions
//!
//! This module provides internal metric functions that operate on
//! `&PrivacyLossDistribution`. Each function handles adjacency worst-case
//! logic (max over REMOVE/ADD for delta/epsilon/advantage, min for beta)
//! and dense conversion internally.
//!
//! Users interact with these metrics through `PrivacyLossDistribution` methods.

use super::pmf::Pmf;
use super::PrivacyLossDistribution;

// ---------------------------------------------------------------------------
// Public (crate) API — operates on &PrivacyLossDistribution
// ---------------------------------------------------------------------------

/// Compute delta for a given epsilon (worst-case over adjacencies)
pub(crate) fn delta(pld: &PrivacyLossDistribution, epsilon: f64) -> f64 {
    let delta_remove = pmf_delta(&pld.pmf_remove, epsilon);

    match &pld.pmf_add {
        None => delta_remove,
        Some(pmf_add) => {
            let delta_add = pmf_delta(pmf_add, epsilon);
            delta_remove.max(delta_add)
        }
    }
}

/// Worst-case infinity_mass over both adjacencies.
pub(crate) fn infinity_mass(pld: &PrivacyLossDistribution) -> f64 {
    match &pld.pmf_add {
        None => pld.pmf_remove.infinity_mass,
        Some(pmf_add) => pld.pmf_remove.infinity_mass.max(pmf_add.infinity_mass),
    }
}

/// Compute the minimum epsilon for a target delta (worst-case over adjacencies)
pub(crate) fn epsilon(pld: &PrivacyLossDistribution, target_delta: f64) -> f64 {
    let epsilon_remove = pmf_epsilon(&pld.pmf_remove, target_delta);

    match &pld.pmf_add {
        None => epsilon_remove,
        Some(pmf_add) => {
            let epsilon_add = pmf_epsilon(pmf_add, target_delta);
            epsilon_remove.max(epsilon_add)
        }
    }
}

/// Compute the advantage / TV privacy (worst-case over adjacencies)
pub(crate) fn advantage(pld: &PrivacyLossDistribution) -> f64 {
    let advantage_remove = pmf_delta(&pld.pmf_remove, 0.0);

    match &pld.pmf_add {
        None => advantage_remove,
        Some(pmf_add) => {
            let advantage_add = pmf_delta(pmf_add, 0.0);
            advantage_remove.max(advantage_add)
        }
    }
}

/// Compute the trade-off function beta(alpha) (worst-case over adjacencies)
///
/// For symmetric PLDs, uses the single-distribution Neyman-Pearson trade-off.
/// For asymmetric PLDs, uses the symmetrized two-distribution trade-off.
pub(crate) fn beta(pld: &PrivacyLossDistribution, target_alpha: f64) -> f64 {
    match &pld.pmf_add {
        None => pmf_beta(&pld.pmf_remove, target_alpha),
        Some(pmf_add) => pmf_beta_symmetrized(&pld.pmf_remove, pmf_add, target_alpha),
    }
}

/// Compute the Bayes risk for a given prior probability
///
/// Uses golden section search to minimize `prior * alpha + (1 - prior) * beta(alpha)`.
pub(crate) fn bayes_risk(pld: &PrivacyLossDistribution, prior: f64) -> f64 {
    if prior <= 0.0 || prior >= 1.0 {
        return 0.0;
    }

    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let tolerance = 1e-10;

    let mut a = 0.0;
    let mut b = 1.0;
    let mut c = b - golden_ratio * (b - a);
    let mut d = a + golden_ratio * (b - a);

    let objective = |alpha: f64| -> f64 { prior * alpha + (1.0 - prior) * beta(pld, alpha) };

    let mut fc = objective(c);
    let mut fd = objective(d);

    while (b - a).abs() > tolerance {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - golden_ratio * (b - a);
            fc = objective(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + golden_ratio * (b - a);
            fd = objective(d);
        }
    }

    let alpha_opt = (a + b) / 2.0;
    objective(alpha_opt)
}

// ---------------------------------------------------------------------------
// Private helpers — operates on &Pmf (actual math)
// ---------------------------------------------------------------------------

/// Clean FFT artifacts from PMF probabilities and renormalize.
///
/// After FFT-based composition, small negative probabilities can appear from
/// floating-point rounding. This function clamps negatives to zero and
/// renormalizes so that `sum(probs) = 1 - infinity_mass - negative_infinity_mass`.
/// This preserves explicit ±∞ atoms carried by [`Pmf`] and keeps beta/risk
/// computations conservative when left-tail truncation accumulates at −∞.
fn clean_probs(probs: &[f64], infinity_mass: f64, negative_infinity_mass: f64) -> Vec<f64> {
    let mut cleaned: Vec<f64> = probs.iter().map(|&p| p.max(0.0)).collect();
    let sum: f64 = cleaned.iter().sum();
    let target = (1.0 - infinity_mass - negative_infinity_mass).max(0.0);
    if target == 0.0 {
        cleaned.fill(0.0);
        return cleaned;
    }
    if sum > 0.0 && target > 0.0 {
        let scale = target / sum;
        for p in cleaned.iter_mut() {
            *p *= scale;
        }
    }
    cleaned
}

/// Compute epsilon-hockey stick divergence on a single dense PMF
///
/// delta(epsilon) = infinity_mass + sum_{loss_i > epsilon} [(1 - exp(epsilon - loss_i)) * prob_i]
fn pmf_delta(pmf: &Pmf, epsilon: f64) -> f64 {
    let mut delta = pmf.infinity_mass;

    for (i, &prob) in pmf.probs.iter().enumerate() {
        let loss = pmf.loss_at_index(i as i64);

        if loss > epsilon {
            delta += -(epsilon - loss).exp_m1() * prob;
        }
    }

    delta
}

/// Compute the minimum epsilon for a target delta on a single dense PMF
fn pmf_epsilon(pmf: &Pmf, target_delta: f64) -> f64 {
    if pmf.infinity_mass > target_delta {
        return f64::INFINITY;
    }

    let mut mass_upper = pmf.infinity_mass;
    let mut mass_lower = 0.0;

    for i in (0..pmf.probs.len()).rev() {
        let loss = pmf.loss_at_index(i as i64);
        let prob = pmf.probs[i];

        if mass_upper > target_delta
            && mass_lower > 0.0
            && ((mass_upper - target_delta) / mass_lower).ln() >= loss
        {
            break;
        }

        mass_upper += prob;
        mass_lower += (-loss).exp() * prob;

        if mass_upper >= target_delta && mass_lower == 0.0 {
            return loss.max(0.0);
        }
    }

    if mass_upper <= mass_lower + target_delta {
        return 0.0;
    }

    ((mass_upper - target_delta) / mass_lower).ln()
}

/// Compute the Neyman-Pearson trade-off function beta(alpha) for a symmetric PMF
fn pmf_beta(pmf: &Pmf, target_alpha: f64) -> f64 {
    if target_alpha <= 0.0 {
        return 1.0 - pmf.infinity_mass;
    }
    if target_alpha >= 1.0 {
        return 0.0;
    }

    let n = pmf.probs.len();
    if n == 0 {
        return if target_alpha >= pmf.infinity_mass {
            0.0
        } else {
            1.0
        };
    }

    let y0 = pmf.lower_loss_index;
    let x0 = -(pmf.lower_loss_index + n as i64 - 1);

    // Build CDF of Y, starting from negative_infinity_mass (mass at −∞).
    // This ensures left-tail mass truncated during Chernoff composition
    // is accounted for, keeping beta estimates conservative.
    let mut cdf_y = vec![0.0; n];
    cdf_y[0] = pmf.negative_infinity_mass + pmf.probs[0];
    for i in 1..n {
        cdf_y[i] = cdf_y[i - 1] + pmf.probs[i];
    }

    // Build complement CDF of X (X = -Y, so pmf_X[i] = probs[n-1-i])
    // negative_infinity_mass propagates automatically through cdf_y:
    // X's +∞ mass (= Y's −∞ mass) is already included in cdf_y values.
    let mut ccdf_x = vec![0.0; n + 1];
    for i in 0..n {
        ccdf_x[i] = cdf_y[n - 1 - i];
    }
    ccdf_x[n] = 0.0;

    if ccdf_x[0] <= target_alpha {
        return 0.0;
    }

    // Find threshold t
    let mut t: i64 = -1;
    for i in 0..n {
        if ccdf_x[i + 1] <= target_alpha {
            t = i as i64;
            break;
        }
    }

    if t == -1 {
        return 1.0 - pmf.infinity_mass;
    }

    let t_idx = t as usize;
    let j = x0 + t - y0;

    if j < 0 {
        return 0.0;
    }
    if j >= n as i64 {
        return 1.0 - pmf.infinity_mass;
    }

    let j_idx = j as usize;

    let pmf_x_t = pmf.probs[n - 1 - t_idx];
    if pmf_x_t <= 1e-300 {
        return cdf_y[j_idx];
    }
    let gamma = (target_alpha - ccdf_x[t_idx + 1]) / pmf_x_t;

    let pmf_y_j = pmf.probs[j_idx];
    let beta = cdf_y[j_idx] - gamma * pmf_y_j;

    beta.clamp(0.0, 1.0 - pmf.infinity_mass)
}

/// Compute the Neyman-Pearson trade-off function for asymmetric PLDs
///
/// Uses pmf_remove as Y and pmf_add (negated) as X.
/// Applies FFT-artifact cleanup, while preserving explicit ±∞ atoms.
fn pmf_beta_asymmetric(pmf_remove: &Pmf, pmf_add: &Pmf, target_alpha: f64) -> f64 {
    if target_alpha <= 0.0 {
        return 1.0 - pmf_remove.infinity_mass;
    }
    if target_alpha >= 1.0 {
        return 0.0;
    }

    let n_y = pmf_remove.probs.len();
    let n_x = pmf_add.probs.len();

    if n_y == 0 || n_x == 0 {
        return if target_alpha >= pmf_remove.infinity_mass {
            0.0
        } else {
            1.0
        };
    }

    // Clean FFT artifacts and preserve finite-vs-infinite mass split.
    let probs_y = clean_probs(
        &pmf_remove.probs,
        pmf_remove.infinity_mass,
        pmf_remove.negative_infinity_mass,
    );
    let probs_x_add = clean_probs(
        &pmf_add.probs,
        pmf_add.infinity_mass,
        pmf_add.negative_infinity_mass,
    );

    let y0 = pmf_remove.lower_loss_index;
    let upper_loss_add = pmf_add.lower_loss_index + n_x as i64 - 1;
    let x0 = -upper_loss_add;

    // Build CDF of Y from cleaned finite probabilities, plus atom at −∞.
    let mut cdf_y = vec![0.0; n_y];
    cdf_y[0] = pmf_remove.negative_infinity_mass + probs_y[0];
    for i in 1..n_y {
        cdf_y[i] = cdf_y[i - 1] + probs_y[i];
    }

    // Build CDF of L_add from cleaned finite probabilities, plus atom at −∞.
    // Under X = −L_add, this contributes to X's +∞ atom and therefore to ccdf_x.
    let mut cdf_add = vec![0.0; n_x];
    cdf_add[0] = pmf_add.negative_infinity_mass + probs_x_add[0];
    for i in 1..n_x {
        cdf_add[i] = cdf_add[i - 1] + probs_x_add[i];
    }

    let mut ccdf_x = vec![0.0; n_x + 1];
    for i in 0..n_x {
        ccdf_x[i] = cdf_add[n_x - 1 - i];
    }
    ccdf_x[n_x] = 0.0;

    if ccdf_x[0] <= target_alpha {
        return 0.0;
    }

    let mut t: i64 = -1;
    for i in 0..n_x {
        if ccdf_x[i + 1] <= target_alpha {
            t = i as i64;
            break;
        }
    }

    if t == -1 {
        return 1.0 - pmf_remove.infinity_mass;
    }

    let t_idx = t as usize;
    let j = x0 + t - y0;

    if j < 0 {
        return 0.0;
    }
    if j >= n_y as i64 {
        return 1.0 - pmf_remove.infinity_mass;
    }

    let j_idx = j as usize;

    let pmf_x_t = probs_x_add[n_x - 1 - t_idx];
    if pmf_x_t <= 1e-300 {
        return cdf_y[j_idx];
    }
    let gamma = (target_alpha - ccdf_x[t_idx + 1]) / pmf_x_t;

    let pmf_y_j = probs_y[j_idx];
    let beta = cdf_y[j_idx] - gamma * pmf_y_j;

    beta.clamp(0.0, 1.0 - pmf_remove.infinity_mass)
}

/// Compute the inverse trade-off function for asymmetric PLDs
///
/// Computes T(Q,P)(alpha) by swapping the roles of X and Y.
fn pmf_beta_inverse(pmf_remove: &Pmf, pmf_add: &Pmf, target_alpha: f64) -> f64 {
    pmf_beta_asymmetric(pmf_add, pmf_remove, target_alpha)
}

/// Compute the symmetrized trade-off function for asymmetric PLDs
///
/// Implements Definition F.1 from Dong et al. (2022).
/// "Gaussian Differential Privacy" JRSS-B.
///
/// Uses riskcal-style cleanup: clamp negative probabilities and renormalize
/// before computing alpha_bar and f_alpha_bar.
fn pmf_beta_symmetrized(pmf_remove: &Pmf, pmf_add: &Pmf, target_alpha: f64) -> f64 {
    if target_alpha <= 0.0 {
        return 1.0 - pmf_remove.infinity_mass;
    }
    if target_alpha >= 1.0 {
        return 0.0;
    }

    let n_y = pmf_remove.probs.len();
    let n_x = pmf_add.probs.len();

    if n_y == 0 || n_x == 0 {
        return if target_alpha >= pmf_remove.infinity_mass {
            0.0
        } else {
            1.0
        };
    }

    // Clean FFT artifacts while preserving explicit ±∞ atoms.
    let probs_y = clean_probs(
        &pmf_remove.probs,
        pmf_remove.infinity_mass,
        pmf_remove.negative_infinity_mass,
    );
    let probs_x_add = clean_probs(
        &pmf_add.probs,
        pmf_add.infinity_mass,
        pmf_add.negative_infinity_mass,
    );

    // Compute alpha_bar = Pr[X > 0], including X's +∞ atom.
    // Under X = -L_add, L_add = -∞ maps to X = +∞, which always contributes.
    let upper_loss_add = pmf_add.lower_loss_index + n_x as i64 - 1;
    let x0 = -upper_loss_add;

    let start_idx = ((-x0 + 1).max(0) as usize).min(n_x);
    let mut alpha_bar = pmf_add.negative_infinity_mass;
    for i in start_idx..n_x {
        alpha_bar += probs_x_add[n_x - 1 - i];
    }

    // f_alpha_bar = Pr[Y <= 0], including Y's atom at −∞.
    let y0 = pmf_remove.lower_loss_index;
    let end_idx = ((-y0 + 1).max(0) as usize).min(n_y);
    let mut f_alpha_bar = pmf_remove.negative_infinity_mass;
    for prob in probs_y.iter().take(end_idx) {
        f_alpha_bar += prob;
    }

    if alpha_bar <= f_alpha_bar {
        if target_alpha < alpha_bar {
            pmf_beta_asymmetric(pmf_remove, pmf_add, target_alpha)
        } else if target_alpha <= f_alpha_bar {
            alpha_bar + f_alpha_bar - target_alpha
        } else {
            pmf_beta_inverse(pmf_remove, pmf_add, target_alpha)
        }
    } else {
        let beta_forward = pmf_beta_asymmetric(pmf_remove, pmf_add, target_alpha);
        let beta_inverse = pmf_beta_inverse(pmf_remove, pmf_add, target_alpha);
        beta_forward.max(beta_inverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pld::pmf::Pmf;
    use std::collections::BTreeMap;

    /// Helper: build a symmetric PLD from a Pmf
    fn sym_pld(pmf: Pmf) -> PrivacyLossDistribution {
        PrivacyLossDistribution::new_symmetric(pmf)
    }

    /// Helper: build an asymmetric PLD from two Pmfs
    fn asym_pld(remove: Pmf, add: Pmf) -> PrivacyLossDistribution {
        PrivacyLossDistribution::new_asymmetric(remove, add)
    }

    /// Helper: identity PLD (all mass at loss=0)
    fn identity_pld() -> PrivacyLossDistribution {
        let mut masses = BTreeMap::new();
        masses.insert(0, 1.0);
        PrivacyLossDistribution::new_symmetric(Pmf::from_sparse(0.1, masses, 0.0, usize::MAX))
    }

    // -----------------------------------------------------------------------
    // pmf_delta tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pmf_delta_all_mass_below_epsilon() {
        // All losses <= epsilon → delta = infinity_mass only
        let pmf = Pmf::new(0.1, -5, vec![0.3, 0.4, 0.3], 0.0, usize::MAX);
        // Losses: -0.5, -0.4, -0.3. Query epsilon=0.0 → all below
        let d = pmf_delta(&pmf, 0.0);
        assert!((d - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_pmf_delta_all_mass_above_epsilon() {
        // All losses > epsilon → delta is large
        let pmf = Pmf::new(0.1, 10, vec![0.5, 0.5], 0.0, usize::MAX);
        // Losses: 1.0, 1.1. Query epsilon=0.0 → all above
        let d = pmf_delta(&pmf, 0.0);
        // delta = (1 - exp(-1.0))*0.5 + (1 - exp(-1.1))*0.5
        let expected = (1.0 - (-1.0_f64).exp()) * 0.5 + (1.0 - (-1.1_f64).exp()) * 0.5;
        assert!((d - expected).abs() < 1e-12);
    }

    #[test]
    fn test_pmf_delta_includes_conservative_right_tail_mass() {
        let pmf = Pmf::new(0.1, 0, vec![0.7], 0.3, usize::MAX);
        let d = pmf_delta(&pmf, 10.0);
        // All finite losses (0.0) < epsilon=10.0, so only infinity_mass contributes
        assert!((d - 0.3).abs() < 1e-15);
    }

    #[test]
    fn test_pmf_delta_monotone_decreasing_in_epsilon() {
        let pmf = Pmf::new(
            0.1,
            -5,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let epsilons = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let deltas: Vec<f64> = epsilons.iter().map(|&e| pmf_delta(&pmf, e)).collect();

        for i in 1..deltas.len() {
            assert!(
                deltas[i] <= deltas[i - 1] + 1e-15,
                "delta({}) = {} > delta({}) = {}",
                epsilons[i],
                deltas[i],
                epsilons[i - 1],
                deltas[i - 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // pmf_epsilon tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pmf_epsilon_returns_infinity_when_infmass_exceeds_target() {
        let pmf = Pmf::new(0.1, 0, vec![0.5], 0.6, usize::MAX);
        let eps = pmf_epsilon(&pmf, 0.3);
        assert!(eps.is_infinite());
    }

    #[test]
    fn test_pmf_epsilon_returns_zero_for_large_target() {
        let pmf = Pmf::new(0.1, 0, vec![0.3, 0.5, 0.2], 0.0, usize::MAX);
        let eps = pmf_epsilon(&pmf, 1.0);
        assert!((eps - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pmf_epsilon_monotone_decreasing_in_delta() {
        let pmf = Pmf::new(
            0.1,
            -5,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let targets = [0.01, 0.05, 0.1, 0.2, 0.5];
        let epsilons: Vec<f64> = targets.iter().map(|&d| pmf_epsilon(&pmf, d)).collect();

        for i in 1..epsilons.len() {
            assert!(
                epsilons[i] <= epsilons[i - 1] + 1e-10,
                "eps(delta={}) = {} > eps(delta={}) = {}",
                targets[i],
                epsilons[i],
                targets[i - 1],
                epsilons[i - 1]
            );
        }
    }

    #[test]
    fn test_pmf_epsilon_and_delta_are_consistent() {
        let pmf = Pmf::new(0.1, -5, vec![0.1, 0.2, 0.4, 0.2, 0.1], 0.0, usize::MAX);

        for &target_delta in &[0.01, 0.05, 0.1, 0.2, 0.4] {
            let eps = pmf_epsilon(&pmf, target_delta);
            if eps.is_finite() {
                let achieved_delta = pmf_delta(&pmf, eps);
                assert!(
                    achieved_delta <= target_delta + 1e-12,
                    "At eps={}, achieved delta {} > target {}",
                    eps,
                    achieved_delta,
                    target_delta
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // pmf_beta tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pmf_beta_boundary_alpha_zero() {
        let pmf = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.1], 0.1, usize::MAX);
        let b = pmf_beta(&pmf, 0.0);
        assert!((b - (1.0 - pmf.infinity_mass)).abs() < 1e-12);
    }

    #[test]
    fn test_pmf_beta_boundary_alpha_one() {
        let pmf = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);
        let b = pmf_beta(&pmf, 1.0);
        assert!((b - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pmf_beta_decreases_with_alpha() {
        let pmf = Pmf::new(
            0.1,
            -5,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let betas: Vec<f64> = alphas.iter().map(|&a| pmf_beta(&pmf, a)).collect();

        for i in 1..betas.len() {
            assert!(
                betas[i] <= betas[i - 1] + 1e-12,
                "beta({}) = {} > beta({}) = {}",
                alphas[i],
                betas[i],
                alphas[i - 1],
                betas[i - 1]
            );
        }
    }

    #[test]
    fn test_pmf_beta_alpha_plus_beta_leq_one() {
        // For a PLD with non-negative losses, alpha + beta(alpha) <= 1
        let pmf = Pmf::new(
            0.1,
            0,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );

        for alpha in (0..=100).map(|i| i as f64 / 100.0) {
            let b = pmf_beta(&pmf, alpha);
            assert!(
                alpha + b <= 1.0 + 1e-10,
                "alpha={}, beta={}, sum={}",
                alpha,
                b,
                alpha + b
            );
        }
    }

    // -----------------------------------------------------------------------
    // Crate-level API tests (delta, epsilon, advantage, beta, bayes_risk)
    // -----------------------------------------------------------------------

    #[test]
    fn test_delta_identity_is_zero() {
        let pld = identity_pld();
        assert!((delta(&pld, 0.0) - 0.0).abs() < 1e-15);
        assert!((delta(&pld, 1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_epsilon_identity_is_zero() {
        let pld = identity_pld();
        assert!((epsilon(&pld, 1e-6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_advantage_identity_is_zero() {
        let pld = identity_pld();
        assert!((advantage(&pld) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_advantage_equals_delta_at_zero() {
        let pmf = Pmf::new(
            0.1,
            -3,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let pld = sym_pld(pmf);
        assert!((advantage(&pld) - delta(&pld, 0.0)).abs() < 1e-15);
    }

    #[test]
    fn test_delta_asymmetric_takes_worst_case() {
        // Create asymmetric PLD where ADD has larger delta than REMOVE
        let remove = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);
        let add = Pmf::new(0.1, 5, vec![0.3, 0.4, 0.3], 0.0, usize::MAX);
        let pld = asym_pld(remove.clone(), add.clone());

        let d = delta(&pld, 0.0);
        let d_remove = pmf_delta(&remove, 0.0);
        let d_add = pmf_delta(&add, 0.0);

        assert!((d - d_remove.max(d_add)).abs() < 1e-15);
    }

    #[test]
    fn test_epsilon_asymmetric_takes_worst_case() {
        let remove = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);
        let add = Pmf::new(0.1, 5, vec![0.3, 0.4, 0.3], 0.0, usize::MAX);
        let pld = asym_pld(remove.clone(), add.clone());

        let eps = epsilon(&pld, 0.1);
        let eps_remove = pmf_epsilon(&remove, 0.1);
        let eps_add = pmf_epsilon(&add, 0.1);

        assert!((eps - eps_remove.max(eps_add)).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_risk_boundary_priors() {
        let pmf = Pmf::new(
            0.1,
            -3,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let pld = sym_pld(pmf);

        // prior = 0 or 1 → risk = 0
        assert!((bayes_risk(&pld, 0.0) - 0.0).abs() < 1e-15);
        assert!((bayes_risk(&pld, 1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_bayes_risk_identity_equals_min_prior() {
        let pld = identity_pld();
        // For identity: beta(alpha) = 1 - alpha, so risk = prior * alpha + (1-prior) * (1 - alpha)
        // Minimum at alpha = prior if prior < 0.5 → risk = prior
        // Minimum at alpha = 1-prior if prior >= 0.5 → risk = 1-prior
        for &prior in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let risk = bayes_risk(&pld, prior);
            let expected = prior.min(1.0 - prior);
            assert!(
                (risk - expected).abs() < 1e-8,
                "prior={}, risk={}, expected={}",
                prior,
                risk,
                expected
            );
        }
    }

    #[test]
    fn test_bayes_risk_leq_min_prior() {
        // Bayes risk is always <= min(prior, 1-prior) (the identity bound)
        let pmf = Pmf::new(0.1, 0, vec![0.3, 0.4, 0.2], 0.1, usize::MAX);
        let pld = sym_pld(pmf);

        for &prior in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let risk = bayes_risk(&pld, prior);
            assert!(
                risk <= prior.min(1.0 - prior) + 1e-10,
                "prior={}, risk={}, bound={}",
                prior,
                risk,
                prior.min(1.0 - prior)
            );
        }
    }

    #[test]
    fn test_bayes_risk_symmetric_in_prior() {
        // risk(prior) == risk(1-prior) for symmetric PLDs with losses centered at 0
        // lower_loss_index=-2, 5 elements → losses: -0.2, -0.1, 0.0, 0.1, 0.2
        // Symmetric weights around center
        let pmf = Pmf::new(0.1, -2, vec![0.1, 0.2, 0.4, 0.2, 0.1], 0.0, usize::MAX);
        let pld = sym_pld(pmf);

        for &prior in &[0.1, 0.2, 0.3, 0.4] {
            let risk = bayes_risk(&pld, prior);
            let risk_complement = bayes_risk(&pld, 1.0 - prior);
            assert!(
                (risk - risk_complement).abs() < 1e-8,
                "prior={}, risk={}, risk(1-prior)={}",
                prior,
                risk,
                risk_complement
            );
        }
    }

    // -----------------------------------------------------------------------
    // pmf_beta_asymmetric tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_beta_asymmetric_boundary_alpha_zero() {
        let remove = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.1], 0.1, usize::MAX);
        let add = Pmf::new(0.1, 0, vec![0.4, 0.4, 0.2], 0.0, usize::MAX);
        let b = pmf_beta_asymmetric(&remove, &add, 0.0);
        assert!((b - (1.0 - remove.infinity_mass)).abs() < 1e-12);
    }

    #[test]
    fn test_beta_asymmetric_boundary_alpha_one() {
        let remove = Pmf::new(0.1, 0, vec![0.5, 0.3, 0.2], 0.0, usize::MAX);
        let add = Pmf::new(0.1, 0, vec![0.4, 0.4, 0.2], 0.0, usize::MAX);
        let b = pmf_beta_asymmetric(&remove, &add, 1.0);
        assert!((b - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_beta_symmetrized_decreases_with_alpha() {
        let remove = Pmf::new(
            0.1,
            -3,
            vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
            0.0,
            usize::MAX,
        );
        let add = Pmf::new(
            0.1,
            -2,
            vec![0.05, 0.15, 0.4, 0.25, 0.1, 0.05],
            0.0,
            usize::MAX,
        );

        let alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let betas: Vec<f64> = alphas
            .iter()
            .map(|&a| pmf_beta_symmetrized(&remove, &add, a))
            .collect();

        for i in 1..betas.len() {
            assert!(
                betas[i] <= betas[i - 1] + 1e-12,
                "beta_sym({}) = {} > beta_sym({}) = {}",
                alphas[i],
                betas[i],
                alphas[i - 1],
                betas[i - 1]
            );
        }
    }

    #[test]
    fn test_beta_inverse_swaps_roles() {
        let remove = Pmf::new(0.1, -3, vec![0.1, 0.3, 0.4, 0.2], 0.0, usize::MAX);
        let add = Pmf::new(0.1, -2, vec![0.2, 0.3, 0.3, 0.2], 0.0, usize::MAX);

        // beta_inverse(remove, add, alpha) == beta_asymmetric(add, remove, alpha)
        for &alpha in &[0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let inv = pmf_beta_inverse(&remove, &add, alpha);
            let direct = pmf_beta_asymmetric(&add, &remove, alpha);
            assert!(
                (inv - direct).abs() < 1e-12,
                "alpha={}: inverse={}, direct={}",
                alpha,
                inv,
                direct
            );
        }
    }

    #[test]
    fn test_beta_asymmetric_includes_negative_infinity_mass() {
        let remove = Pmf {
            discretization: 0.1,
            lower_loss_index: 0,
            probs: vec![0.7],
            infinity_mass: 0.0,
            negative_infinity_mass: 0.3,
            max_grid_size: usize::MAX,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        };
        let add = Pmf::new(0.1, 0, vec![1.0], 0.0, usize::MAX);

        let b = pmf_beta_asymmetric(&remove, &add, 0.5);
        assert!((b - 0.65).abs() < 1e-12);
    }

    #[test]
    fn test_clean_probs_zero_target_returns_zero_distribution() {
        let cleaned = clean_probs(&[1e-16, -1e-16, 2e-16], 0.6, 0.4);
        assert!(cleaned.iter().all(|&p| p == 0.0));
    }
}
