//! Poisson-subsampled mechanism PLDs.

use crate::adjacency::Adjacency;
use crate::discretization::connect_the_dots::discretize_from_deltas;
use crate::discretization::{DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::numerics::logspace::{log_a_times_exp_b_plus_c, log_add, log_sub};
use crate::numerics::special::{arcsinh_exp, gaussian_log_cdf, log_sinh};
use crate::pld::realization::{disc_dist, LossRealization, NormalLoss};
use crate::pld::{Pmf, PrivacyLossDistribution};
use statrs::distribution::{ContinuousCDF, Normal};

use super::{validate_noise_multiplier, validate_rate};

/// Apply Poisson subsampling to an existing PLD realization.
///
/// This implements Theorem 3.3 and Algorithms 8--9 of Feldman and Shenfeld,
/// "Efficient privacy loss accounting for subsampling and random allocation"
/// (arXiv:2602.17284). The input PLD may be a conservative approximation of a
/// mechanism; the pointwise transform preserves that domination.
///
/// The result is asymmetric even when `base` is symmetric: REMOVE requires the
/// PLD dual while ADD is a different increasing relabeling of the input loss.
pub fn poisson_pld(base: &PrivacyLossDistribution, rate: f64) -> Result<PrivacyLossDistribution> {
    validate_rate(rate)?;

    if let Some((noise_multiplier, tail_mass)) = base.gaussian_source() {
        return poisson_from_gaussian(noise_multiplier, tail_mass, &base.pmf_remove, rate);
    }

    let remove_source = realization_source(&base.pmf_remove)?;
    let pmf_remove = poisson_remove_pmf(&base.pmf_remove, &remove_source, rate)?;
    let pmf_add = if let Some(base_add) = &base.pmf_add {
        let add_source = realization_source(base_add)?;
        poisson_add_pmf(base_add, &add_source, rate)?
    } else {
        poisson_add_pmf(&base.pmf_remove, &remove_source, rate)?
    };

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add)
        .with_monte_carlo_guarantee(base.estimation_failure_probability(), base.mc_resolution()))
}

/// Apply plain Poisson subsampling to a Gaussian base through [`poisson_pld`].
fn poisson_from_gaussian(
    noise_multiplier: f64,
    tail_mass: f64,
    source: &Pmf,
    rate: f64,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;

    let loss = NormalLoss::gaussian(noise_multiplier);
    let log_tail_mass = tail_mass.ln();
    let bounds = EpsilonBounds {
        epsilon_lower: loss.quantile(log_tail_mass, false),
        epsilon_upper: loss.quantile(log_tail_mass, true),
    };
    let source_config = DiscretizationConfig {
        discretization: source.discretization,
        max_grid_size: source.max_grid_size,
        ..DiscretizationConfig::default()
    };
    let pmf = disc_dist(
        &loss,
        source_config.effective_discretization(&bounds),
        tail_mass,
        source.max_grid_size,
    )?;
    let base = PrivacyLossDistribution::new_symmetric(pmf)
        .with_tail_budgets(source.right_tail_budget, source.left_tail_budget);

    poisson_pld(&base, rate)
}

struct RealizationSource {
    atoms: Vec<(f64, f64)>,
    infinity_mass: f64,
}

const REALIZATION_MASS_TOLERANCE: f64 = 1e-8;

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let updated = sum + adjusted;
        correction = (updated - sum) - adjusted;
        sum = updated;
    }
    sum
}

fn log_reciprocal_moment(atoms: &[(f64, f64)]) -> f64 {
    let max_term = atoms
        .iter()
        .filter_map(|&(loss, mass)| (mass > 0.0).then_some(mass.ln() - loss))
        .max_by(f64::total_cmp)
        .unwrap_or(f64::NEG_INFINITY);
    if max_term == f64::NEG_INFINITY {
        return max_term;
    }
    max_term
        + compensated_sum(
            atoms
                .iter()
                .filter(|&&(_, mass)| mass > 0.0)
                .map(|&(loss, mass)| (mass.ln() - loss - max_term).exp()),
        )
        .ln()
}

/// Convert `pmf` into a dominating PLD realization suitable for Theorem 3.3.
///
/// `negative_infinity_mass` is beta-only bookkeeping for left-tail truncation:
/// composition has already folded the actual left-tail probability into the
/// first finite bucket. Missing numerical mass is moved to `+∞`; excess above
/// the probability simplex is accepted only within the FFT residue tolerance
/// and removed from the least privacy-relevant finite atoms. A hockey-stick PLD
/// approximation can also exceed the realization invariant `E[exp(-L)] <= 1`.
/// Move the necessary lowest-loss mass to `+∞`, preserving a valid dominating
/// pair for the pointwise transform.
fn realization_source(pmf: &Pmf) -> Result<RealizationSource> {
    let mut support = Vec::with_capacity(pmf.probs.len());
    for (index, &mass) in pmf.probs.iter().enumerate() {
        if !mass.is_finite() || mass < -REALIZATION_MASS_TOLERANCE {
            return Err(PldError::NumericalError(format!(
                "cannot transform a PLD with invalid finite mass {mass}"
            )));
        }
        if mass > 0.0 {
            support.push((pmf.loss_at_index(index as i64), mass));
        }
    }
    if !pmf.infinity_mass.is_finite() || !(0.0..=1.0).contains(&pmf.infinity_mass) {
        return Err(PldError::NumericalError(format!(
            "cannot transform a PLD with invalid infinity mass {}",
            pmf.infinity_mass
        )));
    }
    let mut infinity_mass = pmf.infinity_mass;
    let finite_mass = compensated_sum(support.iter().map(|&(_, mass)| mass));
    let total_mass = finite_mass + infinity_mass;
    if total_mass < 1.0 {
        infinity_mass += 1.0 - total_mass;
    } else if total_mass > 1.0 {
        let mut excess = total_mass - 1.0;
        if excess > REALIZATION_MASS_TOLERANCE {
            return Err(PldError::NumericalError(format!(
                "input PLD mass is {total_mass}, exceeding one by more than \
                 {REALIZATION_MASS_TOLERANCE}"
            )));
        }
        for (loss, mass) in support.iter_mut().rev() {
            if *loss > 0.0 || excess == 0.0 {
                continue;
            }
            let removed = excess.min(*mass);
            *mass -= removed;
            excess -= removed;
        }
        for (loss, mass) in &mut support {
            if *loss <= 0.0 || excess == 0.0 {
                continue;
            }
            let removed = excess.min(*mass);
            *mass -= removed;
            excess -= removed;
        }
        if excess > REALIZATION_MASS_TOLERANCE * f64::EPSILON {
            return Err(PldError::NumericalError(
                "cannot remove numerical mass excess from the finite PLD".into(),
            ));
        }
        support.retain(|&(_, mass)| mass > 0.0);
    }

    const MOMENT_BUFFER: f64 = 1e-12;
    for _ in 0..3 {
        let log_exp_neg_expectation = log_reciprocal_moment(&support);
        if log_exp_neg_expectation <= 0.0 {
            break;
        }
        let log_target = (-MOMENT_BUFFER).ln_1p();
        let mut log_excess = log_sub(log_exp_neg_expectation, log_target)
            .map_err(|message| PldError::NumericalError(message.into()))?;
        let mut moved_mass = 0.0;

        for (loss, mass) in &mut support {
            if log_excess == f64::NEG_INFINITY {
                break;
            }
            let log_contribution = mass.ln() - *loss;
            if log_contribution <= log_excess {
                moved_mass += *mass;
                *mass = 0.0;
                log_excess = log_sub(log_excess, log_contribution)
                    .map_err(|message| PldError::NumericalError(message.into()))?;
            } else {
                let partial_mass = (log_excess + *loss).exp().min(*mass);
                *mass -= partial_mass;
                moved_mass += partial_mass;
                break;
            }
        }

        support.retain(|&(_, mass)| mass > 0.0);
        infinity_mass += moved_mass;
    }
    let repaired_log_moment = log_reciprocal_moment(&support);
    if repaired_log_moment > 0.0 {
        return Err(PldError::NumericalError(format!(
            "failed to repair PLD realization: E[exp(-L)] = {}",
            repaired_log_moment.exp()
        )));
    }

    Ok(RealizationSource {
        atoms: support,
        infinity_mass,
    })
}

/// Project an exact transformed atomic law onto a fixed loss grid with
/// Connect-the-Dots.
///
/// Directly moving every transformed atom to the next grid point is valid but
/// introduces a systematic one-bucket shift that grows under composition.
/// CtD instead evaluates the exact hockey-stick profile at the target knots and
/// reconstructs a PLD whose interpolation is conservative between them.
fn project_transformed_atoms(
    source: &Pmf,
    mut atoms: Vec<(f64, f64)>,
    infinity_mass: f64,
    refinement: usize,
) -> Result<Pmf> {
    if atoms
        .iter()
        .any(|&(loss, mass)| !loss.is_finite() || !mass.is_finite() || mass < 0.0)
    {
        return Err(PldError::NumericalError(
            "Poisson transform produced invalid finite support".into(),
        ));
    }
    atoms.retain(|&(_, mass)| mass > 0.0);
    if atoms.is_empty() {
        return Err(PldError::NumericalError(
            "Poisson transform produced no finite support".into(),
        ));
    }
    if atoms.windows(2).any(|window| window[0].0 > window[1].0) {
        return Err(PldError::NumericalError(
            "Poisson transform produced unordered finite support".into(),
        ));
    }

    let bounds = EpsilonBounds {
        epsilon_lower: atoms.first().unwrap().0,
        epsilon_upper: atoms.last().unwrap().0,
    };
    let mut config = DiscretizationConfig {
        discretization: source.discretization / refinement as f64,
        max_grid_size: source.max_grid_size,
        ..DiscretizationConfig::default()
    };
    config.discretization = config.effective_discretization(&bounds);

    let lower_index = (bounds.epsilon_lower / config.discretization).floor() as i64;
    let upper_index = (bounds.epsilon_upper / config.discretization).ceil() as i64;
    let mut deltas = vec![0.0; (upper_index - lower_index + 1) as usize];
    let mut first_included = atoms.len();
    let mut tail_mass = 0.0;
    let mut mass_correction = 0.0;
    let mut log_tail_exp_neg = f64::NEG_INFINITY;
    for index in (lower_index..=upper_index).rev() {
        let epsilon = index as f64 * config.discretization;
        while first_included > 0 && atoms[first_included - 1].0 > epsilon {
            first_included -= 1;
            let (loss, mass) = atoms[first_included];
            let adjusted_mass = mass - mass_correction;
            let updated_mass = tail_mass + adjusted_mass;
            mass_correction = (updated_mass - tail_mass) - adjusted_mass;
            tail_mass = updated_mass;
            log_tail_exp_neg = log_add(log_tail_exp_neg, mass.ln() - loss);
        }
        let finite_delta = if tail_mass == 0.0 {
            0.0
        } else {
            let log_mass = tail_mass.ln();
            let log_discount = epsilon + log_tail_exp_neg;
            if log_discount >= log_mass {
                0.0
            } else {
                (log_mass + (-(log_discount - log_mass).exp()).ln_1p()).exp()
            }
        };
        let numerical_buffer = 8.0 * f64::EPSILON * tail_mass;
        let delta = infinity_mass + (finite_delta + numerical_buffer).min(tail_mass);
        deltas[(index - lower_index) as usize] = delta.clamp(infinity_mass, 1.0);
    }

    let mut projected = discretize_from_deltas(bounds, &deltas, &config, Adjacency::Remove)?;
    projected.right_tail_budget = source.right_tail_budget;
    projected.left_tail_budget = source.left_tail_budget;
    Ok(projected)
}

/// Algorithm 8 (REMOVE): map `L` to `ln(1 + q * (exp(L) - 1))`.
fn poisson_remove_pmf(source: &Pmf, realized: &RealizationSource, rate: f64) -> Result<Pmf> {
    let mut atoms = Vec::with_capacity(realized.atoms.len() + 1);
    let log_exp_neg_expectation = log_reciprocal_moment(&realized.atoms);

    for &(loss, mass) in &realized.atoms {
        let transformed_loss = log_a_times_exp_b_plus_c(rate, loss, 1.0 - rate);
        let log_dual_weight = log_a_times_exp_b_plus_c(1.0 - rate, -loss, rate);
        let transformed_mass = (mass.ln() + log_dual_weight).exp();
        atoms.push((transformed_loss, transformed_mass));
    }

    if log_exp_neg_expectation > 1e-12 {
        return Err(PldError::NumericalError(format!(
            "input PLD is not a realization: E[exp(-L)] = {}",
            log_exp_neg_expectation.exp()
        )));
    }

    // The PLD dual's residual atom maps from -∞ to ln(1 - q). It is present
    // whenever Q has support that P does not, including (epsilon, delta) PLDs.
    let dual_residual = (-log_exp_neg_expectation.exp_m1()).max(0.0);
    if dual_residual > 0.0 {
        atoms.push(((1.0 - rate).ln(), (1.0 - rate) * dual_residual));
        atoms.rotate_right(1);
    }

    project_transformed_atoms(source, atoms, rate * realized.infinity_mass, 1)
}

/// Algorithm 9 (ADD): map `L` to `-ln(1 + q * (exp(-L) - 1))`.
fn poisson_add_pmf(source: &Pmf, realized: &RealizationSource, rate: f64) -> Result<Pmf> {
    let mut atoms = Vec::with_capacity(realized.atoms.len() + 1);

    for &(loss, mass) in &realized.atoms {
        let transformed_loss = -log_a_times_exp_b_plus_c(rate, -loss, 1.0 - rate);
        atoms.push((transformed_loss, mass));
    }

    // An input +∞ atom has the finite maximal ADD loss -ln(1 - q).
    atoms.push((-(1.0 - rate).ln(), realized.infinity_mass));

    project_transformed_atoms(source, atoms, 0.0, 1)
}

// ===========================================================================
// Poisson math: privacy loss at a point
// ===========================================================================

/// Poisson-transformed privacy loss for REMOVE adjacency.
///
/// `L_rem(x) = log(1−q + q·exp(L_raw(x)))` where `L_raw(x) = Δ·(−0.5Δ − x) / σ²`
fn privacy_loss_remove(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let l_raw = sensitivity * (-0.5 * sensitivity - x) / sigma_sq;

    if (rate - 1.0).abs() < 1e-15 {
        return l_raw;
    }

    log_a_times_exp_b_plus_c(rate, l_raw, 1.0 - rate)
}

/// Poisson-transformed privacy loss for ADD adjacency.
///
/// By symmetry: `L_add(x) = −L_rem(−x)`
fn privacy_loss_add(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    -privacy_loss_remove(-x, sigma, sensitivity, rate)
}

/// Poisson-transformed privacy loss for REPLACE adjacency.
fn privacy_loss_replace(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let q = rate;

    let log_ratio_plus = -(2.0 * x * sensitivity + sensitivity * sensitivity) / (2.0 * sigma_sq);
    let log_ratio_minus = (2.0 * x * sensitivity - sensitivity * sensitivity) / (2.0 * sigma_sq);

    let log_num = log_a_times_exp_b_plus_c(q, log_ratio_plus, 1.0 - q);
    let log_den = log_a_times_exp_b_plus_c(q, log_ratio_minus, 1.0 - q);

    log_num - log_den
}

// ===========================================================================
// Poisson math: epsilon bounds
// ===========================================================================

/// X-space truncation → epsilon bounds for Poisson-subsampled Gaussian.
pub(super) fn poisson_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let lower_x_base = sigma * standard_normal.inverse_cdf(half_mass);
    let upper_x_base = -lower_x_base;

    match adjacency {
        Adjacency::Remove => {
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base;
            EpsilonBounds {
                epsilon_lower: privacy_loss_remove(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_remove(lower_x, sigma, sensitivity, rate),
            }
        }
        Adjacency::Add => {
            let lower_x = lower_x_base;
            let upper_x = upper_x_base + sensitivity;
            EpsilonBounds {
                epsilon_lower: privacy_loss_add(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_add(lower_x, sigma, sensitivity, rate),
            }
        }
        Adjacency::Replace => {
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base + sensitivity;
            EpsilonBounds {
                epsilon_lower: privacy_loss_replace(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_replace(lower_x, sigma, sensitivity, rate),
            }
        }
    }
}

// ===========================================================================
// Poisson math: inverse privacy loss
// ===========================================================================

/// Inverse privacy loss for ADD/REMOVE (Gaussian base).
pub(super) fn inverse_privacy_loss_gaussian(
    privacy_loss: f64,
    sigma: f64,
    sensitivity: f64,
) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - privacy_loss * (sigma_sq / sensitivity)
}

/// Inverse privacy loss for REPLACE adjacency (arcsinh formula).
fn inverse_privacy_loss_replace(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> Result<f64> {
    let sigma_sq = sigma * sigma;

    if epsilon == 0.0 {
        return Ok(0.0);
    }

    if (rate - 1.0).abs() < 1e-15 {
        return Ok(-epsilon * sigma_sq / (2.0 * sensitivity));
    }

    let abs_eps = epsilon.abs();
    let sign_eps = epsilon.signum();

    let ds = sensitivity / sigma;
    let log_alpha = 0.5 * ds * ds + (1.0 - rate).ln() - rate.ln();
    let log_sinh_term = log_alpha + log_sinh(abs_eps / 2.0);
    let asinh_term = arcsinh_exp(log_sinh_term, -sign_eps);

    Ok((sigma_sq / sensitivity) * (asinh_term - epsilon / 2.0))
}

// ===========================================================================
// Poisson math: hockey-stick divergence (get_delta)
// ===========================================================================

/// Hockey-stick divergence for Poisson-subsampled Gaussian.
pub(super) fn poisson_gaussian_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> f64 {
    let q = rate;

    if (q - 1.0).abs() < 1e-15 {
        return base_gaussian_get_delta(epsilon, sigma, sensitivity, adjacency);
    }

    match adjacency {
        Adjacency::Add => get_delta_add(epsilon, sigma, sensitivity, q),
        Adjacency::Remove => get_delta_remove(epsilon, sigma, sensitivity, q),
        Adjacency::Replace => get_delta_replace(epsilon, sigma, sensitivity, q).unwrap_or(0.0),
    }
}

fn base_gaussian_get_delta(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    adjacency: Adjacency,
) -> f64 {
    let delta_tilde = sensitivity / sigma;
    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let x_upper = 0.5 * delta_tilde - epsilon / delta_tilde;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - delta_tilde);
            (cdf_x - epsilon.exp() * cdf_shifted).max(0.0)
        }
        Adjacency::Replace => {
            let dt2 = 2.0 * delta_tilde;
            let x_upper = 0.5 * dt2 - epsilon / dt2;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - dt2);
            (cdf_x - epsilon.exp() * cdf_shifted).max(0.0)
        }
    }
}

fn gaussian_cdf(z: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(z)
}

fn get_delta_add(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    let theoretical_upper = -(1.0 - q).ln();
    if epsilon >= theoretical_upper - 1e-10 {
        return 0.0;
    }

    let exp_neg_eps = (-epsilon).exp();
    let ratio = (exp_neg_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return 0.0;
    }
    let l_base = -ratio.ln();

    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);
    let mu_upper = gaussian_cdf(x_cutoff / sigma);

    let log_mu_upper = gaussian_log_cdf(x_cutoff / sigma);
    let log_cdf_lower = gaussian_log_cdf((x_cutoff - sensitivity) / sigma);
    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_lower = log_add(log_1_minus_q + log_mu_upper, log_q + log_cdf_lower);

    (mu_upper - (epsilon + log_mu_lower).exp()).max(0.0)
}

fn get_delta_remove(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    let theoretical_lower = (1.0 - q).ln();
    if epsilon <= theoretical_lower {
        return (-epsilon.exp_m1()).max(0.0);
    }

    let exp_eps = epsilon.exp();
    let ratio = (exp_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return (-epsilon.exp_m1()).max(0.0);
    }
    let l_base = -ratio.ln();

    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);

    let log_tail_upper = gaussian_log_cdf(-x_cutoff / sigma);
    let log_tail_shifted = gaussian_log_cdf((sensitivity - x_cutoff) / sigma);

    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_upper = log_add(log_1_minus_q + log_tail_upper, log_q + log_tail_shifted);

    (log_mu_upper.exp() - (epsilon + log_tail_upper).exp()).max(0.0)
}

fn get_delta_replace(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> Result<f64> {
    let x_cutoff = inverse_privacy_loss_replace(epsilon, sigma, sensitivity, q)?;

    let cdf_center = gaussian_cdf(x_cutoff / sigma);
    let cdf_plus = gaussian_cdf((x_cutoff + sensitivity) / sigma);
    let cdf_minus = gaussian_cdf((x_cutoff - sensitivity) / sigma);

    let mu_upper = q * cdf_plus + (1.0 - q) * cdf_center;
    let mu_lower = q * cdf_minus + (1.0 - q) * cdf_center;

    Ok((mu_upper - epsilon.exp() * mu_lower).max(0.0))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const ANALYTIC_DELTA_NUMERICAL_TOLERANCE: f64 = 1e-10;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_poisson_rejects_bad_rate() {
        let cfg = default_config();
        let identity = crate::mechanisms::identity_pld(&cfg).unwrap();
        assert!(poisson_pld(&identity, 0.0).is_err());
        assert!(poisson_pld(&identity, -0.1).is_err());
        assert!(poisson_pld(&identity, 1.0).is_err());
    }

    #[test]
    fn test_poisson_amplification_reduces_epsilon() {
        let cfg = default_config();
        let pld_full = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
        let pld_sub = poisson_pld(&pld_full, 0.01).unwrap();

        let eps_full = pld_full.epsilon_at(1e-5);
        let eps_sub = pld_sub.epsilon_at(1e-5);

        assert!(
            eps_sub < eps_full,
            "Poisson should reduce epsilon: {} vs {}",
            eps_sub,
            eps_full
        );
    }

    #[test]
    fn test_poisson_rate_monotonicity() {
        let cfg = default_config();
        let rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5];
        let epsilons: Vec<f64> = rates
            .iter()
            .map(|&q| {
                let base = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
                poisson_pld(&base, q).unwrap().epsilon_at(1e-5)
            })
            .collect();

        for w in epsilons.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-9,
                "higher rate should give higher epsilon: q changed: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    // ---- Poisson math ----

    #[test]
    fn test_privacy_loss_remove_no_subsampling() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let x = 0.5;
        let l_raw = sensitivity * (-0.5 * sensitivity - x) / (sigma * sigma);
        let l_sub = privacy_loss_remove(x, sigma, sensitivity, 1.0);
        assert!((l_raw - l_sub).abs() < 1e-12);
    }

    #[test]
    fn test_privacy_loss_add_remove_symmetry() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.01;
        let x = 0.5;
        let l_add = privacy_loss_add(x, sigma, sensitivity, rate);
        let l_rem = privacy_loss_remove(-x, sigma, sensitivity, rate);
        assert!((l_add + l_rem).abs() < 1e-12);
    }

    #[test]
    fn test_privacy_loss_replace_odd_function() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.1;
        for x in [0.1, 0.5, 1.0, 2.0] {
            let l_pos = privacy_loss_replace(x, sigma, sensitivity, rate);
            let l_neg = privacy_loss_replace(-x, sigma, sensitivity, rate);
            assert!((l_pos + l_neg).abs() < 1e-12);
        }
    }

    #[test]
    fn test_inverse_privacy_loss_replace_roundtrip() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.1;

        for eps in [0.01, 0.1, 0.5, 1.0, -0.1, -0.5] {
            let x = inverse_privacy_loss_replace(eps, sigma, sensitivity, rate).unwrap();
            let l = privacy_loss_replace(x, sigma, sensitivity, rate);
            assert!(
                (l - eps).abs() < 1e-8,
                "Round-trip failed: eps={}, x={}, L(x)={}, diff={}",
                eps,
                x,
                l,
                (l - eps).abs()
            );
        }
    }

    fn pmf_delta(pmf: &Pmf, epsilon: f64) -> f64 {
        pmf.infinity_mass
            + pmf
                .probs
                .iter()
                .enumerate()
                .filter_map(|(index, &mass)| {
                    let loss = pmf.loss_at_index(index as i64);
                    (loss > epsilon).then_some(-(epsilon - loss).exp_m1() * mass)
                })
                .sum::<f64>()
    }

    fn exact_delta(epsilon: f64, support: &[(f64, f64)], infinity_mass: f64) -> f64 {
        infinity_mass
            + support
                .iter()
                .filter_map(|&(loss, mass)| {
                    (loss > epsilon).then_some(-(epsilon - loss).exp_m1() * mass)
                })
                .sum::<f64>()
    }

    fn self_compose_support(support: &[(f64, f64)], count: usize) -> Vec<(f64, f64)> {
        let mut composed = vec![(0.0, 1.0)];
        for _ in 0..count {
            composed = composed
                .iter()
                .flat_map(|&(total_loss, total_mass)| {
                    support
                        .iter()
                        .map(move |&(loss, mass)| (total_loss + loss, total_mass * mass))
                })
                .collect();
        }
        composed
    }

    fn total_mass(pmf: &Pmf) -> f64 {
        pmf.probs.iter().sum::<f64>() + pmf.infinity_mass + pmf.negative_infinity_mass
    }

    #[test]
    fn test_generic_poisson_keeps_identity_private() {
        let identity = crate::mechanisms::identity_pld(&default_config()).unwrap();
        let pld = poisson_pld(&identity, 0.2).unwrap();

        assert_eq!(pld.delta_at(0.0), 0.0);
        assert_eq!(pld.epsilon_at(1e-8), 0.0);
        assert!((total_mass(&pld.pmf_remove) - 1.0).abs() < 1e-12);
        assert!((total_mass(pld.pmf_add.as_ref().unwrap()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_generic_poisson_maps_nonprivate_infinity_mass() {
        let base = crate::mechanisms::non_private_pld(&default_config()).unwrap();
        let pld = poisson_pld(&base, 0.2).unwrap();

        assert!((pld.pmf_remove.infinity_mass - 0.2).abs() < 1e-12);
        assert_eq!(pld.pmf_add.as_ref().unwrap().infinity_mass, 0.0);
        assert!((pld.delta_at(100.0) - 0.2).abs() < 1e-12);
        assert!(pld.epsilon_at(0.1).is_infinite());
        assert!((total_mass(&pld.pmf_remove) - 1.0).abs() < 1e-12);
        assert!((total_mass(pld.pmf_add.as_ref().unwrap()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_generic_poisson_preserves_pld_metadata() {
        let base = crate::mechanisms::identity_pld(&default_config())
            .unwrap()
            .with_tail_budgets(1e-8, 2e-8)
            .with_monte_carlo_guarantee(1e-6, 1e-5);
        let pld = poisson_pld(&base, 0.2).unwrap();

        assert_eq!(pld.estimation_failure_probability(), 1e-6);
        assert_eq!(pld.mc_resolution(), 1e-5);
        assert_eq!(pld.pmf_remove.right_tail_budget, 1e-8);
        assert_eq!(pld.pmf_remove.left_tail_budget, 2e-8);
        let pmf_add = pld.pmf_add.as_ref().unwrap();
        assert_eq!(pmf_add.right_tail_budget, 1e-8);
        assert_eq!(pmf_add.left_tail_budget, 2e-8);
    }

    #[test]
    fn test_realization_repair_dominates_nonnegative_privacy_profile() {
        let base = crate::mechanisms::gaussian_pld(0.6, &default_config()).unwrap();
        let original_atoms: Vec<_> = base
            .pmf_remove
            .probs
            .iter()
            .enumerate()
            .filter_map(|(index, &mass)| {
                (mass > 0.0).then_some((base.pmf_remove.loss_at_index(index as i64), mass))
            })
            .collect();
        let original_log_moment = log_reciprocal_moment(&original_atoms);
        assert!(original_log_moment > 0.1);

        let repaired = realization_source(&base.pmf_remove).unwrap();
        let repaired_log_moment = repaired
            .atoms
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &(loss, mass)| {
                log_add(acc, mass.ln() - loss)
            });
        assert!(repaired_log_moment <= 0.0);
        assert!(repaired.infinity_mass > base.pmf_remove.infinity_mass);

        for epsilon in [0.0, 0.1, 1.0, 3.0, 10.0] {
            assert!(
                exact_delta(epsilon, &repaired.atoms, repaired.infinity_mass)
                    >= exact_delta(epsilon, &original_atoms, base.pmf_remove.infinity_mass),
                "repair underestimates the non-negative privacy profile at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_generic_poisson_ignores_beta_only_left_tail_bookkeeping() {
        let base = crate::mechanisms::identity_pld(&default_config())
            .unwrap()
            .with_tail_budgets(0.0, 1e-8)
            .self_compose(2)
            .unwrap();
        assert!(base.pmf_remove.negative_infinity_mass > 0.0);

        let pld = poisson_pld(&base, 0.2).unwrap();
        assert_eq!(pld.delta_at(0.0), 0.0);
        assert_eq!(pld.epsilon_at(1e-8), 0.0);
        assert!((total_mass(&pld.pmf_remove) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_generic_poisson_is_conservative_for_two_atom_base() {
        let mut config = default_config();
        config.discretization = 0.1;
        let epsilon_base = 0.3;
        let rate = 0.2;
        let base = crate::mechanisms::eps_delta_pld(epsilon_base, 0.0, &config).unwrap();
        let pld = poisson_pld(&base, rate).unwrap();

        let remove_support = [
            (
                (1.0 - rate + rate * epsilon_base.exp()).ln(),
                rate + (1.0 - rate) * (-epsilon_base).exp(),
            ),
            (
                (1.0 - rate).ln(),
                (1.0 - rate) * (1.0 - (-epsilon_base).exp()),
            ),
        ];
        let add_support = [(-(1.0 - rate + rate * (-epsilon_base).exp()).ln(), 1.0)];

        for epsilon in [-0.15, -0.05, 0.0, 0.025, 0.075, 0.15, 0.3] {
            let exact_remove = exact_delta(epsilon, &remove_support, 0.0);
            let exact_add = exact_delta(epsilon, &add_support, 0.0);
            assert!(
                pmf_delta(&pld.pmf_remove, epsilon)
                    >= exact_remove - ANALYTIC_DELTA_NUMERICAL_TOLERANCE,
                "REMOVE underestimates at ε={epsilon}"
            );
            assert!(
                pmf_delta(pld.pmf_add.as_ref().unwrap(), epsilon) >= exact_add - 1e-12,
                "ADD underestimates at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_generic_poisson_bounds_enumerated_binary_pair_after_composition() {
        let loss = 0.1_f64;
        let p_positive = (loss.exp() - 1.0) / (loss.exp() - (-loss).exp());
        let p_negative = 1.0 - p_positive;
        let q_negative = p_positive * (-loss).exp();
        let q_positive = p_negative * loss.exp();
        assert!((q_negative + q_positive - 1.0).abs() < 1e-12);
        let base = PrivacyLossDistribution::new_asymmetric(
            Pmf::new(loss, -1, vec![p_negative, p_positive], 0.0, usize::MAX),
            Pmf::new(loss, -1, vec![q_negative, q_positive], 0.0, usize::MAX),
        );
        let rate = 0.2;
        let count = 3;
        let pld = poisson_pld(&base, rate)
            .unwrap()
            .self_compose(count)
            .unwrap();

        let remove: Vec<_> = [(-loss, p_negative), (loss, p_positive)]
            .iter()
            .map(|&(base_loss, base_mass)| {
                (
                    (1.0 - rate + rate * base_loss.exp()).ln(),
                    base_mass * (rate + (1.0 - rate) * (-base_loss).exp()),
                )
            })
            .collect();
        let add: Vec<_> = [(-loss, q_negative), (loss, q_positive)]
            .iter()
            .map(|&(base_loss, base_mass)| {
                (
                    -(1.0 - rate + rate * (-base_loss).exp()).ln(),
                    base_mass * (-base_loss).exp(),
                )
            })
            .collect();
        let exact_remove = self_compose_support(&remove, count);
        let exact_add = self_compose_support(&add, count);
        let pmf_add = pld.pmf_add.as_ref().unwrap();

        for epsilon in [0.0, 0.025, 0.05, 0.1, 0.2, 0.5] {
            assert!(
                pmf_delta(&pld.pmf_remove, epsilon)
                    >= exact_delta(epsilon, &exact_remove, 0.0) - 1e-12,
                "REMOVE underestimates the enumerated pair at ε={epsilon}"
            );
            assert!(
                pmf_delta(pmf_add, epsilon) >= exact_delta(epsilon, &exact_add, 0.0) - 1e-12,
                "ADD underestimates the enumerated pair at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_generic_poisson_add_uses_connect_the_dots_projection() {
        let mut config = default_config();
        config.discretization = 0.1;
        let rate = 0.2;
        let base = crate::mechanisms::eps_delta_pld(0.3, 0.0, &config).unwrap();
        let pld = poisson_pld(&base, rate).unwrap();
        let pmf_add = pld.pmf_add.as_ref().unwrap();

        let exact_loss = -(1.0 - rate + rate * (-0.3f64).exp()).ln();
        assert!(exact_loss > 0.0 && exact_loss < config.discretization);
        assert_eq!(pmf_add.lower_loss_index, 0);
        assert_eq!(pmf_add.probs.len(), 2);
        assert!(pmf_add.probs[0] > 0.0);
        assert!(pmf_add.probs[1] > 0.0);
        assert!(
            (pmf_delta(pmf_add, 0.0) - exact_delta(0.0, &[(exact_loss, 1.0)], 0.0)).abs() < 1e-12
        );
        assert_eq!(pmf_delta(pmf_add, 0.1), 0.0);
        assert!(
            pmf_delta(pmf_add, exact_loss / 2.0)
                >= exact_delta(exact_loss / 2.0, &[(exact_loss, 1.0)], 0.0),
            "CtD interpolation must remain conservative between grid knots"
        );
    }

    #[test]
    fn test_generic_poisson_moves_add_infinity_to_finite_support() {
        let mut config = default_config();
        config.discretization = 0.1;
        let base = crate::mechanisms::eps_delta_pld(0.3, 0.2, &config).unwrap();
        let pld = poisson_pld(&base, 0.2).unwrap();
        let pmf_add = pld.pmf_add.as_ref().unwrap();

        assert_eq!(pmf_add.infinity_mass, 0.0);
        assert!((total_mass(pmf_add) - 1.0).abs() < 1e-12);
        assert!(pmf_add.loss_at_index(pmf_add.probs.len() as i64 - 1) >= -(0.8f64).ln());
    }

    #[test]
    fn test_generic_poisson_is_conservative_for_two_atom_base_with_delta() {
        let mut config = default_config();
        config.discretization = 0.1;
        let epsilon_base = 0.3;
        let delta_base = 0.2;
        let rate = 0.2;
        let base = crate::mechanisms::eps_delta_pld(epsilon_base, delta_base, &config).unwrap();
        let pld = poisson_pld(&base, rate).unwrap();

        let base_finite_mass = 1.0 - delta_base;
        let remove_support = [
            (
                (1.0 - rate + rate * epsilon_base.exp()).ln(),
                base_finite_mass * (rate + (1.0 - rate) * (-epsilon_base).exp()),
            ),
            (
                (1.0 - rate).ln(),
                (1.0 - rate) * (1.0 - base_finite_mass * (-epsilon_base).exp()),
            ),
        ];
        let add_support = [
            (
                -(1.0 - rate + rate * (-epsilon_base).exp()).ln(),
                base_finite_mass,
            ),
            (-(1.0 - rate).ln(), delta_base),
        ];

        for epsilon in [-0.15, -0.05, 0.0, 0.025, 0.075, 0.15, 0.3] {
            let exact_remove = exact_delta(epsilon, &remove_support, rate * delta_base);
            let exact_add = exact_delta(epsilon, &add_support, 0.0);
            assert!(
                pmf_delta(&pld.pmf_remove, epsilon) >= exact_remove - 1e-12,
                "REMOVE underestimates the two-atom base at ε={epsilon}"
            );
            assert!(
                pmf_delta(pld.pmf_add.as_ref().unwrap(), epsilon) >= exact_add - 1e-12,
                "ADD underestimates the two-atom base at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_generic_poisson_gaussian_directions_bound_analytic_delta() {
        let mut config = default_config();
        config.discretization = 1e-2;
        config.log_mass_truncation_bound = -25.0;
        let sigma = 1.0;
        let rate = 0.02;
        let base = crate::mechanisms::gaussian_pld(sigma, &config).unwrap();
        let pld = poisson_pld(&base, rate).unwrap();
        let pmf_add = pld.pmf_add.as_ref().unwrap();

        for epsilon in [0.0, 0.05, 0.2, 0.5] {
            let exact_remove =
                poisson_gaussian_get_delta(epsilon, Adjacency::Remove, sigma, 1.0, rate);
            let exact_add = poisson_gaussian_get_delta(epsilon, Adjacency::Add, sigma, 1.0, rate);
            assert!(
                pmf_delta(&pld.pmf_remove, epsilon) >= exact_remove - 1e-12,
                "REMOVE underestimates Gaussian delta at ε={epsilon}"
            );
            assert!(
                pmf_delta(pmf_add, epsilon) >= exact_add - ANALYTIC_DELTA_NUMERICAL_TOLERANCE,
                "ADD underestimates Gaussian delta at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_generic_low_noise_gaussian_bounds_analytic_delta() {
        let config = default_config();
        for &(sigma, rate) in &[
            (0.1, 0.0001),
            (0.3, 0.001),
            (0.45, 0.001),
            (0.6, 0.01),
            (0.8, 0.0001),
            (1.2, 0.01),
            (2.0, 0.2),
        ] {
            let base = crate::mechanisms::gaussian_pld(sigma, &config).unwrap();
            let generic = poisson_pld(&base, rate).unwrap();
            let pmf_add = generic.pmf_add.as_ref().unwrap();

            for epsilon in [0.0, 0.001, 0.01, 0.1, 1.0, 3.0] {
                let exact_remove =
                    poisson_gaussian_get_delta(epsilon, Adjacency::Remove, sigma, 1.0, rate);
                let exact_add =
                    poisson_gaussian_get_delta(epsilon, Adjacency::Add, sigma, 1.0, rate);
                let got_remove = pmf_delta(&generic.pmf_remove, epsilon);
                let got_add = pmf_delta(pmf_add, epsilon);
                assert!(
                    got_remove >= exact_remove - ANALYTIC_DELTA_NUMERICAL_TOLERANCE,
                    "REMOVE underestimates at σ={sigma}, q={rate}, ε={epsilon}: \
                     generic={got_remove:.17e}, exact={exact_remove:.17e}"
                );
                assert!(
                    got_add >= exact_add - ANALYTIC_DELTA_NUMERICAL_TOLERANCE,
                    "ADD underestimates at σ={sigma}, q={rate}, ε={epsilon}: \
                     generic={got_add:.17e}, exact={exact_add:.17e}"
                );
            }
        }
    }

    #[test]
    fn test_untagged_gaussian_pmf_repair_bounds_analytic_delta() {
        let config = default_config();
        let sigma = 0.45;
        let rate = 0.001;
        let tagged = crate::mechanisms::gaussian_pld(sigma, &config).unwrap();
        let untagged = PrivacyLossDistribution::new_symmetric(tagged.pmf_remove.clone());
        let generic = poisson_pld(&untagged, rate).unwrap();
        let pmf_add = generic.pmf_add.as_ref().unwrap();

        for epsilon in [0.0, 0.01, 0.1, 1.0] {
            let exact_remove =
                poisson_gaussian_get_delta(epsilon, Adjacency::Remove, sigma, 1.0, rate);
            let exact_add = poisson_gaussian_get_delta(epsilon, Adjacency::Add, sigma, 1.0, rate);
            assert!(
                pmf_delta(&generic.pmf_remove, epsilon) >= exact_remove - 1e-12,
                "REMOVE underestimates at ε={epsilon}"
            );
            assert!(
                pmf_delta(pmf_add, epsilon) >= exact_add - 1e-12,
                "ADD underestimates at ε={epsilon}"
            );
        }
    }

    #[test]
    fn test_gaussian_source_respects_max_grid_size_override() {
        let base = crate::mechanisms::gaussian_pld(0.6, &default_config())
            .unwrap()
            .with_max_grid_size(1_024);
        let pld = poisson_pld(&base, 0.001).unwrap();

        assert!(pld.pmf_remove.probs.len() <= 1_024);
        assert!(pld.pmf_add.as_ref().unwrap().probs.len() <= 1_024);
    }

    #[test]
    fn test_transformed_grid_stays_composition_compatible() {
        let config = default_config();
        let base = crate::mechanisms::gaussian_pld(0.6, &config).unwrap();
        let generic = poisson_pld(&base, 0.001).unwrap();

        for transformed in [&generic.pmf_remove, generic.pmf_add.as_ref().unwrap()] {
            let factor =
                (base.pmf_remove.discretization / transformed.discretization).round() as usize;
            assert!(factor.is_power_of_two());
        }

        let pure_dp = crate::mechanisms::eps_delta_pld(0.3, 0.0, &config).unwrap();
        let other = poisson_pld(&pure_dp, 0.2).unwrap();
        assert!(generic.compose(&other).is_ok());
    }

    #[test]
    fn test_generic_poisson_accepts_composed_base() {
        let base = crate::mechanisms::gaussian_pld(0.6, &default_config())
            .unwrap()
            .self_compose(100)
            .unwrap();
        let transformed = poisson_pld(&base, 0.01).unwrap();
        assert!(transformed.epsilon_at(1e-5).is_finite());
        assert!((total_mass(&transformed.pmf_remove) - 1.0).abs() < 1e-9);
        assert!((total_mass(transformed.pmf_add.as_ref().unwrap()) - 1.0).abs() < 1e-9);
    }
}
