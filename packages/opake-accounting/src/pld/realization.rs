//! Privacy-loss realizations and directional rounding.
//!
//! Connect-the-Dots builds a PMF by inverting a δ(ε) curve, which is tight in
//! the **hockey-stick** sense. The random-allocation transform of Feldman &
//! Shenfeld (arXiv:2602.17284) is stated for **first-order stochastic**
//! domination, which is strictly stronger — so its inputs cannot come from
//! Connect-the-Dots. This module supplies the alternative: a privacy loss
//! known analytically and discretised by CDF binning, which is stochastically
//! dominating by construction.
//!
//! This is *not* a competitor to Connect-the-Dots. CTD remains the way
//! mechanism PLDs are built and is better at it; `disc_dist` exists only for
//! transform inputs.

use crate::error::{PldError, Result};
use crate::numerics::special::gaussian_log_cdf;
use statrs::distribution::{ContinuousCDF, Normal};

/// Direction for geometric-grid operations inside the allocation transform.
///
/// `Down` is only used to lower-bound an exponentiated sum before the
/// add-direction transform applies `-ln`, which reverses the ordering. It is
/// never exposed as an accounting result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rounding {
    /// Round toward +∞.
    Up,
    /// Round toward −∞ for a required intermediate calculation.
    Down,
}

/// A privacy-loss random variable known in closed form.
///
/// Quantiles are taken in log-space: at the tail masses this transform needs
/// (`β/t` with `β ≈ 1e-15`), a naive `ppf(1 - β)` has already rounded its
/// argument to exactly 1.0 and returns `inf`.
pub(crate) trait LossRealization: Sync {
    /// `log P[L ≤ l]`.
    fn log_cdf(&self, l: f64) -> f64;

    /// `log P[L > l]`.
    fn log_survival(&self, l: f64) -> f64;

    /// Quantile in log-space. With `upper_tail = false` returns
    /// `inf{ l : P[L ≤ l] ≥ exp(log_p) }`; with `upper_tail = true` returns the
    /// symmetric upper quantile, i.e. the value with `P[L > l] = exp(log_p)`.
    fn quantile(&self, log_p: f64, upper_tail: bool) -> f64;
}

/// A normal privacy loss `N(mean, sd²)`.
///
/// The Gaussian mechanism's remove-direction loss is `N(1/(2σ²), 1/σ²)`, and —
/// usefully — its PLD dual has the *same* law, so the dual never has to be
/// taken numerically on this path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NormalLoss {
    pub mean: f64,
    pub sd: f64,
}

impl NormalLoss {
    /// `ℓ(P‖Q)` for the Gaussian mechanism: `N(1/(2σ²), 1/σ²)`.
    pub(crate) fn gaussian(sigma: f64) -> Self {
        let var = 1.0 / (sigma * sigma);
        NormalLoss {
            mean: 0.5 * var,
            sd: 1.0 / sigma,
        }
    }

    /// `-L̃`, the negated PLD dual, which `rand-alloc-rem` discretises.
    ///
    /// For the Gaussian, `L̃ = ℓ(Q‖P)` has the same law as `L = ℓ(P‖Q)` —
    /// both `N(1/(2σ²), 1/σ²)` — so `-L̃ ~ N(-1/(2σ²), 1/σ²)`. Deriving it in
    /// closed form here keeps the numerically delicate `e^{-l}` reweighting of
    /// a discrete dual off the critical path entirely.
    pub(crate) fn gaussian_neg_dual(sigma: f64) -> Self {
        let g = Self::gaussian(sigma);
        NormalLoss {
            mean: -g.mean,
            sd: g.sd,
        }
    }
}

impl LossRealization for NormalLoss {
    fn log_cdf(&self, l: f64) -> f64 {
        gaussian_log_cdf((l - self.mean) / self.sd)
    }

    fn log_survival(&self, l: f64) -> f64 {
        gaussian_log_cdf((self.mean - l) / self.sd)
    }

    fn quantile(&self, log_p: f64, upper_tail: bool) -> f64 {
        // exp(log_p) stays representable well past the tail masses used here
        // (f64 min normal ≈ 2.2e-308), whereas `1 - exp(log_p)` saturates to
        // 1.0 by log_p ≈ -37. So always invert the *lower* tail and reflect.
        let p = log_p.exp().clamp(f64::MIN_POSITIVE, 0.5);
        let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(p); // z < 0
        if upper_tail {
            self.mean - self.sd * z
        } else {
            self.mean + self.sd * z
        }
    }
}

/// Discretise an analytic privacy loss onto Opake's uniform loss grid.
///
/// The result stochastically dominates the analytic law: every unit of mass
/// moves up to the nearest grid point, while the right tail becomes `+∞`.
///
/// The grid must sit on multiples of `alpha` to satisfy `Pmf`'s convention
/// that bucket `i` carries loss `(lower_loss_index + i) · discretization`.
///
/// `beta` is the tail mass discarded at each end. The left tail moves to the
/// first grid point and the right tail moves to the conservative `+∞` atom.
pub(crate) fn disc_dist<L: LossRealization + ?Sized>(
    loss: &L,
    alpha: f64,
    beta: f64,
    max_grid_size: usize,
) -> Result<crate::pld::pmf::Pmf> {
    if alpha.is_nan() || alpha <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "alpha must be > 0, got {}",
            alpha
        )));
    }
    if beta.is_nan() || beta <= 0.0 || beta >= 0.5 {
        return Err(PldError::InvalidParameter(format!(
            "beta must be in (0, 0.5), got {}",
            beta
        )));
    }

    let log_beta = beta.ln();
    let l_min = loss.quantile(log_beta, false);
    let l_max = loss.quantile(log_beta, true);
    if !l_min.is_finite() || !l_max.is_finite() || l_max <= l_min {
        return Err(PldError::NumericalError(format!(
            "degenerate quantile range [{}, {}] at beta={}",
            l_min, l_max, beta
        )));
    }

    // Snap to the grid: index k carries loss k·alpha.
    let lo_idx = (l_min / alpha).floor() as i64;
    let hi_idx = (l_max / alpha).ceil() as i64;
    let n = (hi_idx - lo_idx + 1) as usize;
    if n > max_grid_size {
        return Err(PldError::InvalidParameter(format!(
            "grid of {} points exceeds max_grid_size {} — widen `discretization` \
             or raise `log_x_mass_truncation_bound`",
            n, max_grid_size
        )));
    }

    let log_cdf: Vec<f64> = (0..n)
        .map(|i| loss.log_cdf((lo_idx + i as i64) as f64 * alpha))
        .collect();
    let log_survival: Vec<f64> = (0..n)
        .map(|i| loss.log_survival((lo_idx + i as i64) as f64 * alpha))
        .collect();

    let mut probs = vec![0.0f64; n];
    // Mass on (l_{i-1}, l_i] lands on l_i; everything below l_0 lands
    // on l_0; everything above l_{n-1} goes to +∞. In the right tail,
    // subtract survival probabilities instead of `1 - CDF`: CDF values there
    // round to one long before the privacy-relevant tail is empty.
    probs[0] = log_cdf[0].exp();
    for i in 1..n {
        probs[i] = if log_cdf[i] <= -std::f64::consts::LN_2 {
            exp_log_difference(log_cdf[i], log_cdf[i - 1])
        } else {
            exp_log_difference(log_survival[i - 1], log_survival[i])
        };
    }

    Ok(crate::pld::pmf::Pmf {
        discretization: alpha,
        lower_loss_index: lo_idx,
        probs,
        infinity_mass: log_survival[n - 1].exp(),
        negative_infinity_mass: 0.0,
        max_grid_size,
        right_tail_budget: 0.0,
        left_tail_budget: 0.0,
    })
}

/// `exp(log_large) - exp(log_small)` without cancellation.
///
/// The inputs should satisfy `log_large >= log_small`; equal or inverted
/// values can only arise from floating-point ties, and represent zero mass.
fn exp_log_difference(log_large: f64, log_small: f64) -> f64 {
    if log_large <= log_small {
        0.0
    } else {
        (log_large + (-((log_small - log_large).exp())).ln_1p()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn total_mass(p: &crate::pld::pmf::Pmf) -> f64 {
        p.probs.iter().sum::<f64>() + p.infinity_mass + p.negative_infinity_mass
    }

    #[test]
    fn test_gaussian_dual_has_same_law() {
        // ℓ(P‖Q) and ℓ(Q‖P) are both N(1/(2σ²), 1/σ²) for the Gaussian.
        for &sigma in &[0.3, 0.5, 1.0, 2.0, 5.0] {
            let l = NormalLoss::gaussian(sigma);
            let neg_dual = NormalLoss::gaussian_neg_dual(sigma);
            assert_relative_eq!(neg_dual.mean, -l.mean, epsilon = 1e-15);
            assert_relative_eq!(neg_dual.sd, l.sd, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_quantiles_survive_extreme_tails() {
        let l = NormalLoss::gaussian(1.0);
        for &log_p in &[-10.0, -30.0, -50.0, -80.0] {
            let lo = l.quantile(log_p, false);
            let hi = l.quantile(log_p, true);
            assert!(lo.is_finite(), "lower quantile inf at log_p={}", log_p);
            assert!(hi.is_finite(), "upper quantile inf at log_p={}", log_p);
            assert!(lo < l.mean && hi > l.mean);
            // Symmetric about the mean.
            assert_relative_eq!(hi - l.mean, l.mean - lo, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_disc_dist_conserves_mass() {
        for &sigma in &[0.5, 1.0, 3.0] {
            let pmf = disc_dist(&NormalLoss::gaussian(sigma), 1e-3, 1e-12, 10_000_000).unwrap();
            assert_relative_eq!(total_mass(&pmf), 1.0, epsilon = 1e-9);
        }
    }

    /// The discretized PMF stochastically dominates the exact law, checked on
    /// the CCDF pointwise.
    #[test]
    fn test_disc_dist_dominates_exact_ccdf() {
        let sigma = 1.0;
        let loss = NormalLoss::gaussian(sigma);
        let alpha = 1e-2;
        let pmf = disc_dist(&loss, alpha, 1e-12, 10_000_000).unwrap();

        let ccdf = |p: &crate::pld::pmf::Pmf, x: f64| -> f64 {
            let mut acc = p.infinity_mass;
            for (i, &m) in p.probs.iter().enumerate() {
                if (p.lower_loss_index + i as i64) as f64 * p.discretization > x {
                    acc += m;
                }
            }
            acc
        };

        for k in -30..30 {
            let x = k as f64 * 0.1;
            let exact = loss.log_survival(x).exp();
            assert!(
                ccdf(&pmf, x) >= exact - 1e-9,
                "discretization must dominate at x={}: {} < {}",
                x,
                ccdf(&pmf, x),
                exact
            );
        }
    }

    /// Right-tail mass must survive grids where `1 - CDF` would round to zero.
    #[test]
    fn test_disc_dist_preserves_right_tail_mass() {
        let loss = NormalLoss::gaussian(1.0);
        let pmf = disc_dist(&loss, 1e-2, 1e-50, 10_000_000).unwrap();
        assert!(pmf.infinity_mass > 0.0);
        assert_relative_eq!(total_mass(&pmf), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_rejects_bad_params() {
        let l = NormalLoss::gaussian(1.0);
        assert!(disc_dist(&l, 0.0, 1e-12, 1000).is_err());
        assert!(disc_dist(&l, 1e-3, 0.0, 1000).is_err());
        assert!(disc_dist(&l, 1e-3, 0.9, 1000).is_err());
        // Grid too large for the cap.
        assert!(disc_dist(&l, 1e-9, 1e-12, 100).is_err());
    }
}
