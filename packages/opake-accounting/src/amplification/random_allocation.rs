//! Privacy amplification by random allocation, as a PLD transform.
//!
//! In `k`-out-of-`t` random allocation each record is used in `k` steps chosen
//! uniformly at random from `t`. Feldman & Shenfeld (arXiv:2602.17284) show the
//! resulting privacy loss is a transform of the base PLD:
//!
//! ```text
//! remove:  ψ⃗_t(L) = ln( (1/t)( e^{L₀} + Σ_{i=1}^{t-1} e^{-L̃ᵢ} ) )
//! add:     ψ⃖_t(L) = -ln( (1/t) Σ_{i=1}^{t} e^{-Lᵢ} )
//! ```
//!
//! where `L̃` is the PLD dual. Both are convolutions of *exponentiated* PLDs,
//! computed on a geometric grid (`crate::pld::pmf::geom`) by exponentiation by
//! squaring — `O(log t)` convolutions. The grid size does not grow across them,
//! which makes large `t` tractable. Each pairwise convolution contributes
//! at most `α` of privacy-loss discretisation error, so the total error for a
//! `t`-fold convolution is bounded by
//! `(⌊log₂ t⌋ + popcount(t) − 1) · α`.
//!
//! For the Gaussian the dual is closed form — `L̃ ~ N(1/(2σ²), 1/σ²)`, the same
//! law as `L` — so it is taken analytically rather than by reweighting a
//! discretised PLD by `e^{-l}`, which would amplify FFT residue.
//!
//! # Relation to balls-in-bins
//!
//! Opake's fixed-assignment balls-in-bins with the identity strategy is
//! *exactly* 1-out-of-`b` random allocation at `σ_eff = σ/√E`: with `C = I` the
//! mixture means are orthogonal with equal norm `√E`, so `G = E·I_b` and the
//! dominating pair collapses onto `(P̄_b, Q^b)` after rescaling.

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::geom::GeomPmf;
use crate::pld::realization::{disc_dist, LossRealization, NormalLoss, Rounding};
use crate::pld::PrivacyLossDistribution;

/// Accuracy knobs derived from the caller's existing discretisation config.
///
/// `(α, β)` are numerical accuracy parameters, not privacy parameters, so they
/// are not exposed as separate knobs — they follow `discretization` and
/// `log_mass_truncation_bound`, the dials the user already sets.
struct Accuracy {
    /// Interior grid width used inside the transform.
    alpha: f64,
    /// Per-step tail mass discarded at each end.
    beta: f64,
}

impl Accuracy {
    fn derive<L: LossRealization + ?Sized>(
        loss: &L,
        config: &DiscretizationConfig,
        t: usize,
    ) -> Result<Self> {
        if config.max_conv_grid == 0 {
            return Err(PldError::InvalidParameter(
                "max_conv_grid must be > 0".into(),
            ));
        }
        // β is split across the t copies that get convolved together.
        let beta_total = (0.5 * config.log_mass_truncation_bound.exp()).clamp(1e-300, 0.49);
        let beta = (beta_total / t as f64).clamp(1e-300, 0.49);

        // Widen α if the requested resolution would exceed the work budget.
        // config.max_conv_grid is the O(G²) grid cap for this transform;
        // it is separate from config.max_grid_size (the FFT-composition cap).
        let log_beta = beta.ln();
        let span = loss.quantile(log_beta, true) - loss.quantile(log_beta, false);
        let alpha_floor = span / config.max_conv_grid as f64;
        Ok(Accuracy {
            alpha: config.discretization.max(alpha_floor),
            beta,
        })
    }
}

/// Remove-direction PLD of 1-out-of-`t` random allocation.
fn alloc_remove<L: LossRealization + ?Sized>(
    loss: &L,
    neg_dual: &L,
    t: usize,
    acc: &Accuracy,
    config: &DiscretizationConfig,
    out_disc: f64,
) -> Result<crate::pld::pmf::Pmf> {
    let l = disc_dist(loss, acc.alpha, acc.beta, config.max_grid_size)?;
    if t == 1 {
        return GeomPmf::from_pmf_exp(&l)?.into_pmf_log(
            1.0,
            out_disc,
            config.max_grid_size,
            Rounding::Up,
        );
    }
    // disc-dist(-D): the negated dual, taken analytically.
    let d = disc_dist(neg_dual, acc.alpha, acc.beta, config.max_grid_size)?;

    let e_l = GeomPmf::from_pmf_exp(&l)?;
    let e_d = GeomPmf::from_pmf_exp(&d)?;

    let e_d_t1 = e_d.self_conv(t - 1, Rounding::Up)?;
    let e_t = e_d_t1.conv(&e_l, Rounding::Up)?;

    e_t.into_pmf_log(t as f64, out_disc, config.max_grid_size, Rounding::Up)
}

/// Add-direction PLD of 1-out-of-`t` random allocation.
fn alloc_add<L: LossRealization + ?Sized>(
    loss: &L,
    t: usize,
    acc: &Accuracy,
    config: &DiscretizationConfig,
    out_disc: f64,
) -> Result<crate::pld::pmf::Pmf> {
    let l = disc_dist(loss, acc.alpha, acc.beta, config.max_grid_size)?;
    if t == 1 {
        return GeomPmf::from_pmf_exp_neg(&l)?.into_pmf_neg_log(
            1.0,
            out_disc,
            config.max_grid_size,
            Rounding::Up,
        );
    }
    let e_l = GeomPmf::from_pmf_exp_neg(&l)?;
    // An upper bound on -ln(S) needs a *lower* bound on S, so the inner
    // convolution rounds down. This is a fixed internal step in constructing
    // the safe result, not a caller-selectable optimistic estimate.
    let e_t = e_l.self_conv(t, Rounding::Down)?;
    e_t.into_pmf_neg_log(t as f64, out_disc, config.max_grid_size, Rounding::Up)
}

/// PLD of `k`-out-of-`t` random allocation applied to the Gaussian mechanism.
///
/// Exact for `k = 1`. For `k > 1` the result is a valid **upper bound**, not
/// the exact `k`-out-of-`t` PLD: the `t` steps are split into `k` blocks and
/// the record is placed once per block, which is a strictly smaller family of
/// participation patterns than "any `k` of `t`".
///
/// That bound is sound by the same joint-convexity argument that makes
/// per-epoch redraw dominate fixed assignment. Drawing a uniformly random
/// partition of `[t]` into `k` blocks and then picking one step per block
/// induces exactly the uniform distribution over `k`-subsets — the whole
/// construction is invariant under permutations of `[t]`, which act
/// transitively on `k`-subsets. So the true mixture `P` is an average of the
/// block-scheme mixtures `P_π` over partitions `π`, every `P_π` gives the same
/// divergence by symmetry, and joint convexity of the hockey-stick divergence
/// gives `D_ε(P ‖ Q) ≤ D_ε(P_π ‖ Q)`.
///
/// The block scheme itself factorises across blocks (independent picks,
/// independent noise), so composing the per-block PLDs is exact.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0.
/// * `t` — steps per allocation round (the number of bins).
/// * `k` — steps each record is used in, in `[1, t]`. Values above 1 return
///   the block upper bound described above.
/// * `config` — discretisation configuration.
///
/// # Errors
///
/// `InvalidParameter` for out-of-range arguments or a grid exceeding
/// `max_grid_size`.
pub fn random_allocation_gaussian_pld(
    noise_multiplier: f64,
    t: usize,
    k: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if noise_multiplier.is_nan() || noise_multiplier <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be > 0, got {}",
            noise_multiplier
        )));
    }
    if t == 0 {
        return Err(PldError::InvalidParameter("t must be >= 1".into()));
    }
    if k == 0 || k > t {
        return Err(PldError::InvalidParameter(format!(
            "k must be in [1, t={}], got {}",
            t, k
        )));
    }

    // k > 1 reduces to a composition of single allocations: split t into
    // m_f rounds of ⌊t/k⌋ steps and m_c rounds of ⌈t/k⌉.
    let (t_floor, t_ceil) = (t / k, t.div_ceil(k));
    let (m_f, m_c) = if t_floor == t_ceil {
        (k, 0)
    } else {
        let m_c = t - t_floor * k;
        (k - m_c, m_c)
    };
    debug_assert_eq!(m_f + m_c, k);
    debug_assert_eq!(m_f * t_floor + m_c * t_ceil, t);

    let sigma = noise_multiplier;
    let loss = NormalLoss::gaussian(sigma);
    let neg_dual = NormalLoss::gaussian_neg_dual(sigma);

    let mut out: Option<PrivacyLossDistribution> = None;
    for &(rounds, steps) in &[(m_f, t_floor), (m_c, t_ceil)] {
        if rounds == 0 || steps == 0 {
            continue;
        }
        let acc = Accuracy::derive(&loss, config, steps)?;
        let pmf_remove =
            alloc_remove(&loss, &neg_dual, steps, &acc, config, config.discretization)?;
        let pmf_add = alloc_add(&loss, steps, &acc, config, config.discretization)?;
        let single = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);
        let composed = if rounds == 1 {
            single
        } else {
            single.self_compose(rounds)?
        };
        out = Some(match out {
            None => composed,
            Some(prev) => prev.compose(&composed)?,
        });
    }

    out.ok_or_else(|| PldError::InvalidParameter("empty allocation decomposition".into()))
        .map(|pld| {
            let tail = config.tail_mass_truncation / 2.0;
            pld.with_tail_budgets(tail, tail)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplification::poisson::poisson_pld;
    use crate::mechanisms::gaussian_pld;

    fn cfg() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    /// t = 1 is no allocation at all: it must be the base Gaussian.
    #[test]
    fn test_t_one_is_base_gaussian() {
        let c = cfg();
        for &sigma in &[0.5, 1.0, 2.0] {
            let ra = random_allocation_gaussian_pld(sigma, 1, 1, &c).unwrap();
            let base = gaussian_pld(sigma, &c).unwrap();
            let (a, b) = (ra.epsilon_at(1e-8), base.epsilon_at(1e-8));
            assert!(
                (a - b).abs() < 1e-3 * b.max(1.0),
                "σ={}: allocation {} vs base {}",
                sigma,
                a,
                b
            );
        }
    }

    /// More noise, less privacy loss.
    #[test]
    #[ignore = "slow"]
    fn test_monotone_in_sigma() {
        let c = cfg();
        let a = random_allocation_gaussian_pld(1.0, 8, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let b = random_allocation_gaussian_pld(2.0, 8, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(b < a, "more noise should lower eps: {} vs {}", b, a);
    }

    /// Golden values, cross-validated against an independent Python
    /// implementation of the same paper (which bracketed
    /// [3.711, 3.726], [1.792, 1.806], [0.315, 0.345], [1.253, 1.270]).
    /// Opake's safe bound must land inside that independent bracket.
    #[test]
    #[ignore = "slow"]
    fn test_golden_epsilons() {
        let c = cfg();
        for &(sigma, t, want_lo, want_hi) in &[
            (1.0, 8usize, 3.711, 3.727),
            (1.0, 64, 1.792, 1.807),
            (2.0, 64, 0.314, 0.346),
            (1.0, 128, 1.252, 1.271),
        ] {
            let got = random_allocation_gaussian_pld(sigma, t, 1, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                got >= want_lo && got <= want_hi,
                "σ={} t={}: {} outside the cross-validated [{}, {}]",
                sigma,
                t,
                got,
                want_lo,
                want_hi
            );
        }
    }

    /// More steps to hide among, less privacy loss.
    #[test]
    #[ignore = "slow"]
    fn test_monotone_in_t() {
        let c = cfg();
        let mut prev = f64::INFINITY;
        for &t in &[4usize, 8, 16] {
            let e = random_allocation_gaussian_pld(1.0, t, 1, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(e < prev, "eps should fall with t: {} at t={}", e, t);
            prev = e;
        }
    }

    /// More participations, more privacy loss.
    #[test]
    #[ignore = "slow"]
    fn test_monotone_in_k() {
        let c = cfg();
        let e1 = random_allocation_gaussian_pld(1.0, 16, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let e4 = random_allocation_gaussian_pld(1.0, 16, 4, &c)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(e4 > e1, "eps should rise with k: {} vs {}", e4, e1);
    }

    /// `k = t` is the one `k > 1` point with a known answer, so it pins the
    /// block arithmetic against an independent construction.
    ///
    /// Every block has size 1, the record is in every step, and there is no
    /// allocation left to amplify — the answer is the base Gaussian composed
    /// `t` times. An `m_f`/`m_c` split that mis-sized the blocks or mis-counted
    /// the rounds would land nowhere near it.
    ///
    /// Not an equality: at block size 1 this path returns `disc_dist`'s
    /// directionally-rounded grid while `gaussian_pld` uses connect-the-dots,
    /// so the two agree only up to discretisation (~3e-4 relative here). What
    /// must hold exactly is the *direction* — the upper variant may never fall
    /// below the truth.
    #[test]
    fn test_k_equals_t_is_full_participation() {
        let c = cfg();
        for &t in &[2usize, 5, 8] {
            let alloc = random_allocation_gaussian_pld(1.0, t, t, &c)
                .unwrap()
                .epsilon_at(1e-8);
            let every_step = gaussian_pld(1.0, &c)
                .unwrap()
                .self_compose(t)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                alloc >= every_step - 1e-9,
                "t={}: k=t gave {}, below full participation {}",
                t,
                alloc,
                every_step
            );
            assert!(
                alloc <= every_step * 1.01,
                "t={}: k=t gave {}, far above full participation {}",
                t,
                alloc,
                every_step
            );
        }
    }

    /// The `k > 1` block decomposition must cover exactly `t` steps in exactly
    /// `k` rounds for every divisibility case, including `k ∤ t`.
    #[test]
    fn test_block_decomposition_covers_t() {
        for t in 1usize..=24 {
            for k in 1..=t {
                let (t_floor, t_ceil) = (t / k, (t + k - 1) / k);
                let (m_f, m_c) = if t_floor == t_ceil {
                    (k, 0)
                } else {
                    let m_c = t - t_floor * k;
                    (k - m_c, m_c)
                };
                assert_eq!(m_f + m_c, k, "t={} k={}: round count", t, k);
                assert_eq!(
                    m_f * t_floor + m_c * t_ceil,
                    t,
                    "t={} k={}: step coverage",
                    t,
                    k
                );
            }
        }
    }

    /// Random allocation should be at least competitive with Poisson at the
    /// matched rate 1/t — the paper's headline claim.
    #[test]
    fn test_competitive_with_poisson() {
        let c = cfg();
        let (sigma, t) = (1.0, 32usize);
        let ra = random_allocation_gaussian_pld(sigma, t, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let base = gaussian_pld(sigma, &c).unwrap();
        let po = poisson_pld(&base, 1.0 / t as f64)
            .unwrap()
            .self_compose(t)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(
            ra <= po * 1.10,
            "σ={} t={}: allocation {} vs Poisson {}",
            sigma,
            t,
            ra,
            po
        );
    }

    #[test]
    fn test_rejects_bad_params() {
        let c = cfg();
        assert!(random_allocation_gaussian_pld(0.0, 8, 1, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 0, 1, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 8, 0, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 8, 9, &c).is_err());
        // max_conv_grid = 0 must be rejected before the span/0 division.
        let c_zero = DiscretizationConfig::default().with_max_conv_grid(0);
        assert!(random_allocation_gaussian_pld(1.0, 8, 1, &c_zero).is_err());
    }

    /// At t=1 the transform equals the base Gaussian (no allocation
    /// amplification), so the upper bound must be ≥ the Gaussian ε regardless
    /// of grid size (soundness check from the issue).
    #[test]
    fn test_t_one_upper_bound_sound_across_grid_sizes() {
        for &sigma in &[0.5, 1.0, 2.0] {
            let exact = crate::mechanisms::gaussian_pld(sigma, &cfg())
                .unwrap()
                .epsilon_at(1e-8);
            for &g in &[4096_usize, 8192, 16384, 32768] {
                let c = DiscretizationConfig::default().with_max_conv_grid(g);
                let upper = random_allocation_gaussian_pld(sigma, 1, 1, &c)
                    .unwrap()
                    .epsilon_at(1e-8);
                assert!(
                    upper >= exact - 1e-9,
                    "σ={} G={}: upper bound {} below exact Gaussian {}",
                    sigma,
                    g,
                    upper,
                    exact
                );
            }
        }
    }

    /// A larger max_conv_grid cap produces a tighter (lower or equal) ε
    /// upper bound — it never loosens the bound.
    #[test]
    #[ignore = "slow"]
    fn test_larger_max_conv_grid_tightens_bound() {
        let c_coarse = DiscretizationConfig::default().with_max_conv_grid(8_192);
        let c_fine = DiscretizationConfig::default().with_max_conv_grid(32_768);
        for &(sigma, t) in &[(2.0_f64, 64_usize), (1.0, 128)] {
            let eps_coarse = random_allocation_gaussian_pld(sigma, t, 1, &c_coarse)
                .unwrap()
                .epsilon_at(1e-8);
            let eps_fine = random_allocation_gaussian_pld(sigma, t, 1, &c_fine)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                eps_fine <= eps_coarse + 1e-9,
                "σ={} t={}: finer grid gave looser bound: {} > {}",
                sigma,
                t,
                eps_fine,
                eps_coarse
            );
        }
    }
}
