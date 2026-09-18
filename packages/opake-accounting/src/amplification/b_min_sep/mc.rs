//! Monte Carlo privacy accounting for BandMF with warm-start b-min-sep subsampling.
//!
//! Implements the likelihood-ratio dynamic program from Dong & Ganesh (2026),
//! "Privacy Amplification for BandMF via b-Min-Sep Subsampling" (arXiv:2602.09338),
//! Section 5, Equation (1), and the warm-start corollary after Theorem 5.1:
//!
//! `P(y)/Q(y) = (f_1(y) + p * sum_{i=2}^{b} f_i(y)) / (1 + (b-1)p)`
//!
//! where `f_i` is the backward recursion on iteration index (1-based in the paper).
//! Remove samples use `y ~ P` and `ln(P/Q)`; add samples use `y ~ Q` and `ln(Q/P)`.

use crate::amplification::balls_in_bins::monte_carlo::samples_to_pmf;
use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::numerics::logspace::{log_add, log_sumexp};
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// `log N(μ, σ²I)(y) - log N(0, σ²I)(y)`.
/// Samples per deterministic Monte Carlo stream.
const SAMPLES_PER_SHARD: usize = 1024;

#[derive(Clone, Copy)]
#[repr(u64)]
enum AdjacencyDirection {
    Remove = 0,
    Add = 1,
}

fn shard_seed(seed: u64, shard: usize, direction: AdjacencyDirection) -> u64 {
    seed.wrapping_add((shard as u64).wrapping_mul(2))
        .wrapping_add(direction as u64)
}

fn checked_sample_cells(num_samples: usize, n_steps: usize) -> Result<usize> {
    num_samples
        .checked_mul(n_steps)
        .ok_or_else(|| PldError::InvalidParameter("sample count and horizon are too large".into()))
}

/// Evaluates warm-start b-min-sep `ln(P(y)/Q(y))` with reusable storage.
struct WarmLogLikelihoodRatio<'a> {
    coef: &'a [f64],
    prefix_norm_sq: Vec<f64>,
    sigma2: f64,
    log_p: f64,
    log_one_minus_p: f64,
    log_denominator: f64,
    log_f: Vec<f64>,
}

impl<'a> WarmLogLikelihoodRatio<'a> {
    fn new(coef: &'a [f64], p: f64, sigma2: f64) -> Self {
        Self {
            coef,
            prefix_norm_sq: coef
                .iter()
                .scan(0.0, |sum, &mean| {
                    *sum += mean * mean;
                    Some(*sum)
                })
                .collect(),
            sigma2,
            log_p: p.ln(),
            log_one_minus_p: (-p).ln_1p(),
            log_denominator: ((coef.len().saturating_sub(1)) as f64 * p).ln_1p(),
            log_f: Vec::new(),
        }
    }

    fn evaluate(&mut self, y: &[f64]) -> f64 {
        let n = y.len();
        let bands = self.coef.len();
        if bands == 0 {
            return 0.0;
        }

        self.log_f.resize(n + bands, 0.0);
        // Boundary: log f_i = 0 for i >= n.
        self.log_f[n..].fill(0.0);

        for i in (0..n).rev() {
            let block_len = bands.min(n - i);
            let dot = self.coef[..block_len]
                .iter()
                .zip(&y[i..i + block_len])
                .fold(0.0, |sum, (&mean, &value)| sum + mean * value);
            let log_block = (dot - 0.5 * self.prefix_norm_sq[block_len - 1]) / self.sigma2;
            self.log_f[i] = log_add(
                self.log_one_minus_p + self.log_f[i + 1],
                self.log_p + log_block + self.log_f[i + bands],
            );
        }

        let log_numerator = log_add(
            self.log_f[0],
            self.log_p + log_sumexp(&self.log_f[1..bands]),
        );
        log_numerator - self.log_denominator
    }
}

fn sample_y_under_q(n: usize, sigma: f64, rng: &mut impl Rng, buf: &mut [f64]) {
    for v in buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal) * sigma;
    }
}

/// Draw participation `x` under warm-start b-min-sep at fixed `p` (noise independent).
fn sample_x_under_p(n: usize, bands: usize, p: f64, rng: &mut impl Rng, x_buf: &mut [f64]) {
    x_buf.fill(0.0);
    let mut barred_remaining: usize = if bands <= 1 {
        0
    } else {
        let denom = 1.0 + (bands - 1) as f64 * p;
        if rng.random::<f64>() * denom < 1.0 {
            0
        } else {
            1 + rng.random_range(0..bands - 1)
        }
    };

    for xb in x_buf.iter_mut().take(n) {
        let eligible = barred_remaining == 0;
        let participate = eligible && rng.random::<f64>() < p;
        *xb = if participate { 1.0 } else { 0.0 };
        if participate {
            barred_remaining = bands.saturating_sub(1);
        } else {
            barred_remaining = barred_remaining.saturating_sub(1);
        }
    }
}

/// `y = Cx + σ ζ`, visiting only participating columns of `C`.
fn y_from_x_and_zeta(
    coef: &[f64],
    n: usize,
    x: &[f64],
    zeta: &[f64],
    sigma: f64,
    y_out: &mut [f64],
) {
    for (value, &noise) in y_out[..n].iter_mut().zip(&zeta[..n]) {
        *value = sigma * noise;
    }
    for (j, &participation) in x[..n].iter().enumerate() {
        if participation == 0.0 {
            continue;
        }
        for (&mean, value) in coef.iter().zip(&mut y_out[j..n]) {
            *value += mean * participation;
        }
    }
}

/// Standard-normal draws for the Q branch (`y = σ η`).
fn sample_eta_under_q(n: usize, rng: &mut impl Rng, buf: &mut [f64]) {
    for v in buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal);
    }
}

/// Prepare Monte Carlo transcripts for reuse across many `σ` values (calibration).
///
/// Returns flattened row-major arrays of length `num_samples * n_steps`:
/// - `remove_x`, `remove_zeta` for the P-branch (`y = Cx + σ ζ`)
/// - `add_eta` for the Q-branch (`y = σ η`)
///
/// Rows use fixed, seed-indexed shards shared with one-shot sampling. Shards are
/// independent of the Rayon pool size, so changing worker count preserves the
/// generated corpus for a fixed build.
pub fn bandmf_b_min_sep_prepare_transcripts(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    num_samples: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be > 0".into()));
    }
    if !(p > 0.0 && p <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "p must be in (0, 1], got {}",
            p
        )));
    }
    if num_samples == 0 {
        return Err(PldError::InvalidParameter("num_samples must be > 0".into()));
    }

    let bands = strategy_coef.len();
    let sample_cells = checked_sample_cells(num_samples, n_steps)?;
    let shard_cells = checked_sample_cells(SAMPLES_PER_SHARD, n_steps)?;
    let mut remove_x = vec![0.0; sample_cells];
    let mut remove_zeta = vec![0.0; sample_cells];
    let mut add_eta = vec![0.0; sample_cells];

    rayon::join(
        || {
            remove_x
                .par_chunks_mut(shard_cells)
                .zip(remove_zeta.par_chunks_mut(shard_cells))
                .enumerate()
                .for_each(|(shard, (x_shard, zeta_shard))| {
                    let mut rng =
                        StdRng::seed_from_u64(shard_seed(seed, shard, AdjacencyDirection::Remove));
                    for (x, zeta) in x_shard
                        .chunks_exact_mut(n_steps)
                        .zip(zeta_shard.chunks_exact_mut(n_steps))
                    {
                        sample_x_under_p(n_steps, bands, p, &mut rng, x);
                        sample_eta_under_q(n_steps, &mut rng, zeta);
                    }
                });
        },
        || {
            add_eta
                .par_chunks_mut(shard_cells)
                .enumerate()
                .for_each(|(shard, eta_shard)| {
                    let mut rng =
                        StdRng::seed_from_u64(shard_seed(seed, shard, AdjacencyDirection::Add));
                    for eta in eta_shard.chunks_exact_mut(n_steps) {
                        sample_eta_under_q(n_steps, &mut rng, eta);
                    }
                });
        },
    );

    Ok((remove_x, remove_zeta, add_eta))
}

/// Build PLD from precomputed transcripts at noise multiplier `sigma`.
#[allow(clippy::too_many_arguments)]
pub fn bandmf_b_min_sep_pld_from_transcripts(
    remove_x: &[f64],
    remove_zeta: &[f64],
    add_eta: &[f64],
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be > 0".into()));
    }
    if add_eta.is_empty() || add_eta.len() % n_steps != 0 {
        return Err(PldError::InvalidParameter(
            "add_eta length must be positive multiple of n_steps".into(),
        ));
    }
    let num_samples = add_eta.len() / n_steps;
    if remove_x.len() != add_eta.len() || remove_zeta.len() != add_eta.len() {
        return Err(PldError::InvalidParameter(
            "remove_x/remove_zeta/add_eta length mismatch".into(),
        ));
    }
    if sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }
    let sigma2 = sigma * sigma;
    let mut remove_samples = vec![0.0; num_samples];
    let mut add_samples = vec![0.0; num_samples];

    rayon::join(
        || {
            remove_samples
                .par_iter_mut()
                .zip(
                    remove_x
                        .par_chunks(n_steps)
                        .zip(remove_zeta.par_chunks(n_steps)),
                )
                .for_each_init(
                    || {
                        (
                            vec![0.0; n_steps],
                            WarmLogLikelihoodRatio::new(strategy_coef, p, sigma2),
                        )
                    },
                    |(y, evaluator), (sample, (x, zeta))| {
                        y_from_x_and_zeta(strategy_coef, n_steps, x, zeta, sigma, y);
                        *sample = evaluator.evaluate(y);
                    },
                );
        },
        || {
            add_samples
                .par_iter_mut()
                .zip(add_eta.par_chunks(n_steps))
                .for_each_init(
                    || {
                        (
                            vec![0.0; n_steps],
                            WarmLogLikelihoodRatio::new(strategy_coef, p, sigma2),
                        )
                    },
                    |(y, evaluator), (sample, eta)| {
                        for (value, noise) in y.iter_mut().zip(eta) {
                            *value = sigma * noise;
                        }
                        *sample = -evaluator.evaluate(y);
                    },
                );
        },
    );

    let (pmf_remove, remove_resolution) = samples_to_pmf(&mut remove_samples, config, 2)?;
    let (pmf_add, add_resolution) = samples_to_pmf(&mut add_samples, config, 2)?;

    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_monte_carlo_guarantee(
            config.mc_failure_probability,
            remove_resolution.max(add_resolution),
        ),
    )
}

/// Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling (single-example adjacent analysis).
///
/// * `strategy_coef` — first column of strategy matrix `C` (length = bandwidth).
/// * `n_steps` — number of iterations `n`.
/// * `p` — per-iteration inclusion probability `p` in Algorithm 2 (not `p_0`).
/// * `sigma` — raw noise multiplier σ (same units as [`crate::amplification::bnb_mc_pld`]).
/// * `config` — discretization and Monte Carlo confidence configuration.
pub fn bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let num_samples = config.resolved_num_mc_samples(2)?;
    let seed = config.seed;

    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be > 0".into()));
    }
    if !(p > 0.0 && p <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "p must be in (0, 1], got {}",
            p
        )));
    }
    if sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }
    let bands = strategy_coef.len();
    let sigma2 = sigma * sigma;
    let mut remove_samples = vec![0.0; num_samples];
    let mut add_samples = vec![0.0; num_samples];

    rayon::join(
        || {
            remove_samples
                .par_chunks_mut(SAMPLES_PER_SHARD)
                .enumerate()
                .for_each_init(
                    || {
                        (
                            vec![0.0; n_steps],
                            vec![0.0; n_steps],
                            vec![0.0; n_steps],
                            WarmLogLikelihoodRatio::new(strategy_coef, p, sigma2),
                        )
                    },
                    |state, (shard, samples)| {
                        let (x, zeta, y, evaluator) = state;
                        let mut rng = StdRng::seed_from_u64(shard_seed(
                            seed,
                            shard,
                            AdjacencyDirection::Remove,
                        ));
                        for sample in samples {
                            sample_x_under_p(n_steps, bands, p, &mut rng, x);
                            sample_eta_under_q(n_steps, &mut rng, zeta);
                            y_from_x_and_zeta(strategy_coef, n_steps, x, zeta, sigma, y);
                            *sample = evaluator.evaluate(y);
                        }
                    },
                );
        },
        || {
            add_samples
                .par_chunks_mut(SAMPLES_PER_SHARD)
                .enumerate()
                .for_each_init(
                    || {
                        (
                            vec![0.0; n_steps],
                            WarmLogLikelihoodRatio::new(strategy_coef, p, sigma2),
                        )
                    },
                    |(y, evaluator), (shard, samples)| {
                        let mut rng =
                            StdRng::seed_from_u64(shard_seed(seed, shard, AdjacencyDirection::Add));
                        for sample in samples {
                            sample_y_under_q(n_steps, sigma, &mut rng, y);
                            *sample = -evaluator.evaluate(y);
                        }
                    },
                );
        },
    );

    let (pmf_remove, remove_resolution) = samples_to_pmf(&mut remove_samples, config, 2)?;
    let (pmf_add, add_resolution) = samples_to_pmf(&mut add_samples, config, 2)?;

    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_monte_carlo_guarantee(
            config.mc_failure_probability,
            remove_resolution.max(add_resolution),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplification::poisson_pld;
    use crate::discretization::DiscretizationConfig;
    use crate::mechanisms::gaussian_pld;
    use crate::pld::Pmf;
    use approx::assert_abs_diff_eq;
    use std::collections::HashSet;

    fn warm_path_probability(mask: u64, n: usize, bands: usize, p: f64) -> f64 {
        let denominator = 1.0 + (bands - 1) as f64 * p;
        (0..bands)
            .map(|initial_state| {
                let mut probability = if initial_state == 0 { 1.0 } else { p } / denominator;
                let mut barred_remaining = initial_state;

                for i in 0..n {
                    let participates = mask & (1 << i) != 0;
                    if barred_remaining > 0 {
                        if participates {
                            return 0.0;
                        }
                        barred_remaining -= 1;
                    } else if participates {
                        probability *= p;
                        barred_remaining = bands - 1;
                    } else {
                        probability *= 1.0 - p;
                    }
                }
                probability
            })
            .sum()
    }

    fn exact_warm_ratio(y: &[f64], coef: &[f64], p: f64, sigma2: f64) -> f64 {
        let n = y.len();
        assert!(n < u64::BITS as usize);

        (0..1_u64 << n)
            .map(|mask| {
                let probability = warm_path_probability(mask, n, coef.len(), p);
                if probability == 0.0 {
                    return 0.0;
                }
                let mut mean = vec![0.0; n];
                for col in 0..n {
                    if mask & (1 << col) != 0 {
                        for (offset, value) in coef.iter().take(n - col).enumerate() {
                            mean[col + offset] += *value;
                        }
                    }
                }
                let log_ratio: f64 = mean
                    .iter()
                    .zip(y)
                    .map(|(mu, value)| mu * value - 0.5 * mu * mu)
                    .sum::<f64>()
                    / sigma2;
                probability * log_ratio.exp()
            })
            .sum()
    }

    fn assert_float_slices_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert_eq!(left.to_bits(), right.to_bits(), "index {index}");
        }
    }

    fn with_threads<T: Send>(num_threads: usize, op: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap()
            .install(op)
    }

    fn assert_pmf_eq(left: &Pmf, right: &Pmf) {
        assert_eq!(
            left.discretization.to_bits(),
            right.discretization.to_bits()
        );
        assert_eq!(left.lower_loss_index, right.lower_loss_index);
        assert_float_slices_eq(&left.probs, &right.probs);
        assert_eq!(left.infinity_mass.to_bits(), right.infinity_mass.to_bits());
        assert_eq!(
            left.negative_infinity_mass.to_bits(),
            right.negative_infinity_mass.to_bits()
        );
        assert_eq!(left.max_grid_size, right.max_grid_size);
        assert_eq!(
            left.right_tail_budget.to_bits(),
            right.right_tail_budget.to_bits()
        );
        assert_eq!(
            left.left_tail_budget.to_bits(),
            right.left_tail_budget.to_bits()
        );
    }

    fn assert_pld_eq(left: &PrivacyLossDistribution, right: &PrivacyLossDistribution) {
        assert_pmf_eq(&left.pmf_remove, &right.pmf_remove);
        match (&left.pmf_add, &right.pmf_add) {
            (Some(left), Some(right)) => assert_pmf_eq(left, right),
            (None, None) => {}
            _ => panic!("adjacency mismatch"),
        }
        assert_eq!(
            left.estimation_failure_probability().to_bits(),
            right.estimation_failure_probability().to_bits()
        );
        assert_eq!(
            left.mc_resolution().to_bits(),
            right.mc_resolution().to_bits()
        );
        assert_eq!(left.gaussian_source(), right.gaussian_source());
    }

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig {
            mc_resolution: 5e-3,
            mc_failure_probability: 1e-2,
            ..DiscretizationConfig::default()
        }
    }

    #[test]
    fn privacy_loss_matches_jax_reference() {
        let loss = WarmLogLikelihoodRatio::new(&[1.0, 0.5], 0.5, 1.0).evaluate(&[1.0, 1.0, 1.0]);
        assert_abs_diff_eq!(loss, 0.832_939_838_080_925_2, epsilon = 1e-14);
    }

    #[test]
    fn privacy_loss_matches_exact_mixture() {
        let cases = [
            (
                vec![0.8, 0.3],
                vec![-0.7, 0.1, 0.9, -0.2, 0.4, 0.6, -0.5, 0.3],
                0.17,
                1.44,
            ),
            (
                vec![0.7, 0.4, 0.2],
                vec![0.2, -0.4, 0.8, 0.1, -0.6, 0.5, 0.9, -0.3, 0.7],
                0.5,
                0.81,
            ),
            (
                vec![0.6, 0.5, 0.4, 0.3],
                vec![0.2, 0.4, -0.1, 0.8],
                1.0,
                1.0,
            ),
            (vec![1.0], vec![0.3, -0.2, 0.7, 0.1, -0.4], 0.999, 1.21),
        ];

        for (coef, y, p, sigma2) in cases {
            let expected = exact_warm_ratio(&y, &coef, p, sigma2).ln();
            let actual = WarmLogLikelihoodRatio::new(&coef, p, sigma2).evaluate(&y);
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn noisy_trajectory_applies_truncated_toeplitz_columns() {
        let coef = [1.0, 0.5, 0.25];
        let noise = [1.0, -2.0, 0.0, 2.0, -1.0, 0.5, -1.0, 0.0];
        let mut y = [0.0; 8];
        let x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        y_from_x_and_zeta(&coef, y.len(), &x, &noise, 0.5, &mut y);
        assert_eq!(y, [1.5, -0.5, 0.25, 2.0, 0.0, 0.5, 0.5, 0.5]);

        y_from_x_and_zeta(&coef, y.len(), &[0.0; 8], &noise, 1.0, &mut y);
        assert_eq!(y, noise);
    }

    #[test]
    fn zero_signal_has_zero_privacy_loss() {
        for bands in [2, 4, 8] {
            let coef = vec![0.0; bands];
            let y = vec![0.3; 12];
            assert_abs_diff_eq!(
                WarmLogLikelihoodRatio::new(&coef, 0.2, 4.0).evaluate(&y),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn bandwidth_one_reduces_to_poisson_subsampling() {
        let y = [0.3, -0.2, 0.7, 0.1, -0.4];
        let coef = [1.1];
        let p: f64 = 0.23;
        let sigma2 = 1.21;
        let expected = y
            .iter()
            .map(|value| {
                let log_gaussian_ratio = (coef[0] * value - 0.5 * coef[0] * coef[0]) / sigma2;
                log_add((-p).ln_1p(), p.ln() + log_gaussian_ratio)
            })
            .sum::<f64>();

        assert_abs_diff_eq!(
            WarmLogLikelihoodRatio::new(&coef, p, sigma2).evaluate(&y),
            expected,
            epsilon = 1e-14
        );
    }

    #[test]
    fn privacy_loss_remains_finite_when_ratio_overflows() {
        let n = 1_000;
        let y = vec![1.0; n];
        let loss = WarmLogLikelihoodRatio::new(&[1.0, 0.0], 1.0, 0.25).evaluate(&y);
        assert_abs_diff_eq!(loss, 1_000.0, epsilon = 1e-10);
    }

    #[test]
    fn likelihood_ratio_normalizes_under_q() {
        let n = 10;
        let p = 0.2;
        let sigma = 2.0;
        let mut rng = StdRng::seed_from_u64(779);
        let mut y = vec![0.0; n];

        for bands in [2, 4, 8] {
            let coef = vec![1.0 / (bands as f64).sqrt(); bands];
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma * sigma);
            let mean = (0..20_000)
                .map(|_| {
                    sample_y_under_q(n, sigma, &mut rng, &mut y);
                    evaluator.evaluate(&y).exp()
                })
                .sum::<f64>()
                / 20_000.0;
            assert_abs_diff_eq!(mean, 1.0, epsilon = 0.02);
        }
    }

    #[test]
    fn mc_pld_smoke() {
        let coef = vec![0.7_f64.sqrt(), 0.3_f64.sqrt()];
        let cfg = default_config();
        let pld = bandmf_b_min_sep_warm_mc_pld(&coef, 50, 0.05, 1.0, &cfg).unwrap();
        let eps = pld.epsilon_at(1e-2);
        assert!(eps > 0.0 && eps.is_finite());
    }

    #[test]
    fn bandwidth_one_mc_bounds_poisson_gaussian_composition() {
        let n_steps = 8;
        let p = 0.15;
        let sigma = 1.2;
        let mut cfg = default_config();
        cfg.seed = 779;

        let mc = bandmf_b_min_sep_warm_mc_pld(&[1.0], n_steps, p, sigma, &cfg).unwrap();
        let gaussian = gaussian_pld(sigma, &cfg).unwrap();
        let analytic = poisson_pld(&gaussian, p)
            .unwrap()
            .self_compose(n_steps)
            .unwrap();

        for epsilon in [0.0, 0.25, 0.5, 1.0, 2.0] {
            let mc_delta = mc.delta_at(epsilon);
            let analytic_delta = analytic.delta_at(epsilon);
            assert!(
                mc_delta >= analytic_delta,
                "bandwidth-one MC bound underestimates at ε={epsilon}: \
                 mc={mc_delta:.17e}, analytic={analytic_delta:.17e}"
            );
        }
    }

    #[test]
    fn shard_seeds_separate_adjacency_directions() {
        let num_samples = DiscretizationConfig::default()
            .resolved_num_mc_samples(2)
            .unwrap();
        let num_shards = num_samples.div_ceil(SAMPLES_PER_SHARD);
        let mut seeds = HashSet::with_capacity(2 * num_shards);

        for shard in 0..num_shards {
            for direction in [AdjacencyDirection::Remove, AdjacencyDirection::Add] {
                assert!(seeds.insert(shard_seed(u64::MAX - 10, shard, direction)));
            }
        }
    }

    #[test]
    fn transcripts_match_one_shot() {
        let coef = vec![0.8_f64.sqrt(), 0.2_f64.sqrt(), 0.0];
        let mut cfg = default_config();
        cfg.seed = 99;
        let sigma = 1.15;
        let n = 40;
        let p = 0.06;
        let s = cfg.resolved_num_mc_samples(2).unwrap();
        let (rx, rz, ae) = bandmf_b_min_sep_prepare_transcripts(&coef, n, p, s, cfg.seed).unwrap();
        let pld_t =
            bandmf_b_min_sep_pld_from_transcripts(&rx, &rz, &ae, &coef, n, p, sigma, &cfg).unwrap();
        let pld_1 = bandmf_b_min_sep_warm_mc_pld(&coef, n, p, sigma, &cfg).unwrap();
        assert_pld_eq(&pld_1, &pld_t);
    }

    #[test]
    fn transcripts_are_identical_across_thread_counts() {
        let coef = vec![0.7, 0.2, 0.1];
        let prepare = || bandmf_b_min_sep_prepare_transcripts(&coef, 7, 0.08, 2_065, 173).unwrap();
        let single_thread = with_threads(1, prepare);
        let many_threads = with_threads(8, prepare);

        assert_float_slices_eq(&single_thread.0, &many_threads.0);
        assert_float_slices_eq(&single_thread.1, &many_threads.1);
        assert_float_slices_eq(&single_thread.2, &many_threads.2);
    }

    #[test]
    fn transcript_evaluation_is_identical_across_thread_counts() {
        let coef = vec![0.7, 0.2, 0.1];
        let mut cfg = default_config();
        cfg.seed = 173;
        let n = 12;
        let p = 0.08;
        let sigma = 1.2;
        let num_samples = cfg.resolved_num_mc_samples(2).unwrap();
        let (x, zeta, eta) =
            bandmf_b_min_sep_prepare_transcripts(&coef, n, p, num_samples, cfg.seed).unwrap();
        let evaluate = || {
            bandmf_b_min_sep_pld_from_transcripts(&x, &zeta, &eta, &coef, n, p, sigma, &cfg)
                .unwrap()
        };

        let single_thread = with_threads(1, evaluate);
        let many_threads = with_threads(8, evaluate);
        assert_pld_eq(&single_thread, &many_threads);
    }

    #[test]
    fn mc_pld_is_identical_across_thread_counts() {
        let coef = vec![0.8_f64.sqrt(), 0.2_f64.sqrt(), 0.0];
        let mut cfg = default_config();
        cfg.seed = 173;
        let build = || bandmf_b_min_sep_warm_mc_pld(&coef, 25, 0.06, 1.15, &cfg).unwrap();
        let single_thread = with_threads(1, build);
        let many_threads = with_threads(8, build);

        assert_pld_eq(&single_thread, &many_threads);
    }
}
