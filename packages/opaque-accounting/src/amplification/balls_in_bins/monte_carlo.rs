//! Monte Carlo BnB accountant for matrix mechanisms.
//!
//! Implements Algorithm 2 of Choquette-Choo et al. (2024) "Near Exact Privacy
//! Amplification for Matrix Mechanisms" (arxiv:2410.06266).
//!
//! Given a Gram matrix G of the BnB dominating pair mixture means, samples the
//! privacy loss distribution via Monte Carlo and returns a discretized PLD.
//!
//! The dominating pair is:
//!   P = (1/b) Σ_{i=1}^{b} N(m_i, σ²I)
//!   Q = N(0, σ²I)
//!
//! After Gram matrix reduction, the privacy loss depends on G only:
//!   Y = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
//! where u ~ N(G_i, σ²G) with i ~ Uniform([b]).
//!
//! # Performance
//!
//! For DP-λCGD the Gram is *cyclically* banded — `G_{ij} ≈ E·λ^d +
//! (E-1)·λ^{b-d}` for `d = |i-j|`, so entries decay away from the diagonal and
//! then rise again at the corner. We therefore use a **cyclically banded
//! Cholesky** (`super::cyclic_cholesky`), which keeps the per-sample cost at
//! O(b·p) while retaining the wrap that a linear band would discard.
//!
//! # References
//!
//! - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification for Matrix
//!   Mechanisms" <https://arxiv.org/abs/2410.06266>

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::collections::BTreeMap;

use super::cyclic_cholesky::CyclicBandedCholesky;

/// Number of samples assigned to one deterministic Monte Carlo stream.
///
/// The shard count must depend only on the resolved sample count, not on the Rayon
/// worker count: users expect a fixed seed to select the same samples on every
/// machine.
const SAMPLES_PER_SHARD: usize = 1024;

fn shard_seed(seed: u64, shard: usize, add_direction: bool) -> u64 {
    // Every possible shard has an even offset, reserving the adjacent odd
    // offset for the add direction. `num_samples` bounds shard well below
    // u64::MAX / 2, so the multiplication cannot overflow.
    seed.wrapping_add((shard as u64) * 2 + u64::from(add_direction))
}

fn validate_and_factor(gram: &[f64], b: usize, sigma: Option<f64>) -> Result<CyclicBandedCholesky> {
    if gram.len() != b * b {
        return Err(PldError::InvalidParameter(format!(
            "Gram matrix size {} doesn't match num_bins²={}",
            gram.len(),
            b * b
        )));
    }
    if let Some(value) = sigma {
        if value <= 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "sigma must be > 0, got {}",
                value
            )));
        }
    }

    let chol = CyclicBandedCholesky::compute(gram, b, 1e-6)?;
    let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
    let residual = chol.max_residual(gram);
    if residual > 1e-8 * max_diag.max(1.0) {
        return Err(PldError::NumericalError(format!(
            "Cholesky does not reproduce the Gram matrix: max|G - LLᵀ| = {} \
             (tolerance {}, b={}). The sampled covariance would not be G.",
            residual,
            1e-8 * max_diag.max(1.0),
            b
        )));
    }
    Ok(chol)
}

/// Sample one privacy loss value from the BnB dominating pair.
///
/// For the "remove" direction: X ~ P, Y = log(P(X)/Q(X))
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn sample_privacy_loss_remove(
    gram: &[f64],
    chol: &CyclicBandedCholesky,
    b: usize,
    sigma: f64,
    inv_2sig2: f64,
    diag_terms: &[f64],
    z_buf: &mut Vec<f64>,
    u_buf: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> f64 {
    // Step 1: Sample bin i ~ Uniform([b])
    let i: usize = rng.random_range(0..b);

    // Step 2: Sample z ~ N(0, I_b)
    for v in z_buf.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal);
    }

    // Step 3: Compute u = G[i,:] + σ * L * z (banded, O(b*bw))
    let mean = &gram[i * b..i * b + b];
    chol.sample_gaussian(mean, sigma, z_buf, u_buf);

    // Step 4: Y = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
    //        = log_sum_exp(2*u_k*inv_2sig2 + diag_terms) - log(b)
    let mut max_val = f64::NEG_INFINITY;
    for k in 0..b {
        let t = u_buf[k] * inv_2sig2 * 2.0 + diag_terms[k];
        u_buf[k] = t; // reuse buffer for terms
        if t > max_val {
            max_val = t;
        }
    }
    let mut sum_exp = 0.0f64;
    for val in &u_buf[..b] {
        sum_exp += (val - max_val).exp();
    }
    max_val + sum_exp.ln() - (b as f64).ln()
}

/// Sample one privacy loss value for the "add" direction.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn sample_privacy_loss_add(
    chol: &CyclicBandedCholesky,
    b: usize,
    sigma: f64,
    inv_2sig2: f64,
    diag_terms: &[f64],
    zero_mean: &[f64],
    z_buf: &mut Vec<f64>,
    u_buf: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> f64 {
    // Sample z and compute u = 0 + σ * L * z (mean is 0 under Q)
    for v in z_buf.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal);
    }
    chol.sample_gaussian(zero_mean, sigma, z_buf, u_buf);

    // Y_add = -log(P(X)/Q(X))
    let mut max_val = f64::NEG_INFINITY;
    for k in 0..b {
        let t = u_buf[k] * inv_2sig2 * 2.0 + diag_terms[k];
        u_buf[k] = t;
        if t > max_val {
            max_val = t;
        }
    }
    let mut sum_exp = 0.0f64;
    for val in &u_buf[..b] {
        sum_exp += (val - max_val).exp();
    }
    -(max_val + sum_exp.ln() - (b as f64).ln())
}

/// Compute the BnB PLD via Monte Carlo sampling.
///
/// Samples privacy loss values from the dominating pair, builds a histogram
/// on the PLD discretization grid, and returns a standard
/// `PrivacyLossDistribution` (asymmetric: separate remove/add PLDs).
///
/// # Arguments
///
/// * `gram` — b×b Gram matrix (row-major, symmetric positive definite)
/// * `num_bins` — Number of bins b
/// * `sigma` — Noise multiplier
/// * `config` — Discretization and Monte Carlo confidence configuration
pub fn bnb_mc_pld(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let b = num_bins;
    let num_samples = config.resolved_num_mc_samples(2)?;
    let seed = config.seed;

    // Cyclically banded Cholesky: the Gram wraps, so a linear band would
    // discard the corner (up to ~68% of the diagonal for λ-CGD).
    let chol = validate_and_factor(gram, b, Some(sigma))?;

    let sigma2 = sigma * sigma;
    let inv_2sig2 = 1.0 / (2.0 * sigma2);

    // Precompute -G_kk / (2σ²) for the log-sum-exp
    let diag_terms: Vec<f64> = (0..b).map(|k| -gram[k * b + k] * inv_2sig2).collect();

    // Partition samples into fixed, seed-indexed shards. Rayon may execute
    // these shards on any number of workers without changing the streams or
    // their output order.
    // Sample "remove" direction
    let mut remove_samples = vec![0.0f64; num_samples];
    remove_samples
        .par_chunks_mut(SAMPLES_PER_SHARD)
        .enumerate()
        .for_each(|(shard, samples)| {
            let mut rng = StdRng::seed_from_u64(shard_seed(seed, shard, false));
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            for sample in samples {
                *sample = sample_privacy_loss_remove(
                    gram,
                    &chol,
                    b,
                    sigma,
                    inv_2sig2,
                    &diag_terms,
                    &mut z_buf,
                    &mut u_buf,
                    &mut rng,
                );
            }
        });

    // Sample "add" direction
    let mut add_samples = vec![0.0f64; num_samples];
    add_samples
        .par_chunks_mut(SAMPLES_PER_SHARD)
        .enumerate()
        .for_each(|(shard, samples)| {
            let mut rng = StdRng::seed_from_u64(shard_seed(seed, shard, true));
            let zero_mean = vec![0.0; b];
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            for sample in samples {
                *sample = sample_privacy_loss_add(
                    &chol,
                    b,
                    sigma,
                    inv_2sig2,
                    &diag_terms,
                    &zero_mean,
                    &mut z_buf,
                    &mut u_buf,
                    &mut rng,
                );
            }
        });

    // Build PMFs from samples
    let (pmf_remove, remove_resolution) = samples_to_pmf(&mut remove_samples, config, 2)?;
    let (pmf_add, add_resolution) = samples_to_pmf(&mut add_samples, config, 2)?;

    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_monte_carlo_guarantee(
            config.mc_failure_probability,
            remove_resolution.max(add_resolution),
        ),
    )
}

/// Prepare sigma-independent BnB draws for reuse across calibration probes.
///
/// The projected standard-normal draws `Lz` are retained rather than raw `z`,
/// avoiding both RNG and cyclic Cholesky work on every subsequent sigma.
pub(crate) fn bnb_prepare_transcripts(
    gram: &[f64],
    num_bins: usize,
    num_samples: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    if num_samples == 0 {
        return Err(PldError::InvalidParameter("num_samples must be > 0".into()));
    }
    let b = num_bins;
    let cells = num_samples
        .checked_mul(b)
        .ok_or_else(|| PldError::InvalidParameter("sample count and bins are too large".into()))?;
    let chol = validate_and_factor(gram, b, None)?;
    let shard_cells = SAMPLES_PER_SHARD
        .checked_mul(b)
        .ok_or_else(|| PldError::InvalidParameter("sample shard is too large".into()))?;

    let mut remove_components = vec![0usize; num_samples];
    let mut remove_lz = vec![0.0; cells];
    let mut add_lz = vec![0.0; cells];
    let zero_mean = vec![0.0; b];

    rayon::join(
        || {
            remove_components
                .par_chunks_mut(SAMPLES_PER_SHARD)
                .zip(remove_lz.par_chunks_mut(shard_cells))
                .enumerate()
                .for_each(|(shard, (components, projected))| {
                    let mut rng = StdRng::seed_from_u64(shard_seed(seed, shard, false));
                    let mut z = vec![0.0; b];
                    for (component, lz) in components.iter_mut().zip(projected.chunks_mut(b)) {
                        *component = rng.random_range(0..b);
                        for value in &mut z {
                            *value = rng.sample::<f64, _>(StandardNormal);
                        }
                        chol.sample_gaussian(&zero_mean, 1.0, &z, lz);
                    }
                });
        },
        || {
            add_lz
                .par_chunks_mut(shard_cells)
                .enumerate()
                .for_each(|(shard, projected)| {
                    let mut rng = StdRng::seed_from_u64(shard_seed(seed, shard, true));
                    let mut z = vec![0.0; b];
                    for lz in projected.chunks_mut(b) {
                        for value in &mut z {
                            *value = rng.sample::<f64, _>(StandardNormal);
                        }
                        chol.sample_gaussian(&zero_mean, 1.0, &z, lz);
                    }
                });
        },
    );
    Ok((remove_components, remove_lz, add_lz))
}

/// Build a BnB PLD from sigma-independent projected draws.
pub(crate) fn bnb_pld_from_transcripts(
    gram: &[f64],
    num_bins: usize,
    remove_components: &[usize],
    remove_lz: &[f64],
    add_lz: &[f64],
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let b = num_bins;
    if gram.len() != b * b {
        return Err(PldError::InvalidParameter(format!(
            "Gram matrix size {} doesn't match num_bins²={}",
            gram.len(),
            b * b
        )));
    }
    if sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }
    let num_samples = remove_components.len();
    let cells = num_samples
        .checked_mul(b)
        .ok_or_else(|| PldError::InvalidParameter("sample count and bins are too large".into()))?;
    if num_samples == 0 || remove_lz.len() != cells || add_lz.len() != cells {
        return Err(PldError::InvalidParameter(
            "BnB transcript dimensions do not match num_bins".into(),
        ));
    }
    if remove_components.iter().any(|&component| component >= b) {
        return Err(PldError::InvalidParameter(
            "BnB transcript component is outside num_bins".into(),
        ));
    }

    let sigma2 = sigma * sigma;
    let inv_2sig2 = 1.0 / (2.0 * sigma2);
    let diag_terms: Vec<f64> = (0..b).map(|k| -gram[k * b + k] * inv_2sig2).collect();
    let log_b = (b as f64).ln();
    let mut remove_samples = vec![0.0; num_samples];
    let mut add_samples = vec![0.0; num_samples];

    rayon::join(
        || {
            remove_samples
                .par_iter_mut()
                .zip(remove_components.par_iter().zip(remove_lz.par_chunks(b)))
                .for_each(|(sample, (&component, lz))| {
                    let mean = &gram[component * b..component * b + b];
                    let mut max_val = f64::NEG_INFINITY;
                    for k in 0..b {
                        let u = mean[k] + sigma * lz[k];
                        max_val = max_val.max(u * inv_2sig2 * 2.0 + diag_terms[k]);
                    }
                    let mut sum_exp = 0.0;
                    for k in 0..b {
                        let u = mean[k] + sigma * lz[k];
                        let term = u * inv_2sig2 * 2.0 + diag_terms[k];
                        sum_exp += (term - max_val).exp();
                    }
                    *sample = max_val + sum_exp.ln() - log_b;
                });
        },
        || {
            add_samples
                .par_iter_mut()
                .zip(add_lz.par_chunks(b))
                .for_each(|(sample, lz)| {
                    let mut max_val = f64::NEG_INFINITY;
                    for k in 0..b {
                        let u = sigma * lz[k];
                        max_val = max_val.max(u * inv_2sig2 * 2.0 + diag_terms[k]);
                    }
                    let mut sum_exp = 0.0;
                    for k in 0..b {
                        let u = sigma * lz[k];
                        let term = u * inv_2sig2 * 2.0 + diag_terms[k];
                        sum_exp += (term - max_val).exp();
                    }
                    *sample = -(max_val + sum_exp.ln() - log_b);
                });
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

/// Convert MC samples into a simultaneous upper-confidence PLD.
///
/// # Errors
///
/// Returns `NumericalError` on any non-finite sample. The privacy loss here is
/// `log((1/b) Σ exp(·))` of a strictly positive, finite sum, so it is finite
/// whenever the sampler is well-conditioned — a `NaN` or `±inf` means the
/// Cholesky factor blew up, not that the true privacy loss is infinite.
/// Previously `NaN` (which is neither `is_finite()` nor `> 0.0`) was dropped
/// outright and `-inf` vanished, so the PMF silently lost mass and `Pmf::new`
/// does no renormalisation.
pub(crate) fn samples_to_pmf(
    samples: &mut [f64],
    config: &DiscretizationConfig,
    num_directions: usize,
) -> Result<(Pmf, f64)> {
    if samples.is_empty() {
        return Err(PldError::InvalidParameter(
            "Monte Carlo samples must be non-empty".into(),
        ));
    }

    if let Some((idx, bad)) = samples
        .iter()
        .enumerate()
        .find(|(_, y)| !y.is_finite())
        .map(|(i, y)| (i, *y))
    {
        return Err(PldError::NumericalError(format!(
            "non-finite privacy loss sample {} at index {} of {}; the sampler \
             is ill-conditioned (check the Cholesky factor and the Gram matrix)",
            bad,
            idx,
            samples.len()
        )));
    }

    if num_directions == 0 {
        return Err(PldError::InvalidParameter(
            "num_directions must be > 0".into(),
        ));
    }
    let required = config.resolved_num_mc_samples(num_directions)?;
    if samples.len() < required {
        return Err(PldError::InvalidParameter(format!(
            "{} samples are insufficient; {} are required for \
             mc_resolution={} and mc_failure_probability={}",
            samples.len(),
            required,
            config.mc_resolution,
            config.mc_failure_probability
        )));
    }

    samples.par_sort_unstable_by(f64::total_cmp);
    let n = samples.len();
    let min_sample = samples[0];
    let max_sample = samples[n - 1];

    let grid_lo = (min_sample / config.discretization).floor() as i64 - 1;
    let grid_hi = (max_sample / config.discretization).ceil() as i64 + 1;
    let num_buckets = (grid_hi - grid_lo + 1) as usize;

    let effective_disc = if num_buckets > config.max_grid_size {
        if config.max_grid_size < 3 {
            return Err(PldError::InvalidParameter(
                "max_grid_size must be at least 3 for Monte Carlo PLDs".into(),
            ));
        }
        let ratio = (num_buckets as f64 / config.max_grid_size as f64).ceil() as usize;
        config.discretization * ratio.next_power_of_two() as f64
    } else {
        config.discretization
    };

    // For rank j and p < j/n, the binomial Chernoff bound gives
    //
    // P[F(X_(j)) < p] <= exp(-n * KL(j/n || p)).
    //
    // Setting the right side to failure/(directions*n) and union-bounding over
    // every rank and adjacency direction yields a simultaneous lower CDF band.
    // Its final-rank residual is exactly the expression used by
    // `resolved_num_mc_samples`, but unlike exact Beta inversion it remains
    // fast and numerically stable for multi-million-sample shapes.
    let log_inverse_rank_failure =
        (num_directions as f64 * n as f64 / config.mc_failure_probability).ln();
    let mut endpoints = Vec::new();
    let mut start = 0usize;
    while start < n {
        let bucket = (samples[start] / effective_disc).ceil() as i64;
        let mut end = start + 1;
        while end < n && (samples[end] / effective_disc).ceil() as i64 == bucket {
            end += 1;
        }
        endpoints.push((bucket, end));
        start = end;
    }

    let cdf_bounds: Vec<f64> = endpoints
        .par_iter()
        .map(|(_, end)| kl_lower_cdf_bound(*end, n, log_inverse_rank_failure).clamp(0.0, 1.0))
        .collect();

    let mut masses = BTreeMap::new();
    let mut previous_cdf_bound = 0.0f64;
    for ((bucket, _), cdf_bound) in endpoints.into_iter().zip(cdf_bounds) {
        let cdf_bound = cdf_bound.max(previous_cdf_bound);
        let mass = cdf_bound - previous_cdf_bound;
        if mass > 0.0 {
            masses.insert(bucket, mass);
        }
        previous_cdf_bound = cdf_bound;
    }

    let infinity_mass = (1.0 - previous_cdf_bound).clamp(0.0, 1.0);
    if infinity_mass > config.mc_resolution * (1.0 + 1e-10) {
        return Err(PldError::NumericalError(format!(
            "Monte Carlo confidence construction achieved resolution {}, \
             exceeding requested {}",
            infinity_mass, config.mc_resolution
        )));
    }
    let pmf = Pmf::from_sparse(effective_disc, masses, infinity_mass, config.max_grid_size);
    let total: f64 = pmf.probs.iter().sum::<f64>() + pmf.infinity_mass;
    if (total - 1.0).abs() > 1e-9 {
        return Err(PldError::NumericalError(format!(
            "confidence PLD lost mass: total={total}"
        )));
    }
    Ok((pmf, infinity_mass))
}

/// Lower confidence bound on `F(X_(rank))` from binary KL inversion.
///
/// The returned value is kept on the conservative (lower) side of the root.
fn kl_lower_cdf_bound(rank: usize, n: usize, log_inverse_failure: f64) -> f64 {
    let q = rank as f64 / n as f64;
    let target = log_inverse_failure / n as f64;
    let mut lower = 0.0f64;
    let mut upper = q;

    for _ in 0..64 {
        let midpoint = 0.5 * (lower + upper);
        if binary_kl(q, midpoint) >= target {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    lower
}

fn binary_kl(q: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return f64::INFINITY;
    }
    if q >= 1.0 {
        return -p.ln();
    }
    q * (q / p).ln() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{Beta, ContinuousCDF};

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig {
            mc_resolution: 5e-3,
            mc_failure_probability: 1e-2,
            ..DiscretizationConfig::default()
        }
    }

    /// The window that used to produce |L| ~ 1e219 and `inf`.
    ///
    /// At λ=0.9, E=4, momentum=0 and b in [278, 295] the *linearly* banded
    /// factorisation of the (now corner-retaining) Gram is not PSD. The old
    /// code regularised the pivot to sqrt(1e-30)=1e-15 and then divided by it;
    /// the resulting non-finite losses were discarded silently by
    /// `samples_to_pmf`, so the PMF quietly lost mass.
    ///
    /// The cyclic factorisation is PSD here, so this must now simply work —
    /// and in no case may it return a non-finite ε.
    #[test]
    fn test_no_blowup_in_former_non_psd_window() {
        use crate::matrix_factorization::lambda_cgd_gram_matrix;

        let cfg = default_config();
        for &b in &[278usize, 280, 282, 285, 290, 295] {
            let e = 4;
            let gram = lambda_cgd_gram_matrix(0.9, b * e, b, Some(e), true, 0.0).unwrap();
            let pld = bnb_mc_pld(&gram, b, 1.5, &cfg)
                .unwrap_or_else(|err| panic!("b={} failed: {}", b, err));
            let eps = pld.epsilon_at(1e-2);
            assert!(eps.is_finite() && eps > 0.0, "b={}: eps={}", b, eps);
        }
    }

    /// Non-finite samples must error, not vanish.
    #[test]
    fn test_samples_to_pmf_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut samples = vec![0.1, 0.2, bad, 0.3];
            let mut config = default_config();
            config.mc_resolution = 0.99;
            let err = samples_to_pmf(&mut samples, &config, 2);
            assert!(err.is_err(), "{} must be rejected", bad);
        }
    }

    /// Every finite sample lands somewhere: the histogram conserves mass.
    #[test]
    fn test_samples_to_pmf_conserves_mass() {
        let mut samples: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) * 0.01).collect();
        let mut config = default_config();
        config.mc_resolution = 0.02;
        let (pmf, resolution) = samples_to_pmf(&mut samples, &config, 2).unwrap();
        let total: f64 = pmf.probs.iter().sum::<f64>() + pmf.infinity_mass;
        assert!((total - 1.0).abs() < 1e-9, "total mass = {}", total);
        assert!(resolution > 0.0);
        assert!(resolution <= config.mc_resolution);
        assert!(PrivacyLossDistribution::new_symmetric(pmf)
            .with_monte_carlo_guarantee(config.mc_failure_probability, resolution)
            .epsilon_at(resolution)
            .is_infinite());
    }

    #[test]
    fn test_kl_band_is_below_exact_order_statistic_bound() {
        let n = 10_000usize;
        let directions = 2usize;
        let failure = 1e-6;
        let per_rank_failure = failure / (directions * n) as f64;
        let log_inverse_failure = (1.0 / per_rank_failure).ln();
        let mut previous = 0.0;

        for rank in [1usize, 10, 100, 1_000, 5_000, 9_999, 10_000] {
            let kl = kl_lower_cdf_bound(rank, n, log_inverse_failure);
            let exact = Beta::new(rank as f64, (n - rank + 1) as f64)
                .unwrap()
                .inverse_cdf(per_rank_failure);
            assert!(
                kl <= exact + 1e-12,
                "rank={rank}: KL lower bound {kl} exceeded exact bound {exact}"
            );
            assert!(kl >= previous, "lower CDF band must be monotone");
            previous = kl;
        }
    }

    #[test]
    fn test_cyclic_cholesky_identity() {
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let chol = CyclicBandedCholesky::compute(&gram, 2, 1e-6).unwrap();
        // L should be identity → L*z = z
        let mut out = vec![0.0; 2];
        chol.sample_gaussian(&[0.0, 0.0], 1.0, &[1.0, 2.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-8, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-8, "out[1]={}", out[1]);
    }

    #[test]
    fn test_cyclic_cholesky_2x2() {
        let gram = vec![4.0, 2.0, 2.0, 3.0];
        let chol = CyclicBandedCholesky::compute(&gram, 2, 1e-6).unwrap();
        // L = [[2, 0], [1, sqrt(2)]]
        // L * [1, 0] = [2, 1]
        let mut out = vec![0.0; 2];
        chol.sample_gaussian(&[0.0, 0.0], 1.0, &[1.0, 0.0], &mut out);
        assert!((out[0] - 2.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bnb_mc_pld_dpsgd_small() {
        let b = 10;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();
        let pld = bnb_mc_pld(&gram, b, 1.0, &config).unwrap();
        let eps = pld.epsilon_at(1e-2);
        assert!(eps > 0.0 && eps.is_finite(), "eps = {}", eps);
    }

    #[test]
    fn test_bnb_mc_pld_is_identical_across_thread_counts() {
        let b = 8;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            for j in 0..b {
                let distance = i.abs_diff(j).min(b - i.abs_diff(j));
                gram[i * b + j] = 0.2f64.powi(distance as i32);
            }
        }
        let mut config = default_config();
        config.discretization = 1e-3;
        config.seed = 173;

        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| bnb_mc_pld(&gram, b, 1.3, &config).unwrap());
        let many_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(|| bnb_mc_pld(&gram, b, 1.3, &config).unwrap());

        assert_eq!(
            single_thread.pmf_remove.discretization,
            many_threads.pmf_remove.discretization
        );
        assert_eq!(
            single_thread.pmf_remove.lower_loss_index,
            many_threads.pmf_remove.lower_loss_index
        );
        assert_eq!(
            single_thread.pmf_remove.probs,
            many_threads.pmf_remove.probs
        );
        assert_eq!(
            single_thread.pmf_remove.infinity_mass,
            many_threads.pmf_remove.infinity_mass
        );

        let single_add = single_thread.pmf_add.as_ref().unwrap();
        let many_add = many_threads.pmf_add.as_ref().unwrap();
        assert_eq!(single_add.discretization, many_add.discretization);
        assert_eq!(single_add.lower_loss_index, many_add.lower_loss_index);
        assert_eq!(single_add.probs, many_add.probs);
        assert_eq!(single_add.infinity_mass, many_add.infinity_mass);
    }

    #[test]
    fn test_bnb_mc_pld_more_noise_less_epsilon() {
        let b = 5;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();

        let eps_low = bnb_mc_pld(&gram, b, 0.5, &config).unwrap().epsilon_at(1e-2);
        let eps_high = bnb_mc_pld(&gram, b, 2.0, &config).unwrap().epsilon_at(1e-2);
        assert!(
            eps_high < eps_low,
            "More noise: {} should be < {}",
            eps_high,
            eps_low
        );
    }

    #[test]
    fn test_bnb_mc_pld_rejects_bad_params() {
        let config = default_config();
        assert!(bnb_mc_pld(&[1.0], 2, 1.0, &config).is_err());
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 0.0, &config).is_err());
        let mut invalid_mc = default_config();
        invalid_mc.mc_resolution = 0.0;
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 1.0, &invalid_mc).is_err());
    }

    #[test]
    fn test_bnb_mc_pld_lambda_cgd() {
        let b = 5;
        let lambda: f64 = 0.5;
        let e = 2;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            for j in 0..b {
                let gap = (i as i32 - j as i32).unsigned_abs() as i32;
                gram[i * b + j] = e as f64 * lambda.powi(gap);
            }
        }
        let config = default_config();
        let pld = bnb_mc_pld(&gram, b, 1.0, &config).unwrap();
        let eps = pld.epsilon_at(1e-2);
        assert!(eps > 0.0 && eps.is_finite());
    }

    #[test]
    fn test_identity_gram_confidence_bound_is_above_exact_oracle() {
        use crate::amplification::random_allocation_gaussian_pld;

        let b = 4;
        let epochs: f64 = 2.0;
        let sigma = 1.5;
        let gram: Vec<f64> = (0..b)
            .flat_map(|i| (0..b).map(move |j| if i == j { epochs } else { 0.0 }))
            .collect();
        let config = default_config();
        let bounded = bnb_mc_pld(&gram, b, sigma, &config).unwrap();
        let exact = random_allocation_gaussian_pld(sigma / epochs.sqrt(), b, 1, &config).unwrap();
        for delta in [0.01, 0.02, 0.05] {
            assert!(
                bounded.epsilon_at(delta) >= exact.epsilon_at(delta) - 1e-10,
                "delta={delta}: bounded={} exact={}",
                bounded.epsilon_at(delta),
                exact.epsilon_at(delta)
            );
        }
        assert_eq!(
            bounded.estimation_failure_probability(),
            config.mc_failure_probability
        );
        assert!(bounded.mc_resolution() <= config.mc_resolution);
    }
}
