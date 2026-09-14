//! Amplification PLD constructors: Poisson, truncated-Poisson, parallel Poisson.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Apply plain Poisson subsampling to an existing PLD.
///
/// Args:
///     base (Pld): PLD of the base mechanism.
///     rate (float): Poisson sampling probability, in (0, 1).
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_pld", signature = (base, rate))]
pub fn py_poisson_pld(base: &PyPld, rate: f64) -> PyResult<PyPld> {
    let pld = crate::amplification::poisson_pld(&base.inner, rate)?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.
///
/// This is the actual sampling used in production DP-SGD. Unlike standard
/// Poisson (variable batch), truncated capping at batch_size_max gives
/// predictable memory usage.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     rate (float): Poisson sampling probability, in (0, 1].
///     batch_size_max (int): Maximum batch size.
///     dataset_size (int): Total dataset size.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "truncated_poisson_gaussian_pld", signature = (noise_multiplier, rate, batch_size_max, dataset_size, config))]
pub fn py_truncated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::truncated_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        batch_size_max,
        dataset_size,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism.
///
/// Models summing multiple independent Poisson samples before adding noise once.
/// Use cases: gradient accumulation (m microbatches) or parallel workers (K workers).
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     rate (float): Poisson sampling probability, in (0, 1].
///     microbatches (int): Number of independent samples, > 0.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "parallel_poisson_gaussian_pld", signature = (noise_multiplier, rate, microbatches, config))]
pub fn py_parallel_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::parallel_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        microbatches,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}

/// Compute the BnB PLD via Monte Carlo sampling of the dominating pair.
///
/// Uses Algorithm 2 of Choquette-Choo et al. (2024) "Near Exact Privacy
/// Amplification for Matrix Mechanisms" (arxiv:2410.06266).
///
/// The dominating pair is P = (1/b) Σ N(m_i, σ²I), Q = N(0, σ²I), where
/// the Gram matrix G captures the inner products ⟨m_i, m_j⟩.
///
/// Args:
///     gram (list[float]): Flattened row-major b×b Gram matrix.
///     num_bins (int): Number of bins b.
///     sigma (float): Noise multiplier σ, must be > 0.
///     config (DiscretizationConfig): Discretization and Monte Carlo confidence configuration.
///
/// Returns:
///     Pld: The privacy loss distribution (asymmetric, remove + add).
///
/// Raises:
///     ValueError: If parameters or configuration are invalid.
///     RuntimeError: If sampling or PLD construction encounters a numerical failure.
#[pyfunction]
#[pyo3(name = "bnb_mc_pld", signature = (gram, num_bins, sigma, config))]
pub fn py_bnb_mc_pld(
    gram: Vec<f64>,
    num_bins: usize,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bnb_mc_pld(&gram, num_bins, sigma, &config.inner)?;
    Ok(PyPld::new(pld))
}

#[pyfunction]
#[pyo3(
    name = "register_bnb_transcript_corpus",
    signature = (gram, num_bins, num_samples, seed)
)]
pub fn py_register_bnb_transcript_corpus(
    gram: Vec<f64>,
    num_bins: usize,
    num_samples: usize,
    seed: u64,
) -> PyResult<u64> {
    Ok(crate::amplification::register_bnb_transcripts(
        &gram,
        num_bins,
        num_samples,
        seed,
    )?)
}

#[pyfunction]
#[pyo3(name = "drop_bnb_transcript_corpus", signature = (handle))]
pub fn py_drop_bnb_transcript_corpus(handle: u64) {
    crate::amplification::drop_bnb_transcript_handle(handle);
}

#[pyfunction]
#[pyo3(
    name = "bnb_pld_from_transcript_handle",
    signature = (handle, gram, num_bins, sigma, config)
)]
pub fn py_bnb_pld_from_transcript_handle(
    handle: u64,
    gram: Vec<f64>,
    num_bins: usize,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bnb_pld_from_transcript_handle(
        handle,
        &gram,
        num_bins,
        sigma,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}

/// Monte Carlo PLD for BandMF with warm-start b-min-sep subsampling (Dong & Ganesh, arXiv:2602.09338).
///
/// Args:
///     strategy_coef: First column of the BandMF strategy matrix C (length = bandwidth).
///     n_steps: Total training iterations n.
///     p: Per-iteration Poisson inclusion probability p (Algorithm 2), not the per-example rate p_0.
///     sigma: Raw noise multiplier σ (same as BandMf noise_multiplier).
///     config: Discretization and Monte Carlo confidence configuration.
#[pyfunction]
#[pyo3(name = "bandmf_b_min_sep_warm_mc_pld", signature = (strategy_coef, n_steps, p, sigma, config))]
pub fn py_bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bandmf_b_min_sep_warm_mc_pld(
        &strategy_coef,
        n_steps,
        p,
        sigma,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}

#[pyfunction]
#[pyo3(
    name = "register_b_min_sep_transcript_corpus",
    signature = (strategy_coef, n_steps, p, num_samples, seed)
)]
pub fn py_register_b_min_sep_transcript_corpus(
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
    num_samples: usize,
    seed: u64,
) -> PyResult<u64> {
    Ok(crate::amplification::register_b_min_sep_transcripts(
        &strategy_coef,
        n_steps,
        p,
        num_samples,
        seed,
    )?)
}

#[pyfunction]
#[pyo3(name = "drop_b_min_sep_transcript_corpus", signature = (handle))]
pub fn py_drop_b_min_sep_transcript_corpus(handle: u64) {
    crate::amplification::drop_b_min_sep_transcript_handle(handle);
}

#[pyfunction]
#[pyo3(
    name = "bandmf_b_min_sep_pld_from_transcript_handle",
    signature = (handle, strategy_coef, p, sigma, config)
)]
pub fn py_bandmf_b_min_sep_pld_from_transcript_handle(
    handle: u64,
    strategy_coef: Vec<f64>,
    p: f64,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::pld_from_transcript_handle(
        handle,
        &strategy_coef,
        p,
        sigma,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for random allocation applied to the Gaussian mechanism.
///
/// In k-out-of-t random allocation each record is used in k steps chosen
/// uniformly at random from t. Unlike Monte Carlo, this is deterministic,
/// composable, and reproducible across thread counts.
///
/// Exact for k = 1. For k > 1 the result is a valid **upper bound** rather
/// than the exact k-out-of-t PLD: the t steps are split into k blocks and the
/// record is placed once per block. Joint convexity of the hockey-stick
/// divergence makes that an upper bound, because a uniformly random partition
/// into blocks induces exactly the uniform distribution over k-subsets.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     t (int): Steps per allocation round (number of bins), > 0.
///     k (int): Steps each record participates in, in [1, t]. Values above 1
///         return the block upper bound described above.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
///
/// Raises:
///     ValueError: If any parameter is out of range or the grid is too large.
#[pyfunction]
#[pyo3(name = "random_allocation_gaussian_pld", signature = (noise_multiplier, t, k, config))]
pub fn py_random_allocation_gaussian_pld(
    noise_multiplier: f64,
    t: usize,
    k: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::random_allocation_gaussian_pld(
        noise_multiplier,
        t,
        k,
        &config.inner,
    )?;
    Ok(PyPld::new(pld))
}
