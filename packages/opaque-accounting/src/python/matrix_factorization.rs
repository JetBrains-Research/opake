//! PyO3 bindings for matrix factorization privacy accounting.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a matrix factorization Gaussian mechanism.
///
/// Computes the privacy guarantee for the entire MF training run as a
/// single Gaussian mechanism with effective noise multiplier σ/S.
///
/// The sensitivity S should be pre-computed based on the MF strategy
/// (BandMF, BLT) and participation pattern.
///
/// Args:
///     noise_multiplier (float): Raw noise std σ (before MF). Must be positive.
///     sensitivity (float): L2 sensitivity of the encoder matrix. Must be positive.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution for the entire MF training run.
///
/// Raises:
///     ValueError: If parameters or configuration are invalid.
///     RuntimeError: If PLD construction encounters a numerical failure.
#[pyfunction]
#[pyo3(name = "mf_gaussian_pld", signature = (noise_multiplier, sensitivity, config))]
pub fn py_mf_gaussian_pld(
    noise_multiplier: f64,
    sensitivity: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld =
        crate::matrix_factorization::mf_gaussian_pld(noise_multiplier, sensitivity, &config.inner)?;
    Ok(PyPld::new(pld))
}

/// Solve max_u <x, u> where u respects min-sep participation.
///
/// Uses dynamic programming (Algorithm 3, VecSens) from
/// Choquette-Choo et al. (2023).
///
/// Args:
///     x (list[float]): Vector of values to optimize over.
///     min_sep (int): Minimum separation between selections (>= 1).
///     max_participations (int | None): Optional upper bound on selections.
///
/// Returns:
///     float: The optimal inner product.
#[pyfunction]
#[pyo3(name = "max_participation_for_linear_fn", signature = (x, min_sep=1, max_participations=None))]
pub fn py_max_participation_for_linear_fn(
    x: Vec<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    crate::matrix_factorization::max_participation_for_linear_fn(&x, min_sep, max_participations)
}

/// Maximum participations under a min-sep constraint.
///
/// Args:
///     n (int): Number of rounds.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     int: Effective maximum participations.
#[pyfunction]
#[pyo3(name = "minsep_true_max_participations", signature = (n, min_sep, max_participations=None))]
pub fn py_minsep_true_max_participations(
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> usize {
    crate::matrix_factorization::minsep_true_max_participations(n, min_sep, max_participations)
}

/// L2 sensitivity under single participation.
///
/// Args:
///     column_norms (list[float]): L2 norms of encoder matrix columns.
///
/// Returns:
///     float: Maximum column norm (the sensitivity).
///
/// Raises:
///     ValueError: If column_norms is empty or contains non-finite values.
#[pyfunction]
#[pyo3(name = "single_participation_sensitivity", signature = (column_norms,))]
pub fn py_single_participation_sensitivity(column_norms: Vec<f64>) -> PyResult<f64> {
    Ok(crate::matrix_factorization::single_participation_sensitivity(&column_norms)?)
}

/// Exact L2 sensitivity for banded Gram matrices under min-sep participation.
///
/// Args:
///     gram_diag (list[float]): Diagonal of Gram matrix X = C^T C.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: The exact L2 sensitivity.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "banded_sensitivity", signature = (gram_diag, min_sep=1, max_participations=None))]
pub fn py_banded_sensitivity(
    gram_diag: Vec<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    Ok(crate::matrix_factorization::banded_sensitivity(
        &gram_diag,
        min_sep,
        max_participations,
    )?)
}

/// Upper bound on L2 sensitivity for general Gram matrices.
///
/// Args:
///     gram_matrix (list[float]): Flattened row-major Gram matrix X = C^T C.
///     n (int): Matrix dimension.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: An upper bound on the L2 sensitivity.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "general_sensitivity_upper_bound", signature = (gram_matrix, n, min_sep=1, max_participations=None))]
pub fn py_general_sensitivity_upper_bound(
    gram_matrix: Vec<f64>,
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    Ok(
        crate::matrix_factorization::general_sensitivity_upper_bound(
            &gram_matrix,
            n,
            min_sep,
            max_participations,
        )?,
    )
}

/// L2 sensitivity under fixed-epoch participation.
///
/// Args:
///     gram_matrix (list[float]): Flattened row-major Gram matrix X = C^T C.
///     n (int): Matrix dimension (total rounds).
///     epochs (int): Number of epochs (must divide n).
///
/// Returns:
///     float: The L2 sensitivity under fixed-epoch participation.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "fixed_epoch_sensitivity", signature = (gram_matrix, n, epochs))]
pub fn py_fixed_epoch_sensitivity(gram_matrix: Vec<f64>, n: usize, epochs: usize) -> PyResult<f64> {
    Ok(crate::matrix_factorization::fixed_epoch_sensitivity(
        &gram_matrix,
        n,
        epochs,
    )?)
}

/// Sensitivity squared for a BLT strategy matrix.
///
/// Implements Lemma 5.3 of the BLT paper (https://arxiv.org/abs/2404.16706).
///
/// Args:
///     buf_decay (list[float]): Decay factors for each buffer, each in (0, 1).
///     output_scale (list[float]): Scale factors for each buffer.
///     n (float): Number of iterations (use float('inf') for asymptotic).
///
/// Returns:
///     float: The sensitivity squared.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "blt_sensitivity_squared", signature = (buf_decay, output_scale, n))]
pub fn py_blt_sensitivity_squared(
    buf_decay: Vec<f64>,
    output_scale: Vec<f64>,
    n: f64,
) -> PyResult<f64> {
    Ok(crate::matrix_factorization::blt_sensitivity_squared(
        &buf_decay,
        &output_scale,
        n,
    )?)
}

/// Sensitivity squared for a Toeplitz matrix under min-sep participation.
///
/// Implements BSR Theorem 2 closed-form for non-negative, non-increasing
/// Toeplitz coefficients.
///
/// Args:
///     strategy_coef (list[float]): Toeplitz coefficients (non-negative, non-increasing).
///     n (int): Matrix dimension (total rounds).
///     min_sep (int): Minimum separation between participations (>= 1).
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: The sensitivity squared.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "toeplitz_minsep_sensitivity_squared", signature = (strategy_coef, n, min_sep=1, max_participations=None))]
pub fn py_toeplitz_minsep_sensitivity_squared(
    strategy_coef: Vec<f64>,
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    Ok(
        crate::matrix_factorization::toeplitz_minsep_sensitivity_squared(
            &strategy_coef,
            n,
            min_sep,
            max_participations,
        )?,
    )
}

/// Squared L2 sensitivity of the DP-λCGD strategy matrix.
///
/// Uses the closed-form expression completed in Appendix C, equation (16) of
/// arXiv:2601.22334v1. With momentum β > 0, computes
/// via momentum-aware column inner products.
///
/// Args:
///     lambda_ (float): Correlation coefficient in [0, 1). λ=0 is DP-SGD.
///     n_steps (int): Total number of training steps.
///     min_sep (int): Minimum separation between participations (>= 1).
///     max_participations (int | None): Optional upper bound on participations.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     float: The squared L2 sensitivity.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "lambda_cgd_sensitivity_squared", signature = (lambda_, n_steps, min_sep=1, max_participations=None, momentum=0.0))]
pub fn py_lambda_cgd_sensitivity_squared(
    lambda_: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> PyResult<f64> {
    Ok(crate::matrix_factorization::lambda_cgd_sensitivity_squared(
        lambda_,
        n_steps,
        min_sep,
        max_participations,
        momentum,
    )?)
}

/// Squared L2 sensitivity of the column-normalized DP-λCGD.
///
/// Column normalization: C̃_λ = C_λ · D⁻¹ where D = diag(‖C_λ[:,j]‖).
/// For single participation (k=1), always returns 1.0.
/// The normalized result is covered by Lemma 8, equation (12) of the current
/// arXiv:2601.22334v2, rather than its Toeplitz-only Theorem 1.
///
/// Args:
///     lambda_ (float): Correlation coefficient in [0, 1).
///     n_steps (int): Total number of training steps.
///     min_sep (int): Minimum separation between participations (>= 1).
///     max_participations (int | None): Optional upper bound on participations.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     float: The squared L2 sensitivity of the column-normalized matrix.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "lambda_cgd_normalized_sensitivity_squared", signature = (lambda_, n_steps, min_sep=1, max_participations=None, momentum=0.0))]
pub fn py_lambda_cgd_normalized_sensitivity_squared(
    lambda_: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> PyResult<f64> {
    Ok(
        crate::matrix_factorization::lambda_cgd_normalized_sensitivity_squared(
            lambda_,
            n_steps,
            min_sep,
            max_participations,
            momentum,
        )?,
    )
}

/// Compute the BnB Gram matrix for DP-λCGD.
///
/// For the BnB dominating pair, computes G_{ij} = ⟨m_i, m_j⟩ where
/// m_i = Σ_{epoch} m^β_{b·epoch+i} (momentum-accumulated columns,
/// or normalized C̃_λ).
///
/// Args:
///     lambda_ (float): Correlation coefficient in [0, 1).
///     n_steps (int): Total steps (= bins_per_epoch × num_epochs).
///     min_sep (int): Bins per epoch (= b).
///     max_participations (int | None): Number of epochs. None infers.
///     normalized (bool): Whether to use column-normalized matrix.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     list[float]: Flattened row-major b×b Gram matrix.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "lambda_cgd_gram_matrix", signature = (lambda_, n_steps, min_sep=1, max_participations=None, normalized=true, momentum=0.0))]
pub fn py_lambda_cgd_gram_matrix(
    lambda_: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::lambda_cgd_gram_matrix(
        lambda_,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        momentum,
    )?)
}

/// Compatibility tombstone for LR-weighted DP-λCGD accounting.
///
/// Opaque's deployed λ-CGD mechanism does not apply LR weights to its
/// encoder/noise path, so this function always raises ``ValueError``.  It is
/// retained temporarily to give old callers a migration error.
///
/// Args:
///     lambda_ (float): Correlation coefficient in [0, 1).
///     momentum (float): Optimizer momentum β in [0, 1).
///     n_steps (int): Total steps (= bins_per_epoch × num_epochs).
///     min_sep (int): Bins per epoch (= b).
///     max_participations (int | None): Number of epochs. None infers.
///     normalized (bool): Whether to use column-normalized matrix.
///     lr_weights (list[float]): Per-step LR weights, length = n_steps.
///
/// Raises:
///     ValueError: Always; omit the schedule and recalibrate, or use BandMF/BLT.
#[pyfunction]
#[pyo3(name = "lambda_cgd_gram_matrix_lr", signature = (lambda_, momentum, n_steps, min_sep, max_participations, normalized, lr_weights))]
pub fn py_lambda_cgd_gram_matrix_lr(
    lambda_: f64,
    momentum: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    lr_weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::lambda_cgd_gram_matrix_lr(
        lambda_,
        momentum,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        &lr_weights,
    )?)
}

/// Max column L2 norm of the DP-λCGD strategy matrix.
///
/// The first column has the largest norm: sqrt((1 - λ^{2n}) / (1 - λ²)).
///
/// Args:
///     lambda_ (float): Correlation coefficient in [0, 1).
///     n_steps (int): Total number of steps.
///
/// Returns:
///     float: The max column L2 norm.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "lambda_cgd_max_column_norm", signature = (lambda_, n_steps))]
pub fn py_lambda_cgd_max_column_norm(lambda_: f64, n_steps: usize) -> PyResult<f64> {
    Ok(crate::matrix_factorization::lambda_cgd_max_column_norm(
        lambda_, n_steps,
    )?)
}

// ── BISR (Banded Inverse Square Root) ────────────────────────────

/// Squared L2 sensitivity upper bound for BISR under min-sep participation.
///
/// Args:
///     coefficients (list[float]): Banded C^{-1} coefficients [c̃_0, ..., c̃_{p-1}].
///     n_steps (int): Total number of training steps.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     float: The squared L2 sensitivity upper bound. Exact when the effective
///         forward coefficients are non-negative and non-increasing.
#[pyfunction]
#[pyo3(name = "bisr_sensitivity_squared", signature = (coefficients, n_steps, min_sep=1, max_participations=None, momentum=0.0))]
pub fn py_bisr_sensitivity_squared(
    coefficients: Vec<f64>,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> PyResult<f64> {
    Ok(crate::matrix_factorization::bisr_sensitivity_squared(
        &coefficients,
        n_steps,
        min_sep,
        max_participations,
        momentum,
    )?)
}

/// Squared L2 sensitivity upper bound for column-normalized BISR.
///
/// Args:
///     coefficients (list[float]): Banded C^{-1} coefficients.
///     n_steps (int): Total number of training steps.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     float: A conservative squared L2 sensitivity upper bound for the
///         column-normalized matrix.
#[pyfunction]
#[pyo3(name = "bisr_normalized_sensitivity_squared", signature = (coefficients, n_steps, min_sep=1, max_participations=None, momentum=0.0))]
pub fn py_bisr_normalized_sensitivity_squared(
    coefficients: Vec<f64>,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> PyResult<f64> {
    Ok(
        crate::matrix_factorization::bisr_normalized_sensitivity_squared(
            &coefficients,
            n_steps,
            min_sep,
            max_participations,
            momentum,
        )?,
    )
}

/// BnB Gram matrix for BISR with optional momentum.
///
/// Args:
///     coefficients (list[float]): Banded C^{-1} coefficients.
///     n_steps (int): Total steps.
///     min_sep (int): Bins per epoch (= b).
///     max_participations (int | None): Number of epochs. None infers.
///     normalized (bool): Whether to use column-normalized matrix.
///     momentum (float): Optimizer momentum β in [0, 1). Default 0.
///
/// Returns:
///     list[float]: Flattened row-major b×b Gram matrix.
#[pyfunction]
#[pyo3(name = "bisr_gram_matrix", signature = (coefficients, n_steps, min_sep=1, max_participations=None, normalized=true, momentum=0.0))]
pub fn py_bisr_gram_matrix(
    coefficients: Vec<f64>,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::bisr_gram_matrix(
        &coefficients,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        momentum,
    )?)
}

/// Compatibility tombstone for LR-weighted BISR accounting.
///
/// Opaque's deployed BISR mechanism does not apply LR weights to its
/// encoder/noise path, so this function always raises ``ValueError``. It is
/// retained temporarily to give old callers a migration error.
///
/// Args:
///     coefficients (list[float]): Banded C^{-1} coefficients.
///     momentum (float): Optimizer momentum β in [0, 1).
///     n_steps (int): Total steps.
///     min_sep (int): Bins per epoch.
///     max_participations (int | None): Number of epochs.
///     normalized (bool): Whether to use column-normalized matrix.
///     lr_weights (list[float]): Per-step LR weights, length = n_steps.
///
/// Raises:
///     ValueError: Always; omit the schedule and recalibrate, or use BandMF/BLT.
#[pyfunction]
#[pyo3(name = "bisr_gram_matrix_lr", signature = (coefficients, momentum, n_steps, min_sep, max_participations, normalized, lr_weights))]
pub fn py_bisr_gram_matrix_lr(
    coefficients: Vec<f64>,
    momentum: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    lr_weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::bisr_gram_matrix_lr(
        &coefficients,
        momentum,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        &lr_weights,
    )?)
}

/// BnB Gram matrix for a banded Toeplitz strategy (known forward coefficients).
///
/// For BandMF/BLT where the optimized strategy coefficients are known directly.
///
/// Args:
///     strategy_coef (list[float]): Toeplitz strategy coefficients.
///     n_steps (int): Total steps.
///     min_sep (int): Bins per epoch.
///     max_participations (int | None): Number of epochs.
///     normalized (bool): Whether to column-normalize.
///
/// Returns:
///     list[float]: Flattened row-major b×b Gram matrix.
#[pyfunction]
#[pyo3(name = "toeplitz_gram_matrix", signature = (strategy_coef, n_steps, min_sep=1, max_participations=None, normalized=true))]
pub fn py_toeplitz_gram_matrix(
    strategy_coef: Vec<f64>,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::toeplitz_gram_matrix(
        &strategy_coef,
        n_steps,
        min_sep,
        max_participations,
        normalized,
    )?)
}

/// Compute BISR strategy coefficients from inverse coefficients.
///
/// Args:
///     coefficients (list[float]): Inverse coefficients [c̃_0, ..., c̃_{p-1}].
///     n (int): Number of entries to compute.
///
/// Returns:
///     list[float]: First n entries of column 0 of the strategy matrix.
///
/// Raises:
///     ValueError: If coefficients are invalid or n is zero.
#[pyfunction]
#[pyo3(name = "bisr_strategy_coefficients", signature = (coefficients, n))]
pub fn py_bisr_strategy_coefficients(coefficients: Vec<f64>, n: usize) -> PyResult<Vec<f64>> {
    Ok(crate::matrix_factorization::bisr::try_bisr_column_zero(
        &coefficients,
        n,
    )?)
}
