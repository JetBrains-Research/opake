//! PyO3 Python bindings for the PLD accounting engine.
//!
//! Exposes flat functions (scalars in → Pld handles out) and the Pld class
//! with privacy metric methods, composition, and self-composition operators.

mod adaclip;
mod amplification;
mod config;
mod matrix_factorization;
mod mechanisms;
mod pld;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::PldError;

impl From<PldError> for PyErr {
    fn from(error: PldError) -> Self {
        let message = error.to_string();

        match error {
            PldError::InvalidParameter(_)
            | PldError::DiscretizationMismatch(_, _)
            | PldError::TypeMismatch(_, _)
            | PldError::CalibrationInvalidConfig(_)
            | PldError::CalibrationOutOfBounds { .. }
            | PldError::UnsupportedAdjacency { .. }
            | PldError::EmptyCollection(_)
            | PldError::InfiniteBounds(_)
            | PldError::UnsupportedEvent(_) => PyValueError::new_err(message),
            PldError::NumericalError(_)
            | PldError::SelfCompositionTooLarge { .. }
            | PldError::InsufficientMass(_, _)
            | PldError::CalibrationEvaluationFailed(_)
            | PldError::CalibrationConvergenceFailed { .. }
            | PldError::CalibrationMetricUnavailable(_)
            | PldError::LogSubtractionError { .. }
            | PldError::EmptyAccountant(_) => PyRuntimeError::new_err(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pld_errors_map_to_stable_python_categories() {
        Python::initialize();
        Python::attach(|py| {
            let cases = [
                (
                    PldError::DiscretizationMismatch(0.1, 0.3),
                    true,
                    "Discretization intervals differ",
                ),
                (
                    PldError::TypeMismatch("dense".into(), "sparse".into()),
                    true,
                    "Cannot compose PMFs",
                ),
                (
                    PldError::InvalidParameter("invalid value".into()),
                    true,
                    "invalid value",
                ),
                (
                    PldError::NumericalError("unstable convolution".into()),
                    false,
                    "unstable convolution",
                ),
                (
                    PldError::SelfCompositionTooLarge {
                        requested: 64,
                        maximum: 32,
                    },
                    false,
                    "exceeding the 32-element limit",
                ),
                (
                    PldError::InsufficientMass(0.2, 0.1),
                    false,
                    "Insufficient probability mass",
                ),
                (
                    PldError::CalibrationEvaluationFailed("evaluation failed".into()),
                    false,
                    "evaluation failed",
                ),
                (
                    PldError::CalibrationConvergenceFailed {
                        iterations: 8,
                        last_param: 0.75,
                    },
                    false,
                    "8 iterations",
                ),
                (
                    PldError::CalibrationInvalidConfig("invalid calibration".into()),
                    true,
                    "invalid calibration",
                ),
                (
                    PldError::CalibrationMetricUnavailable("epsilon".into()),
                    false,
                    "epsilon",
                ),
                (
                    PldError::CalibrationOutOfBounds {
                        param: 2.0,
                        min: 0.0,
                        max: 1.0,
                    },
                    true,
                    "out of bounds",
                ),
                (
                    PldError::UnsupportedAdjacency {
                        mechanism: "Gaussian",
                        adjacency: "ADD",
                    },
                    true,
                    "Gaussian",
                ),
                (
                    PldError::LogSubtractionError { a: 0.0, b: 1.0 },
                    false,
                    "Log subtraction error",
                ),
                (
                    PldError::EmptyCollection("empty strategy"),
                    true,
                    "empty strategy",
                ),
                (
                    PldError::InfiniteBounds("unbounded epsilon".into()),
                    true,
                    "unbounded epsilon",
                ),
                (
                    PldError::EmptyAccountant("missing PLD"),
                    false,
                    "missing PLD",
                ),
                (
                    PldError::UnsupportedEvent("custom event".into()),
                    true,
                    "custom event",
                ),
            ];

            for (error, expects_value_error, message) in cases {
                let py_error: PyErr = error.into();
                assert!(py_error.to_string().contains(message));
                if expects_value_error {
                    assert!(py_error.is_instance_of::<PyValueError>(py));
                } else {
                    assert!(py_error.is_instance_of::<PyRuntimeError>(py));
                }
            }
        });
    }
}

/// Register all Python-visible types and functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<pld::PyPld>()?;
    m.add_class::<config::PyDiscretizationConfig>()?;

    // Mechanisms
    m.add_function(wrap_pyfunction!(mechanisms::py_gaussian_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_eps_delta_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_identity_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_non_private_pld, m)?)?;

    // Amplification
    m.add_function(wrap_pyfunction!(amplification::py_poisson_pld, m)?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_truncated_poisson_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_parallel_poisson_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_random_allocation_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(amplification::py_bnb_mc_pld, m)?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_register_bnb_transcript_corpus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_drop_bnb_transcript_corpus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_bnb_pld_from_transcript_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_bandmf_b_min_sep_warm_mc_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_register_b_min_sep_transcript_corpus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_drop_b_min_sep_transcript_corpus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_bandmf_b_min_sep_pld_from_transcript_handle,
        m
    )?)?;

    // Matrix Factorization
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_mf_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_max_participation_for_linear_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_minsep_true_max_participations,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_single_participation_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_banded_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_general_sensitivity_upper_bound,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_fixed_epoch_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_blt_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_toeplitz_minsep_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_normalized_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_max_column_norm,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_gram_matrix_lr,
        m
    )?)?;

    // BISR (Banded Inverse Square Root)
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_normalized_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_gram_matrix_lr,
        m
    )?)?;

    // Toeplitz Gram matrix (for BnB with BandMF/BLT strategy coefs)
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_toeplitz_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_strategy_coefficients,
        m
    )?)?;

    // AdaClip
    m.add_function(wrap_pyfunction!(adaclip::py_adaclip_sensitivity, m)?)?;

    Ok(())
}
