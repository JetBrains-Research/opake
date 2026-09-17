//! Unified error types for the opake-accounting library
//!
//! This module provides a single error type [`PldError`] that covers all error
//! conditions in the library, including PLD operations, calibration, mechanisms,
//! and accountants.

use thiserror::Error;

/// Unified error type for all library operations
///
/// This enum covers errors from:
/// - PLD composition and discretization
/// - Calibration operations
/// - Mechanism computations
/// - Accountant operations
#[derive(Error, Debug, Clone)]
pub enum PldError {
    // =========================================================================
    // PLD/PMF errors
    // =========================================================================
    /// Discretization intervals must match for composition
    #[error("Discretization intervals differ: {0} != {1}")]
    DiscretizationMismatch(f64, f64),

    /// Cannot compose PMFs of different types
    #[error("Cannot compose PMFs of different types: {0} with {1}")]
    TypeMismatch(String, String),

    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Exact self-composition exceeds the bounded FFT size.
    #[error(
        "Exact self-composition requires an FFT buffer of {requested} elements, exceeding the {maximum}-element limit; configure a positive tail-mass truncation budget or use a coarser grid"
    )]
    SelfCompositionTooLarge { requested: usize, maximum: usize },

    /// Insufficient probability mass for the requested delta
    #[error("Insufficient probability mass: {0} > {1}")]
    InsufficientMass(f64, f64),

    // =========================================================================
    // Calibration errors (merged from CalibrationError)
    // =========================================================================
    /// Privacy evaluation failed during calibration
    #[error("Calibration evaluation failed: {0}")]
    CalibrationEvaluationFailed(String),

    /// Calibration did not converge within max iterations
    #[error(
        "Calibration did not converge after {iterations} iterations (last param: {last_param:.6})"
    )]
    CalibrationConvergenceFailed {
        /// Number of iterations attempted
        iterations: usize,
        /// Last parameter value tried
        last_param: f64,
    },

    /// Invalid calibration configuration
    #[error("Invalid calibration config: {0}")]
    CalibrationInvalidConfig(String),

    /// Required metric not available from evaluator
    #[error("Required metric unavailable: {0}")]
    CalibrationMetricUnavailable(String),

    /// Parameter out of allowed bounds
    #[error("Parameter {param:.6} out of bounds [{min:.6}, {max:.6}]")]
    CalibrationOutOfBounds {
        /// The parameter value
        param: f64,
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },

    // =========================================================================
    // Mechanism errors
    // =========================================================================
    /// Mechanism does not support the requested adjacency type
    #[error("Unsupported adjacency: {mechanism} does not support {adjacency}")]
    UnsupportedAdjacency {
        /// Name of the mechanism
        mechanism: &'static str,
        /// Name of the adjacency type
        adjacency: &'static str,
    },

    // =========================================================================
    // Numerical/math errors
    // =========================================================================
    /// Log subtraction error (a must be > b)
    #[error("Log subtraction error: log_sub requires a > b, got a={a}, b={b}")]
    LogSubtractionError {
        /// First operand
        a: f64,
        /// Second operand (must be less than a)
        b: f64,
    },

    // =========================================================================
    // Collection/state errors
    // =========================================================================
    /// Operation on empty collection
    #[error("Empty collection: {0}")]
    EmptyCollection(&'static str),

    /// Infinite bounds not supported
    #[error("Infinite bounds: {0}")]
    InfiniteBounds(String),

    /// Accountant has no recorded events
    #[error("Accountant is empty: {0}")]
    EmptyAccountant(&'static str),

    /// Unsupported event type for accounting
    #[error("Unsupported event type: {0}")]
    UnsupportedEvent(String),
}

/// Type alias for Results using PldError
pub type Result<T> = std::result::Result<T, PldError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PldError::InvalidParameter("sigma must be positive".to_string());
        assert!(err.to_string().contains("sigma must be positive"));
    }

    #[test]
    fn test_calibration_error_display() {
        let err = PldError::CalibrationConvergenceFailed {
            iterations: 50,
            last_param: 1.5,
        };
        let display = err.to_string();
        assert!(display.contains("50"));
        assert!(display.contains("1.5"));
    }

    #[test]
    fn test_unsupported_adjacency_display() {
        let err = PldError::UnsupportedAdjacency {
            mechanism: "TestMechanism",
            adjacency: "REPLACE",
        };
        assert!(err.to_string().contains("TestMechanism"));
        assert!(err.to_string().contains("REPLACE"));
    }
}
