//! # opake-accounting
//!
//! Privacy Loss Distribution (PLD) accounting for differential privacy.
//!
//! This crate provides privacy accounting using the Privacy Loss Distribution
//! framework with Connect-the-Dots discretization.
//!
//! ## Architecture
//!
//! Rust is a **PLD computation engine**: flat functions that take scalar
//! parameters and return opaque PLD handles. Python owns composition,
//! repetition, caching, and calibration.
//!
//! ## Rust API
//!
//! ```rust
//! use opake_accounting::amplification::poisson_pld;
//! use opake_accounting::mechanisms::gaussian_pld;
//! use opake_accounting::DiscretizationConfig;
//!
//! let config = DiscretizationConfig::default();
//! let base = gaussian_pld(1.1, &config).unwrap();
//! let pld = poisson_pld(&base, 0.01).unwrap();
//! let epsilon = pld.epsilon_at(1e-5);
//! ```
//!
//! ## Module Overview
//!
//! - [`mechanisms`]: Gaussian, (ε,δ), identity PLD constructors
//! - [`amplification`]: Poisson, truncated Poisson, accumulated PLDs
//! - [`matrix_factorization`]: MF-DP accounting (BandMF, BLT)
//! - [`transformations`]: Adaptive clipping sensitivity formula
//! - [`pld`]: The `PrivacyLossDistribution` type and metrics
//! - [`discretization`]: Connect-the-Dots discretization
//! - [`error`]: Error types
//! - [`numerics`]: Numerical primitives

pub mod error;
pub mod numerics;

// --- Core infrastructure ---
pub(crate) mod adjacency;
pub(crate) mod discretization;
pub mod pld;

// --- New flat-function modules ---
pub mod amplification;
pub mod matrix_factorization;
pub mod mechanisms;
pub mod transformations;

// --- Public re-exports ---
pub use adjacency::Adjacency;
pub use discretization::DiscretizationConfig;
pub use error::{PldError, Result};
pub use pld::PrivacyLossDistribution;

#[cfg(feature = "python-extension")]
mod python;

#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

#[cfg(feature = "python-extension")]
#[pymodule]
fn opake_accounting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
