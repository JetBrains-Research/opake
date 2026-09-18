//! Base mechanism PLD constructors: Gaussian, eps-delta, identity.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
///
/// Raises:
///     ValueError: If parameters or configuration are invalid.
///     RuntimeError: If PLD construction encounters a numerical failure.
#[pyfunction]
#[pyo3(name = "gaussian_pld", signature = (noise_multiplier, config))]
pub fn py_gaussian_pld(noise_multiplier: f64, config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::gaussian_pld(noise_multiplier, &config.inner)?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for an (epsilon, delta)-mechanism.
///
/// Args:
///     epsilon (float): Privacy loss, >= 0.
///     delta (float): Failure probability, in [0, 1].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
///
/// Raises:
///     ValueError: If parameters or configuration are invalid.
///     RuntimeError: If PLD construction encounters a numerical failure.
#[pyfunction]
#[pyo3(name = "eps_delta_pld", signature = (epsilon, delta, config))]
pub fn py_eps_delta_pld(
    epsilon: f64,
    delta: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::mechanisms::eps_delta_pld(epsilon, delta, &config.inner)?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for the identity (zero-privacy-loss) mechanism.
///
/// Args:
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The identity PLD (neutral element for composition).
///
/// Raises:
///     ValueError: If configuration is invalid.
///     RuntimeError: If PLD construction encounters a numerical failure.
#[pyfunction]
#[pyo3(name = "identity_pld", signature = (config))]
pub fn py_identity_pld(config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::identity_pld(&config.inner)?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a non-private mechanism (ε = ∞, δ = 1).
///
/// Args:
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: A PLD with all mass at +∞ (composition annihilator).
#[pyfunction]
#[pyo3(name = "non_private_pld", signature = (config))]
pub fn py_non_private_pld(config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::non_private_pld(&config.inner)?;
    Ok(PyPld::new(pld))
}
