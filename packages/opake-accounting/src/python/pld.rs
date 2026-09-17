//! PyPld: opaque handle wrapping PrivacyLossDistribution.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

use crate::pld::PrivacyLossDistribution;

/// An opaque Privacy Loss Distribution (PLD).
///
/// Created by mechanism/amplification functions. Supports privacy metric
/// queries, composition, and self-composition (repetition).
///
/// Example::
///
///     import opake.accounting as dp
///     pld = dp.gaussian_pld(1.1)
///     print(pld.epsilon_at(1e-5))
///     composed = pld.self_compose(1000)
///     print(composed.epsilon_at(1e-5))
#[pyclass(name = "Pld", from_py_object)]
#[derive(Clone)]
pub struct PyPld {
    pub(super) inner: PrivacyLossDistribution,
}

impl PyPld {
    pub(super) fn new(inner: PrivacyLossDistribution) -> Self {
        Self { inner }
    }

    fn composition_count(count: i64) -> PyResult<usize> {
        if count <= 0 {
            return Err(PyValueError::new_err("count must be greater than zero"));
        }

        let overflow = || PyOverflowError::new_err(format!("count must not exceed {}", u32::MAX));
        let count = u32::try_from(count).map_err(|_| overflow())?;

        usize::try_from(count).map_err(|_| overflow())
    }
}

#[pymethods]
impl PyPld {
    /// Smallest epsilon achieving (epsilon, delta)-DP.
    ///
    /// Args:
    ///     delta (float): Failure probability (typically 1e-5 to 1e-7).
    ///
    /// Returns:
    ///     float: Epsilon value. Monte Carlo PLDs return infinity when
    ///         ``delta`` is at or below their reported unresolved mass.
    #[pyo3(text_signature = "(self, delta)")]
    fn epsilon_at(&self, delta: f64) -> f64 {
        self.inner.epsilon_at(delta)
    }

    /// Smallest delta achieving (epsilon, delta)-DP.
    ///
    /// Always returns at least ``infinity_mass``; if the returned value equals
    /// ``infinity_mass``, the tail-truncation budget is exhausted and the true
    /// δ may be smaller than what is reported.
    ///
    /// Args:
    ///     epsilon (float): Privacy budget. Must be >= 0.
    ///
    /// Returns:
    ///     float: Delta value.
    #[pyo3(text_signature = "(self, epsilon)")]
    fn delta_at(&self, epsilon: f64) -> f64 {
        self.inner.delta_at(epsilon)
    }

    /// Worst-case infinity mass over both adjacency types.
    ///
    /// ``delta_at(ε)`` is always ≥ ``infinity_mass``; equality means the
    /// configured tail-truncation budget is exhausted and the true δ may be
    /// smaller.
    #[getter]
    fn infinity_mass(&self) -> f64 {
        self.inner.infinity_mass()
    }

    /// Failure probability of the Monte Carlo confidence event.
    ///
    /// Zero for analytic PLDs. This probability is statistical metadata and is
    /// distinct from the mechanism's DP delta.
    #[getter]
    fn mc_failure_probability(&self) -> f64 {
        self.inner.estimation_failure_probability()
    }

    /// Confidence level of the Monte Carlo PLD bound.
    #[getter]
    fn mc_confidence(&self) -> f64 {
        self.inner.estimation_confidence()
    }

    /// Unresolved Monte Carlo mass in delta units.
    ///
    /// Zero for analytic PLDs.
    #[getter]
    fn mc_resolution(&self) -> f64 {
        self.inner.mc_resolution()
    }

    /// Total-variation advantage.
    ///
    /// Returns:
    ///     float: Advantage in [0, 1]. Lower is more private.
    fn advantage(&self) -> f64 {
        self.inner.advantage()
    }

    /// Type-II error (beta) at a given Type-I error (alpha).
    ///
    /// Args:
    ///     alpha (float): Type-I error rate in [0, 1].
    ///
    /// Returns:
    ///     float: Beta value in [0, 1].
    #[pyo3(text_signature = "(self, alpha)")]
    fn beta_at(&self, alpha: f64) -> f64 {
        self.inner.beta_at(alpha)
    }

    /// Bayes risk under an optimal adversary.
    ///
    /// Args:
    ///     prior (float): Prior probability (typically 0.5).
    ///
    /// Returns:
    ///     float: Risk value in [0, 0.5].
    #[pyo3(text_signature = "(self, prior)")]
    fn risk_at(&self, prior: f64) -> f64 {
        self.inner.risk_at(prior)
    }

    /// Compose this PLD with another PLD.
    ///
    /// Returns a new PLD representing the composition (running both
    /// mechanisms sequentially).
    ///
    /// Args:
    ///     other (Pld): The other PLD to compose with.
    ///
    /// Returns:
    ///     Pld: Composed PLD.
    ///
    /// Raises:
    ///     ValueError: If the PLDs have incompatible discretizations or types.
    ///     RuntimeError: If composition encounters an invalid internal state.
    fn compose(&self, other: &PyPld) -> PyResult<PyPld> {
        let pld = self.inner.compose(&other.inner)?;
        Ok(PyPld { inner: pld })
    }

    /// Compose this PLD with itself `count` times (homogeneous repetition).
    ///
    /// Equivalent to running the same mechanism `count` times.
    ///
    /// Args:
    ///     count (int): Repetition count. Must be > 0.
    ///
    /// Raises:
    ///     ValueError: If count is not positive.
    ///     OverflowError: If count exceeds 2**32 - 1.
    ///     RuntimeError: If exact composition exceeds the bounded FFT size.
    ///
    /// Returns:
    ///     Pld: Self-composed PLD.
    #[pyo3(text_signature = "(self, count)")]
    fn self_compose(&self, count: i64) -> PyResult<PyPld> {
        Ok(PyPld {
            inner: self.inner.self_compose(Self::composition_count(count)?)?,
        })
    }

    /// ``pld * k`` is shorthand for ``pld.self_compose(k)``.
    fn __mul__(&self, count: i64) -> PyResult<PyPld> {
        self.self_compose(count)
    }

    /// ``k * pld`` also works.
    fn __rmul__(&self, count: i64) -> PyResult<PyPld> {
        self.self_compose(count)
    }

    /// ``a | b`` is shorthand for ``a.compose(b)``.
    fn __or__(&self, other: &PyPld) -> PyResult<PyPld> {
        self.compose(other)
    }

    fn __repr__(&self) -> String {
        let grid = self.inner.pmf_remove.probs.len();
        let sym = if self.inner.pmf_add.is_none() {
            "symmetric"
        } else {
            "asymmetric"
        };
        format!("Pld({}, {} bins)", sym, grid)
    }

    fn __str__(&self) -> String {
        let eps = self.inner.epsilon_at(1e-5);
        format!("Pld(eps(1e-5)={:.6})", eps)
    }
}
