//! PyDiscretizationConfig: discretization configuration for PLD computation.

use pyo3::prelude::*;

use crate::discretization::DiscretizationConfig;

/// Discretization configuration for PLD computation.
///
/// Args:
///     discretization (float): Grid spacing (default 1e-4). Smaller = tighter.
///     log_mass_truncation_bound (float): Log tail mass (default -50).
///     mc_resolution (float): Maximum unresolved MC mass (default 1e-5).
///     mc_failure_probability (float): Failure probability of the simultaneous
///         MC confidence band (default 1e-6).
///
/// Example::
///
///     config = dp.DiscretizationConfig(discretization=0.001)
///     pld = dp.gaussian_pld(1.1, config=config)
#[pyclass(name = "DiscretizationConfig", from_py_object)]
#[derive(Clone)]
pub struct PyDiscretizationConfig {
    pub(super) inner: DiscretizationConfig,
}

#[pymethods]
impl PyDiscretizationConfig {
    #[new]
    #[pyo3(signature = (discretization=1e-4, log_mass_truncation_bound=-50.0, max_grid_size=10_000_000, tail_mass_truncation=1e-15, seed=42, max_conv_grid=32_768, mc_resolution=1e-5, mc_failure_probability=1e-6))]
    fn new(
        discretization: f64,
        log_mass_truncation_bound: f64,
        max_grid_size: usize,
        tail_mass_truncation: f64,
        seed: u64,
        max_conv_grid: usize,
        mc_resolution: f64,
        mc_failure_probability: f64,
    ) -> PyResult<Self> {
        if max_conv_grid == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_conv_grid must be > 0",
            ));
        }
        let mut inner = DiscretizationConfig::new(discretization, log_mass_truncation_bound)?
            .with_max_grid_size(max_grid_size)
            .with_max_conv_grid(max_conv_grid);
        inner.tail_mass_truncation = tail_mass_truncation;
        inner.seed = seed;
        inner.mc_resolution = mc_resolution;
        inner.mc_failure_probability = mc_failure_probability;
        inner.validate_mc_parameters()?;
        Ok(Self { inner })
    }

    #[getter]
    fn discretization(&self) -> f64 {
        self.inner.discretization
    }

    #[getter]
    fn log_mass_truncation_bound(&self) -> f64 {
        self.inner.log_mass_truncation_bound
    }

    #[getter]
    fn max_grid_size(&self) -> usize {
        self.inner.max_grid_size
    }

    #[getter]
    fn max_conv_grid(&self) -> usize {
        self.inner.max_conv_grid
    }

    #[getter]
    fn tail_mass_truncation(&self) -> f64 {
        self.inner.tail_mass_truncation
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn mc_resolution(&self) -> f64 {
        self.inner.mc_resolution
    }

    #[getter]
    fn mc_failure_probability(&self) -> f64 {
        self.inner.mc_failure_probability
    }

    #[getter]
    fn resolved_num_mc_samples(&self) -> PyResult<usize> {
        Ok(self.inner.resolved_num_mc_samples(2)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscretizationConfig(discretization={}, log_mass_truncation_bound={}, max_grid_size={}, tail_mass_truncation={}, max_conv_grid={}, seed={}, mc_resolution={}, mc_failure_probability={})",
            self.inner.discretization, self.inner.log_mass_truncation_bound,
            self.inner.max_grid_size,
            self.inner.tail_mass_truncation,
            self.inner.max_conv_grid,
            self.inner.seed,
            self.inner.mc_resolution,
            self.inner.mc_failure_probability,
        )
    }

    fn __eq__(&self, other: &PyDiscretizationConfig) -> bool {
        self.inner.discretization == other.inner.discretization
            && self.inner.log_mass_truncation_bound == other.inner.log_mass_truncation_bound
            && self.inner.max_grid_size == other.inner.max_grid_size
            && self.inner.tail_mass_truncation == other.inner.tail_mass_truncation
            && self.inner.max_conv_grid == other.inner.max_conv_grid
            && self.inner.seed == other.inner.seed
            && self.inner.mc_resolution == other.inner.mc_resolution
            && self.inner.mc_failure_probability == other.inner.mc_failure_probability
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.discretization.to_bits().hash(&mut hasher);
        self.inner
            .log_mass_truncation_bound
            .to_bits()
            .hash(&mut hasher);
        self.inner.max_grid_size.hash(&mut hasher);
        self.inner.tail_mass_truncation.to_bits().hash(&mut hasher);
        self.inner.max_conv_grid.hash(&mut hasher);
        self.inner.seed.hash(&mut hasher);
        self.inner.mc_resolution.to_bits().hash(&mut hasher);
        self.inner
            .mc_failure_probability
            .to_bits()
            .hash(&mut hasher);
        hasher.finish()
    }
}
