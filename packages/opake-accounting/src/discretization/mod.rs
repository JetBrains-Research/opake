//! Discretization utilities for Connect-the-Dots algorithm
//!
//! This module provides the core infrastructure for computing Privacy Loss
//! Distributions (PLDs) via the Connect-the-Dots algorithm.

pub mod config;
pub mod connect_the_dots;

pub use config::DiscretizationConfig;
pub(crate) use config::EpsilonBounds;
pub(crate) use connect_the_dots::discretize_asymmetric_mechanism;
pub(crate) use connect_the_dots::discretize_symmetric_mechanism;
