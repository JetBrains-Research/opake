//! Core numerical primitives for differential privacy accounting
//!
//! This module provides low-level numerical operations used throughout
//! the DP accounting library, including FFT-based convolution and
//! special mathematical functions.
//!
//! # Modules
//!
//! - [`fft`]: Fast Fourier Transform operations for convolution
//! - [`logspace`]: Numerically stable log-space arithmetic
//! - [`special`]: Special mathematical functions for subsampling
//! - [`truncation`]: Chernoff bound truncation for efficient composition

pub mod fft;
pub(crate) mod gaussian;
pub mod logspace;
pub mod special;
pub mod truncation;
