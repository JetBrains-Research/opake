//! Balls-in-Bins privacy amplification.
//!
//! Monte Carlo sampling of the dominating pair for correlated-noise matrix
//! mechanisms (DP-λCGD, BISR, BLT).  Uses banded Cholesky on the Gram matrix.
//!
//! For independent (Gaussian / AdaClip) noise, BnB amplification reduces
//! exactly to Poisson-subsampled Gaussian composed `num_bins * num_epochs`
//! times — use `poisson_pld(&gaussian_pld(...), ...).self_compose(...)`.

pub(crate) mod cyclic_cholesky;
pub mod monte_carlo;
mod registry;

pub use monte_carlo::bnb_mc_pld;
pub use registry::{
    bnb_pld_from_transcript_handle, drop_bnb_transcript_handle, register_bnb_transcripts,
};
