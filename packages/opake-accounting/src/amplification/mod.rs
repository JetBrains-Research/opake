//! Privacy amplification: subsampled and accumulated mechanism PLDs.
//!
//! Each public function takes scalar parameters directly (no structs, no traits).
//!
//! - [`poisson_pld`] — Poisson subsampling applied to an existing PLD
//! - [`truncated_poisson_gaussian_pld`] — Truncated Poisson-subsampled Gaussian
//! - [`parallel_poisson_gaussian_pld`] — Parallel Poisson Gaussian (gradient accumulation or parallel workers)
//! - [`bandmf_b_min_sep_warm_mc_pld`] — BandMF + warm-start b-min-sep subsampling (MC); transcript registry for calibration reuse
//! - [`bnb_mc_pld`] — Balls-in-Bins Monte Carlo for correlated-noise matrix
//!   mechanisms (BLT/λCGD/BISR/BSR).  Independent-noise BnB collapses to
//!   `poisson_pld` composed; use that directly for independent noise.

mod b_min_sep;
pub mod balls_in_bins;
mod discrete_mixture;
mod parallel_poisson;
pub(crate) mod poisson;
mod random_allocation;
mod truncated_poisson;

pub use b_min_sep::{
    bandmf_b_min_sep_pld_from_transcripts, bandmf_b_min_sep_prepare_transcripts,
    bandmf_b_min_sep_warm_mc_pld, drop_b_min_sep_transcript_handle, pld_from_transcript_handle,
    register_b_min_sep_transcripts,
};
pub use balls_in_bins::{
    bnb_mc_pld, bnb_pld_from_transcript_handle, drop_bnb_transcript_handle,
    register_bnb_transcripts,
};
pub use parallel_poisson::parallel_poisson_gaussian_pld;
pub use poisson::poisson_pld;
pub use random_allocation::random_allocation_gaussian_pld;
pub use truncated_poisson::truncated_poisson_gaussian_pld;

use crate::error::{PldError, Result};
use crate::mechanisms::validate_noise_multiplier;

fn validate_rate(rate: f64) -> Result<()> {
    if !(rate > 0.0 && rate < 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "sampling rate must be in (0, 1), got {}",
            rate
        )));
    }
    Ok(())
}
