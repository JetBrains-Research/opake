//! BandMF **b-min-sep** subsampling amplification (Monte Carlo + transcript registry).

mod mc;
mod registry;

pub use mc::{
    bandmf_b_min_sep_pld_from_transcripts, bandmf_b_min_sep_prepare_transcripts,
    bandmf_b_min_sep_warm_mc_pld,
};
pub use registry::{
    drop_b_min_sep_transcript_handle, pld_from_transcript_handle, register_b_min_sep_transcripts,
};
