//! Rust-owned BnB transcript corpora for repeated calibration probes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::monte_carlo::{bnb_pld_from_transcripts, bnb_prepare_transcripts};

struct BnbTranscriptCorpus {
    gram: Vec<f64>,
    num_bins: usize,
    remove_components: Vec<usize>,
    remove_lz: Vec<f64>,
    add_lz: Vec<f64>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn registry() -> &'static Mutex<HashMap<u64, Arc<BnbTranscriptCorpus>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<BnbTranscriptCorpus>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_bnb_transcripts(
    gram: &[f64],
    num_bins: usize,
    num_samples: usize,
    seed: u64,
) -> Result<u64> {
    let (remove_components, remove_lz, add_lz) =
        bnb_prepare_transcripts(gram, num_bins, num_samples, seed)?;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    registry().lock().unwrap().insert(
        id,
        Arc::new(BnbTranscriptCorpus {
            gram: gram.to_vec(),
            num_bins,
            remove_components,
            remove_lz,
            add_lz,
        }),
    );
    Ok(id)
}

pub fn drop_bnb_transcript_handle(id: u64) {
    if id != 0 {
        registry().lock().unwrap().remove(&id);
    }
}

pub fn bnb_pld_from_transcript_handle(
    id: u64,
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let corpus = {
        let guard = registry().lock().unwrap();
        guard
            .get(&id)
            .cloned()
            .ok_or_else(|| PldError::InvalidParameter(format!("unknown transcript handle {id}")))?
    };
    if corpus.num_bins != num_bins || corpus.gram != gram {
        return Err(PldError::InvalidParameter(
            "transcript handle does not match gram / num_bins".into(),
        ));
    }
    bnb_pld_from_transcripts(
        &corpus.gram,
        corpus.num_bins,
        &corpus.remove_components,
        &corpus.remove_lz,
        &corpus.add_lz,
        sigma,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplification::bnb_mc_pld;

    #[test]
    fn handle_matches_one_shot_for_multiple_sigmas() {
        let b = 8;
        let gram =
            crate::matrix_factorization::lambda_cgd_gram_matrix(0.7, b * 4, b, Some(4), true, 0.0)
                .unwrap();
        let config = DiscretizationConfig {
            mc_resolution: 5e-3,
            mc_failure_probability: 1e-2,
            ..DiscretizationConfig::default()
        };
        let samples = config.resolved_num_mc_samples(2).unwrap();
        let handle = register_bnb_transcripts(&gram, b, samples, config.seed).unwrap();

        for sigma in [0.8, 1.3, 2.0] {
            let cached = bnb_pld_from_transcript_handle(handle, &gram, b, sigma, &config).unwrap();
            let one_shot = bnb_mc_pld(&gram, b, sigma, &config).unwrap();
            for delta in [1e-2, 2e-2, 5e-2] {
                assert_eq!(
                    cached.epsilon_at(delta).to_bits(),
                    one_shot.epsilon_at(delta).to_bits()
                );
            }
        }
        drop_bnb_transcript_handle(handle);
        assert!(registry().lock().unwrap().get(&handle).is_none());
    }
}
