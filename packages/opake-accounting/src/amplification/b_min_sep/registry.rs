//! Global registry of b-min-sep MC transcripts (Rust-owned `Vec<f64>`).
//!
//! Python list-of-floats would multiply memory; handles keep one copy in Rust
//! for reuse across calibration `sigma` probes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::mc::{bandmf_b_min_sep_pld_from_transcripts, bandmf_b_min_sep_prepare_transcripts};

/// One prepared Monte Carlo corpus.
pub struct BMinSepTranscriptCorpus {
    remove_x: Vec<f64>,
    remove_zeta: Vec<f64>,
    add_eta: Vec<f64>,
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn registry() -> &'static Mutex<HashMap<u64, Arc<BMinSepTranscriptCorpus>>> {
    static R: OnceLock<Mutex<HashMap<u64, Arc<BMinSepTranscriptCorpus>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn coef_matches(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

/// Allocate transcripts and register them. Returns handle `> 0`, or error if invalid.
pub fn register_b_min_sep_transcripts(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    num_samples: usize,
    seed: u64,
) -> Result<u64> {
    // Byte budget and LRU eviction live in Python (`_transcript_cache`); this
    // registry only stores corpora whose handles are still reachable.

    let (remove_x, remove_zeta, add_eta) =
        bandmf_b_min_sep_prepare_transcripts(strategy_coef, n_steps, p, num_samples, seed)?;

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let corpus = Arc::new(BMinSepTranscriptCorpus {
        remove_x,
        remove_zeta,
        add_eta,
        strategy_coef: strategy_coef.to_vec(),
        n_steps,
        p,
    });
    registry().lock().unwrap().insert(id, corpus);
    Ok(id)
}

/// Drop a corpus and free its memory.
pub fn drop_b_min_sep_transcript_handle(id: u64) {
    if id == 0 {
        return;
    }
    registry().lock().unwrap().remove(&id);
}

/// Build the full-horizon PLD from a registered corpus at `sigma`.
pub fn pld_from_transcript_handle(
    id: u64,
    strategy_coef: &[f64],
    p: f64,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if id == 0 {
        return Err(PldError::InvalidParameter(
            "transcript handle must be non-zero".into(),
        ));
    }
    let arc = {
        let guard = registry().lock().unwrap();
        guard
            .get(&id)
            .cloned()
            .ok_or_else(|| PldError::InvalidParameter(format!("unknown transcript handle {id}")))?
    };
    if arc.p != p || !coef_matches(&arc.strategy_coef, strategy_coef) {
        return Err(PldError::InvalidParameter(
            "transcript handle does not match strategy_coef / p".into(),
        ));
    }
    bandmf_b_min_sep_pld_from_transcripts(
        &arc.remove_x,
        &arc.remove_zeta,
        &arc.add_eta,
        strategy_coef,
        arc.n_steps,
        p,
        sigma,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplification::bandmf_b_min_sep_warm_mc_pld;
    use crate::discretization::DiscretizationConfig;

    #[test]
    fn handle_matches_one_shot() {
        let coef = vec![0.9_f64.sqrt(), 0.1_f64.sqrt(), 0.0];
        let n = 35;
        let p = 0.07;
        let sigma = 1.05;
        let cfg = DiscretizationConfig {
            seed: 55,
            mc_resolution: 5e-3,
            mc_failure_probability: 1e-2,
            ..DiscretizationConfig::default()
        };
        let s = cfg.resolved_num_mc_samples(2).unwrap();
        let h = register_b_min_sep_transcripts(&coef, n, p, s, 55).unwrap();
        let p1 = pld_from_transcript_handle(h, &coef, p, sigma, &cfg).unwrap();
        let p2 = bandmf_b_min_sep_warm_mc_pld(&coef, n, p, sigma, &cfg).unwrap();
        for delta in [1e-2, 2e-2, 5e-2] {
            assert_eq!(
                p1.epsilon_at(delta).to_bits(),
                p2.epsilon_at(delta).to_bits()
            );
        }
        assert_eq!(
            p1.estimation_failure_probability().to_bits(),
            p2.estimation_failure_probability().to_bits()
        );
        assert_eq!(p1.mc_resolution().to_bits(), p2.mc_resolution().to_bits());
        drop_b_min_sep_transcript_handle(h);
        assert!(registry().lock().unwrap().get(&h).is_none());
    }
}
