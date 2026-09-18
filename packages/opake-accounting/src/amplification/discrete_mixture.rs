//! Stable discrete-mixture weights and conservative tail truncation.

use crate::numerics::logspace::log_sumexp;

pub(crate) fn binomial_log_probs(trials: usize, probability: f64) -> Vec<f64> {
    let mut log_probs = Vec::with_capacity(trials + 1);
    let log_1mp = (1.0 - probability).ln();
    let log_ratio = (probability / (1.0 - probability)).ln();
    log_probs.push(trials as f64 * log_1mp);
    for k in 1..=trials {
        let ratio = ((trials - k + 1) as f64 / k as f64).ln();
        log_probs.push(log_probs[k - 1] + ratio + log_ratio);
    }
    log_probs
}

pub(crate) fn truncate_upper_tail(
    full_log_probs: &[f64],
    tail_target: f64,
) -> (Vec<f64>, f64, f64) {
    let target = tail_target.clamp(0.0, 1.0);
    let mut head_mass = 0.0;
    let mut last = full_log_probs.len() - 1;
    if target > 0.0 {
        for (index, log_probability) in full_log_probs.iter().enumerate() {
            head_mass += log_probability.exp();
            if (1.0 - head_mass).max(0.0) <= target {
                last = index;
                break;
            }
        }
    }
    let selected = &full_log_probs[..=last];
    let log_head_mass = log_sumexp(selected);
    let head_mass = log_head_mass.exp();
    let tail_mass = (1.0 - head_mass).clamp(0.0, 1.0);
    (
        selected.iter().map(|value| value - log_head_mass).collect(),
        head_mass,
        tail_mass,
    )
}
