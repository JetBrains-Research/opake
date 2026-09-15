# DP-SGD mechanisms

DP-SGD adds calibrated noise to a clipped gradient at every step.
Privacy is composed step-by-step: each call to the noise mechanism is
a separate `DpProcess`, multiplied by the number of training steps in
the accountant.

## Mechanisms

- **[Gaussian](gaussian.md)** — independent Gaussian noise on the
  clipped gradient; the canonical DP-SGD release.
- **[MoE load balancing](moe-load-balancing.md)** — the batch router
  load of a mixture-of-experts model released inside the clipper, priced
  jointly with the gradient.

## Pairing with sampling

DP-SGD typically pairs the Gaussian mechanism with Poisson subsampling
(`opaque.dpsgd.sampling.PoissonSampler`). The accounting form is
`opaque.dpsgd.accounting.poisson(opaque.dpsgd.accounting.gaussian(nm),
sample_rate=q)`; multiply by `* num_steps` for full-training privacy.

## See also

- [DP-SGD end-to-end guide](../../user-guide/dp-sgd.md) — full
  training pipeline.
- [DP-FTRL mechanisms](../dp-ftrl/index.md) — the other family of
  noise mechanisms in Opaque (correlated, whole-process).
