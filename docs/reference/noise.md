# Noise Injection

Opaque's noise mechanisms live next to the training paradigm they support:
`opaque.dpsgd.noise` for independent Gaussian-family noise, and
`opaque.dpftrl.noise` for matrix-factorization (correlated) noise. The base
`NoiseState` type that both build on lives in `opaque.types`.

## Overview

After clipping gradients, DP-SGD requires adding noise proportional to the clip norm and noise multiplier. The
noise obscures individual contributions, providing the actual privacy guarantee.

Opaque provides several noise mechanisms:

### Independent Noise (DP-SGD)

- **`gaussian_noise()`** — Gaussian noise; pass `bound=B` (or
  `bound=(low, high)`) for the bounded Gaussian mechanism (renormalized
  density on the interval — no point masses at the boundaries).
- **`jme_noise()`** — Projected joint release of the normalized aggregate and
  its clean square for supported adaptive optimizers.

### Correlated Noise (DP-FTRL / Matrix Factorization)

- **`mf_gaussian_noise()`** — Unified correlated noise dispatcher. Takes a strategy object and creates the
  corresponding noise mechanism.

Strategy factories (passed to `mf_gaussian_noise()`):

- **`band_mf_strategy()`** — BandMF banded Toeplitz correlated noise
- **`blt_strategy()`** — Buffered Linear Toeplitz (BLT) correlated noise
- **`lambda_cgd_strategy()`** — DP-λCGD correlated noise (PRNG replay, zero extra memory)
- **`bisr_strategy()`** — BISR (Banded Inverse Square Root) correlated noise
- **`identity_strategy()`** — Identity (DP-SGD via MF API, easy to swap)

All noise functions return `(noise_fn, state)`, where
`noise_fn(grads, state) -> (noisy_grads, new_state)`.
`jme_noise` returns a `SecondMomentNoiseOutput` carrying the noised projected
aggregate and its separately noised clean square.

### State Classes

- **`NoiseState`** — Abstract base class for all noise state types. Defines `_step_counter` and `_rng_key`.
- **`SecondMomentNoiseOutput`** — Projected JME first/aggregate-square optimizer handoff.
- **`GaussianNoiseState`** — State for `gaussian_noise()`. Holds step counter and RNG key.
- **`JmeNoiseState`** — State for `jme_noise()`, including RNG position,
  aggregate radius, allocation, and noise multiplier.
- **`JmeAllocation`** — Required paper-reference or first-variance-cap allocation policy.
- **`MFNoiseState`** — State for `mf_gaussian_noise()`. Holds internal correlation state, step counter, and RNG key.

### Distributed Sync Helpers

Use `sync()` from `opaque.distributed` to validate noise state consistency
across ranks. It auto-dispatches based on type:

- **`sync(GaussianNoiseState)`** — Validate RNG key and step counter match across ranks.
  The bounded Gaussian path (`gaussian_noise(..., bound=...)`) also returns
  `GaussianNoiseState`, so `sync()` handles it automatically.
- **`sync(JmeNoiseState)`** — Validate both RNG streams, step counter, radius,
  and allocation configuration.
- **`sync(MFNoiseState)`** — Validate MF noise state matches across ranks.

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## Projected JME

`jme_noise` accepts an already globally aggregated scalar `ClippedPytree`,
projects it to a required public L2 radius, forms the clean element-wise square,
and emits both values as `SecondMomentNoiseOutput`. Its required
`JmeAllocation` selects either the paper-reference allocation or a
first-stream variance cap.

The complete release uses the exact constrained add/remove sensitivity. After
whitening by its two realized standard deviations, it is dominated by
`dpsgd_acc.gaussian(noise_multiplier)`. Plain independent Poisson sampling may
use the ordinary Poisson accountant; bounded Gaussian, `PerGroup`, truncated or
parallel sampling, and MF noise are unsupported.

See [Projected JME](../mechanisms/dp-sgd/jme.md) for the exact statistics,
sensitivity branches, allocation semantics, noise scales, and adjacency
assumptions.

`mf_gaussian_noise` accepts scalar or `PerGroup` `max_norm` on `ClippedPytree`
inputs. Single-stream IID stddevs for `PerGroup` bounds match the
MSE-optimal allocation from `ClippedPytree.noise_stddev_for` (the same
Mahalanobis allocation as `gaussian_noise`). Leaf→group keys are optree
`ParamPath` tuples (build them with `per_group(params, …)`); nested
parameter trees are supported. Trainer/examples keep flat
`named_parameters` by choice.

## Gaussian (optionally bounded)

`gaussian_noise` accepts an optional `bound` argument: `bound=B` for the
symmetric interval `[-B, B]` or `bound=(low, high)` for an asymmetric
one (with `low <= 0 <= high`). The per-coordinate sample is then drawn
from a Gaussian renormalized over the interval (Chen and Hale, 2024).
Bounds are absolute, in the same units as the gradient / clip norm.

::: opaque.dpsgd.noise.gaussian_noise

## Projected JME

::: opaque.dpsgd.noise.jme_noise

## Matrix Factorization Noise

### Dispatcher

::: opaque.dpftrl.noise.mf_gaussian_noise

### Strategies

::: opaque.dpftrl.noise.band_mf_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.blt_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.lambda_cgd_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.bisr_strategy
    options:
      heading_level: 4

::: opaque.dpftrl.noise.identity_strategy
    options:
      heading_level: 4

## State Classes

::: opaque.types.NoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.types.SecondMomentNoiseOutput
    options:
      show_source: true
      heading_level: 3

::: opaque.dpsgd.noise.types.GaussianNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.dpsgd.noise.types.JmeNoiseState
    options:
      show_source: true
      heading_level: 3

::: opaque.dpsgd.noise.types.JmeAllocation
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.noise.types.MFNoiseState
    options:
      show_source: true
      heading_level: 3

## Distributed Synchronization

Use `opaque.distributed.sync(state)` — it auto-dispatches on the state's
type to the right sync function. `GaussianNoiseState`, `JmeNoiseState`, and
`MFNoiseState` register handlers at import time, so no named sync call is
required.
