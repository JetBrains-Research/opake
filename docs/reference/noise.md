# Noise Injection

Opake's noise mechanisms live next to the training paradigm they support:
`opake.dpsgd.noise` for independent Gaussian-family noise, and
`opake.dpftrl.noise` for matrix-factorization (correlated) noise. The base
`NoiseState` type that both build on lives in `opake.types`.

## Overview

After clipping gradients, DP-SGD requires adding noise proportional to the clip norm and noise multiplier. The
noise obscures individual contributions, providing the actual privacy guarantee.

Opake provides several noise mechanisms:

### Independent Noise (DP-SGD)

- **`gaussian_noise()`** — Gaussian noise.

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
When private second moments are enabled, the noisy value is a
`SecondMomentNoiseOutput` carrying both `noisy_grads` and
`noisy_squared_grads`.

### State Classes

- **`NoiseState`** — Abstract base class for all noise state types. Defines `_step_counter` and `_rng_key`.
- **`SecondMomentNoiseOutput`** — Paired first/squared-gradient output for private second moments.
- **`GaussianNoiseState`** — State for `gaussian_noise()`. Holds step counter and RNG key.
- **`MFNoiseState`** — State for `mf_gaussian_noise()`. Holds internal correlation state, step counter, and RNG key.
- **`SecondMomentMFNoiseState`** — Paired-stream MF state used by `mf_gaussian_noise(..., second_moment_strategy=...)`.

### Distributed Sync Helpers

Use `sync()` from `opake.distributed` to validate noise state consistency
across ranks. It auto-dispatches based on type:

- **`sync(GaussianNoiseState)`** — Validate RNG key and step counter match across ranks.
- **`sync(MFNoiseState)`** — Validate MF noise state matches across ranks.

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## Paired second-moment release

When `clipped_grad(..., second_moment=True)` produces a
`SecondMomentClippingOutput`, both `gaussian_noise` (DP-SGD) and
`mf_gaussian_noise(..., second_moment_strategy=...)` (DP-FTRL) consume it and
emit a `SecondMomentNoiseOutput` with paired noise on both streams.

The runtime σ allocation is **sensitivity-proportional** and works
identically for scalar `max_norm` and per-group `PerGroup` `max_norm`. For
each group \(g\) (single-group `K=1` for scalar clipping), let
\(\Delta^{(1)}_g\) be the first-stream per-record bound (`C_g / n` for
averaged clipping, or `ζ · ‖C₁‖` strategy-amplified for DP-FTRL) and
\(\Delta^{(2)}_g\) the second-stream per-record bound (`C_g² / n` derived
from the same clip; `ζ² · ‖C₂‖` for DP-FTRL). Then with noise multiplier
\(\text{nm}\), set

\[
  S := \sum_h \bigl(\Delta^{(1)}_h + \Delta^{(2)}_h\bigr),\qquad
  \sigma^{(1)}_g := \text{nm}\sqrt{\Delta^{(1)}_g\,S},\qquad
  \sigma^{(2)}_g := \text{nm}\sqrt{\Delta^{(2)}_g\,S}.
\]

This satisfies the joint Mahalanobis budget with equality:

\[
  \sum_{g,i} \Bigl(\frac{\Delta^{(i)}_g}{\sigma^{(i)}_g}\Bigr)^2
  = \frac{1}{\text{nm}^2}.
\]

So the paired release has **the same PLD as a single sensitivity-1
Gaussian release at the joint effective multiplier** passed to
`paired_noise_stddevs` (see its docstring: no-op for DP-SGD; divide by
`‖C₁‖` for DP-FTRL). Accounting is plain `gaussian(nm)` for DP-SGD and the
underlying `mf_gaussian(nm, …)` for DP-FTRL; there is no separate
transformation wrapper and no `ρ` knob.

`mf_gaussian_noise` accepts scalar or `PerGroup` `max_norm` on `ClippedPytree`
inputs. Single-stream IID stddevs for `PerGroup` bounds match the
MSE-optimal allocation from `ClippedPytree.noise_stddev_for` (the same
Mahalanobis allocation as `gaussian_noise`). Leaf→group keys are optree
`ParamPath` tuples (build them with `per_group(params, …)`); nested
parameter trees are supported. Trainer/examples keep flat
`named_parameters` by choice.

## Gaussian

::: opake.dpsgd.noise.gaussian_noise

## Matrix Factorization Noise

### Dispatcher

::: opake.dpftrl.noise.mf_gaussian_noise

### Strategies

::: opake.dpftrl.noise.band_mf_strategy
    options:
      heading_level: 4

::: opake.dpftrl.noise.blt_strategy
    options:
      heading_level: 4

::: opake.dpftrl.noise.lambda_cgd_strategy
    options:
      heading_level: 4

::: opake.dpftrl.noise.bisr_strategy
    options:
      heading_level: 4

::: opake.dpftrl.noise.identity_strategy
    options:
      heading_level: 4

## State Classes

::: opake.types.NoiseState
    options:
      show_source: true
      heading_level: 3

::: opake.types.SecondMomentNoiseOutput
    options:
      show_source: true
      heading_level: 3

::: opake.dpsgd.noise.types.GaussianNoiseState
    options:
      show_source: true
      heading_level: 3

::: opake.dpftrl.noise.types.MFNoiseState
    options:
      show_source: true
      heading_level: 3

::: opake.dpftrl.noise.types.SecondMomentMFNoiseState
    options:
      show_source: true
      heading_level: 3

## Distributed Synchronization

Use `opake.distributed.sync(state)` — it auto-dispatches on the state's
type to the right sync function. `GaussianNoiseState` and `MFNoiseState`
both register handlers at import time, so no named sync call is required.
