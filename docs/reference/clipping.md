# Gradient Clipping

Per-example clipping APIs for DP training: fixed norm, AUTO-S scaling, and
(on the DP-SGD side) adaptive clipping. DP-FTRL users import fixed and
AUTO-S helpers from `opaque.dpftrl.clipping`; DP-SGD users use
`opaque.dpsgd.clipping` for all three.

## Overview

Per-sample gradient clipping bounds the influence that any single training
example can have on the model, enabling differential privacy.

### Clipping functions

1. **[`clipped_grad()`](../user-guide/clipping.md#clipped_grad-recommended-api)** — High-level API: differentiates, clips, and sums gradients.
2. **[`clipped_fun()`](../user-guide/clipping.md)** — Clip and sum arbitrary function outputs (PyTrees).
3. **[`auto_clipped_grad()`](../user-guide/clipping.md)** — AUTO-S (Bu et al. NeurIPS 2023) automatic per-example gradient scaling. Constant per-record sensitivity ⇒ composes with both DP-SGD's Gaussian mechanism and DP-FTRL's matrix-factorization mechanisms.
4. **[`auto_clipped_fun()`](../user-guide/clipping.md)** — AUTO-S for arbitrary function outputs.
5. **[`adaptive_clipped_grad()`](../user-guide/clipping.md)** — Adaptive clipping (Andrew et al. 2021) with automatic threshold tuning; DP-SGD-only.
6. **[`moe_clipped_grad()`](../mechanisms/dp-sgd/moe-load-balancing.md)** — MoE clipping with the batch router load released inside the clipper; accounted by `moe_aux`.
7. **[`clip_pytree()`](../user-guide/clipping.md)** — Low-level: clip an existing PyTree of gradients.
8. **[`auto_scale_pytree()`](../user-guide/clipping.md)** — Low-level: AUTO-S scale an existing PyTree.

### State types

- **`ClipState`** — Base class for clipping state.
- **`FixedClipState`** — Marker state for fixed `clipped_grad` / `clipped_fun`.
- **`AdaptiveClipState`** — Internal execution state for `adaptive_clipped_grad`.
- **`AutoClipState`** — Marker state for `auto_clipped_grad` / `auto_clipped_fun`.
- **`MoeClipState`** — State of `moe_clipped_grad`: the load estimate `f_tilde` plus the `step`, `load_noise_std`, `filtered_noise_std` and `imbalance` properties.

### Auxiliary output types

- **`ClippedGradAux`** — Per-example `loss_values`, `grad_norms`, `clipped_grad_norms`, `loss_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_grad`).
- **`ClippedFunAux`** — Per-example `values`, `norms`, `clipped_norms`, `value_aux`, plus aggregate `clipping_rate` and `batch_size` (from `clipped_fun`).
- **`AdaptiveClippedGradAux`** — Inherits all fields from `ClippedGradAux` (from `adaptive_clipped_grad`).
- **`AutoClippedGradAux`** — Inherits `ClippedGradAux` fields with AUTO-S semantics: `grad_norms` (pre-scale), `clipped_grad_norms` (post-scale, bounded by R), `clipping_rate` (fraction with ‖g‖ > R), `loss_values`, `loss_aux`, `group_norms`, `batch_size`.
- **`AutoClippedFunAux`** — Inherits `ClippedFunAux` fields with AUTO-S semantics: `norms` (pre-scale), `clipped_norms` (post-scale, bounded by R), `clipping_rate` (fraction with ‖v‖ > R), `values`, `value_aux`, `group_norms`, `batch_size`.

### PyTree wrappers

- **`ClippedPytree`** — Wraps a pytree of clipped gradients with calibration metadata (`max_norm`). Carries a `max_norm` which is either a scalar `float` or a `PerGroup` for per-parameter-group clipping.

  **`noise_stddev_for(*, noise_multiplier, allocation="optimal") -> float | PerGroup`**
  Returns the noise standard deviation that `gaussian_noise(noise_multiplier)` would apply to this
  clipped pytree.

  - For scalar `max_norm`: returns `noise_multiplier * max_norm`.
  - For `PerGroup` `max_norm` with `allocation="optimal"` (default): returns `PerGroup` of per-group
    standard deviations `σᵢ = noise_multiplier · √(Cᵢ · ΣⱼCⱼ)` (MSE-optimal Mahalanobis allocation).
  - For `PerGroup` with `allocation="isotropic"`: returns scalar `noise_multiplier * max_norm.effective`.

  Privacy accounting is `gaussian(noise_multiplier)` regardless of `allocation` — per-group allocation
  costs nothing in the accountant because the Mahalanobis constraint is satisfied with equality.

  ```python
  grads, state = grad_fn(params, x, state=clip_state)
  stddev = grads.noise_stddev_for(noise_multiplier=0.8)
  noise_fn, noise_state = gaussian_noise(noise_multiplier=0.8, ...)
  ```

- **`NoisedPytree`** — Subclass of `ClippedPytree` that also carries a `noise_stddev` field (scalar or `PerGroup`). Returned by all noise functions.

### Distributed sync helpers

Use `sync()` from `opaque.distributed` to synchronize any clipping state or aux
object. It auto-dispatches to the right function based on type:

- **`sync(FixedClipState | AutoClipState)`** → marker-state passthrough.
- **`sync(AdaptiveClipState)`** → sums local clipped counts and batch sizes across ranks, recomputes the global clipping rate, and updates the internal next adaptive threshold identically on every rank.
- **`sync(ClippedFunAux | ClippedGradAux | AutoClippedFunAux | AutoClippedGradAux | AdaptiveClippedGradAux)`** → gathers aux across ranks.

Dispatch resolves by exact type and then by `__mro__`, so any subclass of a
registered clipping state or aux type reaches its base handler; an unregistered
type raises `InputTypeError` rather than passing through unsynchronized.

**See also**: [Per-Sample Gradient Clipping User Guide](../user-guide/clipping.md)

## API Documentation

::: opaque.dpsgd.clipping
    options:
      show_source: true
      heading_level: 2

::: opaque.dpsgd.clipping.fun
    options:
      show_source: true
      heading_level: 2

::: opaque.dpsgd.clipping.types
    options:
      show_source: true
      heading_level: 2

::: opaque.dpftrl.clipping
    options:
      show_source: true
      heading_level: 2

::: opaque.dpftrl.clipping.fun
    options:
      show_source: true
      heading_level: 2

::: opaque.dpftrl.clipping.types
    options:
      show_source: true
      heading_level: 2
