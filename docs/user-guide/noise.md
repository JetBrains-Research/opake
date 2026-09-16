# Noise Addition

After clipping gradients, the next step in DP-SGD is adding calibrated noise
to the sum of clipped gradients. The noise obscures individual contributions,
providing the differential privacy guarantee. The amount of noise is
proportional to the sensitivity (from clipping) and the desired privacy level.

All noise functions in Opaque follow the same pattern: they return a
`(noise_fn, state)` tuple with immutable state.

For mathematical details, privacy analysis, and parameter guidance for
each mechanism, see the [Mechanisms](../mechanisms/index.md) reference.

For MF-specific assumptions (workload fidelity vs DP correctness, LR schedules,
and BSR scope), see [Correlated noise (DP-FTRL)](dp-ftrl.md).

## Gaussian noise

`gaussian_noise` is the standard noise mechanism for DP-SGD. It adds
independent Gaussian noise to each gradient tensor.

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key(42),
)

noisy_grads, noise_state = noise_fn(grads, noise_state)
```

`grads` must be a `ClippedPytree` from a clipping transform. The noise function
reads `grads.max_norm`, adds Gaussian noise with stddev
`noise_multiplier * grads.max_norm`, and returns a `NoisedPytree` carrying the
realized `noise_stddev` metadata for optimizers.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `noise_multiplier` | `float` | Gaussian noise multiplier. The realized stddev is `noise_multiplier * grads.max_norm`. |
| `key` | `RngKey` | Explicit RNG key for deterministic noise. Create with `key(seed)`. |

### Calibrating the noise multiplier

The accountant calibrates `noise_multiplier`; the clipped output supplies the
per-step bound at runtime:

- `noise_multiplier` is determined by the target privacy budget (use
  `acc.calibrate()` to find it)
- `grads.max_norm` comes from `clipped_grad`, `adaptive_clipped_grad`, or
    `auto_clipped_grad`

```python
import opaque.accounting as acc

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1,
    param_max=10.0,
)

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=result.param,
    key=key(42),
)
```

See [Privacy Accounting](accounting.md) for details on calibration.

### State flow

Each call to `noise_fn` returns a new state with an incremented step counter.
Always use the returned state for the next call.

```python
noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, state = noise_fn(grads, state)  # state advances
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)
```

Internally, noise at step t is generated from `fold_in(base_key, t)`, ensuring
deterministic per-step noise regardless of execution order.

### Zero noise multiplier

When `noise_multiplier=0`, `gaussian_noise` returns `NoisedPytree` updates with
zero noise. This is useful for toggling DP on and off without changing the
training loop.

### Per-group noise

When using [per-group clipping](clipping.md#per-group-clipping),
`gaussian_noise` (bounded or not) uses MSE-optimal allocation automatically.
To inspect the realized allocation, call
`ClippedPytree.noise_stddev_for(...)` directly on the clipped output:

```python
# Default 'optimal' allocation: PerGroup with σ_i = nm · √(C_i · Σⱼ C_j)
stddev = grads.noise_stddev_for(noise_multiplier=noise_multiplier)

# Or 'isotropic' (uniform): scalar nm · ‖C‖₂
uniform_stddev = grads.noise_stddev_for(
    noise_multiplier=noise_multiplier,
    allocation="isotropic",
)
```

For a bare `PerGroup` bound (for example restored via
`opaque.serialization.from_state_dict` with a matching template), wrap it
in a `ClippedPytree` with placeholder tensors that share the same parameter
keys, then call `noise_stddev_for` as above.

Both routes return a `PerGroup` of per-group standard deviations with
$\sigma_i \propto \sqrt{C_i}$.  `gaussian_noise` applies the optimal
allocation automatically when `grads.max_norm` is a `PerGroup`.  Privacy
accounting is identical under either allocation — just `gaussian(nm)`.

The [training script](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd.py) uses this by default
when per-group clipping is active.

Alternatively, isotropic noise (same σ everywhere) also works:

```python
noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

`noisy_grads.noise_stddev` records the realized scalar or per-group noise
scale used for that step.

## Bounded Gaussian noise

Standard Gaussian noise has unbounded support, which means privatized outputs
can land arbitrarily far from the input.  Pass ``bound`` to `gaussian_noise`
to restrict the per-coordinate support to a finite interval — the
*bounded Gaussian mechanism* of Chen and Hale (2024).

`gaussian_noise(..., bound=...)` draws from a Gaussian renormalized over the
interval. No probability mass sits at the boundaries — the density is smooth
but slightly taller than the original Gaussian.  ``bound`` accepts a positive
scalar ``B`` (interpreted as ``[-B, B]``) or an asymmetric ``(low, high)``
tuple/list with ``low <= 0 <= high``.  Bounds are absolute (same units as
the gradient / clip norm), not multiples of σ.

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

# Symmetric bound [-2, 2]
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    bound=2.0,
    key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)

# Asymmetric bound [-1, 4]
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    bound=(-1.0, 4.0),
    key=key(42),
)
```

The implementation uses an inverse-CDF method: for each gradient element,
noise is sampled from a Gaussian centred on that element and truncated to
the per-coordinate interval.

Treat `bound=` as experimental: `dpsgd_acc.gaussian()` does **not** cover the
bounded output.

## Projected JME (DP-SGD)

`jme_noise` privately releases a normalized clipped aggregate and its clean
element-wise square for Adam-family optimizers. The aggregate is first projected
to an explicit public L2 radius, so variable Poisson batch sizes cannot make the
square query unbounded.

```python
from opaque.dpsgd.noise import jme_noise
from opaque.dpsgd.noise.types import JmeAllocation

noise_fn, noise_state = jme_noise(
    noise_multiplier=noise_multiplier,
    aggregate_norm=1.0,
    allocation=JmeAllocation.first_variance_cap(1.5),
    key=key(42),
)

# `grads` is the scalar ClippedPytree returned by clipped_grad after any
# distributed sum.
moments, noise_state = noise_fn(grads, noise_state)
updates, opt_state = optimizer.update(moments, opt_state, params=params)
```

The allocation is explicit because no nonzero lambda is universally optimal
under add/remove adjacency. `paper_reference()` reproduces the paper's
identity-strategy convention; `first_variance_cap(kappa)` minimizes
second-stream variance while limiting first-stream variance inflation to
`kappa`.

After whitening, the joint release is dominated by
`dpsgd_acc.gaussian(noise_multiplier)`. This accounting statement applies to
plain independent Poisson sampling. `PerGroup`, bounded Gaussian, truncated or
parallel Poisson, horizon allocation, and MF noise are unsupported.

See [Projected JME](../mechanisms/dp-sgd/jme.md) for the exact clean statistics,
normalization, sensitivity, lambda, realized scales, and adjacency assumptions.

## Matrix-factorization noise (DP-FTRL)

Standard Gaussian noise is independent across training steps. Matrix-
factorization (MF) noise introduces correlations between steps so that noise
partially cancels when aggregated over the training run. This reduces the
effective noise on cumulative updates, improving accuracy for the same privacy
budget.

MF noise implements the DP-FTRL framework (Kairouz et al. 2021, Denisov et al.
2022). Instead of adding independent noise z_t at each step, the mechanism
adds correlated noise n_t = sum_i C_inv(t, i) * z_i, where C_inv is the
inverse of a strategy matrix chosen to minimize cumulative error.

The mechanism factors a **workload matrix** A (what the optimizer computes,
e.g., prefix sums) into:

- **C** (strategy matrix): Encodes the privacy mechanism
- **C^{-1}** (noising matrix): Generates correlated noise at each step
- **B = A C^{-1}** (decoder matrix): Relates noisy outputs back to workload queries

The sensitivity of C determines how much noise is needed, while the error of B
determines the effective noise on the output.

### When to use MF noise

Use MF noise when:

- Training runs many steps (hundreds to thousands)
- The privacy budget is tight and independent noise degrades accuracy too much
- You want better privacy-utility trade-offs at the same epsilon

MF noise has higher per-step overhead (maintaining correlation buffers) and
requires knowing the total number of steps in advance.

### Compatible clipping

`mf_gaussian_noise` requires a constant per-step record sensitivity (the strategy
matrix is optimized offline against a fixed `Δ`). Both `clipped_grad`
(fixed threshold) and `auto_clipped_grad` (AUTO-S smooth scaling) satisfy
this — their per-record bound is set at construction and does not depend
on data — so either can be wired into the loop interchangeably:

```python
from opaque.dpsgd.clipping import auto_clipped_grad
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.random import key

grad_fn, clip_state = auto_clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    R=1.0,
    normalize_by=batch_size,
)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    band_mf_strategy(bands=4),
    n_steps=num_steps,
    noise_multiplier=noise_multiplier,
    key=key(0),
)

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
```

`adaptive_clipped_grad` is *not* compatible: its threshold drifts across
steps, the dispatcher's `_validate_constant_max_norm` latch rejects the
varying `max_norm`, and the standard MF privacy proof would not apply.

### Variants

Opaque provides five MF strategies, all used through the unified `mf_gaussian_noise()` dispatcher:

| Strategy factory | Memory | Best for |
|----------|--------|----------|
| `band_mf_strategy()` | O(bands) | General use with ``dpftrl_acc.poisson`` amplification |
| `blt_strategy()` | O(buffers) | Long training runs (n > 5000), multi-epoch |
| `lambda_cgd_strategy()` | O(1) | Zero extra memory (PRNG replay) |
| `bisr_strategy()` | O(bandwidth) | Asymptotically optimal, arbitrary bandwidth |
| `identity_strategy()` | O(1) | Testing MF infrastructure with standard noise |

All strategies are created by factory functions and passed to `mf_gaussian_noise()`:

```python
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.random import key

strategy = band_mf_strategy(bands=10)
noise_fn, noise_state = mf_gaussian_noise(
    grad_template=params,
    strategy=strategy,
    n_steps=1000,
    noise_multiplier=noise_multiplier,
    key=key(42),
)

for step in range(1000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
```

The `grad_template` argument provides shape and dtype information for
pre-allocating noise buffers. Pass any pytree with the same structure as
the gradients (e.g., the model parameters).

### Per-group clipping

`mf_gaussian_noise` accepts `ClippedPytree` metadata where `max_norm` is a
`PerGroup` (from `opaque.dpsgd.clipping.per_group`), not only a scalar. The
per-leaf IID noise scale follows the same MSE-optimal Mahalanobis allocation
as `gaussian_noise` (bounded or not) on DP-SGD: no extra privacy
cost versus scalar clipping at the same `noise_multiplier`, and the MF
Gaussian accountant is unchanged. Leaf→group assignment is keyed by
optree `ParamPath` tuples (the same contract as `per_group(params, …)`),
so nested parameter pytrees work as well as flat `named_parameters`
dicts. Prefer `per_group(params, …)` over hand-building `PerGroup` with
dotted string keys — a bare string normalizes to a one-segment path and
will not match nested leaves.

`DPTrainer` / examples still use flat trainable params from
`make_functional(..., partition_trainable=True)` by choice; custom loops
may pass any tensor pytree.

In distributed training, pass the same `key(seed)` on all ranks to produce
identical noise. See [Distributed Training](distributed.md) and
[RNG Key](rng-key.md) for details.

### `band_mf_strategy`

Banded Toeplitz strategy. Optimizes banded Toeplitz coefficients for the
workload. Uses ``dpftrl_acc.poisson`` for privacy accounting.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.random import key

strategy = band_mf_strategy(bands=10, momentum=0.95)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    strategy,
    n_steps=1000,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### `blt_strategy`

Buffered Linear Toeplitz strategy. More memory-efficient than BandMF for
long training runs, using a parametric representation via exponential decay
buffers. Supports multi-epoch training via `min_sep` and `max_participations`.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, blt_strategy
from opaque.random import key

strategy = blt_strategy(max_buffers=10)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    strategy,
    n_steps=10000,
    min_sep=100,
    max_participations=5,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### `lambda_cgd_strategy`

DP-λCGD strategy — uses PRNG seed replay instead of storing previous noise
vectors. Zero extra memory overhead compared to DP-SGD.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, lambda_cgd_strategy
from opaque.random import key

strategy = lambda_cgd_strategy(lambda_=0.9)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### `bisr_strategy`

BISR (Banded Inverse Square Root) strategy — generalises λCGD to arbitrary
bandwidth p ≥ 2. Asymptotically optimal.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, bisr_strategy
from opaque.random import key

strategy = bisr_strategy(
    bandwidth=4,
    momentum=0.95,
)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### `identity_strategy`

Identity strategy — equivalent to standard DP-SGD (independent noise at each
step) but using the MF API. Useful for testing or as a baseline.

```python
from opaque.dpftrl.noise import mf_gaussian_noise, identity_strategy
from opaque.random import key

strategy = identity_strategy()
noise_fn, noise_state = mf_gaussian_noise(
    params,
    strategy,
    n_steps=1000,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### Privacy accounting for MF noise

MF noise has different sensitivity than standard Gaussian noise because
the correlated strategy matrix amplifies or attenuates individual
contributions. The noise **strategy** computes `sensitivity` and
`gram_matrix` from the mechanism parameters; the accounting constructor
receives these values rather than recomputing them. This ensures that
noise generation and privacy accounting always agree on the mechanism.

```python
import opaque.accounting as acc  # cross-cutting calibration / composition
import opaque.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories
from opaque.dpftrl.noise import band_mf_strategy, lambda_cgd_strategy

# BandMF — strategy provides sensitivity and coefficients
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(1e-5)

# DP-λCGD / BISR / BLT — strategy.as_mechanism populates the accounting
strategy = lambda_cgd_strategy(lambda_=0.9)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch,
    n_steps=steps_per_epoch * num_epochs,
)

```

See [Privacy Accounting — Matrix factorization mechanisms](accounting.md#matrix-factorization-mechanisms)
for the full API.

### Multi-participation (multi-epoch)

When training for multiple epochs, each example participates multiple times.
Strategies that support multi-epoch patterns (`blt_strategy`, `lambda_cgd_strategy`,
`bisr_strategy`) accept `min_sep` and `max_participations` to compute tight
sensitivity bounds:

```python
from opaque.dpftrl.noise import mf_gaussian_noise, blt_strategy
from opaque.random import key

strategy = blt_strategy(max_buffers=10)
noise_fn, state = mf_gaussian_noise(
    grad_template,
    strategy,
    n_steps=5000,
    min_sep=100,  # minimum steps between participations
    max_participations=5,  # 5 epochs
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### Sensitivity

The sensitivity is computed internally by each strategy factory. You can
inspect it by calling the `sensitivity()` method with the participation context:

```python
strategy = band_mf_strategy(bands=10)
print(strategy.sensitivity(n_steps=500))  # typically 1.0 for normalized strategies
# Some strategies (BiSR, BSR, lambda_CGD) also require min_sep and max_participations:
# print(strategy.sensitivity(n_steps=500, min_sep=10, max_participations=5))
```

### Comparison: DP-SGD vs MF strategies

For a linear regression with n=1000 steps, epsilon=1.0:

| Method | Strategy | Relative MSE | Memory |
|--------|-----------|-------------|--------|
| DP-SGD | `identity_strategy()` | Baseline | O(1) |
| BandMF | `band_mf_strategy(bands=4)` | ~0.7x | O(bands) |
| BLT | `blt_strategy(max_buffers=3)` | ~0.6x | O(buffers) |
| λCGD | `lambda_cgd_strategy(lambda_=0.9)` | ~0.7x | O(1) |
| BISR | `bisr_strategy(bandwidth=4)` | ~0.6x | O(bandwidth) |

Values are illustrative; actual results depend on problem specifics.

### MF noise with Poisson sampling

`CyclicPoissonSampler` splits the data into `bands` groups and, at step
`i`, samples only group `i % bands` with per-example probability
`sample_rate`. Use `bands=1` with `identity_strategy` / `identity_mf`
so each step is plain Poisson on the full dataset; for BandMF, match `bands`
to `band_mf_strategy`. That keeps the data schedule aligned with
`mf_gaussian_noise` and `dpftrl_acc.poisson`:

```python
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.random import key

sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=sample_rate,
    bands=4,
    n_steps=num_steps,
    key=key(0),
)
```

## Distributed noise synchronization

In distributed training, all devices must add the same noise to maintain model
consistency. Pass the **same key** on every rank:

```python
# Same key on all ranks → identical noise → models stay in sync
noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))
```

For independent per-rank noise (not typical for centralized DP-SGD), derive
a per-rank key via `fold_in`:

```python
from opaque.random import key, fold_in
import torch.distributed as dist

rank = dist.get_rank()
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=fold_in(key(42), rank),
)
```

For validation, call `sync(noise_state)` to assert that the RNG key and
step counter match across ranks. The `sync()` dispatcher auto-detects the
noise state type. See [Distributed Training](distributed.md) for details.

## References

- [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153) — BandMF
- [Dvijotham et al., 2024](https://arxiv.org/abs/2404.16706) — BLT
- [McMahan et al., 2024](https://arxiv.org/abs/2408.08868) — Multi-epoch BLT
- [Kalinin et al., 2026](https://arxiv.org/abs/2601.22334) — DP-λCGD
- [Kalinin et al., 2026](https://arxiv.org/abs/2505.12128) — BISR
- [McMahan et al., 2025](https://arxiv.org/abs/2504.21413) — Inversion theorem
- [Kairouz et al., 2021](https://arxiv.org/abs/2103.00039) — DP-FTRL

## API reference

See [Noise API Reference](../reference/noise.md) for complete function signatures
and return types.
