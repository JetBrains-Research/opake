# BLT Mechanism

BLT (Buffered Linear Toeplitz) is a correlated noise mechanism optimized
for **long training runs** and **multi-epoch training**. It uses a parametric
representation of the Toeplitz coefficients via exponential decay buffers,
giving $O(\text{buffers})$ memory regardless of training length. Unlike
[BandMF](band-mf.md), BLT natively supports multi-participation patterns
where each user's data is seen multiple times.

## Idea

BandMF stores explicit Toeplitz coefficients and is limited by the `bands`
parameter. For very long training runs ($n > 5000$), you would need many
bands for good noise reduction, consuming proportional memory.

BLT solves this by **parametrizing** the Toeplitz coefficients through a
small number of exponential decay buffers. Each buffer $i$ has a decay
factor $\theta_i$ and a scale factor $\omega_i$. The effective coefficient
at lag $k$ is:

$$c_k = \sum_i \omega_i \cdot \theta_i^k$$

This sum of exponentials can represent long-range correlations with just a
few buffers (typically 3–10), regardless of $n$.

**Multi-epoch support**: When training for multiple epochs, each example
participates multiple times. BLT's sensitivity computation accounts for
this via `min_sep` (minimum steps between participations) and
`max_participations`, giving tight privacy bounds for multi-epoch training.

**When to use**: Long training runs ($n > 5000$) where BandMF's banded
approach would require too many bands, or multi-epoch training where the
sensitivity analysis needs to account for repeated participation.

## Mathematics

### BLT parametrization

The encoder is a lower-triangular Toeplitz matrix with coefficients
derived from $B$ exponential buffers:

$$C_{t,s} = \sum_{i=1}^{B} \omega_i \cdot \theta_i^{t-s} \quad \text{for } t \geq s$$

The parameters $\{(\omega_i, \theta_i)\}_{i=1}^{B}$ are optimized to
minimize workload error (max or mean MSE on prefix sums).

### Sensitivity

The sensitivity depends on the participation pattern.

**Single participation** ($\text{min\_sep}=1$, $\text{max\_participations}=1$):

Sensitivity squared is computed from the BLT parameters directly:

$$S^2 = 1 + \sum_{i,j} \omega_i \cdot \omega_j \cdot G(\theta_i \cdot \theta_j, n-1)$$

where $G(\rho, n) = \sum_{k=1}^{n} \rho^k = \rho \cdot \frac{1 - \rho^n}{1 - \rho}$ is the geometric sum.

For $n \to \infty$:

$$S^2 = 1 + \sum_{i,j} \frac{\omega_i \cdot \omega_j}{1 - \theta_i \cdot \theta_j}$$

**Min-sep participation** ($\text{min\_sep} > 1$):

When examples can participate multiple times with at least `min_sep` steps
between participations, sensitivity uses the Toeplitz min-sep algorithm
(Theorem 2 of the BSR paper). The algorithm:

1. Pads coefficients to min-sep blocks.
2. Computes cumulative sums within blocks.
3. Uses sliding-window subtraction to find the worst-case participation.

**Fixed-epoch participation** is not natively supported by BLT.

### Privacy analysis

The entire training run reduces to a single Gaussian mechanism with
effective noise multiplier:

$$\sigma_{\text{eff}} = \frac{\sigma}{S}$$

where $S$ is the sensitivity. The PLD is a single Gaussian PLD.

## Assumptions and limitations

- BLT targets long runs via a **buffered** Toeplitz parameterization; privacy is for the optimized strategy you instantiate.
- Optional **`lr_schedule`** weights the optimizer objective on the training-step axis, matching \(W_{t,s}=\eta_t\beta^{t-s}\) while retaining BLT's Toeplitz strategy representation. Use the identical schedule in the optimizer; the strategy cannot validate an external optimizer's updates. Custom callables must be deterministic, side-effect-free, and immutable for the strategy's lifetime.
- **Subsampling**: BLT does not use `dpftrl_acc.poisson` as BandMF does; combine it with Balls-in-Bins when using correlated MF with epoch structure (see examples).
- Overview: [Correlated noise (DP-FTRL)](../../user-guide/dp-ftrl.md).

## Supported amplifications

BLT handles multi-participation patterns **internally** via the
sensitivity computation. There is no external amplification wrapper.

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | No | Not applicable |
| `poisson()` (truncated) | No | Not applicable |
| `dpftrl_acc.poisson` | No | For BandMF / identity MF |

If you need subsampling amplification with correlated noise, use
[BandMF](band-mf.md) with `opake.dpftrl.accounting.poisson` instead.

!!! note "Multi-epoch vs subsampling"
    BLT and Poisson subsampling solve different problems. Poisson subsampling
    amplifies privacy by using a random subset at each step. BLT handles the
    privacy cost of the same example appearing in multiple steps via its
    multi-participation (min-sep / max-participations) sensitivity computation.
    BLT itself does **not** model subsampling amplification and has no
    `sample_rate` parameter. If you need subsampling with correlated noise,
    use [BandMF](band-mf.md) with `opake.dpftrl.accounting.poisson` instead.

## Code examples

### Noise injection

```python
from opake.dpftrl.noise import mf_gaussian_noise, blt_strategy
from opake.random import key

# Single participation
strategy = blt_strategy(max_buffers=10)
noise_fn, noise_state = mf_gaussian_noise(
    grad_template=params,
    strategy=strategy,
    n_steps=10000,
    min_sep=1,
    max_participations=1,
    noise_multiplier=noise_multiplier,
    key=key(42),
)

for step in range(10000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
```

```python
# Multi-epoch: each user participates up to 5 times, ≥100 steps apart
strategy = blt_strategy(max_buffers=10)
noise_fn, noise_state = mf_gaussian_noise(
    grad_template=params,
    strategy=strategy,
    n_steps=5000,
    min_sep=100,
    max_participations=5,
    noise_multiplier=noise_multiplier,
    key=key(42),
)
```

### Privacy accounting

Use the same strategy, horizon, and participation bounds as noise generation.
The bare accountant below prices the 5000-round example above, assuming each
protected unit contributes at most five times, at least 100 rounds apart:

```python
import opake.dpftrl.accounting as dpftrl_acc
from opake.dpftrl.noise import blt_strategy

strategy = blt_strategy(max_buffers=10)

# Unamplified accounting for the multi-participation context above.
proc = dpftrl_acc.mf_gaussian(
    noise_multiplier, strategy, n_steps=5000, min_sep=100, max_participations=5,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"

# Separate run: a fixed partition into 100 bins, repeated for five epochs.
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=100, n_steps=500,
)
eps = proc.epsilon_at(delta=1e-5)
```

!!! note
    BLT optimizes its encoder for the participation context. Sharing a strategy
    recipe is sufficient only when accounting and noise generation use the same
    `n_steps`, `min_sep`, and `max_participations`. Amplifiers supply these bounds
    for their sampling scheme; leave the inner `mf_gaussian` horizon unspecified.

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `n_steps` | Must be known in advance | Total training iterations. |
| `min_sep` | 1 – `n_steps` (default 1) | Minimum steps between participations. Higher = lower sensitivity. |
| `max_participations` | Positive integer or `None` (default) | Upper bound on contributions per protected unit; `None` permits all contributions allowed by the horizon and separation. |
| `error` | `"max"` or `"mean"` | Error metric to optimize. `"max"` is conservative. |
| `max_buffers` | 1 – 20 (default 10) | Number of exponential decay buffers. More = better noise reduction. |

**Tips**:

- **`max_buffers` = 10** is a good default. Going above 10 rarely helps.
- **`min_sep`** must be guaranteed by the sampler. Repeating a fixed ordering
  or fixed bin assignment gives one epoch of separation. Independently
  shuffling each epoch can place two appearances in consecutive rounds.
- **`max_participations`** bounds contributions per protected unit. It equals
  the number of epochs when that unit appears once per epoch.
- **`error = "max"`** is the safe choice for worst-case guarantees.
  `"mean"` optimizes average error, which may be better for practical
  accuracy but gives worse worst-case behavior.
- BLT requires knowing `n_steps`, `min_sep`, and `max_participations`
  before training starts.
- For single-participation training (1 epoch), BLT and BandMF give
  similar results. BLT's advantage is primarily in multi-epoch settings.

## References

- **Dvijotham et al. (2024)** — [Efficient and Near-Optimal Noise Generation for Streaming Differential Privacy](https://arxiv.org/abs/2404.16706).
  BLT mechanism with buffered linear Toeplitz optimization.
- **McMahan et al. (2024)** — [A Hassle-free Algorithm for Private Learning in Practice: Don't Use Tree Aggregation, Use BLTs](https://arxiv.org/abs/2408.08868).
  Multi-epoch BLT extension and practical evaluation.
- **Kalinin, Lampert (2024)** — [Banded Square Root Matrix Factorization for Differentially Private Model Training](https://arxiv.org/abs/2405.13763).
  BSR paper; Theorem 2 gives Toeplitz min-sep sensitivity for multi-participation.
- **Kairouz et al. (2021)** — [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039).
  DP-FTRL framework.
