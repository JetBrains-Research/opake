# BandMF Mechanism

BandMF adds **correlated noise** across training steps using a banded Toeplitz
strategy matrix. Unlike independent Gaussian noise (DP-SGD), the noise at each
step depends on noise from recent steps. This correlation is designed so that
noise partially cancels on cumulative updates, reducing effective noise for the
same privacy budget — the DP-FTRL framework.

## Idea

In standard DP-SGD, each step adds independent noise $z_t \sim \mathcal{N}(0, \sigma^2 I)$.
When computing cumulative sums (as optimizers like SGD with momentum or Adam
implicitly do), these independent noises accumulate: after $n$ steps, the
cumulative noise variance grows as $n \sigma^2$.

Matrix factorization flips this: instead of adding independent noise, the
mechanism adds **correlated** noise $n_t = \sum_i C^{-1}_{t,i} \cdot z_i$ where
$C^{-1}$ is the inverse of a carefully chosen strategy matrix $C$. The
correlations are designed so that:

1. Each individual step is still noisy enough for privacy
2. But cumulative sums see partially-cancelling noise, giving lower MSE

**BandMF** specifically uses a **banded lower-triangular Toeplitz** matrix for
$C$. "Banded" means each step's noise depends only on the last $b$ steps
(controlled by the `bands` parameter). "Toeplitz" means the coefficients
repeat — the correlation pattern is the same at every step.

**Key property**: The Toeplitz coefficients are optimized so that column norms
equal 1, giving single-participation sensitivity = 1.0.

## Mathematics

### Strategy matrix

The encoder $C$ is an $n \times n$ banded lower-triangular Toeplitz matrix:

$$C_{t,s} = \begin{cases} c_{t-s} & \text{if } 0 \leq t - s < b \\ 0 & \text{otherwise} \end{cases}$$

where $c_0, c_1, \ldots, c_{b-1}$ are the optimized Toeplitz coefficients and
$b$ is the number of bands. The coefficients satisfy $\sum_{j=0}^{b-1} c_j^2 = 1$
(column norm normalization).

### Noise generation

At step $t$, the mechanism:

1. Draws fresh noise $z_t \sim \mathcal{N}(0, \sigma^2 I)$
2. Computes correlated noise $n_t = \sum_{j=0}^{\min(t, b-1)} C^{-1}_{t, t-j} \cdot z_{t-j}$
3. Adds $n_t$ to the clipped gradient sum

Only the last $b$ noise vectors need to be stored — memory is $O(b \cdot d)$ where
$d$ is the parameter dimension.

### Sensitivity

Sensitivity is a property of the *participation schema*, not of $C$ alone.
Under single participation (each example appears at most once) it is the
maximum column $\ell_2$ norm, which the Toeplitz optimization normalizes
to 1 by construction:

$$\kappa = \max_j \|C_{\cdot, j}\|_2 = 1$$

Under $b$-min-separation participation an example may contribute to up to
$k' = \lceil n / \texttt{min\_sep} \rceil$ columns.  Because $C$ is
$b$-banded, columns at least $b$ apart are orthogonal, so the schema
sensitivity is

$$S = \max_{\pi \in \Pi_{b}} \Big\| \sum_{j \in \pi} C_{\cdot, j} \Big\|_2
\;\le\; \kappa \sqrt{k'}$$

with equality when every participating column is untruncated
([Choquette-Choo et al. 2023](https://arxiv.org/abs/2306.08153), Theorem 2).
`band_mf_strategy(...).sensitivity(n_steps=..., min_sep=..., max_participations=...)`
returns this value; pass `min_sep=n_steps, max_participations=1` to ask for
$\kappa$ instead.

### Privacy analysis

An **un-amplified** run reduces to a single Gaussian mechanism over the whole
horizon with effective noise multiplier

$$\sigma_{\text{eff}} = \frac{\sigma}{S}$$

evaluated at the `min_sep` / `max_participations` passed to
`mf_gaussian(...)`.  The PLD is computed once for this effective Gaussian
(not per-step).

The amplified routes below do **not** use $S$.  Each prices repeat
participations itself — cyclic Poisson by composing $k$ per-group PLDs,
b-min-sep by its warm-start recursion — so both normalize by the
single-participation $\kappa$.  Folding $\sqrt{k'}$ in there as well would
double-count.

## Supported amplifications

### Poisson subsampling (`opaque.dpftrl.accounting.poisson`)

The primary amplification method for BandMF. The training run is decomposed
into $k = \lceil n / b \rceil$ independent **groups** of $b$ consecutive
steps. Within each group, **cyclic Poisson** participation means each example
in the active group is included independently with probability $q$ (this is
what `opaque.dpftrl.accounting.poisson` composes over; there is no
separate `cyclic_poisson` factory).

**What this means**: instead of analyzing the full $n$-step run as one
mechanism, we analyze $k$ independent Poisson-subsampled Gaussian mechanisms
and compose them. Each group has:

- Effective noise multiplier: $\sigma_{\text{eff}} = \sigma / \kappa$, the
  single-participation column norm — **not** $S$.  Repeat participation is
  priced by the $k$-fold composition below, so normalizing by $S$ here
  would count it twice.
- Sample rate: $q$

The total privacy is the $k$-fold self-composition of the per-group PLD:

$$\text{PLD}_{\text{total}} = \text{PLD}_{\text{group}}^{\otimes k}$$

This is computed efficiently with 2 FFTs (self-composition).

This guarantee (Choquette-Choo et al. 2023, Theorem 4 / Algorithm 2) is
proved under zero-out adjacency with the $(k, b)$-participation schedule
held fixed across neighboring datasets.

```python
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
print(f"Epsilon (δ=1e-5): {eps:.2f}")
```

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `opaque.dpsgd.accounting.poisson` | No | DP-SGD per-step factory; different object |
| `opaque.dpsgd.accounting.poisson` (truncated) | No | DP-SGD only |
| `opaque.dpftrl.accounting.poisson` | Yes | Whole-process MF Poisson; $\lceil n/b \rceil$ groups for `BandMf` |

### b-min-sep subsampling (`b_min_sep`)

[Dong & Ganesh (2026)](https://arxiv.org/abs/2602.09338) introduce **warm-start
b-min-sep** subsampling: Poisson-style inclusion from the full dataset while
excluding examples that appeared in any of the previous $b-1$ batches. For
the same target expected batch size per iteration as cyclic Poisson (per-example
rate $p_0 = \mathbb{E}[|B|]/|D|$), the paper’s per-iteration inclusion probability
is $p = p_0 / (1 - p_0(b-1))$ when $b>1$.

Opaque pairs this with **Monte Carlo PLD** accounting (same family as BnB MC
for matrix mechanisms): pass the BandMF strategy’s first-column coefficients,
`n_steps`, and `p0` to `opaque.dpftrl.accounting.b_min_sep(...)`.
Training scripts can select it with `--band-mf-sampling b_min_sep` (see
`examples/train_dpftrl.py`).

!!! note "Monte Carlo confidence"
    Monte Carlo PLDs use simultaneous one-sided order-statistic bounds. Their
    unresolved mass is reported through `Pld.mc_resolution` and placed at
    `+∞`; queries at or below that delta floor return infinite ε.

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `b_min_sep()` | Yes | Confidence-bounded MC PLD; samples derive from `mc_resolution=1e-5` and `mc_failure_probability=1e-6` |

The transcript corpus needs `24 * n_steps * samples_per_direction` bytes.
Noise calibration reuses it when it fits the
`OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES` cap (default 4 GiB).
Oversized corpora emit a `RuntimeWarning` and use one-shot Monte Carlo;
setting the cap to `0` disables reuse without a warning. The sample count
and privacy bound are identical with either path.

!!! note "Without amplification"
    You can also use BandMF without subsampling by omitting the
    `opaque.dpftrl.accounting.poisson` wrapper. Pass the participation schema
    you actually run under — `mf_gaussian(sigma, strategy, n_steps=n,
    min_sep=b, max_participations=k)` — because the bare PLD is calibrated to
    the schema sensitivity $S$, and the default `min_sep=1` means "an example
    may participate on every step".

## Assumptions and limitations

- **DP correctness** is for the **implemented** banded Toeplitz matrix \(C\) and the sampler you pair with accounting. Monte Carlo guarantees hold with the confidence reported on the returned PLD.
- **`lr_schedule` (optional)**: optimization applies the schedule on the training-step axis, matching \(W_{t,s}=\eta_t\beta^{t-s}\). The strategy remains Toeplitz, while its objective weights each step's squared error by \(\eta_t^2\). Use the identical schedule in the optimizer; the strategy cannot validate an external optimizer's updates. Custom callables must be deterministic, side-effect-free, and immutable for the strategy's lifetime.
- **Momentum \(\beta=0\)**: Opaque warns because the workload becomes essentially identity (little benefit over independent noise).

## Code examples

### Noise injection

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

### Privacy accounting

The accounting constructor receives `sensitivity` and `num_groups` from
the same `band_mf_strategy` used for noise generation. This keeps both
components in sync:

```python
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=10)

# BandMF with Poisson amplification (recommended)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
```

!!! note
    Always use `strategy.sensitivity(n_steps=...)` and `strategy.num_groups` rather than
    hardcoded values. The strategy computes these from the optimized Toeplitz
    coefficients.

### End-to-end BandMF example

BandMF uses `opaque.dpftrl.sampling.CyclicPoissonSampler` with `bands` matching the
strategy so participation lines up with the noise. The same class with
`bands=1` gives plain Poisson on the full dataset each step for an identity MF
baseline (`identity_mf` / `identity_strategy`):

```python
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.random import key

sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=0.01,
    bands=1,
    n_steps=1000,
    key=key(0),
)
```

BandMF training (`bands` matches `band_mf_strategy`):

```python
import torch
from opaque.dpftrl.clipping import clipped_grad
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.random import key, split

n_steps, bands = 1000, 10
sample_rate = 0.01

# Setup
key_samp, key_noise = split(key(42), num=2)
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
strategy = band_mf_strategy(bands=bands)
noise_fn, noise_state = mf_gaussian_noise(
    params, strategy,
    n_steps=n_steps,
    noise_multiplier=result.param,
    key=key_noise,
)
sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=sample_rate,
    bands=bands,
    n_steps=n_steps,
    key=key_samp,
)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# Train
for batch in loader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
```

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `n_steps` | Must be known in advance | Total training iterations. |
| `bands` | 1 – `n_steps` | Number of Toeplitz bands. More = better noise but more memory. |

**Tips**:

- **`bands` = 4–20** covers most workloads. Start with 10.
- **`bands` = 1** reduces to independent noise (like standard DP-SGD).
- **`bands` = `n_steps`** is the full (unbanded) Toeplitz — optimal noise
  reduction but $O(n)$ memory per parameter.
- The number of cyclic groups is $\lceil n / b \rceil$. Fewer groups means
  better per-group amplification but more composition steps.
- BandMF requires knowing `n_steps` before training starts. If the training
  length is uncertain, use standard Gaussian noise with early stopping.
- Pair with `CyclicPoissonSampler` for consistent sampling and accounting —
  both take the per-active-group conditional rate (`q` above), not a
  whole-dataset rate. `opaque.transformers.TrainingArguments` auto-resolves
  `privacy_noise_mechanism="mf_band"` to `sampling_mode="cyclic_poisson"` and
  converts its global `expected_batch_size / len(train_dataset)` rate to that
  conditional rate for you, dividing by the exact per-group population
  `floor(dataset_size / bands)` (or, under DDP, `floor((dataset_size //
  world_size) / bands)`, since each rank independently partitions its own
  shard) so a non-divisible dataset size still matches Algorithm 2 /
  Theorem 4; a plain whole-dataset Poisson sampler does not realize the
  grouped participation this amplification assumes and is rejected.

## References

- **Choquette-Choo et al. (2023)** — [(Amplified) Banded Matrix Factorization: A unified approach to private training](https://arxiv.org/abs/2306.08153).
  BandMF mechanism with banded Toeplitz optimization and cyclic Poisson
  amplification.
- **Kairouz et al. (2021)** — [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039).
  DP-FTRL framework.
- **Denisov et al. (2022)** — [Improved Differential Privacy for SGD via Optimal Private Linear Operators](https://arxiv.org/abs/2202.08312).
  Matrix factorization for DP-SGD.
