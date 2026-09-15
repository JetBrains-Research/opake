# Mechanisms

A **mechanism** is a randomized algorithm that adds noise to a query
result to provide differential privacy. The mechanism determines the
noise distribution, its support, and how privacy loss is computed.
Opaque implements two families:

- **[DP-SGD mechanisms](dp-sgd/index.md)** — independent (per-step)
  noise. Simple, broadly applicable, composes step-by-step.
- **[DP-FTRL mechanisms](dp-ftrl/index.md)** — correlated noise across
  the whole training run via matrix factorization. Reduces effective
  noise on cumulative updates at the cost of fixing the training
  length in advance.

## Independent noise (DP-SGD)

Each training step adds fresh, independent noise to the clipped
gradient sum. Simple and broadly applicable.

| Mechanism | Noise distribution | Support |
|-----------|--------------------|----------|
| [Gaussian](dp-sgd/gaussian.md) | $\mathcal{N}(0, \sigma^2)$ | $(-\infty, +\infty)$ |

## Correlated noise (DP-FTRL)

Instead of independent noise at each step, matrix-factorization (MF)
mechanisms add *correlated* noise designed to partially cancel over
the training run. This reduces effective noise on cumulative updates,
improving accuracy for the same privacy budget — at the cost of
knowing the total number of steps in advance.

For assumptions (workload vs DP correctness, LR schedules, private
second-moment caveats), see the [DP-FTRL user guide](../user-guide/dp-ftrl.md).

| Mechanism | Strategy | Memory | Best for |
|-----------|----------|--------|----------|
| [BandMF](dp-ftrl/band-mf.md) | Banded Toeplitz | $O(\text{bands})$ | General use, moderate runs |
| [BLT](dp-ftrl/blt.md) | Buffered Linear Toeplitz | $O(\text{buffers})$ | Long runs ($n > 5000$), multi-epoch |
| [DP-λCGD](dp-ftrl/lambda-cgd.md) | PRNG replay (exponential decay) | $O(1)$ | Zero extra memory, any run length |
| [BISR](dp-ftrl/bisr.md) | Banded inverse square root | $O(p)$ | Asymptotically optimal; generalizes λCGD |
| [BSR](dp-ftrl/bsr.md) | Banded square root (closed form) | $O(p)$ | Paper `alpha`, `beta` kwargs; no optimizer at init |
| Identity | $I$ (no correlation) | $O(1)$ | Baseline / ablation |

## Which mechanism should I use?

```
Need correlated noise across steps (DP-FTRL)?
│
├─ No ─── Gaussian (standard DP-SGD)
│
└─ Yes ── Constraints?
          ├─ Zero extra memory → DP-λCGD (PRNG replay)
          ├─ Asymptotically optimal → BISR (generalizes λCGD)
          ├─ Closed-form workload (α>β) → BSR (NeurIPS 2024)
          ├─ n < 5000 → BandMF + opaque.dpftrl.accounting.poisson (good default)
          └─ n > 5000, multi-epoch → BLT (memory-efficient)
```

For most DP-SGD workloads, **Gaussian** is the right starting point.
Use an MF mechanism only when you need the privacy-utility improvement
of correlated noise and are willing to fix the training length in
advance.

## Amplification compatibility

Subsampling amplification reduces per-step privacy cost. Not all
mechanisms support all amplification types:

| Mechanism | `dpsgd_acc.poisson` | `dpsgd_acc.poisson` (truncated) | `dpsgd_acc.k_out_of_t` | `dpftrl_acc.poisson` | `dpftrl_acc.balls_in_bins` |
|-----------|:-:|:-:|:-:|:-:|:-:|
| Gaussian | Yes | Yes | Yes | — | — |
| BandMF | — | — | — | Yes | — |
| Identity MF | — | — | — | Yes | Yes |
| BLT | — | — | — | — | Yes |
| DP-λCGD | — | — | — | — | Yes |
| BISR | — | — | — | — | Yes |
| BSR | — | — | — | — | Yes |

- **`opaque.dpsgd.accounting.poisson`**: DP-SGD per-step Poisson
  subsampling ($q$ per example).
- **`opaque.dpsgd.accounting.poisson` (truncated)**: Same factory with
  `truncated_batch_size` and `dataset_size`; caps batches (weaker
  privacy than plain Poisson at the same $q$ unless noise is
  recalibrated).
- **`opaque.dpsgd.accounting.k_out_of_t`**: DP-SGD block or total k-out-of-t
  allocation. Block accounting is exact; total allocation uses the block
  reduction as a conservative upper bound.
- **`opaque.dpftrl.accounting.poisson`**: DP-FTRL whole-process Poisson
  amplification (`BandMf` / `IdentityMf` inner, `n_steps` required).
  For `BandMf` this is the cyclic-participation analysis
  ($\lceil n/b\rceil$ independent groups).
- **`opaque.dpftrl.accounting.balls_in_bins`**: Random-partition
  amplification with the assignment **fixed across epochs**. Used with
  BLT, DP-λCGD, BISR, BSR, and identity MF. Not interchangeable with
  `dpsgd_acc.k_out_of_t(..., allocation="block")`, which draws each block independently — the two
  schemes have different samplers and different accountants.
- **internal**: BLT handles multi-participation patterns (min-sep)
  within its own sensitivity computation — no external amplification
  wrapper needed. BLT also supports `balls_in_bins` with a
  pre-computed Gram matrix.

## Quick comparison

```python
import opaque.accounting as acc                  # cross-cutting primitives
import opaque.dpsgd.accounting as dpsgd_acc      # DP-SGD factories
import opaque.dpftrl.accounting as dpftrl_acc    # DP-FTRL factories
from opaque.dpftrl.noise import band_mf_strategy, lambda_cgd_strategy

# --- Independent noise ---
gauss = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.01) * 1000

# --- Correlated noise ---
# BandMF: strategy computes sensitivity and coefficients
band_s = band_mf_strategy(bands=10)
band = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, band_s),
    sample_rate=0.01,
    n_steps=1000,
)

# DP-λCGD: strategy.as_mechanism populates the accounting
lcgd_s = lambda_cgd_strategy(lambda_=0.9)
lcgd = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, lcgd_s),
    num_bins=100, n_steps=500,
)

for name, proc in [("Gaussian", gauss), ("BandMF", band), ("λCGD", lcgd)]:
    print(f"{name:12s}  ε = {proc.epsilon_at(1e-5):.4f}")
```
