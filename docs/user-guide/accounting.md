# Privacy Accounting

Privacy accounting tracks how much privacy budget is consumed during training.
Opaque uses Privacy Loss Distributions (PLD) computed by a Rust engine for
numerically tight composition bounds. The API is built around composable
`DpProcess` objects that represent privacy mechanisms.

For mathematical details, supported amplifications, and parameter guidance
for each mechanism, see the [Mechanisms](../mechanisms/index.md) reference.

## Namespace organization

The accounting surface is split across three modules so each algorithm's
factories live next to its runtime:

| Module | Provides | Ships with |
|--------|----------|------------|
| `opaque.accounting` | Cross-cutting primitives — composition (`compose`, `repeat`, `cached`), `calibrate`, generic mechanisms (`identity`, `nonprivate`, `eps_delta`), `Accountant`, and the shared PLD / discretization stack. | `opaque-accounting` |
| `opaque.dpsgd.accounting` | DP-SGD factories — `gaussian`, `adaclip`, `poisson` (plain or truncated via `truncated_batch_size` / `dataset_size`), `parallel_poisson`, `k_out_of_t`. | `opaque-dpsgd` |
| `opaque.dpftrl.accounting` | DP-FTRL factories — `mf_gaussian`, `poisson` (whole-process, parameterized by `n_steps`), `b_min_sep`, `balls_in_bins`. | `opaque-dpftrl` |

Private second moments do **not** use a separate accounting wrapper: the joint gradient + squared-gradient release is handled in the runtime σ split (sensitivity-proportional Mahalanobis allocation), so calibration stays on the same underlying mechanism PLD as first-moment-only training. See [Noise API](../reference/noise.md#paired-second-moment-release).

Both algorithm-specific namespaces re-export from the shared `opaque-accounting`
implementation; the split is purely organisational. The `Accountant` interactive
container is on `opaque.accounting` directly (`from opaque.accounting import
Accountant`); calibration helpers (`calibrate`, `epsilon_budget`, etc.) live
there too.

The mechanism factories themselves (`gaussian`, `poisson`, `mf_gaussian`, …) are
**only** on the algorithm-specific namespaces. Use the namespace that matches
your training run (`opaque.dpsgd.accounting` or `opaque.dpftrl.accounting`) —
the per-step (DP-SGD) vs whole-process (DP-FTRL) distinction is part of the
import path, on purpose. The algorithm-specific subpackages are
*lazy-imported* from `opaque.dpsgd` / `opaque.dpftrl`, so the Rust PLD
extension only loads when accounting is actually used.

## Core concepts

### DpProcess

Every mechanism constructor returns a `DpProcess`. Composition operators
produce new `DpProcess` instances. Privacy metrics are computed on demand from
the underlying PLD.

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

# Mechanism constructors return DpProcess instances.  Cross-cutting
# primitives (composition, calibration) live at acc; algorithm-specific
# factories live in the per-algorithm namespace.
g = dpsgd_acc.gaussian(0.8)
p = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)

# Composition produces new DpProcess instances
training = p * 1000

# Privacy metrics (all derived from the same PLD)
eps = training.epsilon_at(delta=1e-5)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
```

### Composition operators

| Operator | Description | Example |
|----------|-------------|---------|
| `proc * k` | Repeat k times (homogeneous) | `step * 1000` |
| `a \| b` | Compose two processes (heterogeneous) | `warmup \| main` |

Homogeneous composition (`*`) is used when the same mechanism is applied
repeatedly (e.g., 1000 identical DP-SGD steps). Heterogeneous composition
(`|`) is used when different mechanisms are applied in sequence (e.g., a
warmup phase with different noise followed by a main phase).

```python
# Homogeneous: same step repeated
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000

# Heterogeneous: different phases
warmup = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.3), sample_rate=0.01) * 100
main = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), sample_rate=0.01) * 900
total = warmup | main

eps = total.epsilon_at(delta=1e-5)
```

Composition is automatically optimized: identical processes are merged into
repeated nodes (2 FFTs instead of N), and identity processes are elided.

Large homogeneous compositions require a positive `tail_mass_truncation`
budget. Setting it to zero requests exact composition; if the requested
composition is too large, accounting raises an error. Use a positive tail
budget or a coarser discretization in that case.

## Mechanisms

### `dpsgd_acc.gaussian(noise_multiplier)`

Gaussian mechanism without subsampling. Rarely used directly since DP-SGD
typically uses Poisson sampling.

```python
g = dpsgd_acc.gaussian(0.8)
eps = g.epsilon_at(delta=1e-5)
```

### `dpsgd_acc.poisson(inner, sample_rate)`

Standard Poisson-subsampled mechanism. Each example is included independently
with probability `sample_rate`. This provides privacy amplification through
subsampling. Plain Poisson accepts any Opaque `DpProcess` as its base
mechanism. Use the capped form only when training truncates draws; it requires
`gaussian()`, `adaclip()`, or `nonprivate()` as its base.

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=256 / 50_000)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

### `dpsgd_acc.poisson(inner, sample_rate, *, truncated_batch_size, dataset_size)` (truncated form)

Setting both `truncated_batch_size` and `dataset_size` (must be both or
neither) selects the truncated-Poisson PLD for a per-step batch cap. That
matches production runs with capped batches but is **no stricter** than plain
Poisson at the same `sample_rate`—typically **weaker** (higher ε at the same
noise) because truncation changes the subsampling distribution.

```python
n = 50_000
batch = 256
step = dpsgd_acc.poisson(
    dpsgd_acc.gaussian(0.8),
    sample_rate=batch / n,
    truncated_batch_size=batch,
    dataset_size=n,
)
```

### `dpsgd_acc.parallel_poisson(inner, sample_rate, num_workers)`

Accounts for Poisson sampling under parallel worker execution. Like
`poisson()` (plain or truncated), this is a full wrapper: pass the
inner Gaussian mechanism and sample rate directly.

```python
step = dpsgd_acc.parallel_poisson(
    dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4,
)
```

### `dpsgd_acc.k_out_of_t(inner, *, k, t, allocation)`

With `allocation="block"`, each example lands in exactly one batch in each of
`k` contiguous, nearly equal blocks. It amplifies strictly more than
`poisson()` at the matched average rate `k / t`, and is computed by an exact
PLD transform rather than Monte Carlo.

```python
process = dpsgd_acc.k_out_of_t(
    dpsgd_acc.gaussian(0.8),
    k=num_epochs,
    t=num_epochs * steps_per_epoch,
    allocation="block",
)
eps = process.epsilon_at(1e-5)
```

The returned process accounts the complete declared horizon and must be
included exactly once.

With `allocation="total"`, each example chooses a uniform `k`-subset of the
whole horizon. The factory currently reports the block reduction as a valid
conservative upper bound.

### `dpsgd_acc.adaclip(inner, *, fraction_noise_std, expected_batch_size, num_groups=1)`

Accounts for the additional privacy cost of adaptive clipping (the noisy
quantile query). Use this when using `adaptive_clipped_grad`. `num_groups`
defaults to 1 for a single adaptive threshold; set it to the number of
independently adaptive groups for per-group clipping, because each group
releases its own noisy quantile query.

```python
expected_batch_size = sample_rate * dataset_size
step = dpsgd_acc.poisson(
    dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8),
                fraction_noise_std=0.05,
                expected_batch_size=expected_batch_size),
    sample_rate=0.01,
)
```

### `acc.eps_delta(epsilon, delta=0.0)`

A fixed (epsilon, delta)-DP mechanism. Useful for composing external privacy
costs (e.g., a hyperparameter tuning step with known privacy cost).

```python
external_cost = acc.eps_delta(1.0, delta=1e-6)
total = external_cost | (step * 1000)
```

### `acc.identity()`

Zero privacy cost. Identity element for composition. Useful as an initial
value when building up a process programmatically.

```python
process = acc.identity()
for step_proc in step_list:
    process = process | step_proc
eps = process.epsilon_at(delta=1e-5)
```

## Matrix factorization mechanisms

MF noise introduces correlations between training steps, reducing
effective noise on cumulative updates. The accounting must handle the
modified sensitivity of the strategy matrix. These mechanisms optimize
the strategy internally and compute the correct sensitivity — do not use
`dpsgd_acc.gaussian()` for MF noise.

### Strategy-driven accounting

`mf_gaussian()` couples a noise multiplier to a strategy recipe. An explicit
horizon is required for bare accounting, which resolves sensitivity using its
participation bounds. When wrapped, the amplifier supplies that context and
resolves the required sensitivity or Gram matrix at PLD evaluation time.

The workflow is:

1. Create a **noise strategy** (for example, `band_mf_strategy()` or
   `lambda_cgd_strategy()`) with its structural knobs: bands, correlation,
   momentum, and any supported workload schedule.
2. Create the inner mechanism with `mf_gaussian(noise_multiplier, strategy)`.
3. Wrap that mechanism in the sampler-matching amplifier, which supplies
   `n_steps` and the participation pattern and derives the strategy's privacy
   quantities internally.

Use the same strategy for noise generation and accounting, and match the
runtime context to the amplifier's horizon and participation pattern. A recipe
can resolve to different encoders under different contexts.

```python
from opaque.dpftrl.noise import band_mf_strategy
import opaque.dpftrl.accounting as dpftrl_acc

# Strategy is a recipe; the amplifier supplies the horizon at PLD time.
strategy = band_mf_strategy(bands=10, momentum=0.95)

proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
```

### `dpftrl_acc.mf_gaussian(noise_multiplier, strategy, *, n_steps=None, min_sep=1, max_participations=None)`

Without `n_steps`, this is an amplifier input: bare `pld()` and `epsilon_at()`
calls raise `ConfigurationError`. The amplifier supplies the horizon once.

For bare accounting, declare the full horizon and the participation bounds used
by `mf_gaussian_noise`. This example assumes one contribution per protected unit
across 1000 rounds:

```python
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.mf_gaussian(
    1.0, strategy, n_steps=1000, min_sep=1, max_participations=1,
)
eps = proc.epsilon_at(delta=1e-5)
```

Bare BandMF and Identity sensitivity currently accounts for one participation.
Do not use bare calls with these strategies to budget repeated participation.
For subsampling amplification, leave the inner horizon unspecified and pass the
full horizon to the sampler-matching amplifier (see below).

### Correlated MF mechanisms (BLT, λCGD, BISR, BSR)

Correlated strategies use the same factory. Bare accounting must match the
runtime horizon and participation bounds. This BLT example assumes at most five
contributions per protected unit, at least 1000 rounds apart:

```python
from opaque.dpftrl.noise import blt_strategy

strategy = blt_strategy(max_buffers=10)

# Unamplified accounting for the full 5000-round context.
proc = dpftrl_acc.mf_gaussian(
    1.0, strategy, n_steps=5000, min_sep=1000, max_participations=5,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"

# With a fixed partition into 1000 bins, repeated for five epochs.
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=1000, n_steps=5000,
)
```

The same pattern works for `lambda_cgd_strategy`, `bisr_strategy`, and
`bsr_strategy`:

```python
strategy = lambda_cgd_strategy(lambda_=0.9)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch, n_steps=steps_per_epoch * num_epochs,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
```

### `dpftrl_acc.poisson(inner, sample_rate, *, n_steps)`

Poisson amplification for DP-FTRL. Whole-process accountant covering all
`n_steps` rounds (do **not** compose with `* num_steps` externally).
Cyclic when the inner is `BandMf` with `bands > 1` (decomposes into
`ceil(n_steps / bands)` independent groups), plain Poisson per round
when the inner is `IdentityMf` or `BandMf` with `bands == 1`.

```python
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
```

### `dpftrl_acc.balls_in_bins(inner, *, num_bins, n_steps)`

Balls-in-Bins (random-partition) amplification. Returns the **total**
privacy cost across all `n_steps` rounds (must be a positive multiple of
`num_bins`; per-bin participation count is `n_steps // num_bins`). Do NOT
compose further externally.

Used with DP-λCGD, BISR, BSR, BLT, and `identity_strategy()`. The identity
case is exact (its Gram is `num_epochs · I`, so the dominating pair collapses
onto random allocation at `σ / √num_epochs`); the correlated cases go through
Monte Carlo.

The DP-SGD analogue is
`dpsgd_acc.k_out_of_t(..., allocation="block")`, which draws an independent
partition in every block. Do not mix the two: this accountant is for the
fixed-assignment sampler `opaque.dpftrl.sampling.BallsInBinsSampler`.

```python
strategy = lambda_cgd_strategy(lambda_=0.9)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch,
    n_steps=steps_per_epoch * num_epochs,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
```

### Calibrating MF noise

Calibration works the same way — create the strategy first, then build
the accounting mechanism from strategy-derived quantities. The strategy is
created once, and the calibration lambda varies only `noise_multiplier`:

```python
strategy = band_mf_strategy(bands=10)

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpftrl_acc.poisson(
        dpftrl_acc.mf_gaussian(nm, strategy),
        sample_rate=0.01,
        n_steps=1000,
    ),
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

## Privacy metrics

All metrics are methods on `DpProcess`. They compute the underlying PLD on
demand and cache the result.

| Method | Returns | Interpretation |
|--------|---------|----------------|
| `.epsilon_at(delta)` | float | Smallest epsilon for (epsilon, delta)-DP |
| `.delta_at(epsilon)` | float | Smallest delta for (epsilon, delta)-DP |
| `.advantage()` | float | f-DP total-variation advantage (0 = perfect privacy) |
| `.beta_at(alpha)` | float | Type-II error at Type-I error alpha (higher = more private) |
| `.risk_at(prior)` | float | Bayes risk under optimal adversary (higher = more private) |

```python
training = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01) * 1000

eps = training.epsilon_at(delta=1e-5)
delta = training.delta_at(epsilon=3.0)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
risk = training.risk_at(prior=0.5)
```

See [DP Concepts](dp-concepts.md#privacy-metrics) for the meaning of each
metric.

## Calibration

Calibration finds the parameter value (typically noise multiplier) that
achieves a target privacy budget. It uses binary search over the parameter
space.

### Basic calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

The `process` argument is a lambda that takes the parameter being calibrated
and returns a `DpProcess`. `calibrate` binary-searches for a privacy-safe
parameter whose metric is within the requested relative tolerance. For
privacy-loss budgets (epsilon, delta, and advantage), the returned metric is
at or below the target. For privacy-gain budgets (beta and risk), it is at or
above the target. Every successfully returned result has `converged=True`.

The `tolerance` argument must be finite and positive, and `max_iterations`
must be positive. Invalid values raise `CalibrationError` before the process is
evaluated. If the search cannot find a safe endpoint satisfying
`math.isclose(achieved, target, rel_tol=tolerance, abs_tol=0.0)`, it raises
`CalibrationError` instead of returning an under-noised parameter.

### Budget types

| Function | Target metric | Metric kind |
|----------|---------------|-------------|
| `acc.epsilon_budget(eps, delta)` | epsilon | privacy loss (safe at-or-below target) |
| `acc.delta_budget(delta, epsilon)` | delta | privacy loss (safe at-or-below) |
| `acc.advantage_budget(adv)` | advantage | privacy loss (safe at-or-below) |
| `acc.beta_budget(beta, alpha)` | beta | privacy gain (safe at-or-above target) |
| `acc.risk_budget(risk, prior)` | risk | privacy gain (safe at-or-above) |

### Search direction

The search direction is derived from the calibrated parameter itself: the
metric is probed at both endpoints and the privacy-safe endpoint detected
automatically. Parameters whose increase improves privacy (noise multiplier)
and parameters whose increase spends privacy (sample rate, step count) are
both supported, for every budget kind. The metric produced by
`process(param)` must be monotone over `[param_min, param_max]`; flat or
detectably non-monotone parameterizations raise `CalibrationError`.

## Accountant

The `Accountant` class provides step-by-step privacy tracking during training.
It wraps a `DpProcess` and provides budget checking.

```python
from opaque.accounting import Accountant

acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), sample_rate)

for batch in dataloader:
    # ... train ...
    acct = acct | step
    if acct.budget_exceeded:
        print("Privacy budget exhausted")
        break

eps = acct.epsilon_at(delta=1e-5)
```

`Accountant` is functional: `acct | step` returns a new `Accountant` without
mutating the original. The `budget_exceeded` property checks whether the
accumulated process exceeds the budget.

### DP-FTRL: whole-process accounting

DP-FTRL accountants are whole-process mechanisms: `dpftrl_acc.poisson`,
`b_min_sep`, and `balls_in_bins` already cover all declared `n_steps`. Include
the process once rather than once per training step:

```python
import opaque.accounting as acc
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.accounting import Accountant
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=64)
process = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    sample_rate=0.01,
    n_steps=15_624,
)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
acct |= process
eps = acct.epsilon_at(delta=1e-5)
```

The resulting epsilon is the declared full-horizon guarantee. These mechanisms
do not support prefix privacy queries or privacy-based early stopping.

### Serialization

```python
from opaque.accounting import Accountant
from opaque.serialization import from_state_dict, state_dict

flat = state_dict(acct)
# ... torch.save(flat, path) / flat = torch.load(path) ...
```

The flat mapping may include `torch.Tensor` and NumPy arrays. Persist it with
`torch.save` / `torch.load` (or another pickle-compatible format), not JSON,
unless every leaf is JSON-serializable.

For a bare `DpProcess`, use `from_state_dict(identity(), flat)` (or any
concrete process as the template).

## Discretization

PLD computation uses a discretized grid. The default parameters are suitable
for most use cases. For tighter bounds at the cost of computation, adjust the
discretization. Each call updates only the named parameters; every other
setting keeps its current value:

```python
from opaque.accounting import set_discretization

# Tighter (slower)
set_discretization(discretization=1e-5, max_grid_size=50_000_000)

# Faster (looser)
set_discretization(discretization=1e-3)
```

Parameters can also be overridden per query:

```python
eps = training.epsilon_at(delta=1e-5, discretization=1e-5)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `discretization` | 1e-4 | Grid spacing. Smaller = tighter, slower. |
| `log_x_mass_truncation_bound` | -50.0 | Log tail mass cutoff. |
| `max_grid_size` | 10,000,000 | Maximum grid bins before coarsening. |
| `seed` | 42 | Monte Carlo RNG seed. |
| `mc_resolution` | 1e-5 | Maximum unresolved MC mass, in delta units. |
| `mc_failure_probability` | 1e-6 | Failure probability of the simultaneous MC bound. |

The default `mc_resolution=1e-5` and `mc_failure_probability=1e-6` require
2,940,252 samples per adjacency direction. Required counts above 50 million
produce an advisory warning but are not capped. An `epsilon_at(delta)` query
uses `min(mc_resolution, delta / 2)`, so smaller requested deltas automatically
increase the sample count.

B-min-sep and correlated Balls-in-Bins reuse Monte Carlo draws across noise
probes when their transcript corpora fit the cache. Each cache defaults to
4 GiB; configure them with `OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES` and
`OPAQUE_BNB_TRANSCRIPT_CACHE_MAX_BYTES`, respectively. Oversized entries emit
a `RuntimeWarning` with the required bytes and cap, then use one-shot accounting.
A cap of `0` disables reuse without a warning. Cache limits do not change
accounting precision or bound total process memory.

Discretization is always conservative: exact privacy-loss atoms, PMF
coarsening, and histogram buckets are rounded upward to the grid. There is no
option to request an optimistic or lower-bound accounting result.

!!! note "Monte Carlo confidence bounds"
    b-min-sep, and Balls-in-Bins with a **correlated** strategy (λCGD, BISR,
    BSR, BLT), use simultaneous one-sided order-statistic bounds for both
    adjacency directions. The sample count is derived from `mc_resolution` and
    `mc_failure_probability`. Remaining uncertainty is placed at `+∞`, and the returned PLD
    exposes `mc_confidence` and `mc_resolution`. Statistical confidence is
    separate from DP delta. Balls-in-Bins with `identity_strategy()`, and
    `random_allocation` and `k_out_of_t`, use deterministic transforms. Monte
    Carlo PLDs bound
    ε/δ and advantage; β and Bayes risk fail closed to zero.

## API reference

See [Accounting API Reference](../reference/accounting.md) for complete function
signatures and return types.
