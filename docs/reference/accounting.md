# opaque.accounting

Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees.
Mechanism constructors return `DpProcess` objects that compose with `*` (repeat)
and `|` (heterogeneous compose). Privacy metrics are queried directly on the
resulting process.

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)
training = step * 1000
epsilon = training.epsilon_at(1e-5)
```

The underlying implementation uses Google's PLD accounting via the
`opaque-accounting` Rust crate (PyO3 bindings).

Published `opaque-accounting` artifacts are an sdist, `manylinux_2_28`
Linux wheels for `x86_64` and `aarch64`, and a macOS 11+ `arm64` wheel.
Windows, macOS `x86_64`, and `musllinux` wheels are intentionally not
published at the moment.

**See also**: [Privacy Accounting User Guide](../user-guide/accounting.md)

---

## Namespace organization

The accounting API is split into three namespaces:

| Namespace | Contents | Import |
|-----------|----------|--------|
| `opaque.accounting` | Cross-cutting: calibration, composition, `Accountant`, `repeat`, `compose` | `import opaque.accounting as acc` |
| `opaque.dpsgd.accounting` | DP-SGD mechanisms: `gaussian`, `adaclip`, `moe_aux`, `poisson` (plain or truncated via `truncated_batch_size` / `dataset_size`), `parallel_poisson`, `k_out_of_t` | `from opaque.dpsgd import accounting as dpsgd_acc` |
| `opaque.dpftrl.accounting` | DP-FTRL mechanisms: `mf_gaussian`, `poisson` (whole-process, parameterized by `n_steps`), `b_min_sep`, `balls_in_bins` | `from opaque.dpftrl import accounting as dpftrl_acc` |

The mechanism factories (`gaussian`, `poisson`, `mf_gaussian`, …) live **only** on
the algorithm-scoped namespaces — use the namespace that matches your training
run. Cross-cutting primitives (`Accountant`, `calibrate`, `epsilon_budget`,
composition operators) live on `opaque.accounting`.

```python
# DP-SGD
from opaque.dpsgd import accounting as dpsgd_acc
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=0.01)

# DP-FTRL
from opaque.dpftrl import accounting as dpftrl_acc
from opaque.dpftrl.noise import band_mf_strategy
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)

# Cross-cutting composition and calibration always go via opaque.accounting
import opaque.accounting as acc
total = step * 1000
eps = total.epsilon_at(1e-5)
```

`opaque.dpsgd.accounting` and `opaque.dpftrl.accounting` are lazily imported:
the Rust PLD extension is not loaded until you access these submodules.

---

## Classes

### Custom budget checkpointing

`Accountant` checkpoints include an optional budget. The built-in budget
factories are registered automatically. If an application supplies its own
`Budget` protocol implementation, register a self-contained codec before
checkpointing:

```python
import opaque.accounting as acc

acc.register_budget_serializer(
    MyBudget,
    lambda budget: {"target": budget.target},
    lambda state: MyBudget(state["target"]),
)
```

The serializer must return JSON-compatible state and the deserializer must
rebuild the budget from that state. Checkpointing an unregistered budget or
restoring an unknown budget checkpoint type raises `CheckpointError`.

### `DpProcess`

Abstract base class for all privacy processes. Subclasses implement `pld()` to
compute the Privacy Loss Distribution on demand. Results are automatically
cached via `@lru_cache` (maxsize=8). Use `cached()` for larger cache
size (16) or as an opaque merge barrier.

**Privacy metrics:**

| Method               | Returns                                         |
|----------------------|-------------------------------------------------|
| `epsilon_at(delta)`  | Smallest epsilon achieving (epsilon, delta)-DP   |
| `delta_at(epsilon)`  | Smallest delta achieving (epsilon, delta)-DP     |
| `advantage()`        | Total-variation advantage (f-DP)                 |
| `beta_at(alpha)`     | Type-II error at given Type-I error alpha        |
| `risk_at(prior)`     | Bayes risk under optimal adversary               |

Monte Carlo-backed `Pld` objects expose `mc_failure_probability`,
`mc_confidence`, and `mc_resolution`; analytic PLDs report zero failure and
zero MC resolution. The confidence construction bounds hockey-stick metrics
(`epsilon_at`, `delta_at`, and `advantage`). `beta_at` and `risk_at` fail
closed to zero for Monte Carlo PLDs because separate directional CDF bounds do
not by themselves establish a hypothesis-testing trade-off curve.

**Composition operators:**

| Operator     | Description                                  | Equivalent             |
|--------------|----------------------------------------------|------------------------|
| `proc * k`   | Homogeneous k-fold composition (repeat)      | `acc.repeat(proc, k)`  |
| `k * proc`   | Same (reflected multiply)                    | `acc.repeat(proc, k)`  |
| `a \| b`     | Heterogeneous composition                    | `acc.compose(a, b)`    |

Composition is optimized at construction time: identical steps are collapsed
via structural equality, nested repeats are flattened, and identity processes
are elided. Composing the same step in a loop produces a single `Repeated` node
with one `self_compose` call (2 FFTs), not `n` heterogeneous composes.

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)

# Homogeneous composition
training = step * 1000

# Heterogeneous composition (multi-phase)
phase1 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01) * 500
phase2 = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.3), 0.01) * 500
total = phase1 | phase2

eps = total.epsilon_at(1e-5)
```

### `DiscretizationConfig`

Controls PLD discretization precision. Configuration is applied at **query time**
when computing privacy metrics via `pld()`, not stored in process structure.

| Parameter                      | Default      | Description                                      |
|--------------------------------|--------------|--------------------------------------------------|
| `discretization`               | `1e-4`       | Grid spacing for PLD PMF. Error scales as O(d^2) |
| `log_x_mass_truncation_bound`  | `-50`        | Tails below exp(bound) are truncated             |
| `max_grid_size`                | `10_000_000` | Coarsen grid if it exceeds this many bins        |
| `tail_mass_truncation`         | `1e-15`      | Tail-mass budget during composition              |
| `seed`                         | `42`         | RNG seed for Monte Carlo PLD builders            |
| `max_conv_grid`                | `32_768`     | Convolution grid cap for random-allocation PLD   |
| `mc_resolution`                | `1e-5`       | Maximum unresolved MC mass, in delta units       |
| `mc_failure_probability`       | `1e-6`       | Failure probability of the simultaneous MC bound |

The Monte Carlo count is derived from binary-KL Chernoff order-statistic bounds
with a Bonferroni allocation over all ranks and both adjacency directions. At
the default `mc_resolution=1e-5` and `mc_failure_probability=1e-6`, this
requires 2,940,252 samples per direction. Counts above 50 million emit an
advisory runtime warning but are not capped. The construction follows
[Hoeffding (1963)](https://doi.org/10.1080/00401706.1963.10490085); the rank
and adjacency allocation makes the entire returned PLD simultaneous.
`epsilon_at(delta)` tightens the effective resolution to
`min(mc_resolution, delta / 2)`, reserving at least half of the requested delta
for finite privacy-loss mass.

Discretization is unconditionally conservative: exact atoms, PMF coarsening,
and histogram buckets are rounded upward. The API has no optimistic or
lower-bound mode.

**Query-time configuration (recommended):**

```python
# Create processes without config
proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)

# Apply config at query time
eps_coarse = proc.epsilon_at(1e-5, discretization=1e-3)  # faster, less accurate
eps_fine = proc.epsilon_at(1e-5, discretization=1e-5)    # slower, more accurate

# Monte Carlo accountants derive enough samples for the requested resolution.
pld = proc.pld(mc_resolution=1e-5, mc_failure_probability=1e-6, seed=123)
eps_mc = pld.epsilon_at(2e-5)
print(pld.mc_confidence, pld.mc_resolution)
```

All `pld()`-family methods (`epsilon_at`, `delta_at`, `advantage`, `beta_at`,
`risk_at`) accept every parameter in the table above except
`tail_mass_truncation` (composition-internal, module-level only). Overrides
are broadcast to every node of a composed process, so one `seed` is shared
by all Monte Carlo nodes — the same semantics as `set_discretization`.

**Module-level discretization defaults:**

Set default config for all queries when not overridden:

```python
acc.set_discretization(discretization=1e-4)  # Apply to all queries
proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
eps = proc.epsilon_at(1e-5)  # Uses 1e-4 default
```

- `acc.set_discretization(discretization=1e-4, ...)` — Update the global
  default for only the named parameters; every omitted parameter keeps its
  current value. A bare call is a no-op.
- `acc.get_discretization()` — Return the current `DiscretizationConfig`

---

## Mechanism functions

All mechanism constructors return a `DpProcess`. Discretization is configured
at query time via `epsilon_at(..., discretization=...)` or module-level via
`set_discretization()`.

### `gaussian(noise_multiplier) -> DpProcess`

Gaussian mechanism with noise multiplier sigma. Adds noise N(0, sigma^2) to
sensitivity-1 queries. Base mechanism for DP-SGD.

- `noise_multiplier` (float): Ratio of noise std to sensitivity. Larger = more private.

### `poisson(inner, sample_rate) -> DpProcess`

Poisson-subsampled mechanism (standard DP-SGD step). `sample_rate` is
`batch_size / dataset_size`. Plain Poisson accepts any Opaque `DpProcess` as
its base mechanism.

- `inner` (DpProcess): Base mechanism
- `sample_rate` (float): Probability of including each example, in (0, 1)

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), sample_rate=256 / 50_000)
```

### `poisson(inner, sample_rate, *, truncated_batch_size=None, dataset_size=None) -> DpProcess` (truncated form)

`poisson()` switches to a truncated Poisson PLD when both
`truncated_batch_size` and `dataset_size` are set together (must be both
or neither). This is the truncated-Poisson PLD for capped batches; it does
**not** improve privacy versus plain Poisson at the same rate—use it when
training actually truncates draws. The capped form accepts a Gaussian,
AdaClip(Gaussian), or `nonprivate()` base.

The guarantee is [Ganesh (2025), Theorem
3.1](https://arxiv.org/abs/2508.15089). It assumes each over-cap provisional
sample is replaced by a uniformly random subset of exactly the cap size.
Its directed add/remove adjacency is indexed by the public size \(n\) of the
larger neighboring dataset. Therefore `dataset_size` must equal the sampler's
pre-sampling size, and composition must preserve the same \(n\)-indexed
adjacency interpretation.

- `inner` (Gaussian | AdaClip | NonPrivate): Base mechanism
- `sample_rate` (float): Expected sampling rate, in (0, 1)
- `truncated_batch_size` (int | None): Optional max batch-size cap
- `dataset_size` (int | None): Public larger-neighbor size \(n\), required when
  `truncated_batch_size` is set

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

### `parallel_poisson(inner, sample_rate, num_workers) -> DpProcess`

Parallel Poisson subsampling. Models independent Poisson sampling on
multiple workers, where the same example can appear on multiple devices.
Like `poisson()` (plain or truncated), this is a full wrapper.

- `inner` (Gaussian | AdaClip): Base mechanism (from `gaussian()` or `adaclip()`)
- `sample_rate` (float): Probability of including each example, in (0, 1]
- `num_workers` (int): Number of parallel workers sampling independently

```python
step = dpsgd_acc.parallel_poisson(
    dpsgd_acc.gaussian(0.5), sample_rate=0.01, num_workers=4,
)
```

### `k_out_of_t(inner, *, k, t, allocation) -> DpHorizonProcess`

With `allocation="block"`, every record participates in exactly one batch in
each of `k` contiguous, nearly equal blocks. Block sizes differ by at most one.

This mode pairs with
`opaque.dpsgd.sampling.KOutOfTSampler(..., allocation="block")` and
is computed for the declared horizon by the exact PLD transform of Feldman &
Shenfeld (2026), with no Monte Carlo sampling.

- `inner` (Gaussian | AdaClip | NonPrivate): Base mechanism
- `k` (int): Number of blocks / participations per record
- `t` (int): Total optimizer-step horizon
- `allocation` (`"block"` or `"total"`): Required allocation mode

```python
process = dpsgd_acc.k_out_of_t(
    dpsgd_acc.gaussian(0.5),
    k=num_epochs,
    t=num_epochs * steps_per_epoch,
    allocation="block",
)
eps = process.epsilon_at(1e-5)
```

With `allocation="total"`, every record chooses a uniform `k`-subset of the
complete horizon. The accountant currently uses the block reduction as a
conservative upper bound.

### `adaclip(inner, *, fraction_noise_std, expected_batch_size, num_groups=1) -> DpProcess`

Accounts for the extra privacy cost of adaptive clipping's noisy
fraction query. Returns an `AdaClip` process composable with `poisson()`
(plain or truncated).

- `inner` (Gaussian): Base mechanism (from `gaussian()`)
- `fraction_noise_std` (float): Noise std on the clipping fraction. Default: 0.05.
- `expected_batch_size` (float): Expected batch size (`sample_rate × dataset_size`), used to compute the absolute noise std for the quantile query.
- `num_groups` (int): Number of independently adaptive clipping groups, and
  therefore independent noisy quantile queries. Default: 1. Set this to the
  number of groups when using per-group adaptive clipping.

```python
step = dpsgd_acc.poisson(dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.5), fraction_noise_std=0.05, expected_batch_size=256), 0.01)
```

### `moe_aux(inner, *, ratio=0.02) -> DpProcess`

Accounts for the MoE router-load release of
[`moe_clipped_grad`](clipping.md): the step is a Gaussian at the joint
multiplier `inner.noise_multiplier / sqrt(1 + ratio)`. Returns a `MoeAux`
process composable with `poisson()`, `parallel_poisson()` and
`k_out_of_t()`. The same `ratio` is given to the clipper.

- `inner` (Gaussian): Base mechanism (from `gaussian()`).
- `ratio` (float): Share of the whitened sensitivity given to the load
  release. Default: 0.02.

```python
step = dpsgd_acc.poisson(dpsgd_acc.moe_aux(dpsgd_acc.gaussian(0.5), ratio=0.02), 0.01)
```

See [MoE load balancing](../mechanisms/dp-sgd/moe-load-balancing.md).

### Private second-moment release

When the noise mechanism produces both gradients **and** squared gradients
(via `clipped_grad(..., second_moment=True)`), the joint paired release uses
sensitivity-proportional Mahalanobis allocation in the runtime σ split (see
the [paired second-moment release](noise.md#paired-second-moment-release)
section). The Mahalanobis
budget collapses to a single sensitivity-1 Gaussian release at the same
noise multiplier, so **privacy accounting is exactly the underlying
first-moment mechanism**:

```python
import opaque.accounting as acc

# DP-SGD: same chain as first-moment-only
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate=batch_size / dataset_size)
training = step * num_steps

# DP-FTRL with BandMF: same chain as first-moment-only
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=batch_size / dataset_size,
    n_steps=num_steps,
)
```

There is no separate `second_moment` transformation to wrap and no `ρ` knob:
the runtime σ on each stream already absorbs the joint cost. Use the
underlying mechanism factories (`dpsgd_acc.gaussian`, `dpftrl_acc.mf_gaussian`,
…) directly.

### `eps_delta(epsilon, delta=0.0) -> DpProcess`

Fixed (epsilon, delta)-DP guarantee. Useful for composing an external mechanism
with known privacy parameters into tracked processes.

- `epsilon` (float): Privacy parameter (>= 0)
- `delta` (float): Failure probability (default 0.0)

```python
external = acc.eps_delta(3.0, 1e-5)
total = external | (dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01) * 1000)
```

### `identity() -> DpProcess`

Identity mechanism (zero privacy loss). Acts as the identity element in
composition: `identity() | a` returns `a`.

### `nonprivate() -> DpProcess`

Non-private mechanism (ε=∞, δ=0). Useful as a baseline or for training without
a privacy guarantee. All Poisson-family amplifications handle `nonprivate()`
inner transparently (zero privacy cost).

```python
step = dpsgd_acc.poisson(acc.nonprivate(), sample_rate=0.01)
training = step * 1000
# training.epsilon_at(1e-5) == inf
```

---

## Matrix factorization mechanisms

MF mechanisms resolve sensitivity and Gram matrices from a noise strategy and
its participation context. Use the same recipe and context for accounting and
noise generation.

### `mf_gaussian(noise_multiplier, strategy, *, n_steps=None, min_sep=1, max_participations=None) -> DpProcess`

Omitting `n_steps` creates a recipe for an amplifier. Calling `pld()` or
`epsilon_at()` on that recipe directly raises `ConfigurationError`.

For bare accounting, `n_steps` is the full release horizon, `min_sep` is the
minimum separation between contributions, and `max_participations` bounds each
protected unit's contributions. If omitted, `max_participations` defaults to
`n_steps`. The resulting Gaussian PLD accounts for the declared context; do not
compose it again by the number of rounds.

```python
from opaque.dpftrl.noise import band_mf_strategy

# One contribution per protected unit across 1000 rounds.
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.mf_gaussian(
    1.0, strategy, n_steps=1000, min_sep=1, max_participations=1,
)
eps = proc.epsilon_at(1e-5)
```

Bare BandMF and Identity sensitivity currently accounts for one participation.
Do not use bare calls with these strategies to budget repeated participation.
For amplified accounting, leave the inner horizon unspecified and supply the
full horizon to the sampler-matching amplifier, which resolves its own context.

### Correlated MF mechanisms (BLT, λCGD, BISR, BSR)

For bare BLT accounting, supply the same horizon and participation bounds as
`mf_gaussian_noise`. This example assumes at most five contributions per
protected unit, at least 1000 rounds apart:

```python
from opaque.dpftrl.noise import blt_strategy
strategy = blt_strategy(max_buffers=10)

# Unamplified accounting for the full 5000-round context.
proc = dpftrl_acc.mf_gaussian(
    1.0, strategy, n_steps=5000, min_sep=1000, max_participations=5,
)

# With a fixed partition into 1000 bins, repeated for five epochs.
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=1000, n_steps=5000,
)
```

The same amplified construction works for `lambda_cgd_strategy`,
`bisr_strategy`, and `bsr_strategy`:

```python
from opaque.dpftrl.noise import lambda_cgd_strategy
strategy = lambda_cgd_strategy(lambda_=0.9)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch, n_steps=steps_per_epoch * num_epochs,
)
```

### `poisson(inner, sample_rate, *, n_steps) -> DpProcess`

Poisson amplification for DP-FTRL. Whole-process accountant covering all
`n_steps` training rounds (do **not** compose with `* num_steps`
externally). Cyclic when the inner is `BandMf` with `bands > 1` (decomposes
into `ceil(n_steps / bands)` independent groups); plain Poisson per round
when the inner is `IdentityMf` or `BandMf` with `bands == 1`.

- `inner` (BandMf | IdentityMf): MF mechanism.
- `sample_rate` (float): Poisson sampling probability per round.
- `n_steps` (int, keyword-only): Total number of training rounds.

```python
strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
```

### `b_min_sep(inner, *, n_steps, p0) -> DpProcess`

Warm-start **b-min-sep** amplification for BandMF (Dong & Ganesh, arXiv:2602.09338).
Uses Monte Carlo PLD accounting. `inner` must be
`mf_gaussian(nm, BandMfStrategy(...))` — strategy coefficients and band width
are read from `inner.strategy`. `p0` is the per-example participation rate per
iteration `E[|B|]/|D|` (match the training sampler’s target batch size).
Control the confidence construction per query with `mc_resolution`,
`mc_failure_probability`, and `seed`, or module-wide via `set_discretization`.
The sample count is derived automatically and exposed as
`get_discretization().resolved_num_mc_samples`.

!!! note "Monte Carlo confidence"
    b-min-sep, and Balls-in-Bins with a **correlated** strategy, build their
    PLDs from simultaneous one-sided order-statistic bounds over both adjacency
    directions. Unresolved probability is placed at `+∞`, so
    `epsilon_at(delta)` returns infinity when `delta` is at or below the
    reported `pld.infinity_mass`. `pld.mc_failure_probability` is statistical
    confidence metadata and is separate from mechanism delta.

### `balls_in_bins(inner, *, num_bins, n_steps) -> DpProcess`

Balls-in-Bins (random-partition) amplification. Returns the **total** privacy
cost across all epochs — do NOT compose further with `* num_epochs`.

With `identity_strategy()` (uncorrelated noise) the Gram is exactly
`num_epochs · I`, so the dominating pair collapses onto 1-out-of-`num_bins`
random allocation at `σ / √num_epochs` and the PLD is computed by the exact
transform — deterministic, no sampling. For correlated-noise strategies
(DP-λCGD, BISR, BSR, BLT) it uses Monte Carlo sampling of the dominating pair.

- `inner` (MfGaussian): `dpftrl_acc.mf_gaussian(nm, strategy)`.
- `num_bins` (int): Number of bins per epoch (typically `dataset_size / batch_size`).
- `n_steps` (int): Total training rounds; must be a multiple of `num_bins`.

```python
# With DP-λCGD
strategy = lambda_cgd_strategy(lambda_=0.9)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    num_bins=steps_per_epoch, n_steps=steps_per_epoch * num_epochs,
)

# With uncorrelated MF noise (analytic identity-strategy path)
proc = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(1.1, identity_strategy()),
    num_bins=100,
    n_steps=1000,
)
```

---

## Composition Functions

Functional equivalents of the `*` and `|` operators. Most users should prefer
the operator syntax.

### `repeat(process, count) -> DpProcess`

Homogeneous k-fold composition. Equivalent to `process * count`.

### `compose(left, right) -> DpProcess`

Heterogeneous two-process composition. Equivalent to `left | right`.

### `cached(process) -> DpProcess`

Increases the LRU cache size from 8 to 16 entries and acts as an opaque merge
barrier: the composition optimizer will not look through a cached node.

**Note**: All `pld()` methods are automatically cached with `maxsize=8` via
`@lru_cache`. Use `cached()` when you need:
- A larger cache (16 entries instead of 8)
- An explicit merge barrier to prevent composition optimizations

```python
# All queries automatically cached (maxsize=8)
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)
eps = step.epsilon_at(1e-5)   # Cached automatically
adv = step.advantage()         # Cache hit

# Use cached() for merge barrier or larger cache
training = acc.cached(step * 1000)
eps = training.epsilon_at(1e-5)   # Cached with maxsize=16
```

---

## Serialization

Processes checkpoint as a **flat** `dict[str, Any]` (string keys with dotted
prefixes for nested composition). Use `opaque.serialization.state_dict`
and `opaque.serialization.from_state_dict` — pass any concrete `DpProcess`
instance as the template (for example, `identity()`); the registered handler
rebuilds from the dict's root `type` field.

```python
import opaque.accounting as acc
from opaque.serialization import from_state_dict, state_dict
import opaque.dpsgd.accounting as dpsgd_acc

step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)
flat = state_dict(step)
step2 = from_state_dict(acc.identity(), flat)
```

---

## Accountant

The `Accountant` class tracks accumulated privacy loss across a training loop.
It provides a functional API: composing a new process returns a fresh
`Accountant` (the original is not modified).

Merge optimization is automatic. Composing the same `step` repeatedly in a loop
produces a single `Repeated` node internally.

```python
from opaque.accounting import Accountant

acct = Accountant()
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)

for i in range(num_steps):
    acct = acct | step

    if i % 100 == 0:
        eps = acct.epsilon_at(1e-5)
        print(f"Step {i}: eps={eps:.2f}")
```

### Budget tracking

Pass an optional `Budget` from the calibration module to enable budget checking:

```python
from opaque.accounting import calibration as cal
from opaque.accounting import Accountant

budget = cal.epsilon_budget(3.0, delta=1e-5)
acct = Accountant(budget=budget)
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.5), 0.01)

for i in range(num_steps):
    acct = acct | step
    if acct.budget_exceeded:
        print("Privacy budget exhausted.")
        break
```

**Methods:** `epsilon_at(delta)`, `delta_at(epsilon)`, `advantage()`,
`beta_at(alpha)`, `risk_at(prior)`, `budget_exceeded` (property).

### Seeding with a prior process

Pass `prefix` to start from an already-executed process instead of the
zero-cost identity. This is sequential composition across runs: use it when a
second training stage (e.g. DPO after SFT) touches the same dataset, so the
new run's budget checks and reported ε include the earlier run's cost. The
prefix composes at the PLD level — much tighter than adding the two stages'
epsilons.

```python
import json

from opaque.accounting import Accountant
from opaque.serialization import from_state_dict

with open("sft_checkpoint/accountant.json") as f:
    sft = from_state_dict(Accountant(), json.load(f))

acct = Accountant(budget=budget, prefix=sft.process)
acct = acct | dpo_step  # composes on top of the SFT prefix
```

If the two stages train on disjoint records (and the privacy unit is the
record), no prefix is needed — parallel composition applies and the overall
guarantee is the pointwise max of the two stages' ε(δ) curves.

### Serialization

```python
from opaque.accounting import Accountant
from opaque.serialization import from_state_dict, state_dict

flat = state_dict(acct)
acct2 = from_state_dict(Accountant(), flat)
```

`process.*` keys hold the composed tree; `budget.*` keys are present when
the accountant was constructed with a budget.

---

## Calibration

Submodule: `opaque.accounting.calibration`

```python
from opaque.accounting import calibration as cal
```

Binary search for finding parameter values that achieve a target privacy budget.

### `calibrate(budget, process, param_min, param_max, tolerance=1e-6, max_iterations=100, prefix=None) -> CalibrateResult`

Binary search for a parameter value such that `process(param)` produces a
`DpProcess` achieving the given privacy budget from the privacy-safe side.

| Parameter        | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `budget`         |         | A `Budget` object from a budget factory (see below)      |
| `process`        |         | Callable: `float -> DpProcess`                           |
| `param_min`      |         | Lower bound for search                                   |
| `param_max`      |         | Upper bound for search                                   |
| `tolerance`      | `1e-6`  | Positive, finite relative convergence tolerance          |
| `max_iterations` | `100`   | Positive maximum number of binary search iterations      |
| `prefix`         | `None`  | Already-executed `DpProcess` composed into every probe   |

The `process` callable takes a single float parameter and returns a `DpProcess`.
Its metric must be monotone in the parameter over `[param_min, param_max]`;
the search direction is derived automatically by probing both endpoints, so
noise-multiplier-like parameters (increase improves privacy) and
sample-rate/step-like parameters (increase spends privacy) are both
supported. Exactly one endpoint must be privacy-safe; which one is detected,
not positional.

```python
import opaque.accounting as acc
from opaque.accounting import calibration as cal

budget = cal.epsilon_budget(3.0, delta=1e-5)
result = cal.calibrate(
    budget,
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=5.0,
)
print(f"noise_multiplier = {result.param:.4f}, epsilon = {result.achieved:.6f}")
```

Every successful result is privacy-safe and relatively close to the target:
decreasing privacy-loss budgets return `achieved <= target`, while increasing
privacy-gain budgets return `achieved >= target`. In both cases,
`math.isclose(achieved, target, rel_tol=tolerance, abs_tol=0.0)` is true.
`tolerance` must be finite and positive, and `max_iterations` must be positive;
invalid values raise `CalibrationError` before the process is evaluated. If no
safe endpoint reaches the requested relative tolerance, `calibrate()` raises
`CalibrationError` instead of returning an under-noised parameter.

When the process uses a Monte Carlo PLD, calibration divides the configured
failure probability across the two endpoint probes and at most
`max_iterations` interior probes. `result.mc_confidence` therefore covers the
adaptive search as a whole rather than only its selected final parameter.
Balls-in-Bins calibration also reuses its seeded, sigma-independent projected
draws across probes when they fit the bounded native cache. Set
`OPAQUE_BNB_TRANSCRIPT_CACHE_MAX_BYTES` to control its memory cap (4 GiB by
default), or to `0` to use the lower-memory one-shot path for every probe.

Calibrating a second stage against the remaining budget (see
[Seeding with a prior process](#seeding-with-a-prior-process)): pass the
earlier run's executed process as `prefix`. Each probe evaluates
`prefix | process(param)`, so the budget is the total across both stages.
The prefix's PLD is computed once for the whole search.

```python
result = cal.calibrate(
    cal.epsilon_budget(8.0, delta=1e-6),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.02) * 2000,
    param_min=0.5,
    param_max=5.0,
    prefix=sft.process,  # from the SFT run's accountant.json
)
```

Multi-phase training:

```python
result = cal.calibrate(
    cal.epsilon_budget(5.0, delta=1e-5),
    lambda nm: (
        dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 500
        | dpsgd_acc.poisson(dpsgd_acc.gaussian(nm * 0.8), 0.01) * 500
        | dpsgd_acc.poisson(dpsgd_acc.gaussian(nm * 0.5), 0.01) * 500
    ),
    param_min=0.2,
    param_max=3.0,
    tolerance=0.01,
)
```

### `CalibrateResult`

Returned by `calibrate()`.

| Attribute   | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `param`     | `float` | Found parameter value                            |
| `achieved`  | `float` | Achieved metric value at `param`                 |
| `target`    | `float` | Target metric value                              |
| `iterations`| `int`   | Number of binary search iterations               |
| `converged` | `bool`  | Always `True` for a successfully returned result |
| `mc_failure_probability` | `float` | Overall failure probability for adaptive MC probes |
| `mc_confidence` | `float` | Confidence covering the complete calibration search |

### Budget Factories

Budget factories create `Budget` objects that define what privacy metric to
optimize and what value to achieve.

| Factory                             | Metric being calibrated                 | Decreasing with noise |
|-------------------------------------|-----------------------------------------|-----------------------|
| `cal.epsilon_budget(eps, delta)`    | epsilon at given delta                  | Yes                   |
| `cal.delta_budget(delta, epsilon)`  | delta at given epsilon                  | Yes                   |
| `cal.advantage_budget(advantage)`   | f-DP total-variation advantage          | Yes                   |
| `cal.beta_budget(beta, alpha)`      | Type-II error at given Type-I error     | No                    |
| `cal.risk_budget(risk, prior)`      | Bayes risk under optimal adversary      | No                    |

"Decreasing with noise" indicates the metric kind: privacy-loss metrics
(epsilon/delta/advantage) are safe at-or-below the target, privacy-gain
metrics (beta/risk) at-or-above. The binary search derives the parameter
direction automatically by probing both bracket endpoints.

```python
# (epsilon, delta)-DP
result = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000,
    0.1, 5.0,
)

# f-DP advantage
result = cal.calibrate(
    cal.advantage_budget(0.1),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)

# (alpha, beta) error rates
result = cal.calibrate(
    cal.beta_budget(0.05, alpha=0.01),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)

# Bayes risk
result = cal.calibrate(
    cal.risk_budget(0.1, prior=0.5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000,
    0.2, 3.0,
)
```
