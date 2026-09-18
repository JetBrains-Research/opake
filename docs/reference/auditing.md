# Auditing API

::: opake.auditing

## Module functions

### coin_flip

```python
auditing.coin_flip(
    dataset, *, num_canaries, key,
    candidate_indices=None,
) -> CoinFlip
```

Create a coin-flip partition. Randomly selects `num_canaries` examples
and flips a fair coin for each to decide inclusion/exclusion. Selection
uses the whole dataset unless `candidate_indices` restricts it to a pool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | any with `len()` | required | Full training dataset |
| `num_canaries` | `int` | required | Number of canaries to designate |
| `key` | `RngKey` | required | RNG key for reproducibility |
| `candidate_indices` | array-like \| `None` | `None` | Unique, in-range integer indices eligible to become canaries; order is ignored |

**Raises** `ConfigurationError` if the candidate pool is malformed or contains fewer
than `num_canaries` indices.

```python
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))
```

To designate a precommitted block of constructed canaries:

```python
pool = range(len(dataset), len(audited_dataset))
cf = auditing.coin_flip(
    audited_dataset,
    num_canaries=len(pool),
    key=key(42),
    candidate_indices=pool,
)
```

See [Stronger canaries](../user-guide/auditing.md#stronger-canaries) for the
statistical and accounting requirements.

---

### loss_scores

```python
auditing.loss_scores(
    loss_fn, *args, *,
    batch_argnums, coin_flip, dataset,
    reference_scores=None,
    batch_size=32, collate_fn=None,
) -> CanaryScores
```

Compute per-example membership scores as negative loss. Higher score =
lower loss = more likely a training member.

Builds an internal loader over the partition's canaries and pairs every
score with the dataset index of the example that produced it. Returns
[`CanaryScores`](#canaryscores) — the form [`one_run`](#one_run) requires.
The pairing is joined by identifier, so no iteration order over the
batches can misalign it. Identifiers are attached per batch *before*
collation, so a `collate_fn` that reorders examples within a batch
misaligns the pairing and cannot be detected; a `collate_fn` that drops or
adds rows raises.

To audit scores computed by some other pipeline, attest their identifiers
with [`canary_scores`](#canary_scores) instead.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_fn` | `Callable` | required | Per-example loss function (vmap-compatible) |
| `*args` | any | — | Non-batched arguments (e.g., model parameters) |
| `batch_argnums` | `tuple[int, ...]` | required | Which `loss_fn` args are batched (same as `clipped_grad`) |
| `coin_flip` | `CoinFlip` | required | The audit partition to score against |
| `dataset` | any | required | The full dataset the partition was created from |
| `reference_scores` | `CanaryScores` | `None` | Baseline scores to subtract (e.g., from untrained model), aligned by identifier |
| `batch_size` | `int` | `32` | Batch size of the internal loader |
| `collate_fn` | `Callable` | default collate | Collates raw canary examples into a batch for `loss_fn`; must emit one row per example, in the order received |

**Returns** [`CanaryScores`](#canaryscores) of shape `(n,)`, one score per
canary. Higher = more likely member.

**Raises** `ConfigurationError` if `batch_argnums` is malformed, `batch_size` is
not positive, or `collate_fn` changes the batch row count; `InputTypeError` if
`reference_scores` does not carry identifiers.

```python
def canary_collate(examples):
    batch = data_collator(examples)
    return (batch["input_ids"].to(device),)

scores = auditing.loss_scores(
    loss_fn, params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
    batch_size=32, collate_fn=canary_collate,
)
estimate = auditing.one_run(scores, coin_flip=cf)
```

---

### gradient_scores

```python
auditing.gradient_scores(
    loss_fn, *args, *,
    batch_argnums, coin_flip, dataset,
    reference_scores=None,
    batch_size=32, collate_fn=None,
) -> CanaryScores
```

Compute per-example membership scores as negative squared gradient norm.
Higher score = smaller gradient norm = more likely a training member.

This is a white-box attack that differentiates with respect to the first
`loss_fn` argument (model parameters). Therefore `0` must not appear in
`batch_argnums`.

Scores the partition's canaries over an internal identifier-carrying
loader, with the same pairing guarantees and `collate_fn` obligation as
[`loss_scores`](#loss_scores).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_fn` | `Callable` | required | Per-example scalar loss function |
| `*args` | any | — | Non-batched arguments; first arg is differentiated |
| `batch_argnums` | `tuple[int, ...]` | required | Batched argument indices; must exclude 0 |
| `coin_flip` | `CoinFlip` | required | The audit partition to score against |
| `dataset` | any | required | The full dataset the partition was created from |
| `reference_scores` | `CanaryScores` | `None` | Baseline scores to subtract, aligned by identifier |
| `batch_size` | `int` | `32` | Batch size of the internal loader |
| `collate_fn` | `Callable` | default collate | Must emit one row per example, in the order received |

**Returns** [`CanaryScores`](#canaryscores) of shape `(n,)`, one score per
canary. Higher = more likely member.

**Raises** `ConfigurationError` if `0 in batch_argnums`, `batch_size` is not
positive, or `collate_fn` changes the batch row count; `InputTypeError` if
`reference_scores` does not carry identifiers.

```python
ref = auditing.gradient_scores(
    loss_fn, initial_params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
)
scores = auditing.gradient_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
    reference_scores=ref,
)
```

The scorer is also available from the attacks namespace:

```python
from opake.auditing.attacks import gradient_scores
```

---

### canary_scores

```python
auditing.canary_scores(
    scores, *, canary_indices,
) -> CanaryScores
```

Attest which canary each externally computed score belongs to. The
built-in scorers already return `CanaryScores` in verified mode; use this
for scores produced by another pipeline. Identifiers may be in any order
— [`one_run`](#one_run) joins on them rather than assuming a position.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scores` | array-like | required | Membership scores, shape `(n,)`, float |
| `canary_indices` | array-like | required | Dataset index behind each score, same order as `scores` |

**Raises** `ConfigurationError` if either array is not 1-D, the identifiers are
not integers, the lengths disagree, or an identifier repeats.

```python
scores = auditing.canary_scores(values, canary_indices=ids)
estimate = auditing.one_run(scores, coin_flip=cf)
```

---

### one_run

```python
auditing.one_run(scores, *, coin_flip) -> OneRunEstimate
```

Build a one-run privacy estimate from canary scores. Joins scores to the
coin-flip partition by canary identifier, precomputes the empirical ROC,
and returns a frozen estimate. The join makes scoring order irrelevant;
identifiers that do not cover the partition's canaries one-to-one raise
instead of silently producing a meaningless estimate.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scores` | [`CanaryScores`](#canaryscores) | required | Per-canary membership scores with identifiers, from verified scoring or explicit attestation |
| `coin_flip` | `CoinFlip` | required | The coin-flip partition |

**Raises** `InputTypeError` for bare arrays without identifiers; `ConfigurationError`
if identifiers are unexpected, duplicated, or missing.

```python
estimate = auditing.one_run(scores, coin_flip=cf)
print(estimate.epsilon_at(delta=1e-5))
```

---

## CoinFlip

Coin-flip partitioning for canary-based privacy auditing. Each canary
is independently included or excluded with probability 0.5.

Create instances with [`coin_flip`](#coin_flip). To control which records are
eligible, pass `candidate_indices` to that factory rather than constructing the
state carrier directly.

| Attribute | Type | Description |
|---|---|---|
| `num_canaries` | `int` | Total canary count |
| `canary_indices` | `np.ndarray` | All canary dataset indices |
| `in_indices` | `np.ndarray` | Included in training (heads) |
| `out_indices` | `np.ndarray` | Excluded from training (tails) |

### train_indices

```python
cf.train_indices(dataset_size) -> list[int]
```

All indices except held-out canaries. Returns `list[int]` for HuggingFace
`dataset.select()`.

### split_scores

```python
cf.split_scores(scores) -> tuple[np.ndarray, np.ndarray]
```

Split per-canary scores into `(in_scores, out_scores)`, joining
[`CanaryScores`](#canaryscores) to the partition by identifier. Bare
arrays raise `InputTypeError`; identifiers that do not join one-to-one onto
`canary_indices` raise `ConfigurationError`.

---

## CanaryScores

```python
class CanaryScores(scores, canary_indices)  # frozen dataclass
```

Membership scores paired with stable canary identifiers: `scores[k]` was
computed for dataset example `canary_indices[k]`. Produced by the scoring
functions in verified mode; build one with
[`canary_scores`](#canary_scores) to attest identifiers for scores
computed elsewhere (any order — [`one_run`](#one_run) joins by
identifier). Arrays are defensively copied, validated (1-D, equal length,
unique integer identifiers), and marked read-only.

| Attribute | Type | Description |
|---|---|---|
| `scores` | `np.ndarray` | Membership scores, shape `(n,)`, float, read-only |
| `canary_indices` | `np.ndarray` | Dataset index of the canary behind each score |

Supports `len()` and `np.asarray()` (yields the score values).

---

## OneRunEstimate

```python
class OneRunEstimate  # frozen dataclass
```

Precomputed one-run audit estimate, returned by `auditing.one_run()`.
Holds the empirical ROC counts and exposes:

- a **default audit-method surface** (`epsilon_at`, `delta_at`, `beta_at`,
  `advantage`) that dispatches to μ-GDP — the paper-recommended default
  for Gaussian-DP mechanisms;
- two **method factories** (`eps_delta`, `gdp`) for explicit override;
- **attack-side empirical metrics** (`attack_auc`, `attack_beta_at`).

| Attribute | Type | Description |
|---|---|---|
| `n_in` | `int` | Number of held-in canaries |
| `n_out` | `int` | Number of held-out canaries |
| `canary_indices` | `np.ndarray \| None` | Stable example identifiers: dataset indices of the audited canaries, in partition order (always populated by `one_run`) |

### epsilon_at

```python
estimate.epsilon_at(*, delta, significance=0.05, threshold=None) -> float
```

Epsilon lower bound from the default μ-GDP audit method. Equivalent to
`self.gdp().epsilon_at(...)`. Requires `delta > 0`. For non-Gaussian-DP
mechanisms use `self.eps_delta().epsilon_at(...)` explicitly.

### delta_at

```python
estimate.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

δ(ε) from the default μ-GDP audit method. Equivalent to
`self.gdp().delta_at(...)`.

### beta_at

```python
estimate.beta_at(*, alpha, significance=0.05, threshold=None) -> float
```

Theoretical f-DP β at α under the inferred μ̂-GDP. Equivalent to
`self.gdp().beta_at(...)`. For the empirical attack ROC β, see
[`attack_beta_at`](#attack_beta_at).

### advantage

```python
estimate.advantage(*, significance=0.05, threshold=None) -> float
```

Total-variation advantage at the inferred μ̂-GDP. Equivalent to
`self.gdp().advantage()`.

### eps_delta

```python
estimate.eps_delta() -> EpsDeltaMethod
```

Mechanism-agnostic (ε, δ)-DP order-statistics audit method
(Xiang et al. 2025). Use for non-Gaussian-DP mechanisms or pure ε-DP
auditing. See [EpsDeltaMethod](#epsdeltamethod) below.

### gdp

```python
estimate.gdp(*, grid_size=10_000) -> GdpMethod
```

μ-GDP order-statistics audit method (Xiang et al. 2025) with a tunable
integration grid. `grid_size` must be at least `1_000`: coarser grids can
resolve a detectable leak as `epsilon=0`. The top-level `epsilon_at` /
`delta_at` / `beta_at` / `advantage` are shortcuts for `gdp()` with the
default `grid_size`. See [GdpMethod](#gdpmethod) below.

### attack_auc

```python
estimate.attack_auc(*, confidence=None, num_samples=1000, key=None) -> float | tuple[float, float]
```

Empirical ROC AUC of the membership inference attack. Returns point
estimate by default, or `(lower, upper)` CI tuple when `confidence` is
provided. A CI requires an explicit `key` so bootstrap resampling is
reproducible and isolated from global NumPy state. Independent of the
audit method.

### attack_beta_at

```python
estimate.attack_beta_at(*, alpha) -> float | np.ndarray
```

Empirical attack β at given FPR: `1 − TPR` interpolated from the
empirical ROC. Distinct from
[`beta_at`](#beta_at) which is the theoretical f-DP β at the
inferred μ̂-GDP.

---

## EpsDeltaMethod

Returned by `estimate.eps_delta()`. Exposes the two query directions
along the audit's (ε, δ) boundary.

### epsilon_at

```python
method.epsilon_at(*, delta=0.0, significance=0.05, threshold=None) -> float
```

Largest ε the audit can certify at the given δ. `delta=0` is allowed
(pure ε-DP). When `threshold` is provided, accuracy is evaluated at that
specific score threshold; otherwise the Pareto-optimal threshold
maximising TP + TN is used.

### delta_at

```python
method.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

Largest δ at which the audit certifies ε ≥ `epsilon`. Returns `0.0` when
`epsilon` is unreachable even at δ=0.

`EpsDeltaMethod` deliberately does not expose `beta_at` / `advantage`:
the (ε, δ)-DP trade-off function is a family envelope, so those metrics
would be worst-case across the family rather than instance-specific —
use `GdpMethod` for sharp f-DP queries on Gaussian-DP mechanisms.

---

## GdpMethod

Returned by `estimate.gdp(grid_size=)`. Mirrors the full
[`Pld`](accounting.md) metric surface: all four methods derive from a
single inferred μ̂.

| Attribute | Type | Default | Description |
|---|---|---|---|
| `grid_size` | `int` | `10_000` | Grid points for the order-statistics integration; minimum `1_000` |

### epsilon_at

```python
method.epsilon_at(*, delta, significance=0.05, threshold=None) -> float
```

Largest ε the audit can certify at the given δ. Requires `delta > 0`
(μ-GDP is incompatible with pure ε-DP).

### delta_at

```python
method.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

δ(ε) under the inferred μ̂-GDP guarantee, via the closed-form GDP relation.

### beta_at

```python
method.beta_at(*, alpha, significance=0.05, threshold=None) -> float
```

f-DP Type-II error at α under the inferred μ̂-GDP:
β(α; μ) = Φ(Φ⁻¹(1 − α) − μ̂). Theoretical β of the post-audit guarantee,
distinct from `OneRunEstimate.attack_beta_at` (empirical attack ROC).

### advantage

```python
method.advantage(*, significance=0.05, threshold=None) -> float
```

Total-variation advantage at the inferred μ̂-GDP: TV(μ) = 2·Φ(μ̂/2) − 1.

---

## Quick reference

| | |
|---|---|
| `auditing.coin_flip(dataset, ...)` | Coin-flip partition → `CoinFlip` |
| `auditing.coin_flip(dataset, ..., candidate_indices=pool)` | Partition a precommitted canary pool |
| `auditing.loss_scores(loss_fn, ..., coin_flip=cf, dataset=ds)` | Verified membership scores → `CanaryScores` |
| `auditing.gradient_scores(loss_fn, ..., coin_flip=cf, dataset=ds)` | Verified white-box scores → `CanaryScores` |
| `auditing.one_run(scores, coin_flip=cf)` | Estimate privacy → `OneRunEstimate` |
| `auditing.canary_scores(values, canary_indices=...)` | Attest identifiers for externally computed scores |
| `cf.train_indices(len(dataset))` | Training indices for `dataset.select()` |
| `estimate.epsilon_at(delta=)` | ε bound (μ-GDP default) |
| `estimate.delta_at(epsilon=)` | δ at given ε |
| `estimate.beta_at(alpha=)` | Theoretical β at α (μ-GDP) |
| `estimate.advantage()` | TV advantage at inferred μ̂ |
| `estimate.eps_delta()` | Mechanism-agnostic (ε, δ)-DP method (override) |
| `estimate.gdp(grid_size=)` | μ-GDP method with tunable grid |
| `estimate.attack_auc()` | Empirical attack AUC |
| `estimate.attack_auc(confidence=, key=)` | Empirical AUC with CI |
| `estimate.attack_beta_at(alpha=)` | Empirical attack β at given FPR |
