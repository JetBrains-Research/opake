# Privacy Auditing

Privacy accounting gives theoretical upper bounds on privacy loss.
Privacy auditing gives empirical lower bounds by running a membership
inference attack. If the audited epsilon exceeds the theoretical epsilon,
there is likely a bug in the implementation.

## How it works

Opaque implements one-run auditing
([Steinke, Nasr, Jagielski 2023](https://arxiv.org/abs/2305.08846))
with the tight order-statistics tests from
[Xiang et al. (2025)](https://arxiv.org/abs/2509.08704):

1. **Designate canaries.** Select `m` examples as canaries. By default they
   are uniformly sampled natural rows; a caller-supplied pool can contain
   preselected or constructed examples.
2. **Flip coins.** For each canary, include or exclude it from training
   with probability 0.5.
3. **Train once.** Train on the resulting subset. Non-canary data is always
   included.
4. **Score.** Compute a membership score for each canary (negative loss by
   default — lower loss means more likely a member). Each score carries the
   dataset index of its canary.
5. **Test.** Join scores to coin flips by those identifiers and convert them
   into an ε lower bound via the audit-method surface on `OneRunEstimate`.

Only one training run is needed, unlike shadow-model approaches.

## Quick start

```python
import opaque.auditing as auditing
from opaque.random import key

# 1. Partition: designate canaries and flip coins
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))

# 2. Train with DP-SGD on train_data ...

# 3. Score: compute membership scores for the canaries
def canary_collate(examples):
    batch = data_collator(examples)
    return (batch["input_ids"].to(device),)

scores = auditing.loss_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
    batch_size=32, collate_fn=canary_collate,
)

# 4. Estimate: build the one-run estimate and query ε
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (audit):  {estimate.epsilon_at(delta=1e-5):.4f}")
print(f"AUC (attack): {estimate.attack_auc():.4f}")
```

Scoring with `coin_flip=` + `dataset=` builds the canary loader internally
and returns `CanaryScores`: every score is paired with the dataset index of
the canary that produced it. `one_run` joins scores to coin-flip labels by
those identifiers — identifiers that are wrong, missing, or duplicated
raise instead of silently pairing scores with the wrong labels, so however
the pipeline orders its batches, it cannot fake a "no leakage" result.

The one thing you still own is `collate_fn`. Identifiers ride alongside
the examples and are captured *before* your collate runs, so a collate
that reorders rows within a batch attaches every score to the wrong
canary — silently, since the count still matches. Keep it
order-preserving; dropping or adding rows raises.

For scores computed outside the built-in scorers, attest their
identifiers explicitly (any order is accepted — the join realigns them):

```python
scores = auditing.canary_scores(values, canary_indices=ids_in_your_scoring_order)
estimate = auditing.one_run(scores, coin_flip=cf)
```

## Default audit method: μ-GDP

`OneRunEstimate.epsilon_at(delta=)`, `.delta_at(epsilon=)`, `.beta_at(alpha=)`,
and `.advantage()` all dispatch to the **μ-GDP order-statistics test** from
Xiang et al. (2025).

Use μ-GDP only when it models the audited mechanism; otherwise use the
mechanism-agnostic method.

The dispatch requires `delta > 0`: μ-GDP is incompatible with pure ε-DP.

```python
estimate.epsilon_at(delta=1e-5)                       # default — μ-GDP
estimate.epsilon_at(delta=1e-5, significance=0.01)    # stricter confidence
estimate.epsilon_at(delta=1e-5, threshold=4.0)        # specific threshold
estimate.delta_at(epsilon=3.0)                        # inverse along the boundary
estimate.beta_at(alpha=0.01)                          # theoretical β under μ̂-GDP
estimate.advantage()                                  # TV(μ̂) = 2·Φ(μ̂/2) − 1
```

## Choosing a different method

`OneRunEstimate.gdp(grid_size=)` returns the underlying μ-GDP method
object — useful if you want to tune the integration grid or reuse a
single μ̂ across multiple queries:

```python
gdp = estimate.gdp(grid_size=20_000)   # tighter numerics
gdp.epsilon_at(delta=1e-5)
gdp.beta_at(alpha=0.01)
```

`grid_size` must be at least `1_000`. Coarser grids are rejected because
they can resolve a detectable leak as `epsilon=0` — silently reporting no
leakage instead of an inaccurate one.

`OneRunEstimate.eps_delta()` returns the mechanism-agnostic (ε,δ)-DP
method object. Use it when:

- You audit a non-Gaussian-DP mechanism (Laplace, discrete Gaussian,
  heavy-tailed noise, …).
- You need pure ε-DP (δ=0) auditing.

```python
estimate.eps_delta().epsilon_at(delta=0.0)   # pure ε-DP
estimate.eps_delta().epsilon_at(delta=1e-5)  # general (ε, δ)-DP fallback
estimate.eps_delta().delta_at(epsilon=3.0)
```

`EpsDeltaMethod` deliberately does *not* expose `beta_at` / `advantage`:
the (ε,δ)-DP trade-off function is a family envelope, so those metrics
would be worst-case across the family rather than instance-specific. The
μ-GDP method's inferred μ̂ pins down a single curve, so its `beta_at` /
`advantage` are sharp.

## Integration with training

The four-step API separates concerns: **partition** (before training),
**train** (unchanged), **score** + **estimate** (after training).

### Step 1: Partition before training

Call `auditing.coin_flip` after preparing the dataset. It derives distinct
audit subkeys for canary selection and coins internally; training mechanisms
should use their own RNG domains:

```python
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))

# Remove held-out canaries from training set
train_dataset = dataset.select(cf.train_indices(len(dataset)))
```

### Step 2: Train normally

No changes to the training loop. The dataset is already filtered.

### Step 3: Score and estimate after training

```python
scores = auditing.loss_scores(
    per_example_loss_fn, trained_params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
    batch_size=32, collate_fn=canary_collate,
)
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (audit): {estimate.epsilon_at(delta=1e-5):.4f}")
```

See [examples/train_dpsgd.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd.py)
and [examples/train_dpftrl.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpftrl.py)
for complete working examples with the `--audit` flag.

### Parameter reference

- **`batch_argnums`**: Which positional arguments of `loss_fn` come from the
  batch. `(1,)` means arg 1 is batched; `(1, 2)` means args 1 and 2 are
  batched. Same convention as `clipped_grad`.
- **`coin_flip` + `dataset`**: required. The scorer builds its own loader
  over the partition's canaries and returns `CanaryScores` with per-score
  identifiers — the form `one_run` requires. `batch_size` and `collate_fn`
  configure the internal loader; `collate_fn` receives the raw canary
  examples and must return a batch for `loss_fn`, one row per example in
  the order it received them. The collated batch should be a tensor (single
  `batch_argnums`) or a tuple of tensors (multiple `batch_argnums`).
- **`reference_scores`**: Baseline scores from an untrained model, over the
  same partition. When provided, returned scores are
  `scores - reference_scores` (loss reduction). The reference must be a
  `CanaryScores` and is aligned by identifier before subtraction.

Scoring a pipeline the built-in scorers cannot express is still supported —
compute the scores yourself and attest their identifiers with
`auditing.canary_scores(values, canary_indices=...)`.

## Attack-side metrics

Independent of the audit method, you can read attack-side empirical
metrics directly from the estimate:

```python
estimate.attack_auc()                  # ROC AUC (0.5 = random, 1.0 = perfect)
estimate.attack_beta_at(alpha=0.01)    # empirical attack β = 1 − TPR at FPR=α
```

AUC confidence intervals:

```python
ci = estimate.attack_auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

`key` is required when requesting a CI: bootstrap resampling must be
reproducible and must not depend on global NumPy state.

The `attack_` prefix distinguishes these from the (audit-side) `beta_at`
that reads the theoretical f-DP β at the inferred μ̂-GDP.

## Number of canaries

For the mechanism-agnostic $(\varepsilon,\delta)$ method, a perfect attack
has an approximate ceiling `ε ≲ ln(m / -ln(α))`. This does not apply to the
μ-GDP estimator. The μ-GDP implementation computes the 2000 most informative
order statistics exactly and uses a conservative, μ-dependent boundary bound
for any remaining ranks, so this numerical cutoff does not limit the supported
number of canary guesses:

| Canaries m | Hard ceiling at α=0.05 |
|---|---|
| 1 000 | ≈ 5.8 |
| 10 000 | ≈ 8.1 |
| 100 000 | ≈ 10.4 |
| 1 000 000 | ≈ 12.7 |

Imperfect attacks lower the mechanism-agnostic estimate further.

In practice:

| Canaries | Use case |
|---|---|
| 100 | Debugging only, very noisy |
| 500 | Detecting large implementation bugs |
| 1000+ | Recommended for reliable bounds |
| 5000+ | Needed for small-epsilon regimes |

## Stronger canaries

The default `coin_flip` call partitions natural rows already present in the
dataset. This is a valid one-run audit, not canary injection. Uniformly random
natural rows often provide less membership signal than deliberately selected or
constructed examples, although influential natural records can also be strong.

Canary design is application-specific. Images with perturbed labels, unusual
text sequences, and sparse gradient-space canaries are different objects with
different model and scoring requirements. Construct or select them in your own
pipeline, then pass their dataset indices as `candidate_indices`:

```python
from datasets import concatenate_datasets

base_size = len(dataset)
audited = concatenate_datasets([dataset, crafted_canaries])
pool = range(base_size, len(audited))

cf = auditing.coin_flip(
    audited,
    num_canaries=len(pool),
    key=key(42),
    candidate_indices=pool,
)
train_data = audited.select(cf.train_indices(len(audited)))
```

The pool must be one-dimensional, unique, integer-valued, and within
`range(len(audited))`. Opaque samples `num_canaries` indices from it without
replacement; passing a pool of exactly that size designates every supplied
index. If the pool is larger, its unselected records remain ordinary always-in
training data. Pool order does not affect a seeded partition.

Fix the pool before target training, independently of that run's membership
coins and outputs. Opaque still generates independent fair membership coins;
do not construct `CoinFlip` directly to choose a favourable in/out mask. Use
the same `audited` dataset for `coin_flip`, subset construction, and scoring so
the stable canary indices continue to identify the same records.

Strength depends on both detectability and diversity. Many similar synthetic
canaries can interfere with one another, so increasing the count or making a
record more unusual does not guarantee a tighter result. The original one-run
paper evaluates relabelled and in-distribution input canaries as well as
gradient-space canaries; more recent work also finds that influence-selected
natural records can outperform random natural records.

Finally, compare the audit only with accounting for the same training
mechanism. Canary partitioning changes the records and cardinality seen by the
sampler. If sampling probability, step horizon, minimum separation, bin count,
or another privacy-relevant setting is derived from the realized partition,
accounting for one fixed configuration may not cover its neighbouring runs.
Do not reuse a theoretical bound computed for the unpartitioned dataset without
checking that it covers the actual audited workload.

## Interpreting results

The audited epsilon should be **below** the theoretical epsilon:

```
  ε (one-run):          1.50
  ε (theoretical):      3.00    ← expected: gap exists
```

**Gap is expected.** The audit runs a particular attack with finite statistical
power; the theoretical bound is a worst-case upper bound. A large gap is normal,
especially in high-noise regimes.

**A low audited epsilon is not a privacy certificate.** It can mean that the
mechanism leaked little, that the scores or canaries did not expose the leakage,
that canaries interfered with one another, or that the sample was too small to
resolve it. Report the canary design and attack with the lower bound. If the
result matters, compare multiple precommitted designs rather than interpreting
a weak attack as evidence of privacy.

**Audited > theoretical is a red flag.** Investigate:

- Gradient clipping applied incorrectly
- Noise scaled to wrong sensitivity
- Privacy accounting doesn't match actual training procedure
- Data leakage outside the private training loop

## References

- Xiang et al. (2025). [Tight Privacy Audit in One Run](https://arxiv.org/abs/2509.08704).
- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Dagréou, Bellet (2026). [Detectability in Diversity: Improved Canary Crafting for Privacy Auditing in One Run](https://arxiv.org/abs/2605.27292).
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

## API reference

See [Auditing API Reference](../reference/auditing.md) for complete function
signatures.
