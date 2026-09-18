# Sampling

Sampling primitives live in `opake.dpsgd.sampling` (Poisson, k-out-of-t allocation)
and `opake.dpftrl.sampling` (cyclic Poisson with optional `bands`, b-min-sep,
balls-in-bins, sequential).
Distributed shard helpers live in `opake.distributed`. They provide
privacy-amplifying sampling mechanisms for DP-SGD and DP-FTRL.

## Overview

Privacy amplification through sampling is a key technique in DP-SGD: training
on randomly selected subsets provides stronger privacy than training on the
full dataset.

Opake provides these sampling strategies:

1. **Poisson Sampling — DP-SGD** (`opake.dpsgd.sampling.PoissonSampler`):
   each example is sampled independently with probability `sample_rate`.
   Optional `truncated_batch_size` caps per-step batch size for more stable
   batches and memory; accounting must use the truncated-Poisson PLD (weaker
   than plain Poisson at the same `sample_rate`).

2. **K-Out-of-T Allocation — DP-SGD** (`opake.dpsgd.sampling.KOutOfTSampler`):
   with `allocation="block"`, each record is assigned to one batch in each of
   `k` contiguous, nearly equal blocks. With
   `allocation="total"`, each record instead chooses `k` positions uniformly
   from the complete horizon.

3. **Cyclic Poisson (DP-FTRL)** (`opake.dpftrl.sampling.CyclicPoissonSampler`):
   `bands` disjoint groups; step `i` samples only group `i % bands`, with
   independent inclusion at `sample_rate`. Use `bands=1` for identity MF
   (full dataset each step); for BandMF, match `bands` to the MF strategy.

4. **Balls-in-Bins Sampling** (`BallsInBinsSampler`): each example is
   independently assigned to a bin once at init; the assignment is **fixed
   across epochs** (required for BnB accounting). Bin sizes are variable
   (Binomial); some bins may be empty. Used with DP-λCGD, BISR, BSR, and
   BLT mechanisms.

5. **Sequential Batch Sampling** (`SequentialBatchSampler`): iterates
   through the dataset in fixed-size contiguous batches with no randomness.
   The dataset should be pre-shuffled once before constructing the sampler.
   Used with the BLT mechanism.

6. **b-min-sep** (`BMinSepSampler`): warm-start minimum-separation Poisson
   subsampling for BandMF (arXiv:2602.09338). Use with `dpftrl_acc.b_min_sep`.

**See also**: [Sampling & Microbatching User Guide](../user-guide/sampling.md)

## PoissonSampler (DP-SGD)

```python
from opake.dpsgd.sampling import PoissonSampler
from opake.random import key

sampler = PoissonSampler(
    data_source,
    sample_rate=batch_size / len(data_source),
    n_steps=None,
    truncated_batch_size=None,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each example, in (0, 1]. At `sample_rate=1.0` there is no amplification — the accountant treats the step as the plain `gaussian(nm)` |
| `n_steps` | `int` or `None` | `None` | Number of batches to yield. `None` = infinite |
| `truncated_batch_size` | `int` or `None` | `None` | Optional per-step batch-size cap. When set, the sampler emits batches truncated to this many examples (uniform random subset of the Poisson draw). |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Plain Poisson — account with
`dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate)`.

Truncated Poisson (when `truncated_batch_size` is set) — account with
`dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate,
truncated_batch_size=batch, dataset_size=n)` to use the matching
truncated-Poisson PLD. Truncated Poisson requires `sample_rate < 1`;
for a full participation step (`sample_rate=1.0`) account directly with
`dpsgd_acc.gaussian(nm)`.

## KOutOfTSampler (DP-SGD)

```python
from opake.dpsgd.sampling import KOutOfTSampler
from opake.random import key

sampler = KOutOfTSampler(
    data_source,
    k=num_epochs,
    t=num_epochs * steps_per_epoch,
    allocation="block",
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `k` | `int` | required | Participations per record |
| `t` | `int` | required | Total number of batches to yield |
| `allocation` | `"block"` or `"total"` | required | Fixed-block or total-horizon allocation |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

With `allocation="block"`, the block sizes differ by at most one. Each block's
batches partition the dataset exactly, with an independent assignment in each
block. Bin sizes are Binomial, so some batches are empty; they are emitted
rather than compacted away, because dropping them changes the accounted
participation schedule.

Account block allocation with
`dpsgd_acc.k_out_of_t(mechanism, k=k, t=t, allocation="block")`.
This accounts the complete declared horizon once. Total allocation uses the
same factory with `allocation="total"` and currently receives the conservative
block bound.

!!! warning "Not the same scheme as `BallsInBinsSampler`"

    `opake.dpftrl.sampling.BallsInBinsSampler` draws the bin assignment
    once and reuses it for every epoch, because the matrix-mechanism
    dominating pair needs a known separation between an example's
    participations. `KOutOfTSampler(..., allocation="block")` draws each block independently, which is
    valid only because DP-SGD noise is uncorrelated across steps — and is
    strictly better there. Pair each sampler only with its own accountant.

## BallsInBinsSampler

```python
from opake.dpftrl.sampling import BallsInBinsSampler
from opake.random import key

sampler = BallsInBinsSampler(
    data_source,
    num_bins=dataset_size // batch_size,
    n_steps=8 * (dataset_size // batch_size),
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `num_bins` | `int` | required | Number of bins per epoch (≥ 2). Typically `dataset_size / batch_size` |
| `n_steps` | `int` or `None` | `None` | Total number of batches to yield. Must be a positive multiple of `num_bins` (per-bin participation count is `n_steps // num_bins`). `None` = infinite |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Bin sizes are variable (Binomial distribution). Assignments are **fixed
across epochs** (required for BnB dominating-pair accounting). Empty bins
are emitted as empty batches so every accounted bin slot executes.

Account with
`dpftrl_acc.balls_in_bins(mechanism, num_bins=num_bins, n_steps=n_steps)` where
`mechanism` is `dpftrl_acc.mf_gaussian(nm, strategy)` for `lambda_cgd_strategy`,
`bisr_strategy`, `blt_strategy`, `bsr_strategy`, or `dpftrl_acc.mf_gaussian(..., identity_strategy())`.

## CyclicPoissonSampler (DP-FTRL)

Partitions the dataset into `bands` groups and, at step `i`, draws only
from group `i % bands`, with each eligible example included independently at
`sample_rate` (Binomial batch size within the group). Identity MF uses
`bands=1`; BandMF uses `bands` equal to the mechanism’s band count — both
pair with `dpftrl_acc.poisson`.

```python
from opake.dpftrl.sampling import CyclicPoissonSampler
from opake.random import key

sampler = CyclicPoissonSampler(
    data_source,
    sample_rate=0.5,
    bands=4,
    n_steps=1000,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each eligible example, in (0, 1] |
| `bands` | `int` | `1` | Groups in the cycle. `1` = identity-style (full dataset each step). `>1` = cyclic BandMF-style (match strategy `bands`). |
| `n_steps` | `int` | `1` | Total batches to yield |
| `partition_type` | `PartitionType` | `EQUAL_SPLIT` | How to partition: `EQUAL_SPLIT` (only used when `bands > 1`) or `INDEPENDENT` |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

In distributed training, shard the dataset with `local_shard()` and pass
a per-rank key via `fold_in(key, rank)`. Best used with `mf_gaussian_noise`
for correlated noise (DP-FTRL); account with
`dpftrl_acc.poisson(mechanism, sample_rate, n_steps=...)`. There is no
batch-size cap on this sampler; `dpftrl_acc.poisson` matches uncapped
Poisson draws only.

## Distributed Helpers

### `local_shard`

Partition a dataset for DDP training. Returns a `Subset` containing the
contiguous shard for the given rank.

```python
from opake.distributed import local_shard
import torch.distributed as dist

shard = local_shard(
    dataset,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
sampler = PoissonSampler(shard, sample_rate=0.01, key=fold_in(key(42), rank))
loader = DataLoader(shard, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | dataset with `len()` | required | The full training dataset |
| `rank` | `int` | `0` | Current device rank |
| `world_size` | `int` | `1` | Total number of devices |

**Returns:** `torch.utils.data.Subset` containing the local shard.

::: opake.distributed.local_shard
    options:
      show_source: true
      heading_level: 3

## API Documentation

::: opake.dpsgd.sampling.PoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opake.dpsgd.sampling.KOutOfTSampler
    options:
      show_source: true
      heading_level: 3

::: opake.dpftrl.sampling.CyclicPoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opake.dpftrl.sampling.BMinSepSampler
    options:
      show_source: true
      heading_level: 3

::: opake.dpftrl.sampling.BallsInBinsSampler
    options:
      show_source: true
      heading_level: 3

## SequentialBatchSampler

```python
from opake.dpftrl.sampling import SequentialBatchSampler

sampler = SequentialBatchSampler(
    data_source,
    batch_size=256,
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset (must not be empty) |
| `batch_size` | `int` | required | Exact number of examples per batch (≥ 1) |

Batch size is deterministic and fixed. The last incomplete batch is
dropped (like `drop_last=True`). This sampler has no RNG key — it is
fully deterministic. Pre-shuffle the dataset before constructing the
sampler.

Used with the BLT mechanism, which requires deterministic batch order
with fixed separation between participations.

::: opake.dpftrl.sampling.SequentialBatchSampler
    options:
      show_source: true
      heading_level: 3
