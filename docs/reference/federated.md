# Federated

Federated-learning twins of the DP training loop's data primitives live in
`opaque.federated`, executed on [IFED](https://github.com/JetBrains-Research/ifed).
The privacy unit is the **client** (a federated agent holding a private
dataset) instead of the example, and one training round replaces one batch —
everything downstream of the clipped gradient (`gaussian_noise`, optimizers,
the `Accountant`) is the unchanged central chain.

IFED's own primitives stay user-facing: you describe the model with
`ifed.build_train` and open the run with `ifed.session`. Opaque adds the two
pieces a DP loop needs — a strategy that releases a per-client-clipped sum, and
a driver that turns each round into a `ClippedPytree`.

## Overview

| central | federated | notes |
|-----------|-----------|-------|
| `Dataset` | `opaque.federated.population(name, version="*")` | a symbolic pool of clients |
| `Sampler` | `opaque.federated.MinSepSampler` | yields Opaque `Cohort` *specs*; the platform enforces the participation constraint |
| `batch` | `opaque.federated.Cohort` | symbolic — per-client gradients depend on the params, so a cohort is resolved by executing its round |
| `DataLoader` | `opaque.federated.DataLoader` | bounds the stream to `rounds` cohorts |
| per-example clip | `opaque.federated.clipped_sum` | the IFED strategy that clips each client and sums |
| `clipped_grad` | `opaque.federated.clipped_grad` | the loop driver; one call = one federated round |

```python
import ifed
import opaque.federated as fed

pop     = fed.population("/hive", version="*")
sampler = fed.MinSepSampler(pop, batch_size=8, bands=4)
loader  = fed.DataLoader(pop, batch_sampler=sampler, rounds=60)

strategy = fed.clipped_sum(clipping_norm=1.0)
plan = ifed.build_train(
    net=net,
    source=MyDataset,
    target="y",
    features=["x"],
    loss=ifed.Loss.mse,
    batch_size=None,          # one client contribution = one full-batch gradient
    strategy=strategy,
)

store = ifed.FederatedDatastore(
    population=pop.name,
    version=pop.version,
    cardinality=sampler.batch_size,   # the cohort size the accounting is stated for
    assign_delta=sampler.assign_delta,
    server="prod",
)
with ifed.session(plan, store) as run:
    params = plan.init_state.params
    grad_fn, clip_state = fed.clipped_grad(run, strategy)
    for cohort in loader:
        grads, clip_state = grad_fn(params, cohort, state=clip_state)
        # gaussian_noise → optimizer.update → apply_updates → accountant: unchanged
```

## What a round releases

`clipped_sum` overrides two of the four IFED server phases:

- `aggregate` scales each client's whole gradient pytree to L2 norm
  `clipping_norm` and releases their **sum**. Clipping per client before
  summation is what bounds the round's sensitivity to `clipping_norm` under
  add-or-remove-one-client adjacency.
- `finalize` carries that sum out untouched instead of taking a server SGD
  step: the step belongs to Opaque's noise → optimizer chain.

It fixes `weighted=False` and `max_skipped=0.0`, so the divisor stays
data-independent: every client counts once, and a round with even one unusable
client (no gradient, a non-finite value, a metric schema the round cannot pool,
or a reported row count of zero) fails instead of quietly averaging over the
survivors. Its metrics bundle is emptied on the way out, so a round's only
release is the clipped sum.

`clipped_grad` divides that sum by `normalize_by` (the cohort size by default)
and returns a `ClippedPytree` carrying
`max_norm = clipping_norm / normalize_by`, exactly like central. Build the plan
with `batch_size=None` so one client's contribution is one gradient over all of
its own rows; with a smaller local batch, what a client sends depends on how the
agent runtime accumulates across batches, which a per-client sensitivity bound
cannot rest on.

The first cohort produced by `DataLoader` binds the cohort size, the separation
and the loader identity. Later cohorts must come from that same loader, in
order, with those values unchanged.

## End-to-end example

`examples/federated_regression.py` runs the whole loop — cohort stream, clipped
sum, Gaussian noise, SGD — either against two agents in this process or against
a driver:

```bash
uv run python examples/federated_regression.py local --rounds 10
uv run python examples/federated_regression.py prod --clients 8 --bands 4
```

## Sampling honesty

`MinSepSampler` deliberately drops two central `BMinSepSampler` parameters,
because federation cannot honestly claim them today:

- **No `sampling_prob` / `key`** — IFED assigns rounds greedily among live
  eligible agents; there is no randomized-selection guarantee, hence **no
  subsampling amplification**. Account with the *non-amplified* BandMF
  mechanism (`opaque.dpftrl.accounting.mf_gaussian` with a
  `band_mf_strategy`): `min_sep = sampler.bands`,
  `max_participations = ceil(loader.rounds / sampler.bands)`.
- **No `n_steps`** — the round horizon lives on the loader (`rounds=`).

Cohorts are **exact**: a round collects contributions from exactly `batch_size`
clients (blocking until it has them), so the batch size is a constant, the
gradient normalizer is fixed, and the sensitivity
`clipping_norm / batch_size` is data-independent.

The store is IFED's own `ifed.FederatedDatastore`: nothing in Opaque wraps it.
Give it the sampler's `batch_size` as `cardinality`, so the round runs with the
cohort size the accounting is stated for.

`sampler.assign_delta` is the `bands - 1` that IFED's assignment-separation
policy takes, so an agent that served round `r` is ineligible until round
`r + bands`. It is a field of the **driver** store: a `LocalDatastore` has no
driver to enforce it and no such field, so local runs reproduce the loop but
not the separation guarantee.

Accounting remains user-owned, as everywhere in Opaque: the sampler and loader
only expose the participation structure (`bands`, `batch_size`, `rounds`).

## API

::: opaque.federated.population

::: opaque.federated.clipped_sum

::: opaque.federated.clipped_grad

::: opaque.federated.MinSepSampler

::: opaque.federated.DataLoader
