# opaque-federated

Federated twins of the central DP training loop's data primitives, executed on
[IFED](https://github.com/JetBrains-Research/ifed):

- `opaque.federated.clipped_sum` — the IFED **strategy** that clips each
  client's gradient to `clipping_norm` and releases their sum. Per-client
  clipping before summation is what bounds the round's sensitivity to
  `clipping_norm` under add-or-remove-one-client adjacency.
- `opaque.federated.clipped_grad` — the loop driver: one call = one federated
  round. Returns the same `(grad_fn, FixedClipState)` / `ClippedPytree` shapes
  as `opaque.dpsgd.clipping.clipped_grad`, so the noise → optimizer →
  accountant chain downstream is the central one unchanged.
- `opaque.federated.MinSepSampler` — the federated `BMinSepSampler`: yields
  Opaque `Cohort` specs and compiles minimum separation onto IFED's
  assignment-delta policy. No sampling randomness is claimed (selection is
  platform-greedy), so it pairs with **non-amplified** BandMF accounting.
- `opaque.federated.DataLoader` — iterates a population for `rounds` cohorts.

IFED's own primitives stay user-facing: you build the plan with
`ifed.build_train` and open the run with `ifed.session`. Nothing here wraps
them.

```python
import ifed
import opaque.federated as fed

pop = fed.population("/hive", version="*")
sampler = fed.MinSepSampler(pop, batch_size=8, bands=4)
loader = fed.DataLoader(pop, batch_sampler=sampler, rounds=60)

strategy = fed.clipped_sum(clipping_norm=1.0)
plan = ifed.build_train(
    net=net,
    source=MyDataset,
    loss=ifed.Loss.mse,
    batch_size=None,  # one client contribution = one full-batch gradient
    strategy=strategy,
)

store = ifed.FederatedDatastore(
    population=pop.name,
    version=pop.version,
    cardinality=sampler.batch_size,
    assign_delta=sampler.assign_delta,
    server="prod",
)
with ifed.session(plan, store) as run:
    params = plan.init_state.params
    grad_fn, clip_state = fed.clipped_grad(run, strategy)
    for cohort in loader:
        grads, clip_state = grad_fn(params, cohort, state=clip_state)
        ...  # gaussian_noise / optimizer / accountant — unchanged central opaque
```

The first loader-produced cohort fixes the cardinality, separation, and
horizon the accounting is stated for; later cohorts that disagree are
rejected.

See `docs/reference/federated.md` in this repository for the complete
contract.
