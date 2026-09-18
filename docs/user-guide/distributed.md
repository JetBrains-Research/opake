# Distributed Training

Opake supports multi-GPU training via PyTorch DistributedDataParallel
(DDP). FSDP, Tensor Parallel, and Pipeline Parallel are not supported.

## How DP-SGD works with DDP

Each device runs the same training loop on a different data shard. Three
operations happen in sequence:

1. **Clip** — each device computes per-example clipped gradients on its
   local batch.
2. **Aggregate** — an AllReduce SUM collects the clipped gradient sums
   from every device.
3. **Noise** — every device independently adds the *same* Gaussian noise.
   Because the noise key is identical on all ranks, the noise is identical,
   and models stay in sync.

```
Device 0:  clip(local_batch) --+
Device 1:  clip(local_batch) --+-- AllReduce SUM -- + noise(key) -- update
Device 2:  clip(local_batch) --+
```

There are two valid approaches to noise in distributed DP-SGD:

- **Independent generation (recommended):** Every rank generates the same
  noise using the same key. No communication is needed — the noise is
  identical because the RNG state is identical. This is simpler and avoids
  an extra broadcast.
- **Rank-0 broadcast:** Rank 0 generates the noise and broadcasts it to
  all other ranks. This is correct but adds a communication step and
  requires special-casing rank 0.

## Minimal example

```python
import torch
import torch.distributed as dist
import torchopt
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler
from opake.functional import make_functional
import opake.distributed as dist_utils
from opake.random import key, fold_in
from opake.distributed import local_shard

# Distributed setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model -> functional form
model = MyModel().to(device)
fmodel, params = make_functional(model)

def loss_fn(params, x, y):
    return ((fmodel(params, x) - y) ** 2).sum()

# Expected global batch size
batch_size = 256

# DP components
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=(1, 2),
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=1.1, key=key(42),
)

# Poisson sampler (shard dataset)
shard = local_shard(dataset, rank=rank, world_size=dist.get_world_size())
sampler = PoissonSampler(
    shard, sample_rate=0.01, key=fold_in(key(0), rank),
)
loader = torch.utils.data.DataLoader(shard, batch_sampler=sampler)

# Optimizer
from opake.optimizers import sgd

optimizer = sgd(lr=0.01)
opt_state = optimizer.init(params)

# Training loop
for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # 1. Clip (local)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    # 2. Aggregate (AllReduce SUM, copy-returning)
    grads = dist_utils.sum_gradients(grads)

    # 3. Noise (identical on every device)
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    # 4. Update
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)

dist.destroy_process_group()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py
```

## Noise synchronization

Noise constructors take a `key` parameter. Pass the **same key** on every
rank to get identical noise (centralized DP-SGD). Use `fold_in(key, rank)`
to get independent per-rank noise streams when needed:

```python
from opake.random import key, fold_in

# Synchronized noise — same key on all ranks
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))

# Independent noise — different key per rank
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=1.1, key=fold_in(key(42), rank)
)
```

In the common centralized DP-SGD pattern, pass the same `key(seed)` on
every rank so that all devices produce identical noise and models stay
in sync.

## Gradient aggregation

`sum_gradients` performs an AllReduce SUM on a PyTree of tensors and returns
a reduced copy:

```python
import opake.distributed as dist_utils

grads = dist_utils.sum_gradients(grads)
```

This sums the clipped gradient contributions from all devices. After this
call, every rank holds the same total clipped gradient sum in `grads`.

For more general reductions, use:

- `reduce_pytree(pytree, op)` to return a reduced copy
- `reduce_pytree_(pytree, op)` for in-place reduction
- `sum_gradients_(grads)` for in-place DP-specific sum

```python
avg_grads = dist_utils.reduce_pytree(grads, op="mean")
dist_utils.reduce_pytree_(grads, op="sum")
dist_utils.sum_gradients_(grads)
```

## Adaptive clipping

`adaptive_clipped_grad` is local-only — it does not communicate or
auto-detect DDP. Each rank clips its own local batch and records local
clipping statistics in `AdaptiveClipState`. To keep the adaptive threshold
consistent across ranks, explicitly synchronize that state after each step:

```python
from opake.dpsgd.clipping import adaptive_clipped_grad
import opake.distributed as dist_utils
from opake.dpsgd.noise import gaussian_noise
from opake.distributed import sync
from opake.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    batch_argnums=(1, 2),
    initial_clipping_norm=1.0,
    normalize_by=batch_size,
    key=key(7),
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))

# In the training loop:
grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
clip_state = sync(clip_state)   # required: aggregate local counts and update next threshold
grads = dist_utils.sum_gradients(grads)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

When syncing several objects in the same phase, `sync` accepts variadic
arguments and returns them in order:

```python
clip_state, aux = sync(clip_state, aux)
noise_state = sync(noise_state)
```

`sync()` auto-dispatches based on the type of the state object. For
`AdaptiveClipState`, it sums `num_clipped` and batch size across ranks,
recomputes the global clipping rate from those aggregated totals, and updates
the internal next threshold identically on every device. This is the required
step that keeps distributed adaptive clipping correct even when ranks see
uneven local batches. The current DP bound is carried by the clipped output's
`.max_norm` metadata.

For fixed clipping (`clipped_grad`), the state is a deterministic marker and
does not need synchronization. `sync(clip_state)` is a passthrough.

## Poisson sampling

Shard the dataset using `local_shard()`, then create a `PoissonSampler`
on the shard. Derive a per-rank key via `fold_in(key, rank)`.

```python
import torch.distributed as dist
from opake.dpsgd.sampling import PoissonSampler
from opake.random import key, fold_in
from opake.distributed import local_shard

rank = dist.get_rank()
world_size = dist.get_world_size()

shard = local_shard(dataset, rank=rank, world_size=world_size)
sampler = PoissonSampler(
    shard,
    sample_rate=0.01,
    key=fold_in(key(42), rank),
)
loader = DataLoader(shard, batch_sampler=sampler)
```

## Privacy accounting

Privacy accounting is the same as single-device training. The effective
sample rate is the global sample rate across all devices:

```python
import opake.accounting as acc

global_sample_rate = batch_size_per_device * world_size / dataset_size
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), global_sample_rate)
training = step * num_steps
epsilon = training.epsilon_at(delta=1e-5)
```

## Optimizer state synchronization

For correlated noise mechanisms (`mf_gaussian_noise` with any MF strategy), each
device must generate the same correlated noise stream. Pass the same
`key=key(seed)` on every rank.

Functional optimizers (TorchOpt) stay synchronized automatically because:

1. After `sum_gradients`, all ranks hold the same gradient sum.
2. After adding synchronized noise, all ranks hold the same noisy gradient.
3. `optimizer.update` is a pure function: same inputs produce same outputs.

So optimizer states evolve identically on all devices without explicit
synchronization.

## `opake.distributed` API summary

Functions are split into copy-returning defaults and explicit in-place `_`
variants. When `torch.distributed` is not initialized, high-level helpers such
as `sum_gradients` and `reduce_pytree` are local no-ops, while direct
collective wrappers such as `all_reduce` / `all_reduce_` require an initialized
process group and raise `RuntimeError`.

| Function | Purpose |
|----------|---------|
| `is_distributed()` | `True` if `torch.distributed` is initialized |
| `get_rank()` | Current rank (0 if not distributed) |
| `get_world_size()` | Number of devices (1 if not distributed) |
| `sum_gradients(grads)` | Return a summed copy of a gradient PyTree |
| `sum_gradients_(grads)` | In-place AllReduce SUM on a gradient PyTree |
| `reduce_pytree(pytree, op)` | Return a reduced copy of a PyTree (op: `"sum"`, `"mean"`, `"max"`, `"min"`, `"product"`) |
| `reduce_pytree_(pytree, op)` | In-place AllReduce on a PyTree (op: `"sum"`, `"mean"`, `"max"`, `"min"`, `"product"`) |
| `reduce_scalar(value, op, device, *, compute_dtype)` | Reduce a Python float or integer across ranks; floats default to fp32 and can request fp64, while integers use an exact int64 path |
| `all_reduce(tensor, op)` | Return an all-reduced tensor copy |
| `all_reduce_(tensor, op)` | In-place AllReduce on a single tensor |
| `gather_tensors(tensor, dim)` | Gather variable-size tensors from all ranks and concatenate |
| `gather_pytree(pytree)` | Gather and concatenate tensor leaves of a PyTree |
| `assert_pytree_equal(pytree, name)` | Assert a PyTree is identical across ranks (fingerprint check) |
| `sync(*states)` | Dispatch to the right sync function for one or more state/aux/profiler objects |
| `sync_object(state, field_ops)` | Synchronize scalar fields of a dataclass across ranks; each `field_ops` entry is an operation name or a callable `fn(value, device)`, invoked once |
| `assert_scalar_equal(v, name)` | Raise `RuntimeError` if a scalar differs across ranks |
| `barrier()` | Blocking barrier across all ranks |

### Type-specific sync behavior

`sync()` auto-dispatches based on the type of the object passed. The
following types are registered:

| Type | Behavior |
|------|----------|
| `FixedClipState` | Marker-state passthrough |
| `AdaptiveClipState` | Aggregate counts, recompute global clipping rate, update the internal next threshold |
| `ClippedFunAux`, `ClippedGradAux`, `AdaptiveClippedGradAux` | Gather aux tensors across ranks |
| `GaussianNoiseState` | Assert seed and step counter match across ranks |
| `MFNoiseState` | Assert seed and step counter match for MF noise |
| `SecondMomentMFNoiseState` | Validate both paired MF streams, in a fixed order |
| `PerfState` | Aggregate step times (max), sample counts (sum), and peak memory (max) across ranks |

### Noise scaling depends on the key, not on the reduction

Summing noised local queries scales `noise_stddev` by `sqrt(world_size)` — but
only when the summands are **independent**. `NoisedPytree` carries no record of
which key produced it, so the reduction cannot tell the two regimes apart and
always publishes the `sqrt(world_size)` figure:

| Key per rank | Aggregation | Realized stddev | Published metadata |
|---|---|---|---|
| `fold_in(key, rank)` — independent | `sum` | `σ·√W` | `σ·√W` ✅ |
| shared `key` — identical draws | `sum` | `σ·W` | `σ·√W` ⚠️ |
| shared `key` | aggregate first, then noise once | `σ` | `σ` ✅ |

`sync(noise_state)` *enforces* a shared seed, so the recommended flow is the
third row: reduce the clipped gradients across ranks, then add noise once. Add
noise per rank only with per-rank keys, and only when you have checked that the
reported stddev is the one you intend to record.

See [API Reference](../reference/distributed.md) for full docstrings.

## Limitations

- **DDP only.** FSDP, Tensor Parallel, and Pipeline Parallel are not
  supported.
- **Backend support tiers.** First-class backends are NCCL, Gloo, and MPI.
  Vendor/runtime-dependent backends require environment-specific stacks and are
  not covered by default CI lanes.
- **Single-node primarily.** Multi-node DDP should work but is not
  extensively tested.

## API reference

See [Distributed API Reference](../reference/distributed.md) for complete
function signatures and return types.
