# Per-Example Gradient Clipping

Per-example gradient clipping bounds the influence of each training example on
the model update. This is the core operation that makes DP-SGD possible:
clipping establishes a known sensitivity, which determines how much noise is
needed for a given privacy guarantee.

Opaque provides three high-level clipping functions:

- **`clipped_grad`** ([`opaque.dpsgd.clipping`](../reference/clipping.md)) — Fixed-threshold clipping (recommended default).
- **`auto_clipped_grad`** ([`opaque.dpsgd.clipping`](../reference/clipping.md)) — AUTO-S automatic scaling, no threshold to tune (Bu et al. NeurIPS 2023). Algorithm-agnostic: composes with both DP-SGD's Gaussian mechanism and DP-FTRL's matrix-factorization mechanisms.
- **`adaptive_clipped_grad`** ([`opaque.dpsgd.clipping`](../reference/clipping.md)) — Auto-tuned threshold via quantile tracking (Andrew et al. 2021); DP-SGD-only because the threshold drifts across steps.

For DP-FTRL, import `clipped_grad`, `auto_clipped_grad`, and `per_group`
from `opaque.dpftrl.clipping`.

Lower-level building blocks (`clipped_fun`, `clip_pytree`, `auto_scale_pytree`)
are documented in the [Clipping API Reference](../reference/clipping.md).

## `clipped_grad` — recommended API

`clipped_grad` wraps a per-example loss function. It computes per-example
gradients, clips each to a maximum L2 norm, and sums the result. This is the
primary API for DP-SGD training.

```python
from opaque.dpsgd.clipping import clipped_grad


def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()


grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,  # differentiate w.r.t. first argument (params)
    batch_argnums=(1, 2),  # second and third arguments are batched
    clipping_norm=1.0,
    normalize_by=batch_size,
)

grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
```

### How it works

1. `torch.func.grad_and_value` computes the gradient of `loss_fn` with respect
   to the argument at position `argnums`.
2. `torch.func.vmap` vectorizes this over the batch dimension of the arguments
   at positions `batch_argnums`, producing one gradient per example.
3. Each per-example gradient is clipped to L2 norm at most `clipping_norm`.
4. The clipped gradients are summed across the batch.

The returned gradients are a `ClippedPytree`. Its `.pytree` holds the clipped
gradient sum, and its `.max_norm` holds the per-step sensitivity used to calibrate
noise.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `Callable` | required | Per-example loss function. Must return a scalar (or `(scalar, aux)` if `has_aux=True`). |
| `argnums` | `int \| tuple[int, ...]` | `0` | Which arguments to differentiate. |
| `has_aux` | `bool` | `False` | If True, `loss_fn` returns `(loss, aux)`. The aux data is returned per-example. |
| `clipping_norm` | `float` | required | Maximum L2 norm for per-example gradients. |
| `batch_argnums` | `int \| tuple[int, ...]` | `1` | Which arguments have a batch dimension. |
| `microbatch_size` | `int \| None` | `None` | Process batch in chunks to reduce memory. |
| `normalize_by` | `float` | `1.0` | Divide the clipped sum and output bound by this constant. Set to expected batch size to get averaged gradients with bound = `clipping_norm / batch_size`. |
| `pre_clipping_transform` | `Callable` | identity | Transform applied to each per-example gradient before clipping. |
| `dtype` | `torch.dtype \| None` | `None` | Accumulation dtype (e.g., float32 for float16 inputs). |
| `return_aux` | `bool` | `False` | Return per-example diagnostics. |

### State flow

`clipped_grad` returns `(grad_fn, clip_state)`. The state must be threaded
through each call:

```python
grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=1)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    # clip_state is immutable; the returned value is the same object
```

With fixed clipping, `FixedClipState` is an immutable marker and the same
instance is returned on every call. The state-passing convention exists for API
consistency with `adaptive_clipped_grad`, where the internal state does change.

### Sensitivity

The sensitivity is the maximum change in the clipped gradient sum when one
example is added, removed, or replaced. Noise is calibrated to this value.

```python
grads, clip_state = grad_fn(params, batch, state=clip_state)
# With clipping_norm=1.0, normalize_by=32: grads.max_norm = 1.0 / 32

noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

### Diagnostics

Set `return_aux=True` to get per-example gradient norms and loss values:

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=1,
    return_aux=True,
)

(grads, aux), clip_state = grad_fn(params, batch, state=clip_state)
# aux.grad_norms: per-example L2 norms before clipping
# aux.clipped_grad_norms: per-example L2 norms after clipping
# aux.loss_values: per-example loss values
# aux.clipping_rate: fraction of per-example gradients that were clipped
# aux.batch_size: number of examples in the batch
```

`adaptive_clipped_grad` returns `AdaptiveClippedGradAux` instead, which adds
a `clipping_rate` field (fraction of gradients clipped).

## Microbatching

Per-example gradient computation via `vmap` requires memory proportional to
`batch_size * model_parameters`. For large models, this may exceed GPU memory.

Microbatching processes the batch in smaller chunks, accumulating clipped
gradients incrementally. The running sum is held in `compute_dtype`, so the
result matches processing the full batch to the precision of that dtype —
`float32` by default for `bfloat16` / `float16` inputs.

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=1,
    microbatch_size=16,  # process 16 examples at a time
)
```

Each microbatch of 16 examples is vmapped, clipped, and summed into a running
accumulator, so peak memory is proportional to
`microbatch_size * model_parameters` instead of `batch_size * model_parameters`.
The accumulator itself is one model-sized buffer in `compute_dtype` — for a
`bfloat16` run that is a `float32` copy, which is the price of the accumulation
precision described in [Precision](precision.md).

### Choosing microbatch size

Use `step_perf` from `opaque.profiling` to run a short sweep and
select the largest stable microbatch that does not OOM:

```python
from opaque.profiling import reset_peak_memory, step_perf

for candidate_mb in [64, 32, 16, 8, 4, 2, 1]:
    grad_fn, state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=candidate_mb,
    )

    reset_peak_memory(device)
    with step_perf(device, batch_size=batch_size) as perf:
        grads, aux = grad_fn(params, batch_x, batch_y, state=state)

    print(candidate_mb, perf.result.memory_peak_gb)
```

See [Memory Optimizations](memory-optimizations.md) for details.

## Adaptive clipping

`adaptive_clipped_grad` automatically adjusts the clip norm during training.
Instead of manually
tuning the clip norm, you specify a target fraction of gradients that should
be clipped (the *target quantile*).

```python
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    batch_argnums=1,
    initial_clipping_norm=1.0,
    target_quantile=0.5,  # aim for 50% of gradients clipped
    normalize_by=batch_size,
    key=key(7),  # required for quantile noise
)

grads, clip_state = grad_fn(params, batch, state=clip_state)
# the returned state carries the next threshold internally
```

### How adaptive clipping works

After each step:

1. Compute the fraction of per-example gradients whose norm exceeded the
   current clip norm.
2. Add calibrated Gaussian noise to this fraction (for privacy).
3. Apply a geometric update: if the noisy clipping rate exceeds the target
   quantile, increase the clip norm; otherwise, decrease it.

The update rule is:

    C_{t+1} = C_t * exp(eta * (noisy_rate_t - target_quantile))

where eta is the `learning_rate` parameter (default 0.2).

### State changes

Unlike `clipped_grad`, the state from `adaptive_clipped_grad` **does change**
on each call. `AdaptiveClipState` keeps the next threshold and counters as
internal execution state. Always use the returned state for the next call; use
`grads.max_norm` when you need the current DP bound.

```python
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    current_bound = grads.max_norm
```

### Privacy accounting for adaptive clipping

Adaptive clipping introduces an additional privacy cost (the noisy clipping
rate query). Account for it using `dpsgd_acc.adaclip()`:

```python
import opaque.dpsgd.accounting as dpsgd_acc

expected_batch_size = sample_rate * dataset_size
step = dpsgd_acc.poisson(
    dpsgd_acc.adaclip(
        dpsgd_acc.gaussian(noise_multiplier),
        fraction_noise_std=0.05,
        expected_batch_size=expected_batch_size,
    ),
    sample_rate,
)
training = step * num_steps
eps = training.epsilon_at(1e-5)
```

### Distributed adaptive clipping

In distributed training, `adaptive_clipped_grad` stays local to each rank:
it clips the local batch and records local clipped counts in the returned
state. To keep the adaptive threshold consistent across devices, synchronize
that state after every step, before the next adaptive-clipping call:

```python
import opaque.distributed as dist_utils
from opaque.distributed import sync

grads, clip_state = grad_fn(params, batch, state=clip_state)
clip_state = sync(clip_state)  # required: aggregate counts and update next threshold
grads = dist_utils.sum_gradients(grads)
```

For `AdaptiveClipState`, `sync()` aggregates clipping counts and batch
sizes, then updates the same next threshold on every rank. This is
required with uneven local batches; the current bound remains
`grads.max_norm`.

## MoE load balancing

`moe_clipped_grad` is the clipper for mixture-of-experts models trained
with the Switch load-balancing loss. It releases the batch router load
privately inside its state, the way adaptive clipping releases its clipped
fraction, and feeds the filtered estimate into the next step's per-example
surrogate. The loss function returns `(loss, router_logits, attention_mask)`
for one example and the accountant is
`dpsgd_acc.moe_aux(dpsgd_acc.gaussian(nm), ratio=ratio)` with the same
`ratio`:

```python
from opaque.dpsgd.clipping import moe_clipped_grad
from opaque.random import key

grad_fn, clip_state = moe_clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    normalize_by=batch_size,
    batch_argnums=(1, 2, 3),
    noise_multiplier=noise_multiplier,  # the value gaussian_noise uses
    ratio=0.02,
    key=key(7),
    top_k=8, num_experts=64, num_layers=28,
    max_tokens=2048,
    alpha=model.config.router_aux_loss_coef,
)
```

Under DDP synchronize the state after every step with
`opaque.distributed.sync`, as for adaptive clipping. See
[MoE load balancing](../mechanisms/dp-sgd/moe-load-balancing.md).

## Loss function requirements

The loss function passed to `clipped_grad` must:

1. **Return a scalar** for each example. Opaque differentiates this scalar to
   produce per-example gradients.
2. **Accept batched arguments** at the positions specified by `batch_argnums`.
   These arguments have a batch dimension that `vmap` maps over.
3. **Be compatible with `torch.func`**. Operations using in-place mutation,
   data-dependent control flow, or non-functional layers may fail under `vmap`
   (see [Known Limitations](../limitations.md)).

## Common patterns

### Functional model conversion

PyTorch models store parameters internally. To use them with `clipped_grad`,
convert to functional form:

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.functional import make_functional

fmodel, params = make_functional(model)


def loss_fn(params, x, y):
    pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (pred - y) ** 2


grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
)
```

### Separating trainable and frozen parameters

For parameter-efficient methods (LoRA, adapters), separate trainable from
frozen parameters:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)


def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(
        trainable_params,
        frozen,
        input_ids=input_ids.unsqueeze(0),
        labels=labels.unsqueeze(0),
    )
    return out.loss


grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
)
```

Only the trainable parameters receive per-example gradients. Frozen parameters
are treated as constants by `vmap`.

## Per-group clipping

Per-group clipping assigns different clipping norms to different parameter
groups. Instead of a single global L2 norm bound, each group is clipped
independently. This can better match the natural gradient scale of different
layers or module types (e.g., attention vs. MLP).

Use per-group clipping when groups have materially different gradient
scales. Set each bound near that group's typical norm; scalar clipping
is usually preferable for homogeneous groups.

### Setup

Use `per_group` to construct a `PerGroup` from parameter keys and substring
patterns:

```python
from opaque.dpsgd.clipping import clipped_grad, per_group

pg = per_group(params, self_attn=1.0, mlp=2.0)

grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    clipping_norm=pg,
    normalize_by=batch_size,
)
```

Each parameter leaf is matched by substring against the patterns, using the
leaf's display path (`".".join` of the optree `ParamPath`). Flat
`named_parameters` keys from `make_functional` are one-segment paths; nested
dicts/lists use multi-segment paths (e.g. `("layer", "attn")` displays as
`layer.attn`). Every leaf must match exactly one pattern, and every explicit
pattern must match at least one leaf. Use `fallback=<value>` as a catch-all
for unmatched parameters. To intentionally share a pattern configuration
across architectures with different parameter names, pass
`allow_unused_patterns=True`.

Prefer `per_group(params, …)` over constructing `PerGroup` by hand: a bare
string key such as `"layer.attn"` normalizes to the one-segment path
`("layer.attn",)`, which does **not** match a nested leaf at
`("layer", "attn")`.

```python
pg = per_group(params, self_attn=1.0, fallback=0.5)
```

### How it works

With per-group clipping norms $C_1, \dots, C_K$:

1. Per-example gradients are computed as usual via `vmap(grad(...))`.
2. Each group's gradient slice is clipped to its own L2 norm bound.
3. The clipped gradients are summed across the batch and divided by
   `normalize_by`.

### Sensitivity and noise

The L2 sensitivity of the full clipped query is a **scalar**:

$$\Delta_2 = \frac{\lVert C \rVert_2}{n} = \frac{\sqrt{\sum_i C_i^2}}{n}$$

This is carried by the clipped output:

```python
noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))
noisy_grads, noise_state = noise_fn(grads, noise_state)
stddev = noisy_grads.noise_stddev
```

Accounting is simply `gaussian(nm)` — no composition penalty, regardless of
the number of groups.

### Per-group noise allocation

For per-group bounds, `gaussian_noise` uses an MSE-optimal allocation that
puts less noise on small-norm groups.  To inspect or pass that allocation
to a mechanism that accepts stddev directly, call
`ClippedPytree.noise_stddev_for(...)`:

```python
# Default 'optimal' allocation — same MSE-optimal Mahalanobis assignment
# that gaussian_noise applies internally.
stddev = grads.noise_stddev_for(noise_multiplier=noise_multiplier)

# Or 'isotropic' for a uniform σ across leaves
uniform = grads.noise_stddev_for(
    noise_multiplier=noise_multiplier,
    allocation="isotropic",
)
```

For a bare `PerGroup` bound, wrap it in a `ClippedPytree` with placeholder
tensors keyed like your parameters, then use `noise_stddev_for` as above.

The optimal allocation sets $\sigma_i \propto \sqrt{C_i}$ instead of a
uniform σ.  Privacy accounting remains `gaussian(nm)` — the allocation
satisfies the same Mahalanobis constraint, just with better MSE.

The same joint allocation supports paired first/second-moment releases.
With a shared noise multiplier, the first-moment standard deviation is
increased by $\sqrt{1+C}$, where `C` is the per-record clipping
bound. Use the smallest bound the optimizer tolerates.

### Diagnostics

With `return_aux=True`, the returned auxiliary data includes per-group norms:

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=pg,
    batch_argnums=1,
    return_aux=True,
)
(grads, aux), clip_state = grad_fn(params, batch, state=clip_state)

# aux.group_norms: dict mapping group name → per-example norms tensor
for name, norms in aux.group_norms.items():
    print(f"{name}: mean_norm={norms.mean():.3f}")
```

### Adaptive per-group clipping

Per-group clipping works with `adaptive_clipped_grad`. Each group's threshold
adapts independently based on its own clipping rate:

```python
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.random import key

pg = per_group(params, self_attn=1.0, mlp=2.0)

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clipping_norm=pg,
    target_quantile=0.5,
    batch_argnums=(1, 2),
    normalize_by=batch_size,
    key=key(7),
)
```

`self_attn` and `mlp` adapt from separate noisy clipping-rate queries.
`pg.num_groups` is 2 here and remains correct if the grouping changes. Pass it
to `dpsgd_acc.adaclip()` to account for every independently adaptive group:

```python
expected_batch_size = sample_rate * dataset_size
step = dpsgd_acc.poisson(
    dpsgd_acc.adaclip(
        dpsgd_acc.gaussian(noise_multiplier),
        fraction_noise_std=0.05,
        expected_batch_size=expected_batch_size,
        num_groups=pg.num_groups,
    ),
    sample_rate,
)
```

## Automatic clipping (AUTO-S)

`auto_clipped_grad` implements the automatic clipping scheme of Bu et al.
(NeurIPS 2023). Instead of a hard threshold, each per-example gradient is
scaled by

    g̃_i = R * g_i / (||g_i|| + gamma)

so the output has L2 norm at most `R` by construction. There is no clip
threshold to tune; the effective step size is absorbed into the learning
rate.

```python
from opaque.dpsgd.clipping import auto_clipped_grad

grad_fn, clip_state = auto_clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    R=1.0,  # sensitivity bound (default 1.0)
    gamma=0.01,  # denominator stabilizer γ (default 0.01)
    normalize_by=batch_size,
)

grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
```

### Why it's "automatic"

The scaling `R / (||g|| + gamma)` depends only on the example's own
gradient, so the sensitivity `R / normalize_by` is guaranteed without any
adaptation or tuning. AUTO-S removes the clip-threshold hyperparameter
from the DP-SGD hyperparameter search.

### Compatibility with DP-FTRL / matrix-factorization noise

Because the per-record sensitivity bound `R` is constant and
data-independent, AUTO-S satisfies the constant per-step sensitivity
assumption that matrix-factorization privacy proofs rely on. The
returned `ClippedPytree.max_norm` is the same value (`R / normalize_by`)
on every step, so it flows through `mf_gaussian_noise` exactly as fixed clipping
does:

```python
from opaque.dpsgd.clipping import auto_clipped_grad
from opaque.dpftrl import band_mf_strategy, mf_gaussian_noise
from opaque.random import key

grad_fn, clip_state = auto_clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    R=1.0,
    normalize_by=batch_size,
)
noise_fn, noise_state = mf_gaussian_noise(
    params,
    band_mf_strategy(bands=4),
    n_steps=num_steps,
    noise_multiplier=noise_multiplier,
    key=key(0),
)

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    # ... optimizer step
```

`adaptive_clipped_grad`, by contrast, *cannot* be used with `mf_gaussian_noise`:
its threshold drifts across steps and the dispatcher's
`_validate_constant_max_norm` latch (rightly) rejects the resulting
varying `max_norm`.

### State and privacy accounting

The returned `AutoClipState` is a fixed marker. `R`, `gamma`, and
`normalize_by` are captured by the clipping closure, while `grads.max_norm`
carries the sensitivity. Privacy accounting is plain Gaussian DP-SGD — AUTO-S
introduces no extra data-dependent query:

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))
noisy_grads, noise_state = noise_fn(grads, noise_state)

step = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), sample_rate)
training = step * num_steps
eps = training.epsilon_at(1e-5)
```

### Per-group AUTO-S

Pass a `PerGroup` as `R` to scale each group independently:

```python
from opaque.dpsgd.clipping import auto_clipped_grad, per_group

pg = per_group(params, self_attn=1.0, mlp=2.0)

grad_fn, clip_state = auto_clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    R=pg,
    normalize_by=batch_size,
)
# grads.max_norm.effective = sqrt(sum R_k^2) / normalize_by
```

### When to choose AUTO-S vs. fixed or adaptive clipping

- Fixed (`clipped_grad`): best when you have a tuned clip norm already.
- AUTO-S (`auto_clipped_grad`): avoids tuning a clip threshold and adds
  no data-dependent accounting query.
- Adaptive (`adaptive_clipped_grad`): when you want the clip threshold to
  track a target quantile of gradient norms, at a small additional
  privacy cost for the quantile query.

## API reference

See [Clipping API Reference](../reference/clipping.md) for complete function
signatures, all parameters, and return types.
