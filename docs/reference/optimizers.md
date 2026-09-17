# Optimizers

Opake ships its own functional optimizer library at
`opake.optimizers`: Opake-built factories with
DP-aware paths, plus a curated set of `torchopt` re-exports for the
stateless primitives where vanilla behaviour is acceptable under DP
noise. All factories return
[TorchOpt](https://torchopt.readthedocs.io/) `GradientTransformation`s
and route DP metadata from `NoisedPytree` or `SecondMomentNoiseOutput` updates.

---

## API shape

Functional: no hidden mutable state, explicit
`(updates, new_state) = optimizer.update(grads, state)` interface.

```python
import torchopt
from opake.optimizers import adamw

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Explicit state in, state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

---

## What's in `opake.optimizers`

### Opake-built factories (DP-aware)

Noise-aware factories accept `noise_bias_correction=True` to subtract the
known Gaussian variance carried by `NoisedPytree` updates (off by default;
flip on to ablate against vanilla). Where applicable they also route
`SecondMomentNoiseOutput` for private squared-gradient substitution —
an alternative answer to the same v-update bias.

| Factory | DP-aware mode | When to use |
|---|---|---|
| **`sgd`** | No second moment; accepts `NoisedPytree` and ignores σ metadata | Canonical DP baseline |
| **`adam`** | Original Adam/L2 variant with the same BC/private-moment paths as AdamW | Adam parity without decoupled WD |
| **`adamw`** | Optional φ-EMA on v̂ when `noise_bias_correction=True`; private second moments via `SecondMomentNoiseOutput` | Adam-family fine-tuning when first-momentum and decoupled WD matter |
| **`radam`** | φ-EMA on v̂ in the rectified phase (`ρ_t > 5`); SGD-of-momentum in warmup | Long runs where you want RAdam's variance rectification with DP correction |
| **`adadelta`** | Two-EMA BC: φ_g on `E[g²]` and per-element φ_dx on `E[Δx²]` | LR-free DP optimizer; useful when learning-rate tuning is hard |
| **`ademamix`** | φ-EMA on v̂ + private second moments | Long-horizon training (slow EMA captures long-range signal) |
| **`adafactor`** | Factored second moment; optional per-factor φ-EMA when `noise_bias_correction=True` | Recommended default for DP LM fine-tuning (relative step scaling); see [user guide](../user-guide/optimizers.md) |
| **`lion`** | No second-moment correction | Smaller state than Adam; vanilla works under noise |
| **`rmsprop`** | φ-EMA on v + private second moments | Adaptive without first moment; cheaper than Adam |
| **`adagrad`** | cumulative `Φ_acc` subtraction | Sparse-gradient settings; **the correction is mandatory** — vanilla Adagrad's denominator runs away under DP noise |
| **`schedule_free`** | post-processing (transparent forward) | Wrapper around any base optimizer; replaces external LR schedules |

Constructor knobs vary per optimizer (e.g. `decoupled_weight_decay`,
`update_rms_clip` on `adamw`, `ademamix`, and `adafactor`). Every
`update_rms_clip` is model-wide/global over the full update pytree, not
per-leaf; see the docstrings.

### Foundational references

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  — Kingma and Ba (2015); the basis for `adam`.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  — Loshchilov and Hutter (2019); introduces AdamW.
- [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013)
  — Wortsman et al. (2023); introduces StableAdamW's update-RMS clipping.
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)
  — Duchi, Hazan, and Singer (2011); introduces Adagrad.
- [Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
  — Tieleman and Hinton (2012) lecture notes; introduces RMSprop.

### Not exposed

- `torchopt.adamax` — the max-norm tracker `v_t = max(β v_{t-1}, |g_t + ξ|)`
  permanently absorbs the half-normal noise mean (`σ·√(2/π)`); per-coordinate
  effective LR is floored by accumulated noise magnitude regardless of the
  true gradient.  No clean DP-aware variant; for non-DP use,
  `from torchopt import adamax` directly.

---

## DP-aware update metadata

Noise-aware behaviour is selected by the update object.

### `NoisedPytree`

`NoisedPytree` carries the realized per-step noise σ. Each optimizer activates
whatever noise-aware machinery it has when `noise_bias_correction=True`:

| Optimizer | What `NoisedPytree.noise_stddev` does |
|---|---|
| `adamw`, `ademamix` | β₂-EMA of σ², subtracted from v̂ before sqrt (Chooi et al., [arXiv:2511.07843](https://arxiv.org/abs/2511.07843)) |
| `radam` | β₂-EMA of σ² advanced every step; subtracted from v̂ only in the rectified phase (`ρ_t > 5`).  Warmup phase uses SGD-of-momentum (no v) and is naturally DP-robust. |
| `adadelta` | Two parallel ρ-EMAs: `φ_g` of σ² (subtracted from `E[g²]`) and `φ_dx` of `coef² σ²` per element (subtracted from `E[Δx²]`).  No published prior — derived by propagating Gaussian variance through the linear scaling step. |
| `rmsprop` | α-EMA of σ², subtracted from v before sqrt (no `(1−α^t)` divide; v and φ accumulate at the same rate) |
| `adagrad` | Cumulative Σ σ², subtracted from v_acc before sqrt (no decay in either) |
| `lion` | No second-moment correction |
| `schedule_free` | Forwarded transparently to the wrapped base |

```python
optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_bias_correction=True)
updates, state = optimizer.update(noisy_grads, state, params=p)
```

Raw pytree updates use standard optimizer math.

### `noisy_squared_grads`

Substitutes a privately-estimated `g²` stream in place of squaring the
(already noised) gradient. `mf_gaussian_noise(..., second_moment_strategy=...)` returns
a paired output that Opake optimizers route automatically:

```python
from opake.dpftrl.noise import blt_strategy, mf_gaussian_noise
from opake.optimizers import adamw

strategy = blt_strategy(max_buffers=10)
second_strategy = blt_strategy(max_buffers=10)
noise_fn, noise_state = mf_gaussian_noise(
  grad_template,
  strategy,
  n_steps=1000,
  noise_multiplier=noise_multiplier,
  key=key(42),
  second_moment_strategy=second_strategy,
)

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Per-step:
noisy_grads, noise_state = noise_fn(grads, noise_state)
updates, opt_state = optimizer.update(
  noisy_grads, opt_state, params=p,
)
```

Reference: Kalinin et al., [arXiv:2502.06597](https://arxiv.org/abs/2502.06597).

`NoisedPytree` metadata and private second-moment outputs are mutually exclusive
at any single `update()` call; passing both routes raises `ValueError`.

---

## Complete pattern (DP-SGD)

```python
import torchopt
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.optimizers import adamw
from opake.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))

# Optimizer with DP-AdamW-BC active
optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_bias_correction=True)
opt_state = optimizer.init(params)

for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

---

## Schedule-free wrapper

`schedule_free` wraps any base `GradientTransformation` (Opake-built
or TorchOpt) with Defazio's schedule-free averaging
([arXiv:2405.15682](https://arxiv.org/abs/2405.15682)). Three weight
sequences internally: `z` (raw iterate), `x` (Polyak-Ruppert average,
the published params), `y = (1-β)z + βx` (forward-pass weights).

```python
from opake.optimizers import adamw, schedule_free

optimizer = schedule_free(adamw(lr=1e-3), beta=0.9, warmup_steps=100)
opt_state = optimizer.init(params)

for step in range(num_steps):
    # Trainer treats params as y_t; wrapper internally maintains z, x.
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

# At save / eval time, read the published x_t directly off the state:
eval_params = opt_state.x
```

**DP-utility note**: under DP, `x_t = (1/n) Σ z_s` is a Polyak-Ruppert
average of noised iterates. When per-step iterate noise is approximately
independent, `Var[x_n] ≈ Var[z]/n` — the published checkpoint has
significantly lower noise than the final iterate at the same privacy
budget.  This is a real DP-utility win specific to averaging-based
methods.

---

## Serialization

`state_dict` / `from_state_dict` live in `opake.serialization`. They walk
chain state, encoding every tensor leaf and Python primitive into a flat
`{path: value}` dict ready for `torch.save`. Restore returns a **new**
object and never mutates the template.

```python
from opake.optimizers import adamw
from opake.serialization import from_state_dict, state_dict

opt = adamw(lr=1e-3, weight_decay=0.01)
state = opt.init(params)
# ... train ...

# Save
torch.save(state_dict(state), "opt.pt")

# Restore — template must have the same shape (init from same params).
template = opt.init(params)
state = from_state_dict(template, torch.load("opt.pt"))
```

Forward-compatible: paths missing from the saved dict keep the
template's value, so optimizers that gain new state fields between
releases load cleanly from older checkpoints.

---

## DDP compatibility

When using `torch.nn.parallel.DistributedDataParallel`, Opake's
functional gradient pipeline runs *inside* each rank. DDP handles the
all-reduce of noisy gradients across ranks; the optimizer state stays
synchronised because `optimizer.update` is a pure function and all
ranks receive identical noisy gradients after `sum_gradients` and noise
addition with the same key on all ranks.

Use `local_shard()` to partition the dataset across ranks and pass a
per-rank key via `fold_in(key, rank)` to each `PoissonSampler`.

---

## See also

- [Optimizers User Guide](../user-guide/optimizers.md) — concept-level
  explanation of the second-moment problem, when to use which DP mode.
- [Gradient Clipping API](clipping.md) — `clipped_grad`,
  `adaptive_clipped_grad`, `auto_clipped_grad`, `PerGroup`.
- [Schedules API](schedules.md) — LR schedules that pair with
  optimizer factories.
- [TorchOpt Documentation](https://torchopt.readthedocs.io/) — for
  re-exports and lower-level transforms.
