# DP-SGD end-to-end

This guide covers the DP-SGD pipeline: calibrate a noise multiplier,
clip per-example gradients, add Gaussian noise, update parameters, and
checkpoint state.

For the conceptual deep-dives on each component, see the topic pages
under [User Guide](index.md): [clipping](clipping.md),
[noise](noise.md), [sampling](sampling.md), [accounting](accounting.md),
[optimizers](optimizers.md). For the full DP-FTRL counterpart, see
[DP-FTRL end-to-end](dp-ftrl.md).

## Why DP-SGD

DP-SGD adds independent Gaussian noise to a clipped gradient at every
training step. The privacy accountant composes per-step costs into a
total budget: each call to the noise mechanism is a separate
`DpProcess`, multiplied by the number of training steps.

Compared with DP-FTRL, DP-SGD needs no fixed training horizon or
matrix strategy; its independent noise can be less useful on cumulative
updates.

## 1. Calibration

Calibrate the noise multiplier to your privacy budget before training:

```python
import opaque.accounting as acc            # cross-cutting (compose, calibrate)
import opaque.dpsgd.accounting as dpsgd_acc  # DP-SGD factories

dataset_size = 50_000
batch_size = 256
sample_rate = batch_size / dataset_size
num_steps = 1000

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param
```

`dpsgd_acc.gaussian(nm)` is the per-step Gaussian mechanism;
`dpsgd_acc.poisson(..., sample_rate)` wraps it with Poisson
amplification; `* num_steps` composes across the run. The resulting
`DpProcess` is what `acc.calibrate` solves for the noise multiplier.

## 2. Clipping

`opaque.dpsgd.clipping.clipped_grad` wraps a per-example loss
function in `vmap(grad(...))` semantics, clips each per-example
gradient to a fixed norm, and sums:

```python
from opaque.dpsgd.clipping import clipped_grad

def loss_fn(params, batch):
    # ... per-example loss (NO mean over batch) ...
    return loss

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    argnums=0,
    batch_argnums=1,
    normalize_by=batch_size,
)
```

For automatic threshold tuning across steps:
`opaque.dpsgd.clipping.adaptive_clipped_grad`
([Andrew et al., 2021](https://arxiv.org/abs/1905.03871)).
For AUTO-S smooth scaling: `opaque.dpsgd.clipping.auto_clipped_grad`
([Bu et al., 2023](https://arxiv.org/abs/2206.07136)).

## 3. Noise

`opaque.dpsgd.noise.gaussian_noise` adds calibrated Gaussian noise to
the clipped gradient sum:

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key, split

key_sampling, key_noise = split(key(42), num=2)

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key_noise,
)
```

For bounded noise support, pass `bound=...` to
`opaque.dpsgd.noise.gaussian_noise`. It uses inverse-CDF sampling and accepts a
positive scalar (symmetric `[-B, B]`) or a `(low, high)` tuple. The standard
Gaussian accountant does not cover the bounded-output mechanism.

### Projected JME for Adam

To release Adam's clean aggregate square separately instead of squaring the
noised gradient, use `jme_noise`:

```python
from opaque.dpsgd.noise import jme_noise
from opaque.dpsgd.noise.types import JmeAllocation

noise_fn, noise_state = jme_noise(
    noise_multiplier=noise_multiplier,
    aggregate_norm=1.0,
    allocation=JmeAllocation.first_variance_cap(1.5),
    key=key_noise,
)
```

The clipping function remains single-stream. After clipping and any distributed
sum, `jme_noise` projects the normalized aggregate to `aggregate_norm`, forms
its clean element-wise square, and noises both values. The resulting
`SecondMomentNoiseOutput` is routed by supported Opaque optimizers without
first-stream variance subtraction.

The fixed denominator in `normalize_by` is part of the sensitivity:
`Delta = clipping_norm / normalize_by`. JME requires scalar clipping, a public
projection radius and allocation, standard Gaussian noise, and plain independent
Poisson sampling. See [Projected JME](../mechanisms/dp-sgd/jme.md) for the exact
add/remove sensitivity and noise scales.

## 4. Sampling

DP-SGD pairs with Poisson subsampling:

```python
from opaque.dpsgd.sampling import PoissonSampler

sampler = PoissonSampler(
    dataset, sample_rate=sample_rate, key=key_sampling,
)
```

`PoissonSampler` accepts `truncated_batch_size=` for the
truncated-Poisson variant; calibration must use
`dpsgd_acc.poisson(..., truncated_batch_size=..., dataset_size=...)`
to match.

## 5. Optimizer

```python
from opaque.optimizers import adamw

optimizer = adamw(
    lr=1e-3,
    weight_decay=0.0,
    noise_bias_correction=True,  # DP-aware second-moment correction
)
opt_state = optimizer.init(params)
```

`opaque.optimizers` ships `adamw`, `adam`, `sgd`, `radam`,
`adafactor`, `lion`, `ademamix`, `schedule_free`, plus a few torchopt
re-exports. The `noise_bias_correction=True` flag corrects the
biased second moment that arises when the optimizer sees noised
gradients.

## 6. End-to-end loop

```python
import torch
from opaque.functional import make_functional
from opaque.serialization import state_dict

fmodel, params = make_functional(model)
for step, batch in enumerate(sampler):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noised, opt_state, params)
    params = torchopt.apply_updates(params, updates)

# Checkpoint at the end (or any step):
ckpt = {
    "params": params,
    "opt_state": opt_state,
    "clip_state": clip_state,
    "noise_state": noise_state,
}
torch.save(state_dict(ckpt), "step.pt")
```

Restore from the same flat state dict with
`opaque.serialization.from_state_dict`.

## Runnable references

- [`examples/train_dpsgd.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd.py)
  — full causal-LM training script.
- `tests/integration/test_dpsgd_pipeline.py` — minimal smoke test
  exercising the same flow on a tiny LlamaConfig + LoRA model
  (and the Qwen2 variant for the real-HF case).

## See also

- [Clipping](clipping.md) — fixed, AUTO-S, adaptive variants and
  per-group norms.
- [Noise](noise.md) — Gaussian (optionally bounded), when to choose
  which, plus projected JME.
- [Sampling](sampling.md) — Poisson sampler details and the
  truncated-Poisson trade-off.
- [Accounting](accounting.md) — `DpProcess`, calibration, budgets.
- [Optimizers](optimizers.md) — DP bias correction and the
  second-moment story.
- [DP-SGD mechanisms](../mechanisms/dp-sgd/index.md) — Gaussian and JME
  reference pages.
- [DP-FTRL end-to-end](dp-ftrl.md) — the correlated-noise companion
  pipeline.
