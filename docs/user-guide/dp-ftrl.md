# DP-FTRL end-to-end

This guide walks through the full DP-FTRL pipeline: pick a
matrix-factorization strategy, calibrate the noise multiplier for the
*whole training run*, clip gradients, add correlated MF noise, run a
torchopt step, and checkpoint state. Every import on this page comes
from the `opake.dpftrl.*` public façade.

For DP-FTRL theory and a side-by-side comparison of mechanisms, see
[DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md). For the DP-SGD
counterpart, see [DP-SGD end-to-end](dp-sgd.md).

## Why DP-FTRL

DP-FTRL adds **correlated** Gaussian noise across training steps via
matrix factorization. Compared to independent noise at each step
(DP-SGD), correlated noise reduces variance on the **cumulative**
updates that the optimizer actually applies, for the same calibrated
privacy guarantee.

The trade-off: DP-FTRL accountants describe **whole training runs**.
The amplification factory takes `n_steps` at calibration time, the
strategy commits to a sensitivity / Gram matrix at construction time,
and the noise mechanism latches the per-step contribution bound on
the first call. Changing the training length, the per-step bound, or
the strategy mid-run breaks the privacy claim.

## Privacy and workload assumptions

Use the same strategy in `mf_gaussian_noise` and accounting, and pair
the sampler with its amplification factory. A strategy's workload model
(optimizer, schedule, and accumulation) affects utility; changing it
does not itself extend the documented privacy analysis.

## 1. Strategy choice

Pick a matrix-factorization strategy by mechanism. The strategy
object holds: coefficients defining the lower-triangular linear map
used for noise, the sensitivity (and sometimes a Gram matrix) used by
the accountant, and a streaming representation for efficient noise
generation.

```python
from opake.dpftrl.noise import (
    band_mf_strategy,    # numerical Toeplitz optimization
    blt_strategy,        # buffered linear Toeplitz, multi-epoch
    bisr_strategy,       # banded inverse square root
    bsr_strategy,        # banded square root, closed-form
    lambda_cgd_strategy, # PRNG replay, O(1) memory
    identity_strategy,   # no correlation; baseline
)

strategy = band_mf_strategy(bands=10)
```

See [DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md) for the
choice criteria.

## 2. Calibration

DP-FTRL accountants describe a whole training run. Build the strategy
first, then build the matching accounting mechanism using its
sensitivity / Gram matrix:

```python
import opake.accounting as acc                  # cross-cutting
import opake.dpftrl.accounting as dpftrl_acc    # DP-FTRL factories

# Same strategy that will go into mf_gaussian_noise below.
strategy = band_mf_strategy(bands=10)

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpftrl_acc.poisson(
        dpftrl_acc.mf_gaussian(nm, strategy),
        sample_rate=0.01,
        n_steps=1000,
    ),
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param
```

Three amplification factories under
`opake.dpftrl.accounting` — pick the one that matches your sampler:

- `dpftrl_acc.poisson(...)` — Poisson subsampling (cyclic-Poisson
  under banded MF).
- `dpftrl_acc.b_min_sep(...)` — b-min-separation participation
  pattern.
- `dpftrl_acc.balls_in_bins(...)` — fixed-partition participation.

Each amplification factory wraps a mechanism into a single
`DpProcess` describing the full training run. **Always pass the same
strategy object** to `mf_gaussian_noise` and the accounting factory — that is
how DP correctness is preserved.

### Whole-process ε reporting

DP-FTRL processes are whole-process mechanisms. Include the process once at
its declared `n_steps`; composing it once per optimizer step would over-count:

```python
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))
acct |= proc
epsilon = acct.epsilon_at(delta=1e-5)
```

The reported epsilon covers the full declared horizon. Prefix queries and
privacy-based early stopping are unsupported for these correlated mechanisms.

## 3. Clipping

Same engine clipping primitives as DP-SGD, just imported from
`opake.dpftrl.clipping`. Adaptive clipping is **not** available
under DP-FTRL — its threshold drifts across steps, violating the
constant per-step sensitivity assumption MF privacy proofs require.

```python
from opake.dpftrl.clipping import clipped_grad

def loss_fn(params, batch):
    return loss

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    argnums=0,
    batch_argnums=1,
    normalize_by=batch_size,
)
```

`auto_clipped_grad` (AUTO-S) is also available and compatible with
DP-FTRL — its per-record sensitivity bound
`sup ‖R · g / (‖g‖ + γ)‖ ≤ R` is constant in the input, satisfying
the same invariant MF accounting requires of fixed clipping.

Use scalar `clipped_grad` by default. Use `per_group` when groups have
materially different gradient scales, with bounds near their typical
norms; use `auto_clipped_grad` when avoiding a fixed threshold is more
important than tuning one.

## 4. Noise

`opake.dpftrl.noise.mf_gaussian_noise` injects correlated noise:

```python
from opake.dpftrl.noise import mf_gaussian_noise
from opake.random import key

# grad_template is the structure of clipped_grad's output —
# typically a ClippedPytree from a single warm-up call.
warmup_grads, _ = grad_fn(params, warmup_batch, state=clip_state)

noise_fn, noise_state = mf_gaussian_noise(
    warmup_grads,
    strategy,                       # same object you used in accounting
    n_steps=1000,                   # total training steps
    noise_multiplier=noise_multiplier,
    key=key(0),
)
```

`mf_gaussian_noise` reads the per-step contribution bound from the
`ClippedPytree` input on the **first call** and latches it for the
rest of the run. The bound is `noise_multiplier × max_norm`, so each
step must produce gradients with the same `max_norm` for the privacy
claim to hold.

For private second-moment estimation, pass
`second_moment_strategy=...`; the joint allocation is accounted with
the same mechanism PLD as the first-moment release. See
[Optimizers](optimizers.md) and [Noise](noise.md).

## 5. Sampling

DP-FTRL has its own sampler family under `opake.dpftrl.sampling`:

```python
from opake.dpftrl.sampling import (
    CyclicPoissonSampler,    # banded MF: cyclic Poisson subsampling
    BMinSepSampler,          # b-min-separation
    BallsInBinsSampler,      # fixed-partition
    SequentialBatchSampler,  # deterministic order, used by BLT
)

sampler = CyclicPoissonSampler(
    dataset, sample_rate=0.01, bands=10, n_steps=1000, key=key(42),
)
```

The sampler must match the amplification factory you used in
calibration.

## 6. Optimizer

Same surface as DP-SGD:

```python
from opake.optimizers import adamw

optimizer = adamw(lr=1e-3, noise_bias_correction=True)
opt_state = optimizer.init(params)
```

Private second-moment AdamW pairs with `mf_gaussian_noise(...,
second_moment_strategy=...)` — the noise mechanism produces a
`SecondMomentNoiseOutput` and the optimizer's DP-aware path consumes
it. See [Optimizers](optimizers.md) for the full second-moment story.

**Stability under the paired release.** Adam-family stability remains
workload-dependent. Watch clipping rate and gradient norms; lowering
the learning rate or using suitable per-group bounds can help.

## 7. End-to-end loop

```python
import torch
from opake.serialization import state_dict
from opake.functional import make_functional

fmodel, params = make_functional(model)
for step, batch in enumerate(sampler):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noised, opt_state, params)
    params = torchopt.apply_updates(params, updates)

# Checkpoint:
ckpt = {
    "params": params,
    "opt_state": opt_state,
    "clip_state": clip_state,
    "noise_state": noise_state,  # carries MF streaming-matrix state
}
torch.save(state_dict(ckpt), "step.pt")
```

## Runnable references

- [`examples/train_dpftrl.py`](https://github.com/JetBrains-Research/opake/blob/main/examples/train_dpftrl.py)
  — full DP-FTRL training script.
- `tests/integration/test_dpftrl_pipeline.py` — minimal smoke test
  exercising the same flow on a tiny LlamaConfig + LoRA model (and
  the Qwen2 variant).

## See also

- [Clipping](clipping.md) — fixed and AUTO-S variants
  (adaptive is DP-SGD-only).
- [Noise](noise.md) — `mf_gaussian_noise` shape, strategy types,
  per-step bound latching.
- [Sampling](sampling.md) — DP-FTRL sampler family.
- [Accounting](accounting.md) — `DpProcess`, the
  whole-process model, MF-specific composition.
- [Optimizers](optimizers.md) — second-moment integration.
- [DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md) — per-mechanism
  reference pages.
- [DP-SGD end-to-end](dp-sgd.md) — the per-step companion pipeline.
