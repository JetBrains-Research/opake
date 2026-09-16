# Optimizers

Opaque ships its own functional optimizer library at
[`opaque.optimizers`](../reference/optimizers.md): Opaque-built factories with a
common wrapper-aware update surface (`sgd`, `adam`, `adamw`, `radam`, `lion`,
`ademamix`, `adafactor`, `rmsprop`, `adagrad`, `adadelta`, `schedule_free`).
Every factory carries DP-aware behaviour selectable at construction time and
activated by the metadata wrappers (`NoisedPytree`, `SecondMomentNoiseOutput`)
landing in `update()`.

All factories return [TorchOpt](https://torchopt.readthedocs.io/)
`GradientTransformation`s, so they compose with TorchOpt's lower-level
transforms and `torchopt.apply_updates`. DP-aware paths (DP-AdamW-BC, private
second moments, Adagrad's mandatory variance subtraction) are selected by
passing `NoisedPytree` or `SecondMomentNoiseOutput` updates.

## Why functional optimizers

Opaque's gradient pipeline is functional: every function takes state
in and returns new state out.  The optimizer factories follow the same
pattern:

```python
import torchopt
from opaque.optimizers import adamw

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Explicit state in -> state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

No hidden mutable state.  Every piece of the training loop is explicit.

## Complete training loop

```python
import torchopt
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import adamw
from opaque.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=noise_multiplier, key=key(42))

# DP-aware AdamW; pass `noise_bias_correction=True` to enable the
# φ-EMA subtraction once the LR is tuned for the workload.
optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

## Choosing an optimizer

**Adafactor** (`adafactor`) is the recommended default for DP training.
Its per-tensor relative-step normalization (factored
`v_row × v_col`) auto-scales effective learning rates per parameter
group, which makes it both LR-robust and naturally resistant to
DP noise inflation of the second moment (see [Choosing from measurements](#choosing-from-measurements)
below). Pass `NoisedPytree` updates from a DP noise mechanism:

```python
from opaque.optimizers import adafactor

optimizer = adafactor(lr=5e-4, weight_decay=0.01)
# `update_rms_clip=1.0` (the default) clips one model-wide/global RMS,
# rather than clipping each parameter leaf independently.
```

**AdamW** (`adamw`) is a good alternative when you want first-moment
momentum, but is more LR-sensitive than Adafactor: tuning matters.
Use `noise_bias_correction=True` only when the LR has been tuned for
the workload (see [Choosing from measurements](#choosing-from-measurements)):

```python
from opaque.optimizers import adamw

optimizer = adamw(lr=1.5e-4, weight_decay=0.01)
# Plain Adam (no decoupled WD): adamw(..., decoupled_weight_decay=False).
# StableAdamW (model-wide/global RMS-clipped update, not per-leaf):
# adamw(..., update_rms_clip=1.0).
```

**SGD** (`sgd`) is the canonical DP baseline. No second
moment is corrected, but the Opaque wrapper accepts `NoisedPytree` updates so
the training loop stays uniform. `E[g + ξ] = g` and momentum's variance is
bounded. Good debugging baseline:

```python
from opaque.optimizers import sgd
optimizer = sgd(lr=0.01, momentum=0.9)
```

**RMSprop** (`rmsprop`) is adaptive but cheaper
than Adam (no first moment). `noise_bias_correction=True` enables
the same flavor of φ-EMA subtraction as AdamW; it is off by default.
Enable it to ablate:

```python
from opaque.optimizers import rmsprop
optimizer = rmsprop(lr=1e-2, alpha=0.99, noise_bias_correction=True)
```

**Adagrad** (`adagrad`) is for sparse-gradient
settings. Its accumulator does not decay, so under DP noise `v_acc`
absorbs `t·σ²` over training; `noise_bias_correction=True`
subtracts a matching cumulative term:

```python
from opaque.optimizers import adagrad
optimizer = adagrad(lr=1e-2, noise_bias_correction=True)
```

Whether the correction helps in practice depends on the workload —
ablate against `noise_bias_correction=False`.

**AdEMAMix**, **Adafactor**, **Lion**, **schedule-free** — see the
[API reference](../reference/optimizers.md#whats-in-opaqueoptimizers) for
their DP modes.

## The second-moment problem in DP training

Adam-style optimizers compute a second moment as
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.  Under DP, the optimizer
sees noised gradients $\tilde{g}_t = g_t + \xi_t$, so:

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t^2 + 2g_t \xi_t + \xi_t^2)$$

The $\xi_t^2$ term inflates the denominator, shrinking effective
learning rates.  The $2g_t \xi_t$ cross-term averages out but adds
variance.  Gaussian noise with variance $\sigma^2$ adds an expected
bias of $\sigma^2$ to every component of $\hat{v}_t$.

Opaque provides two independent corrections, both selected at
`update()` time.  They address the same problem from different
angles and **must not be combined** in the same call — using both
would double-correct the second moment.

### `NoisedPytree`: bias correction by variance subtraction

**Idea:** the noise variance $\Phi_t = \sigma_t^2$ is *known* (we
chose it), so we can subtract it from the biased estimate.

For Adam-family optimizers (`adamw`, `ademamix`):

$$\phi_t = \beta_2 \phi_{t-1} + (1-\beta_2) \Phi_t, \qquad
\hat{v}^{\text{corrected}}_t = \max\!\bigl(\hat{v}_t - \hat{\phi}_t,\; \gamma\bigr)$$

This is Algorithm 2 from
[Chooi et al. (arXiv:2511.07843)](https://arxiv.org/abs/2511.07843).
Cheap (one extra scalar EMA), no changes to the noise mechanism, works
with any i.i.d. Gaussian noise source.

For `rmsprop`: same idea, but $\phi$ accumulates with the same EMA
decay $\alpha$ as $v$ (and there's no `(1−α^t)` divide because
RMSprop doesn't bias-correct $v$ either).  Subtracting $\phi$ from
$v$ directly gives the unbiased estimate.

For `adagrad`: cumulative $\Phi_\text{acc} = \sum_s \sigma_s^2$
(no decay) matches the cumulative $v_\text{acc}$.  **Mandatory** for
DP-Adagrad — without it, the denominator runs away with $t \cdot \sigma^2$
of accumulated noise variance and learning halts.

When a raw pytree update is passed, each optimizer reduces to its standard
math. `NoisedPytree` updates supply the realized per-step σ metadata.

```python
from opaque.optimizers import adamw

# Without correction — standard AdamW math.
optimizer = adamw(lr=1e-3, weight_decay=0.01)

# With correction — pass NoisedPytree updates from gaussian_noise().
optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_bias_correction=True)
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
```

### Projected JME aggregate squares

Instead of squaring the noisy gradient, `jme_noise` separately noises a clean
aggregate square:

\[
a_t=\sum_i\operatorname{clip}(g_{t,i},C)/z,\qquad
x_t=\Pi_R(a_t),
\]

\[
\widetilde x_t=x_t+\xi_t,\qquad
\widetilde{x_t^2}=x_t\odot x_t+\eta_t.
\]

The optimizer receives both values and uses each for its corresponding EMA:

\[
\mu_t=\beta_1\mu_{t-1}+(1-\beta_1)\widetilde x_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)\widetilde{x_t^2}.
\]

The separately noised square is routed through `SecondMomentNoiseOutput`; the
optimizer does not subtract first-stream noise variance in this branch. JME and
noise-variance bias correction are alternatives.

With coupled Adam/L2 weight decay, Opaque deterministically transforms the two
JME streams to estimate `(g + weight_decay * params)²`. Decoupled AdamW applies
weight decay after moment scaling and leaves the JME square unchanged.

Projected JME is DP-SGD-only. It requires scalar clipping, a fixed public
aggregate radius, an explicit allocation policy, standard Gaussian noise, and
plain independent Poisson sampling. See
[Projected JME](../mechanisms/dp-sgd/jme.md).

### Choosing from measurements

No reproducible cross-workload benchmark is available. Tune the optimizer,
learning rate, schedule, and bias correction together.

### When to use which

The bias-correction (BC) and JME paths target the same
v-update bias by different means; they are alternatives.  BC default is
**off** — turn it on once you've tuned LR; see
[Choosing from measurements](#choosing-from-measurements)
above.

| Scenario | Optimizer | Notes |
|---|---|---|
| DP-SGD default (LM fine-tuning) | `adafactor` | Recommended default — relative-step normalization makes BC a no-op |
| Adam-family without LR tuning | `adamw` (BC off) | BC-off auto-shrinks effective LR; forgiving |
| Adam-family with tuned LR | `adamw(noise_bias_correction=True)` | BC reaches min faster at the right LR |
| LR-robust alternative to AdamW | `radam` | Rectification gate flattens LR sensitivity to ±0.1% across ±3× |
| Sign-based, lowest memory | `lion` | No second moment; sharp LR optimum (~AdamW LR / 10) |
| DP-FTRL without an Adam-family update | `sgd` | No second moment to correct |
| DP-SGD with a separately noised clean square | `adamw(...) + jme_noise(...)` | Uses projected JME instead of squaring noised gradients |
| DP-FTRL with Adam | `adamw(...)` | Single correlated gradient stream; JME is not supported for repeated MF rows |
| Sparse gradients under DP | `adagrad` | `noise_bias_correction=True` is essentially mandatory — without it the un-decaying accumulator absorbs `t·σ²` |
| RMSprop user under DP | `rmsprop` | LR-sensitive; tune carefully (lr=1e-4 worked in our sweep, 5e-4 diverged) |

## AdamW with projected JME

```python
from opaque.dpsgd.noise import jme_noise
from opaque.dpsgd.noise.types import JmeAllocation
from opaque.optimizers import adamw
from opaque.random import key

noise_fn, noise_state = jme_noise(
    noise_multiplier=noise_multiplier,
    aggregate_norm=1.0,
    allocation=JmeAllocation.first_variance_cap(1.5),
    key=key(42),
)

optimizer = adamw(lr=lr_schedule_fn, betas=(0.9, 0.999), weight_decay=0.01)
opt_state = optimizer.init(params)
```

### Training loop

```python
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    moments, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(
        moments, opt_state,
        params=params,
    )
    params = torchopt.apply_updates(params, updates)
```

### Accounting

After whitening by its two runtime scales, projected JME is dominated by a
sensitivity-one Gaussian release at the same noise multiplier. Plain independent
Poisson sampling therefore uses the standard DP-SGD chain:

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q)
process = step * num_steps
```

### CLI example

```bash
python examples/train_dpsgd.py --preset smoke --optimizer adamw \
  --jme --jme-aggregate-norm 1.0 \
  --jme-allocation first_variance_cap \
  --jme-max-first-variance-ratio 1.5
```

## DP-specific optimizer considerations

### Why Adam works well with DP

In DP training, different parameters receive different signal-to-noise
ratios.  Parameters with large true gradients (e.g. biases, early
layers) get a better ratio than parameters with small true gradients
(e.g. deep attention layers).  Adam's per-parameter adaptive learning
rate naturally compensates for this, scaling up updates where the
signal is strong relative to noise and scaling down where it is weak.
SGD applies the same learning rate everywhere, which is suboptimal
when noise levels vary across parameters.

### Why vanilla Adagrad fails under DP

Adagrad's cumulative second moment $v_t = \sum_s g_s^2$ has no decay.
Under DP, $E[g̃_s^2] = g_s^2 + \sigma^2$, so:

$$E[v_t] = \sum_s g_s^2 \;+\; t \sigma^2$$

The noise term grows linearly forever.  After enough steps the
denominator is dominated by accumulated noise; updates become
effectively random and the per-coordinate LR shrinks indefinitely.
`adagrad(noise_bias_correction=True, ...)` subtracts a parallel
cumulative $\Phi_\text{acc}$ to counter this.  Whether the corrected
denominator is preferable to vanilla Adagrad in practice depends on
the workload — ablate against `noise_bias_correction=False` rather
than treating BC as a default.

### RAdam under DP

RAdam (Liu et al., [arXiv:1908.03265](https://arxiv.org/abs/1908.03265))
keeps Adam's `m`/`v` accumulators but post-multiplies the update by a
variance-rectification factor `r_t` that depends only on `(β₂, t)`.  Below
the rectification threshold (`ρ_t ≤ 5`, roughly the first
`O(1/(1-β₂))` steps) the rule degenerates to SGD-of-momentum — `v` is
not consumed at all.  This makes the warmup phase naturally DP-robust:
the noise in `v` cannot affect the update because it isn't read.

Once `ρ_t > 5`, the standard Adam DP-BC story applies — subtract a
β₂-EMA of `σ²` from `v̂` before the sqrt. `radam`
advances the φ-EMA every step (warmup included) so that at the first
rectified step the correction reflects all prior noise contributions to
`v`, not just the current step.

### Adadelta under DP

Vanilla Adadelta (Zeiler 2012) is learning-rate-free: the update is
the per-element ratio `RMS[Δx]_{t-1} / RMS[g]_t` times the gradient.
Under DP both EMAs accumulate noise:

- `E[g²]_t` inherits the Adam-shaped `σ²` offset.
- `E[Δx²]_t` accumulates `coef_t² · σ²` per element because
  `Δx_t = -coef_t · g̃_t` is linear in the noised gradient.

`adadelta` maintains two parallel φ-EMAs at the
same decay `ρ` and subtracts both biases. `φ_g` is scalar (or
per-group), while `φ_dx` is per-element because the per-step update-noise
variance varies element-wise even when `σ` is scalar. This adds one scalar
(or per-group) value and one parameter-shaped value to vanilla Adadelta's
state.

The Adadelta two-EMA derivation has no published prior; it falls out
of direct propagation of Gaussian variance through the linear scaling.

### Why Adamax isn't shipped

Adamax tracks $v_t = \max(\beta v_{t-1}, |g_t + \xi|)$.  The max
operator absorbs the half-normal noise mean ($\sigma\sqrt{2/\pi} \approx
0.8\sigma$) and never releases it — even when the true gradient is
tiny, the per-coordinate denominator is permanently floored by
accumulated noise magnitude.  No clean DP-aware variant exists.
For non-DP use, `from torchopt import adamax` directly.

### Schedule-free under DP

Schedule-free averages optimizer iterates. Its DP variance reduction depends
on their covariance, so there is no universal $\sqrt{n}$ reduction.

```python
from opaque.optimizers import adamw, schedule_free

optimizer = schedule_free(
    adamw(lr=1e-3, noise_bias_correction=True), beta=0.9
)
opt_state = optimizer.init(params)

# Train as usual: trainer treats `params` as y_t.
for batch in dataloader:
    ...
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

# At save / eval time, read the published x_t directly off the state:
eval_params = opt_state.x
```

### Weight decay and privacy

Weight decay (L2 regularization) does not consume privacy budget
because it is applied to the model parameters, not the data.  AdamW's
decoupled weight decay is implemented as `params = params - wd *
params` after the gradient step — this is a deterministic function of
the current parameters and does not depend on the training data.

### Gradient accumulation

Opaque does not use gradient accumulation in the HF-style `step ÷ K`
sense.  Instead, `clipped_grad` processes the entire batch (possibly
in microbatches) and returns the sum of clipped gradients in one
call.  You pass the full noisy gradient directly to the optimizer —
no manual accumulation is needed.

## Learning rate schedules

Standard LR schedules work with DP training.  Linear warmup followed
by cosine decay is common for DP fine-tuning.  Use Opaque's
[scheduling primitives](../reference/schedules.md):

```python
from opaque.optimizers import adamw
from opaque.scheduling import cosine_schedule, with_warmup

decay = cosine_schedule(
    init_value=1e-3, end_value=0.0,
    transition_steps=900, transition_begin=100,
)
schedule = with_warmup(decay, transition_steps=100)
optimizer = adamw(lr=schedule, weight_decay=0.01, noise_bias_correction=True)
```

A typical warmup is 5-10% of total training steps.  The warmup helps
stabilize early updates when the model has not yet adapted to the
noisy gradient signal.

For schedule-free training, you don't need an external schedule —
the wrapper's averaging is the implicit schedule.

## Checkpoint round-tripping

The optimizer state is a `torchopt` chain tuple of dataclasses; flatten
it to a serializable dict via `opaque.serialization`:

```python
from opaque.optimizers import adamw
from opaque.serialization import from_state_dict, state_dict

opt = adamw(lr=1e-3, weight_decay=0.01, noise_bias_correction=True)
state = opt.init(params)
# ... train ...

# Save
torch.save(state_dict(state), "opt.pt")

# Restore: re-init a template, then build a new state from the saved dict.
template = opt.init(params)
state = from_state_dict(template, torch.load("opt.pt"))
```

Forward-compatible: paths missing from a saved dict keep the template's
value, so optimizers that gain new state fields between releases load
cleanly from older checkpoints.

## Practical notes

**Do not clip optimizer updates.** Opaque clips *gradients* before the
optimizer sees them.  Clipping after the optimizer would distort the
adaptive state.

**DDP compatibility.** In distributed training, optimizer states stay
synchronized automatically because `optimizer.update` is a pure
function and all ranks receive identical noisy gradients after
`sum_gradients` and noise addition (using the same key on all ranks).
No explicit state synchronization is needed.

**`NoisedPytree` is the unified contract.** Every optimizer that has a
noise-aware path reads realized `noise_stddev` metadata from `NoisedPytree`
updates. The trainer does not pass per-step optimizer kwargs; it just feeds
the output of the DP noise mechanism into `optimizer.update()`.

## API reference

See [Optimizers API Reference](../reference/optimizers.md) for full factory
signatures, knob descriptions, and the `serialization` /
`schedule_free` submodule helpers.
