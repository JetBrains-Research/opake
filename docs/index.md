# Opake

**Opake** (/oʊˈpeɪk/, archaic *opaque*) — Functional DP-SGD and DP-FTRL for PyTorch.

Opake provides composable primitives for differentially private
model training in PyTorch. Built on `torch.func`, every component
uses explicit state — no hooks, no subclassing, no hidden mutation.

Opake ships two complementary training pipelines:

- **[DP-SGD](user-guide/dp-sgd.md)** — independent Gaussian noise at
  every step, per-step privacy composition. The standard DP training
  recipe.
- **[DP-FTRL](user-guide/dp-ftrl.md)** — correlated noise across the
  whole training run via matrix factorization. Reduces effective
  noise on cumulative updates at the cost of fixing the training
  length in advance.

Both pipelines share the same primitives — clipping, noise, sampling,
optimizers, and accounting — but use them differently. Pick the track
that matches your problem.

## Per-example gradient clipping

`clipped_grad` computes per-example gradients via `vmap` + `grad`,
clips each to a maximum L2 norm, and sums the result.

```python
# DP-SGD context:
from opake.dpsgd.clipping import clipped_grad

# DP-FTRL context:
# from opake.dpftrl.clipping import clipped_grad

grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, argnums=0, batch_argnums=1,
    normalize_by=batch_size,
)
grads, clip_state = grad_fn(params, batch, state=clip_state)
```

## Noise injection

DP-SGD adds independent Gaussian noise scaled to `grads.max_norm`:

```python
from opake.dpsgd.noise import gaussian_noise
from opake.random import key

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

DP-FTRL adds correlated noise via a matrix-factorization strategy:

```python
from opake.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opake.random import key

strategy = band_mf_strategy(bands=10)
noise_fn, noise_state = mf_gaussian_noise(
    grads_template, strategy,
    n_steps=1000,
    noise_multiplier=noise_multiplier, key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

## Privacy accounting

Composable `DpProcess` objects built on a Rust PLD engine. Mechanisms
compose with `*` (repeat) and `|` (heterogeneous composition).

```python
import opake.accounting as acc                # cross-cutting
import opake.dpsgd.accounting as dpsgd_acc    # DP-SGD per-step factories
import opake.dpftrl.accounting as dpftrl_acc  # DP-FTRL whole-process factories
from opake.dpftrl.noise import band_mf_strategy

# DP-SGD: per-step Gaussian + Poisson, composed across N steps.
dpsgd_proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.01) * 1000

# DP-FTRL: whole-process MF + Poisson at calibration time.
strategy = band_mf_strategy(bands=10)
dpftrl_proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01, n_steps=1000,
)

print(f"DP-SGD ε    = {dpsgd_proc.epsilon_at(1e-5):.4f}")
print(f"DP-FTRL ε   = {dpftrl_proc.epsilon_at(1e-5):.4f}")
```

## Distributed training

DDP-aware: pass the same key on all ranks for synchronized noise,
use `sum_gradients` for cross-rank reduction.

```python
from opake.distributed import sum_gradients

grads, clip_state = grad_fn(params, local_batch, state=clip_state)
grads = sum_gradients(grads)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

## Privacy auditing

Empirical privacy validation via one-run membership inference
([Steinke et al. 2023](https://arxiv.org/abs/2305.08846)).

```python
import opake.auditing as auditing
from opake.random import key

cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))
# ... train with DP-SGD or DP-FTRL ...
scores = auditing.loss_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    coin_flip=cf, dataset=dataset,
)
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (empirical): {estimate.epsilon_at(delta=1e-5):.4f}")
```

## Next steps

**Getting started**: [Installation](getting-started/installation.md) and
[Quick Start](getting-started/quickstart.md).

**End-to-end pipelines**:
[DP-SGD](user-guide/dp-sgd.md) ·
[DP-FTRL](user-guide/dp-ftrl.md).

**Understanding the API**: [User Guide](user-guide/index.md) covers
each component with detailed explanations and practical guidance.

**Hands-on practice**: [Tutorials](tutorials/README.md) are
task-based Jupyter notebooks.

**API details**: [API Reference](reference/index.md) provides complete
function signatures and docstrings.
