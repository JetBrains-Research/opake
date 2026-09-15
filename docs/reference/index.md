# API Reference

Opaque provides a functional API for differential privacy in PyTorch. This
reference documents all public functions and classes.

Install via `opaque` (and `opaque[...]` extras) when using this API. Module
paths remain under `opaque.*`, but the root package is the supported
user-facing installation target.

## Module organization

Opaque is organized into several modules, each focused on a specific aspect of DP training:

### Core utilities

- **[Serialization](serialization.md)** — Flat `state_dict` / `from_state_dict` for
  explicit state trees (optimizers, accounting, clip/noise state, …);
  template-driven restore; optional `register_serializer` for custom types

- **[Exceptions](exceptions.md)** — Stable semantic categories for configuration,
  calibration, privacy-budget, checkpoint, type, and operation failures

- **[Random Number Generation](rng.md)** — Immutable RNG keys for deterministic DP
  - `RngKey` — Immutable key type
  - `key()`, `random_key()` — Create keys
  - `split()`, `fold_in()` — Manipulate keys
  - `set_reproducible_pytorch_seed()` — PyTorch/cuDNN reproducibility
  - `generator_from_key()` — PyTorch generator bridge

- **[Utilities](utilities.md)** — Functional and PyTree utilities
  - `make_functional()` — Convert `nn.Module` to functional form
  - `tree_map()`, `tree_map_with_path()`, `partition()`, `merge()`, `global_norm()`, `tree_leaves()`

### DP-SGD components

- **[Clipping](clipping.md)** — Per-sample gradient clipping
  - `clipped_grad()` — High-level gradient clipping
  - `clipped_fun()` — Clip and sum function outputs
  - `clip_pytree()` — Low-level PyTree clipping
  - `adaptive_clipped_grad()` — Clipped gradients with an auto-tuned clip norm
  - `auto_clipped_grad()` — AUTO-S automatic per-example gradient scaling (Bu et al. 2023)
  - `moe_clipped_grad()` — MoE clipping with the router load released inside the clipper

- **[Noise](noise.md)** — Noise injection for DP
  - `gaussian_noise()` — Gaussian noise
  - `mf_gaussian_noise()` — Correlated noise dispatcher (DP-FTRL)
  - Strategy factories: `band_mf_strategy()`, `blt_strategy()`, `lambda_cgd_strategy()`, `bisr_strategy()`, `identity_strategy()`

- **[Accounting](accounting.md)** — Privacy budget tracking
  - `gaussian()`, `adaclip()`, `moe_aux()` — DP-SGD mechanisms (also via `opaque.dpsgd.accounting`)
  - `poisson()` (plain or truncated via `truncated_batch_size` / `dataset_size`), `parallel_poisson()` — Poisson-family amplification
  - `mf_gaussian()`, `poisson()`, `b_min_sep()`, `balls_in_bins()` — DP-FTRL whole-process accounting
  - `DpProcess` operators: `*` (repeat), `|` (compose)
  - `.epsilon_at()`, `.delta_at()`, `.advantage()`, `.beta_at()`, `.risk_at()` — Privacy metrics
  - `calibrate()` — Binary-search noise multiplier for target privacy

- **[Sampling](sampling.md)** — Privacy-amplifying sampling
  - `PoissonSampler` — Standard Poisson sampling
  - `PoissonSampler` + `truncated_batch_size` — Bounded Poisson sampling
  - `CyclicPoissonSampler` (`opaque.dpftrl`) — Cyclic Poisson over `bands` groups; `bands=1` = identity (full-data Poisson each step)
  - `BallsInBinsSampler` — Random-partition sampling (λCGD, BISR, BLT)
  - `SequentialBatchSampler` — Deterministic sequential batching (BLT)

- **[Schedules](schedules.md)** — LR schedules for TorchOpt functional optimizers
  - `constant_schedule()` — Constant LR
  - `cosine_schedule()` — Cosine annealing
  - `inverse_sqrt_schedule()` — Inverse-square-root decay
  - `one_minus_sqrt_schedule()` — `1 - sqrt(progress)` decay (concave)
  - `with_warmup()` — Compose a warmup ramp with any decay schedule
  - `with_restarts()` — Replay a schedule N times

### Validation and debugging

- **[Auditing](auditing.md)** — Empirical privacy validation
  - `auditing.coin_flip()`, `auditing.loss_scores()`, `auditing.one_run()` — Three-step workflow
  - `OneRunEstimate.epsilon_at()`, `.delta_at()`, `.beta_at()`, `.advantage()` — μ-GDP-default PLD-mirroring surface
  - `OneRunEstimate.eps_delta()` — Mechanism-agnostic (ε, δ)-DP override
  - `attack_auc()`, `attack_beta_at()` — Attack-side empirical metrics

- **[Distributed](distributed.md)** — Multi-GPU training with DDP
  - `sum_gradients()` / `sum_gradients_()` — Copy-returning and in-place DP gradient summation
  - `all_reduce()` — Generic tensor all-reduce (sum, mean, max, min)
  - `reduce_pytree()` / `reduce_pytree_()` — Copy-returning and in-place generic PyTree reduction
  - `sync()` — Auto-dispatch sync for any state/aux type
  - `local_shard()` — Partition a dataset for DDP training
  - `is_distributed()`, `get_rank()`, `get_world_size()` — Distributed utilities

### Hugging Face integration

- **[Transformers](transformers.md)** — `opaque.transformers` — DP-SGD for Hugging Face models
  - `DPTrainer` — full constructor, methods, callback wiring, overridable hooks
  - `TrainingArguments` — every field grouped by concern (privacy, compute, patches, save, eval, …)
  - `opaque.transformers.trainer.types` — `EvaluationResult`, `TrainOutput` return types
  - `opaque.patches.apply_runtime_patches` / `is_runtime_patched` — install/query the global runtime shims
  - `opaque.transformers.trl` — TRL-style `SFTTrainer` / `DPOTrainer` (+ `SFTConfig` / `DPOConfig`), built on `DPTrainer`

## Quick Reference

### Typical DP-SGD Workflow

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

# Calibrate noise multiplier
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=5.0,
)

# Set up DP components
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=result.param, key=key(42),
)

# Training loop
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)
```

See [Quick Start](../getting-started/quickstart.md) for a complete working example.

## Function index

### Clipping

| Function         | Purpose                             | User Guide                                                        |
|------------------|-------------------------------------|-------------------------------------------------------------------|
| `clipped_grad()` | Differentiate loss with DP clipping | [Guide](../user-guide/clipping.md#clipped_grad-recommended-api) |
| `clipped_fun()`  | Clip and sum function outputs       | [API](clipping.md) |
| `clip_pytree()`  | Clip PyTree to max norm             | [API](clipping.md) |

### Noise

| Function                       | Purpose                                      | User Guide                      |
|--------------------------------|----------------------------------------------|---------------------------------|
| `gaussian_noise()` | Gaussian noise | [Guide](../user-guide/noise.md) |
| `mf_gaussian_noise()`                         | Correlated noise dispatcher (DP-FTRL)        | [Guide](../user-guide/noise.md#matrix-factorization-noise-dp-ftrl) |
| `band_mf_strategy()`                | BandMF banded Toeplitz strategy              | [Guide](../user-guide/noise.md#band_mf_strategy) |
| `blt_strategy()`                     | BLT buffered Toeplitz strategy               | [Guide](../user-guide/noise.md#blt_strategy) |
| `lambda_cgd_strategy()`              | DP-λCGD PRNG-replay strategy                 | [Guide](../user-guide/noise.md#lambda_cgd_strategy) |
| `bisr_strategy()`                    | BISR banded inverse sqrt strategy            | [Guide](../user-guide/noise.md#bisr_strategy) |
| `identity_strategy()`                | Identity (DP-SGD via MF API)                 | [Guide](../user-guide/noise.md#identity_strategy) |

### Accounting (Mechanisms)

| Function                  | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `gaussian()`              | Gaussian mechanism                | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `poisson()`               | Poisson-subsampled mechanism      | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `poisson(..., truncated_batch_size=, dataset_size=)` | Truncated Poisson subsampling | [Guide](../user-guide/accounting.md#mechanisms) |
| `parallel_poisson()`      | Parallel Poisson subsampling      | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `adaclip()`               | Adaptive clipping mechanism       | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `eps_delta()`             | Fixed (ε, δ) guarantee            | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `identity()`              | Zero privacy loss                 | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `nonprivate()`            | Infinite-ε non-private baseline   | [Guide](../user-guide/accounting.md#mechanisms)                         |

### Accounting (Matrix Factorization)

| Function                  | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `mf_gaussian()`          | Strategy-driven MF Gaussian mechanism | [Guide](../user-guide/accounting.md#matrix-factorization-mechanisms) |
| `balls_in_bins()`        | Balls-in-Bins amplification       | [Guide](../user-guide/accounting.md#matrix-factorization-mechanisms)    |
| `poisson()` (`opaque.dpftrl`) | MF Poisson amplification (BandMF / identity) | [Guide](../user-guide/accounting.md#matrix-factorization-mechanisms) |
| `b_min_sep()`            | Warm-start b-min-separation amplification | [Guide](../user-guide/accounting.md#matrix-factorization-mechanisms) |

### Accounting (Composition & Metrics)

| Method / Operator         | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `process * k`             | Repeat k times                    | [Guide](../user-guide/accounting.md#core-concepts)                      |
| `a \| b`                  | Heterogeneous composition         | [Guide](../user-guide/accounting.md#core-concepts)                      |
| `.epsilon_at(delta)`      | Query (ε, δ)-DP                   | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.delta_at(epsilon)`      | Query δ for given ε               | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.advantage()`            | Query f-DP advantage              | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.beta_at(alpha)`         | Query (α, β) error rates          | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.risk_at(prior)`         | Query Bayes risk                  | [Guide](../user-guide/accounting.md#privacy-metrics)                    |

### Calibration

| Function                  | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `calibrate()`             | Find noise for target privacy     | [Guide](../user-guide/accounting.md#calibration)                        |
| `epsilon_budget()`        | Budget for (ε, δ) calibration     | [Guide](../user-guide/accounting.md#calibration)                        |
| `delta_budget()`          | Budget for δ calibration           | [Guide](../user-guide/accounting.md#calibration)                        |
| `advantage_budget()`      | Budget for advantage calibration  | [Guide](../user-guide/accounting.md#calibration)                        |
| `beta_budget()`           | Budget for (α, β) calibration     | [Guide](../user-guide/accounting.md#calibration)                        |
| `risk_budget()`           | Budget for Bayes risk calibration | [Guide](../user-guide/accounting.md#calibration)                        |

### Sampling

| Class                     | Purpose                    | User Guide                                                    |
|---------------------------|----------------------------|---------------------------------------------------------------|
| `PoissonSampler`          | Standard Poisson sampling  | [Guide](../user-guide/sampling.md#poisson-sampling) |
| `PoissonSampler` (with `truncated_batch_size`) | Truncated Poisson sampling | [Guide](../user-guide/sampling.md#poisson-sampling) |
| `CyclicPoissonSampler` (`opaque.dpftrl`) | Cyclic Poisson over `bands` groups; `bands=1` = identity | [Guide](../user-guide/sampling.md#poisson-sampling) |
| `BallsInBinsSampler`      | Random-partition sampling  | [Guide](../user-guide/sampling.md#balls-in-bins-sampling) |
| `SequentialBatchSampler`  | Deterministic sequential batching (BLT) | [Guide](../user-guide/sampling.md#sequential-batch-sampling) |

### Schedules

| Function                       | Purpose                                       | User Guide                                       |
|--------------------------------|-----------------------------------------------|--------------------------------------------------|
| `constant_schedule()`          | Constant LR                                   | [Guide](../user-guide/lr-scheduling.md)          |
| `cosine_schedule()`            | Cosine annealing curve                        | [Guide](../user-guide/lr-scheduling.md)          |
| `inverse_sqrt_schedule()`      | Inverse-square-root decay                     | [Guide](../user-guide/lr-scheduling.md)          |
| `one_minus_sqrt_schedule()`    | `1 - sqrt(progress)` decay (concave)          | [Guide](../user-guide/lr-scheduling.md)          |
| `with_warmup()`                | Compose warmup ramp with a decay schedule     | [Guide](../user-guide/lr-scheduling.md#adding-warmup) |
| `with_restarts()`              | Replay a schedule N times                     | [Guide](../user-guide/lr-scheduling.md)          |

### Privacy Auditing

| Function / Method                    | Purpose                         | User Guide                                     |
|--------------------------------------|---------------------------------|------------------------------------------------|
| `auditing.coin_flip()`               | Designate canaries + partition  | [Guide](../user-guide/auditing.md)             |
| `auditing.loss_scores()`            | Compute membership scores       | [Guide](../user-guide/auditing.md)             |
| `auditing.gradient_scores()`        | Compute white-box membership scores | [Guide](../user-guide/auditing.md)         |
| `auditing.one_run()`                | Build one-run estimate          | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.epsilon_at()`       | ε bound (μ-GDP default)         | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.delta_at()`         | δ at given ε (μ-GDP default)    | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.beta_at()`          | Theoretical β at α (μ-GDP)      | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.advantage()`        | TV at inferred μ̂ (μ-GDP)        | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.eps_delta()`        | (ε, δ)-DP audit method override | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.gdp(grid_size=)`    | μ-GDP method with tunable grid  | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.attack_auc()`       | Empirical attack AUC            | [Guide](../user-guide/auditing.md)             |
| `OneRunEstimate.attack_beta_at()`   | Empirical attack β at given FPR | [Guide](../user-guide/auditing.md)             |

### Distributed

| Function               | Purpose                     | User Guide                                 |
|------------------------|-----------------------------|--------------------------------------------|
| `sum_gradients()` / `sum_gradients_()` | DP gradient summation (copy-returning / in-place) | [Guide](../user-guide/distributed.md) |
| `all_reduce()`         | Generic tensor all-reduce (sum, mean, max, min) | [Guide](../user-guide/distributed.md) |
| `reduce_pytree()` / `reduce_pytree_()` | Generic PyTree reduction (copy-returning / in-place) | [Guide](../user-guide/distributed.md) |
| `sync()`               | Auto-dispatch sync for any state/aux type | [Guide](../user-guide/distributed.md) |
| `local_shard()`        | Partition dataset for DDP training | [Guide](../user-guide/distributed.md) |
| `is_distributed()`     | Check if DDP is active      | [Guide](../user-guide/distributed.md)      |
| `get_rank()`           | Get current GPU index       | [Guide](../user-guide/distributed.md)      |
| `get_world_size()`     | Get total number of GPUs    | [Guide](../user-guide/distributed.md)      |

## Type hints

Opaque uses type hints throughout. Key types:

```python
import opaque.accounting as acc

# PyTree: Nested structure of tensors
PyTree = dict[str, torch.Tensor] | tuple[torch.Tensor, ...]

# DpProcess: Composable privacy process (from Rust PLD engine)
process: DpProcess = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 1000

# Generator for reproducible noise
Generator = torch.Generator | None
```

## Design philosophy

Opaque's API follows these principles:

1. **Functional-first**: Immutable state, pure functions
2. **Composable**: Small, focused functions that combine naturally
3. **Type-safe**: Comprehensive type hints
4. **PyTorch-native**: Built on `torch.func`, works with standard PyTorch
5. **JAX-inspired**: API closely mirrors JAX-Privacy for familiarity

## See also

- **[User Guides](../user-guide/index.md)**: Conceptual explanations and examples
- **[Tutorials](../tutorials/README.md)**: Interactive Jupyter notebooks
- **[Quick Start](../getting-started/quickstart.md)**: End-to-end DP-SGD example

---

**Browse by module:** Use the navigation on the left to explore detailed API documentation.
