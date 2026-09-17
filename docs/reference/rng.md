# Random Number Generation API

Opake provides immutable, JAX-style RNG key semantics with explicit key
threading for deterministic and reproducible DP training.

## Overview

The `opake.random` module provides:

- **Core primitives**: `RngKey`, `split()`, `fold_in()` — Functional RNG with immutable keys
- **Convenience helpers**:
  - `key()` — Create an RngKey from an integer seed
  - `random_key()` — Nondeterministic key from system entropy
  - `set_reproducible_pytorch_seed()` — Configure PyTorch/cuDNN reproducibility
- **Bridge function**: `generator_from_key()` — Create a `torch.Generator` from an RngKey

## Quick Reference

### Essential Imports

```python
# Core primitives
from opake.random import split, fold_in, key, generator_from_key
from opake.random.types import RngKey

# Convenience helpers
from opake.random import random_key, set_reproducible_pytorch_seed
```

### Creating Keys

```python
# From integer seed (reproducible)
k = key(42)

# Nondeterministic (for prototyping)
k = random_key()  # Uses system entropy

# From torch.Generator
gen = torch.Generator().manual_seed(42)
k = RngKey(seed=42)  # Equivalent for reproducible use
```

### Splitting Keys

```python
# Binary split (most common)
k1, k2 = split(k, num=2)

# Multi-way split
keys = split(k, num=10)  # Returns tuple

# Loop pattern — split and advance
for step in range(100):
    k, step_key = split(k, num=2)
    result = do_something_random(step_key)
```

### Per-Step Derivation

```python
# Using fold_in for step-based keys
base_key = key(42)
step_key = fold_in(base_key, 0)

# Multiple values (variadic)
step_rank_key = fold_in(base_key, step, rank)
```

### PyTorch Reproducibility

```python
# Set all PyTorch/CUDNN seeds from RngKey
from opake.random import set_reproducible_pytorch_seed, key, fold_in

set_reproducible_pytorch_seed(key(42))

# Then use fold_in for per-step DP randomness
base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
    # ... training step ...
```

## API Reference

### Classes

#### RngKey

```python
@dataclass(frozen=True)
class RngKey:
    seed: int
    impl: str = "opake_threefry_like"
```

Immutable RNG key. Thread explicitly through functions for deterministic randomness.

**Attributes:**
- `seed`: Integer seed value (main key material)
- `impl`: Implementation identifier (default: "opake_threefry_like")

### Functions

#### key(seed: int) → RngKey

Create an RngKey from an integer seed.

```python
from opake.random import key

k = key(42)
```

**Args:**
- `seed`: Integer seed value

**Returns:** RngKey

**Raises:** `InputTypeError` if seed is not an integer

---

#### random_key() → RngKey

Create a nondeterministic RngKey using system entropy.

```python
from opake.random import random_key

k = random_key()  # Each call returns different key
```

Useful for prototyping and experiments. For reproducible training, use `key()` with an explicit seed.

**Returns:** RngKey with seed from `secrets.randbits(64)`

---

#### split(rng_key: RngKey, num: int = 2) → tuple of RngKey values

Split a key into `num` deterministically derived child keys.

```python
from opake.random import split, key

k = key(42)
k1, k2 = split(k, num=2)

# Multi-way split
keys = split(k, num=10)
```

**Args:**
- `rng_key`: Key to split
- `num`: Number of child keys (default: 2)

**Returns:** Tuple of `num` derived RngKeys

**Raises:** `ConfigurationError` if `num < 1`

**Rule:** Split keys for separate uses. Each derived key identifies a separate
PRNG stream, and consuming one does not advance another.

---

#### fold_in(rng_key: RngKey, *data: int | str) → RngKey

Deterministically derive a new key by folding in one or more values.

Accepts a variable number of int/str arguments. Each value is folded
sequentially, so `fold_in(k, a, b)` equals `fold_in(fold_in(k, a), b)`.

```python
from opake.random import fold_in, key

base_key = key(42)

# Single value
step_key = fold_in(base_key, 0)

# Multiple values (variadic)
step_rank_key = fold_in(base_key, step, rank)

# Full derivation chain
full_key = fold_in(base_key, step, rank, worker_id)

# String data
key_v2 = fold_in(base_key, "v2")
```

**Args:**
- `rng_key`: Base key
- `*data`: One or more int or str values to fold in sequentially

**Returns:** New RngKey derived from `rng_key` and all folded values

**Raises:**
- `InputTypeError` if `rng_key` is not an `RngKey`, or any value is not int or str
  (`bool` is rejected even though it subclasses `int`)
- `ConfigurationError` if no data values are provided

Integers and strings occupy disjoint hash paths: each fold is prefixed with a
type tag before hashing, so `fold_in(k, 1)` is never the same key as
`fold_in(k, "1")`. That split is load-bearing.

- **Integers are the caller's.** Steps, ranks, epochs, leaf and group
  indices, and every key `split` returns are integer folds.
- **Strings root a mechanism.** A mechanism that draws randomness folds one
  unique string tag into the key it was given, then derives steps beneath
  that tag. Skipping the string is the failure the convention exists to
  prevent: `fold_in(key, step)` is what every mechanism writes first, so two
  mechanisms handed the same base key draw byte-identical noise.

Do not reuse these shipped tags:

| Tag | Occupied by |
| --- | --- |
| `opake.dpsgd.gaussian` | `opake.dpsgd.noise.gaussian_noise` (both streams) |
| `opake.dpsgd.adaptive_clipping` | adaptive clipping threshold noise |
| `opake.dpsgd.poisson` | DP-SGD Poisson sampling |
| `opake.dpsgd.k_out_of_t` | DP-SGD k-out-of-t sampling |
| `opake.dpftrl.mf_gaussian` | `opake.dpftrl.noise.mf_gaussian_noise` |
| `opake.dpftrl.second_moment.first` / `.second` | paired MF second-moment streams |
| `opake.dpftrl.cyclic_poisson` | DP-FTRL cyclic Poisson sampling |
| `opake.dpftrl.b_min_sep` | DP-FTRL b-min-separation sampling |
| `opake.dpftrl.balls_in_bins` | DP-FTRL balls-in-bins sampling |
| `opake.paired.first` / `opake.paired.second` | paired first/second-moment streams |
| `opake.auditing.canary_selection` / `opake.auditing.coin_flip` | `opake.auditing.coin_flip` |
| `opake.transformers.ignore_data_skip` | trainer Poisson restart after skipped sampler state |

A component's stream is deterministic for a given caller key, configuration,
and call sequence. Split the key or fold in an instance or rank identifier when
streams must differ.

**Use Cases:**
- Step counters in loops: `fold_in(base, step)`
- Distributed training: `fold_in(base, step, rank)`
- DataLoader workers: `fold_in(base, step, rank, worker_id)`
- Checkpointing (deterministic resume key)
- Versioning DP mechanisms

---

#### generator_from_key(rng_key: RngKey) → torch.Generator

Create a deterministic `torch.Generator` from an RngKey.

```python
from opake.random import generator_from_key, key
import torch

k = key(42)
gen = generator_from_key(k)

# Use with torch operations
tensor = torch.randn(10, generator=gen)
```

**Args:**
- `rng_key`: RngKey to convert

**Returns:** torch.Generator seeded with rng_key.seed

**Note:** The Generator is deterministic. Give framework randomness (for
example, dropout or initialization) a key derived separately from DP operations.

---

#### set_reproducible_pytorch_seed(key_val: RngKey) → None

Configure PyTorch and cuDNN for reproducible training from a single RngKey.

```python
from opake.random import key, fold_in, set_reproducible_pytorch_seed

# At training start
set_reproducible_pytorch_seed(key(42))

# Then use fold_in for per-step DP
base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
    # ... training with deterministic framework + DP randomness
```

Sets:
- `torch.manual_seed()` for CPU operations
- `torch.cuda.manual_seed_all()` for GPU operations
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` (if available)
- `CUBLAS_WORKSPACE_CONFIG` environment variable for cuBLAS determinism

**Args:**
- `key_val`: RngKey (typically from `key()`)

**Returns:** None (side effects only)

**Performance impact:** Deterministic algorithms may be slower; measure on
your workload.

**Example:**

```python
from opake.random import key, fold_in, split, set_reproducible_pytorch_seed
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler

# Setup framework reproducibility once
set_reproducible_pytorch_seed(key(42))

# Training loop with per-step DP randomness
base = key(42)
for step in range(1000):
    # Deterministic per-step key
    step_key = fold_in(base, step)
    step_key, noise_key = split(step_key, num=2)

    # Use for DP
    noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=noise_key)
    grads = compute_dp_grads(model, batch)
    noisy_grads, state = noise_fn(grads, state)
    optimizer.step(noisy_grads)
```

**See Also:**
- torch.use_deterministic_algorithms - PyTorch determinism documentation
- torch.backends.cudnn - CUDNN configuration

## Common Patterns

### Component Splitting

Split master key for different components:

```python
from opake.random import split, key

master = key(42)
sampling_key, noise_key, init_key = split(master, num=3)

sampler = PoissonSampler(..., key=sampling_key)
noise_fn, _ = gaussian_noise(..., key=noise_key)
model = initialize_model(init_key)  # If using jax
```

### Sequential Splitting (Loop Pattern)

Thread key through loop, splitting at each step:

```python
from opake.random import split, key

k = key(42)
for step in range(100):
    k, step_k = split(k, num=2)
    result = train_step(model, batch, key=step_k)
```

### Distributed Training

**Using fold_in()** (recommended):

```python
import torch.distributed as dist
from opake.random import key, fold_in

rank = dist.get_rank()
base = key(42)

for step in range(steps):
    # Synchronized noise — same key on all ranks (no rank folded in)
    noise_key = fold_in(base, step)

    # Per-rank noise — fold in rank for diversity
    sample_key = fold_in(base, step, rank)
```

**Manual approach** (for reference):

```python
from opake.random import split, key

master = key(42)
sampling_key, noise_master = split(master, num=2)

# Noise: per-rank keys
rank_keys = split(noise_master, num=world_size)
my_noise_key = rank_keys[rank]

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=my_noise_key,
)
```

## Troubleshooting

### Results not reproducible?

Ensure all randomness uses RngKey:

```python
# Correct: Use RngKey throughout
from opake.random import set_reproducible_pytorch_seed, key, fold_in

set_reproducible_pytorch_seed(key(42))  # Framework
base = key(42)
for step in range(n):
    k = fold_in(base, step)  # DP operations
    # ... training ...

# Incorrect: Mixing with global seeds
torch.manual_seed(42)  # Framework
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=some_key,
)  # DP
# Different RNG sources can cause issues
```

### Distributed ranks have correlated noise?

Ensure rank-specific keys:

```python
# Incorrect: All ranks share the same key
k = key(42)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=k,
)  # All ranks get identical noise

# Correct: Per-rank keys via fold_in
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=fold_in(key(42), rank),
)

# Also correct: Per-rank keys via split
rank_keys = split(key(42), num=world_size)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, key=rank_keys[rank],
)
```

### Need to resume from checkpoint?

Use `fold_in()` for deterministic resume:

```python
from opake.random import fold_in, key

base_key = key(42)

# Save checkpoint
checkpoint = {"step": 500, ...}

# Resume
for step in range(checkpoint["step"], total_steps):
    step_key = fold_in(base_key, step)
    # ... training ...
```

## Further Reading

- [RngKey User Guide](../user-guide/rng-key.md) - Conceptual guide
- [Noise APIs](noise.md) - Using keys with noise injection
- [Sampling](sampling.md) - Using keys with samplers
- [Distributed Training](distributed.md) - DDP patterns with keys
