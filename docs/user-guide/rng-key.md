# Random Number Generation

Opake uses explicit RNG keys instead of global random state. Every
function that involves randomness — noise injection, sampling, adaptive
clipping, and auditing — takes a `key` parameter. This follows the JAX PRNG
model: no hidden state, fully deterministic, and safe in distributed
settings.

## Why explicit keys

Global random state (`torch.manual_seed`, `np.random.seed`) has two
problems for DP-SGD:

1. **Reproducibility is fragile.** Any library call that consumes random
   numbers silently advances the global counter, making results depend on
   import order, PyTorch version, and GPU scheduling.
2. **Distributed training is error-prone.** You must carefully manage which
   ranks share state and which diverge. A subtle bug can cause all ranks to
   generate the same noise (correct for centralized DP-SGD) or different
   noise (incorrect for centralized DP-SGD) without any visible error.

Explicit keys solve both problems. The key is a value, not hidden state.
You pass it explicitly, split it for separate uses, and fold in additional data
(step counter, rank) to derive new keys deterministically. Other code cannot
advance an immutable key.

## Core primitives

| Function | Purpose |
|----------|---------|
| `key(seed)` | Create a key from an integer seed |
| `split(k, num=2)` | Derive `num` deterministic child keys |
| `fold_in(k, *data)` | Mix a key with one or more int/str values |
| `generator_from_key(k)` | Convert to `torch.Generator` |
| `random_key()` | Non-deterministic key (uses system entropy) |

### `key(seed)`

Create a key from an integer seed. This is the entry point for all
randomness in Opake.

```python
from opake.random import key

k = key(42)
# RngKey(seed=42, impl='opake_threefry_like')
```

The returned `RngKey` is a frozen (immutable) dataclass with a `seed`
field (uint64) and an `impl` field identifying the hash function.

### `split(k, num=2)`

Derive `num` child keys from a parent key. Use this when you need multiple
separate sources of randomness.

```python
from opake.random import key, split

k = key(42)
k_noise, k_sample = split(k)
k1, k2, k3 = split(k, num=3)
```

Each child is deterministically derived via `fold_in(k, i)` for index `i`.
Under Opake's PRNG model, each child identifies a separate stream, and
consuming one child does not advance another.

Opake separates registered component families internally. Split explicitly
when constructing multiple instances that must use distinct streams.

```python
# Same sampler configuration and key: reproducibly identical streams
sampler_a = PoissonSampler(dataset, sample_rate=0.01, key=k)
sampler_b = PoissonSampler(dataset, sample_rate=0.01, key=k)

# Split when the instances require distinct streams
k_a, k_b = split(k)
sampler_a = PoissonSampler(dataset, sample_rate=0.01, key=k_a)
sampler_b = PoissonSampler(dataset, sample_rate=0.01, key=k_b)
```

### `fold_in(k, *data)`

Mix additional data into a key. Variadic — accepts multiple values:

```python
from opake.random import key, fold_in

k = key(42)
step_key = fold_in(k, step)              # single int
rank_key = fold_in(k, rank)               # single int
combined = fold_in(k, step, rank)         # multiple values
full     = fold_in(k, step, rank, worker) # step → rank → worker
```

`fold_in(k, a, b)` is equivalent to `fold_in(fold_in(k, a), b)`.

Integers and strings occupy disjoint hash paths: each fold is prefixed with a
type tag before hashing. That split is load-bearing. Integers are the caller's
space (`split`, steps, ranks).
Strings root a mechanism: fold one unique tag once, then derive steps
beneath it. Opake's shipped tags are listed in the
[RNG reference](../reference/rng.md).

`fold_in` prefixes each typed input and hashes it with BLAKE2b to derive a
64-bit key:

```python
fold_in(k, 0).seed != fold_in(k, 1).seed
fold_in(k, 42).seed != fold_in(k, "42").seed  # int vs str distinguished
```

### `generator_from_key(k)`

Convert an `RngKey` to a `torch.Generator` for use with PyTorch operations
that require one.

```python
from opake.random import key, generator_from_key

gen = generator_from_key(key(42))
tensor = torch.randn(10, generator=gen)
```

This bridges Opake's immutable keys with PyTorch's generator API. Use it
for non-DP operations (dropout, weight initialization) where you want
determinism but don't need Opake's key management.

### `random_key()`

Create a non-deterministic key using system entropy. Each call returns a
different key.

```python
from opake.random import random_key

k = random_key()  # different every time
```

Use for prototyping and exploration when reproducibility is not needed.
For production training, always use `key(seed)` with a fixed seed.

## Using keys with Opake components

### Noise

```python
from opake.dpsgd.noise import gaussian_noise
from opake.random import key

noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

The noise function manages its own step counter internally. Each call derives
from the mechanism's namespaced root, the base key, and the current step, then
increments the counter. Different registered components use different domains.
Instances of one component with the same configuration, key, and call sequence
remain reproducibly identical.

### Sampling

```python
from opake.dpsgd.sampling import PoissonSampler
from opake.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
```

Each randomized sampler derives its stream under a stable component-specific
domain. A sampler's serialized state preserves the stream identity and cursor
needed to continue its sequence. Instances with the same sampler type,
configuration, key, dataset ordering, and call sequence reproduce the same
batches within a compatible NumPy execution environment. Split or derive the
key when sampler instances require distinct streams.

### Adaptive clipping

```python
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn, initial_clipping_norm=1.0, key=key(7),
)
```

### Auditing

`coin_flip()` derives separate audit subkeys for canary selection and inclusion
coins. Training mechanisms use their own RNG domains, so their streams remain
separated from auditing under a shared reproducible root seed.

```python
import opake.auditing as auditing
from opake.random import key

cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
```

## Training loop patterns

### Simple (single device)

In the common case, you create keys once at the start and thread state
through the loop:

```python
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.random import key, split

k_noise, k_sample = split(key(42))

grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=k_noise)

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### With `fold_in()` (per-step keys)

Use `fold_in` to derive per-step keys from a base key. This gives
explicit control over the derivation chain:
`key(seed) → fold_in(step) → fold_in(rank) → fold_in(worker)`.

```python
from opake.random import key, fold_in

base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=step_key)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
```

### Distributed training

In centralized DP-SGD, all ranks must add the **same** noise so that
models stay in sync. Pass the same `key(seed)` on every rank:

```python
# Same key on all ranks → same noise → models stay in sync
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))
```

For per-rank key control, use `fold_in()`:

```python
from opake.random import key, fold_in
import torch.distributed as dist

rank = dist.get_rank()

# Synchronized noise — same key, no rank folded in
step_key = fold_in(key(42), step)

# Independent noise per rank — fold in rank
step_key = fold_in(key(42), step, rank)

# Equivalent explicit chaining
step_key = fold_in(fold_in(key(42), step), rank)
```

## Reproducibility

Within a compatible software, build, and execution environment, the same key
and call sequence reproduce the same output:

```python
from opake.dpsgd.noise import gaussian_noise
from opake.random import key
import torch

grads = {"w": torch.randn(100)}

noise_fn, s1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
noisy1, _ = noise_fn(grads, s1)

noise_fn, s2 = gaussian_noise(noise_multiplier=1.0, key=key(42))
noisy2, _ = noise_fn(grads, s2)

assert torch.equal(noisy1["w"], noisy2["w"])
```

Opake controls only its own randomness. PyTorch operations like
`torch.randn` or `torch.nn.Dropout` use PyTorch's global state and may
vary across platforms. Use `set_reproducible_pytorch_seed` to configure
framework-level determinism:

```python
from opake.random import set_reproducible_pytorch_seed, key

set_reproducible_pytorch_seed(key(42))
# Sets torch.manual_seed, torch.cuda.manual_seed_all
# Enables torch.backends.cudnn.deterministic
# Disables torch.backends.cudnn.benchmark
```

This has a 10-30% performance cost due to deterministic algorithm
selection. Call it once at training startup if you need full reproducibility.

## Checkpoint and resume

Because keys are values (not stateful objects), checkpointing is
straightforward. Save the noise state and restore it:

```python
# Save
state = {"noise_state": noise_state, "step": step, ...}
torch.save(state, "checkpoint.pt")

# Resume
state = torch.load("checkpoint.pt")
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))
# Advance to the saved step by setting the internal counter
noise_state = state["noise_state"]
```

Alternatively, reconstruct the key from the step counter using
`fold_in`:

```python
# Resume from step 500
k = fold_in(key(42), 500)
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=k)
```

## API reference

See [Random API Reference](../reference/rng.md) for complete function signatures
and return types.
