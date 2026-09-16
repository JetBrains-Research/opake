# opaque-dpsgd

DP-SGD mechanisms for Opaque: Gaussian noise (optionally
bounded — [Chen and Hale, 2024](https://arxiv.org/abs/2211.17230)), clipping
(fixed, AUTO-S, adaptive), projected JME, and Poisson subsampling. Functional optimizers
(including the universal `adamw` with optional DP bias correction) live in
[`opaque.optimizers`](../opaque-optimizers/README.md).

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation).
`opaque-dpsgd` is included in the default `opaque` package set.

## Quick start

```python
from opaque.dpsgd.clipping import auto_clipped_grad, clipped_grad
from opaque.random import key
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise, jme_noise
from opaque.dpsgd.noise.types import JmeAllocation
from opaque.dpsgd.sampling import PoissonSampler
```

## Layout

- `opaque.dpsgd.noise` — `gaussian_noise` (optional `bound`) and
  `jme_noise` with explicit `JmeAllocation`
- `opaque.dpsgd.clipping` — `clipped_grad`, `auto_clipped_grad`, `per_group`, `adaptive_clipped_grad`, `.types`, `.fun`
- `opaque.dpsgd.sampling` — `PoissonSampler` (optional `truncated_batch_size`)

RNG keys, pytree helpers, distributed plumbing, and serialization live in
[`opaque-engine`](../opaque-engine/README.md).
