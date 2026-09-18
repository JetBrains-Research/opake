# opake-dpsgd

DP-SGD mechanisms for Opake: Gaussian noise, clipping (fixed, AUTO-S,
adaptive), and Poisson subsampling. Functional optimizers
(including the universal `adamw` with optional DP bias correction) live in
[`opake.optimizers`](../opake-optimizers/README.md).

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation).
`opake-dpsgd` is included in the default `opake` package set.

## Quick start

```python
from opake.dpsgd.clipping import auto_clipped_grad, clipped_grad
from opake.random import key
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler
```

## Layout

- `opake.dpsgd.noise` — `gaussian_noise`
- `opake.dpsgd.clipping` — `clipped_grad`, `auto_clipped_grad`, `per_group`, `adaptive_clipped_grad`, `.types`, `.fun`
- `opake.dpsgd.sampling` — `PoissonSampler` (optional `truncated_batch_size`)

RNG keys, pytree helpers, distributed plumbing, and serialization live in
[`opake-engine`](../opake-engine/README.md).
