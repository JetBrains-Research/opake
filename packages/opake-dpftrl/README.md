# opake-dpftrl

Matrix-factorization noise mechanisms for Opake: BLT, BSR, BiSR,
band-MF, lambda-CGD, identity — plus the MF-specific participation
samplers (b-min-separation, Poisson, balls-in-bins, sequential
batches). Functional optimizers (including the universal `adamw`
that consumes private `noisy_squared_grads` streams) live in
[`opake.optimizers`](../opake-optimizers/README.md).

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation),
using its `dpftrl` extra to include this component.

## Quick start

```python
from opake.random import key
from opake.dpftrl.noise import mf_gaussian_noise, blt_strategy
from opake.dpftrl.sampling import BMinSepSampler
```

## Layout

- `opake.dpftrl.noise` — strategies (band-MF, BLT, BSR, BiSR, identity, lambda-CGD) + dispatchers
- `opake.dpftrl.clipping` — MF-safe `clipped_grad`, `auto_clipped_grad`, `per_group`
- `opake.dpftrl.sampling` — `BMinSepSampler`, `CyclicPoissonSampler`, `BallsInBinsSampler`, `SequentialBatchSampler`

Shared clipping implementation and other cross-cutting primitives live in
[`opake-engine`](../opake-engine/README.md).
