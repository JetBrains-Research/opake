# Quick Start

Train a linear regression model with differential privacy using Opake.

## Prerequisites

Install Opake following the [Installation Guide](installation.md).

## Complete example

```python
--8<-- "examples/quickstart.py:4:"
```

## What this does

1. **`make_functional`** converts the model so parameters are passed
   explicitly, which is required for `torch.func.vmap`.
2. **`clipped_grad`** computes per-example gradients, clips each to an L2
   norm 1.0, and sums the result.
3. **`PoissonSampler`** independently includes each example with the same
   `sample_rate` priced by the accountant.
4. **`acc.calibrate`** performs a binary search for the noise multiplier
   that achieves ε = 3.0 over the exact `num_steps` run.
5. **`gaussian_noise`** adds calibrated Gaussian noise to the clipped
   gradient sum.
6. **`Accountant`** tracks cumulative privacy cost and checks against the
   budget.

## Next steps

- [User Guide](../user-guide/index.md) — detailed explanations of each
  component.
- [Tutorials](../tutorials/README.md) — hands-on Jupyter notebooks.
- [API Reference](../reference/index.md) — complete function signatures.
