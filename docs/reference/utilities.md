# Utilities

## Functional Utilities

The `opake.functional` module provides the `torch.func` bridges that turn
standard `nn.Module` models into the pure-function form DP-SGD needs.
`opake.dpsgd.clipping.clipped_grad` and friends assume their `loss_fn`
argument is functional in this sense.

**Key function**: `make_functional()` — convert a PyTorch module to
functional form.

```python
import torch.nn as nn
from opake.functional import make_functional

model = nn.Linear(10, 1)
fmodel, params = make_functional(model)
output = fmodel(params, input_tensor)
```

Functional models allow `torch.func.vmap` to compute per-example gradients
efficiently, which is essential for DP-SGD.

### Trainable / frozen partition (for PEFT and LoRA)

For parameter-efficient fine-tuning (LoRA, adapters, BitFit, …) only a
small subset of parameters has `requires_grad=True`.  Pass
`partition_trainable=True` to split the parameter dict accordingly:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, x, y):
    out = fmodel({**frozen, **trainable_params}, x)
    return ((out - y) ** 2).mean()
```

Only `trainable_params` flows into the loss closure, so
`vmap(grad(...))` over `loss_fn` produces per-example gradients only
for the trainable subset.  Frozen parameters are broadcast (constant)
under `vmap`, which is what makes LoRA-style DP fine-tuning of
multi-billion-parameter models feasible — per-example gradient memory
scales with trainable params, not total params.  See [Memory
optimizations](../user-guide/memory-optimizations.md) for the memory
arithmetic; fused LoRA Triton kernels are documented under [Model
patches — Fused LoRA operations](../user-guide/huggingface/model-patches.md#fused-lora-operations).

**See also**: [Quick Start Guide](../getting-started/quickstart.md) for
functional API usage.

::: opake.functional
    options:
      show_source: true
      heading_level: 3

---

## PyTree Utilities

The `opake.pytree` module provides helpers for working with PyTrees —
nested structures of tensors used throughout Opake.

**PyTrees** are nested dictionaries or tuples of tensors, commonly used to
represent model parameters:

```python
params = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}
```

This module provides:

- **`tree_map()`** — apply a function to all leaves
- **`tree_map_with_path()`** — apply a function with key paths to all leaves
- **`tree_leaves()`** — extract all leaf tensors
- **`partition()`** — split a tree into two by predicate
- **`merge()`** — recombine partitioned trees
- **`global_norm()`** — compute L2 norm across the entire tree

**See also**: [Gradient Clipping Guide](../user-guide/clipping.md) for PyTree
usage in DP-SGD.

::: opake.pytree
    options:
      show_source: true
      heading_level: 3
