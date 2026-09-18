# opake-transformers

Hugging Face trainer integration for Opake: DP-aware training loop
(`opake.transformers.trainer.DPTrainer`) with TRL-style SFT/DPO support
(`opake.transformers.trl`) and Hugging Face compatibility layers.

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opake#installation).
Use its `transformers` extra for trainer integration or `trl` for TRL config
conversion.

Depends on `opake-engine`, `opake-patches`, `opake-dpsgd`, `opake-dpftrl`,
`opake-accounting`, `opake-optimizers`, `opake-alignment`, plus
`transformers>=5.0`, `peft>=0.13`, and `datasets>=2.0`.

For Triton fused kernels (RoPE, RMSNorm, activation, cross-entropy), install
`opake-patches[transformers]` — kernels are a dependency of `opake-patches`
and gate on CUDA + Triton at runtime.

## Quick start

Runtime compat patches (vmap-safe masking, collator / checkpoint hooks) are
applied when you construct :class:`opake.transformers.trainer.DPTrainer`, or
when you call :func:`opake.patches.apply_runtime_patches` explicitly (e.g. in
a notebook that uses HF primitives without the trainer).

```python
from opake.patches import apply_runtime_patches, is_runtime_patched
from opake.transformers import DPTrainer

apply_runtime_patches(compat=True)  # global runtime shims — idempotent
trainer = DPTrainer(
    model, args, ...
)  # runtime compat + apply_model_patches on the model

assert is_runtime_patched()
```

## Layout

- **`opake.api.transformers.trainer`** — DPTrainer implementation
  (`_dp_trainer.py`, `_config.py`, `_state.py`, `_optim.py`,
  `_scheduler.py`, `_checkpoint.py`, `_distributed.py`,
  `_performance_kernels.py`, …).
- **`opake.transformers`** / **`opake.transformers.trainer`** — thin
  re-export façades (same pattern as `opake-engine`: `opake.api.*` for
  implementation, `opake.*` for stable imports).
- **`opake.patches.transformers`** — vmap-safe runtime patches and optional
  Triton kernel hooks (see `opake.patches`).
