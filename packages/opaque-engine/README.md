# opaque-engine

Torch substrate for the Opaque library. Ships everything that uses PyTorch
that downstream stacks (`opaque-dpsgd`, `opaque-dpftrl`,
`opaque-auditing`, `opaque-patches`, `opaque-transformers`) need:

- `opaque.api.engine.types` — `ClippedPytree`, `NoisedPytree`,
  `PerGroup`, `ClipState`, `NoiseState`, and the JME optimizer handoff.
- `opaque.api.engine.pytree` — torch-pytree ops (`tree_map`,
  `partition`, `merge`, `global_norm`, …).
- `opaque.api.engine.random` — `RngKey` (uint32 tensor), `key`,
  `split`, `fold_in`, `torch.Generator` helpers.
- `opaque.api.engine.serialization` — `torch.Tensor` / `numpy.ndarray`
  handler registration with the base-side serialization registry.
- `opaque.api.engine.distributed` — DDP collectives, sync registry,
  detection helpers.
- `opaque.api.engine.noise_allocation` — per-group noise stddev math,
  shared between DP-SGD and DP-FTRL.
- `opaque.api.engine.clipping` — fixed + AUTO-S clipping primitives
  (constant-sensitivity; usable by both DP-SGD and DP-FTRL).
- `opaque.api.engine.functional` — `make_functional`,
  `with_batch_dim`, `empty_collate`.
- `opaque.api.engine.scheduling` — step-indexed schedules + warmup /
  restarts composition.
- `opaque.api.engine.profiling` — memory + step timer.

User-facing façades live at `opaque.types`, `opaque.pytree`,
`opaque.random`, `opaque.distributed`, `opaque.functional`,
`opaque.scheduling`, `opaque.profiling`. **No `opaque.clipping`
façade** — clipping is reached via stack façades
(`opaque.dpsgd.clipping`, `opaque.dpftrl.clipping`).
