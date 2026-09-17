# opake-engine

Torch substrate for the Opake library. Ships everything that uses PyTorch
that downstream stacks (`opake-dpsgd`, `opake-dpftrl`,
`opake-auditing`, `opake-patches`, `opake-transformers`) need:

- `opake.api.engine.types` — `ClippedPytree`, `NoisedPytree`,
  `PerGroup`, `ClipState`, `NoiseState`, paired second-moment outputs.
- `opake.api.engine.pytree` — torch-pytree ops (`tree_map`,
  `partition`, `merge`, `global_norm`, …).
- `opake.api.engine.random` — `RngKey` (uint32 tensor), `key`,
  `split`, `fold_in`, `torch.Generator` helpers.
- `opake.api.engine.serialization` — `torch.Tensor` / `numpy.ndarray`
  handler registration with the base-side serialization registry.
- `opake.api.engine.distributed` — DDP collectives, sync registry,
  detection helpers.
- `opake.api.engine.noise_allocation` — per-group / paired-stream
  noise stddev math, shared between DP-SGD and DP-FTRL.
- `opake.api.engine.clipping` — fixed + AUTO-S clipping primitives
  (constant-sensitivity; usable by both DP-SGD and DP-FTRL).
- `opake.api.engine.functional` — `make_functional`,
  `with_batch_dim`, `empty_collate`.
- `opake.api.engine.scheduling` — step-indexed schedules + warmup /
  restarts composition.
- `opake.api.engine.profiling` — memory + step timer.

User-facing façades live at `opake.types`, `opake.pytree`,
`opake.random`, `opake.distributed`, `opake.functional`,
`opake.scheduling`, `opake.profiling`. **No `opake.clipping`
façade** — clipping is reached via stack façades
(`opake.dpsgd.clipping`, `opake.dpftrl.clipping`).
