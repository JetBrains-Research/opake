# opake-optimizers

Functional torchopt-based optimizers with a DP-aware update surface.

The implementation lives at `opake.api.optimizers.*`; the user-facing façade
lives at `opake.optimizers`. The wheel ships:

- `opake.api.optimizers.{adamw,adam,sgd,radam,adafactor,adagrad,
  adadelta,rmsprop,lion,ademamix,schedule_free}` — optimizer factories.
- `opake.api.optimizers.types` — state dataclasses for non-trivial
  optimizers.
- `opake.api.optimizers._chain` — `make_optimizer_chain` (DP-aware
  chain wrapper).
- `opake.api.optimizers._bias_correction` — `is_per_group`,
  `resolve_noise_variance` helpers.

Depends on `opake-engine` (for `ClippedPytree` / `NoisedPytree` /
`PerGroup`). It is installed by the `optimizers` extras of
`opake-dpsgd` and `opake-dpftrl`.
