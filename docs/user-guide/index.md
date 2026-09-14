# User Guide

This guide explains each component of Opaque's DP training pipeline:
what it does, how the API works, and the practical decisions you
need to make. For hands-on practice, see the
[Tutorials](../tutorials/README.md). For complete function
signatures, see the [API Reference](../reference/index.md).

Install with `pip install opaque` (plus `opaque[...]` extras).

## End-to-end pipelines

Pick the track that matches your problem:

- **[DP-SGD end-to-end](dp-sgd.md)** — independent Gaussian noise at
  every step, with per-step privacy composition. This is the standard DP
  training recipe. Imports from `opaque.dpsgd.*`.
- **[DP-FTRL end-to-end](dp-ftrl.md)** — correlated noise across the
  whole training run via matrix factorization. Imports from
  `opaque.dpftrl.*`.

Both pipelines share the same primitives (clipping, noise, sampling,
optimizer, accounting). The topic pages below are stack-agnostic
concept references; each end-to-end guide picks the right pieces and
combines them.

## Topics

### Foundations

- **[Differential Privacy Concepts](dp-concepts.md)** — What DP
  guarantees, how DP-SGD works, privacy budgets, composition, and
  amplification.
- **[Random Number Generation](rng-key.md)** — Explicit RNG keys,
  splitting, `fold_in`, reproducibility in distributed training.

### Core pipeline

- **[Per-Example Gradient Clipping](clipping.md)** — `clipped_grad`,
  `auto_clipped_grad`, `adaptive_clipped_grad` (DP-SGD-only),
  microbatching, per-group clipping.
- **[Noise Addition](noise.md)** — `gaussian_noise` (DP-SGD) and
  `mf_gaussian_noise` with strategy factories (DP-FTRL).
- **[Privacy Accounting](accounting.md)** — Composable `DpProcess`
  objects, privacy metrics, calibration, the `Accountant` helper.
- **[Sampling & Microbatching](sampling.md)** — Poisson, truncated
  Poisson, cyclic / b-min-sep / balls-in-bins / sequential samplers.

### Integration

- **[Optimizers](optimizers.md)** — TorchOpt functional optimizers
  with DP-SGD bias correction and the private second-moment story
  for DP-FTRL.
- **[Serialization (API reference)](../reference/serialization.md)** —
  Checkpoint explicit state with
  `opaque.serialization.state_dict` / `from_state_dict`.
- **[LR Scheduling](lr-scheduling.md)** — Warmup, cosine,
  inverse-sqrt schedules; composing `with_warmup` with any decay
  curve.
- **[Distributed Training](distributed.md)** — DDP with synchronized
  noise and gradient aggregation.
- **[Distributed Trainer](distributed-trainer.md)** — DDP setup and
  `ddp_shard` behavior for `Trainer`.
- **[HuggingFace Integration](huggingface/index.md)** — `Trainer`,
  `TrainingArguments`, model patches, and LoRA.
- **[Memory Optimizations](memory-optimizations.md)** — Microbatching,
  gradient checkpointing, fused kernels, profiling.
- **[Privacy Auditing](auditing.md)** — Empirical privacy validation
  via membership inference.

### Mechanism reference

- **[DP-SGD mechanisms](../mechanisms/dp-sgd/index.md)** — Gaussian
  per-step noise.
- **[DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md)** — BandMF,
  BLT, BSR, BISR, λ-CGD strategies.

### Reference

- **[Known Limitations](../limitations.md)** — Flash Attention,
  DDP-only, in-place operations, other constraints.
