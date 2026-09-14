# Distributed Trainer

`Trainer` supports multi-process DDP without Accelerate runtime ownership.
This page documents the backend matrix, DP-specific distributed semantics, and
the safe Hugging Face parity subset.

## Scope

- Supported launcher model: `torchrun` or equivalent process launcher that
  initializes `torch.distributed`.
- Not supported: FSDP, DeepSpeed, TPU/XLA runtime ownership, Accelerate
  `DataLoaderConfiguration` knobs that conflict with Poisson DP semantics.

`Trainer` expects the process group to be initialized externally and then
validates that runtime against `TrainingArguments.ddp_backend`.

## Backend matrix

`ddp_backend` accepts Hugging Face-compatible values:

- First-class in Opaque: `nccl`, `gloo`, `mpi`.
- Environment-dependent parity values: `xccl`, `hccl`, `cncl`, `mccl`.

Environment-dependent values are accepted for argument parity, but they require
vendor runtime stacks and fail fast when unavailable.

## Launch pattern

From the repository root, use one process per rank:

```bash
torchrun --nproc-per-node=4 examples/train_dpsgd_trainer.py
```

```python
args = TrainingArguments(
    output_dir="runs/ddp",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    ddp_backend="nccl",
    ddp_shard="per_rank",
    report_to=[],
)
```

## `ddp_shard` modes

`ddp_shard` controls dataset visibility and accounting when `WORLD_SIZE > 1`:

- `per_rank` (default): disjoint shard per rank, Poisson accounting.
- `global`: full dataset per rank with rank-folded streams and
  `parallel_poisson` accounting.

`per_rank` requires the public, fixed training population length to be evenly
divisible by `WORLD_SIZE`. When it isn't, `Trainer` drops the remaining
tail examples and logs a warning; the dropped tail changes both the
sharding and the accounting sample-rate denominator.

## Metrics and eval gather

- `compute_metrics(EvalPrediction)` runs on gathered cluster-wide payloads.
- `batch_eval_metrics=True` is supported under DDP.
- Gather uses a tensor fastpath when shapes align, with object-gather fallback
  for irregular payloads.
- `eval_use_gather_object=True` remains available as an explicit object gather
  switch for metrics payloads.

## DataLoader parity policy

Safe parity fields kept in `TrainingArguments`:

- `data_seed` (sampler trajectory control).
- `dataloader_drop_last` (eval loader only).
- worker/pin/prefetch/persistent settings.

Intentionally not exposed from Accelerate-style config:

- `split_batches`, `dispatch_batches`, `even_batches`, `use_seedable_sampler`.

These knobs conflict with Poisson-sampling semantics or imply batching behavior
that `Trainer` does not implement.

## Rank-gated side effects

Under DDP, side effects are rank-gated:

- logging by `log_on_each_node` policy,
- checkpoint and metrics writes by `save_on_each_node` policy,
- Hub repo creation/push only on world rank zero.

## CI checklist for distributed changes

- `uv run pytest packages/opaque-transformers/tests/opaque_transformers/test_config.py`
- `CUDA_VISIBLE_DEVICES=0,1,2,3 uv run pytest packages/opaque-transformers/tests/distributed/test_ddp_trainer.py`
- `uv run pytest packages/opaque-engine/tests/distributed/`
- `MASTER_ADDR=127.0.0.1 MASTER_PORT=<port> uv run pytest -k gloo packages/opaque-transformers/tests/distributed/`
- `mpirun -n 2 uv run pytest -k mpi packages/opaque-transformers/tests/distributed/` (when MPI launcher/runtime is available)
