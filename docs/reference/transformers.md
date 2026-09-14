# Transformers Integration

API reference for `opaque.transformers` — `Trainer`,
`TrainingArguments`, and the public state objects. For task-shaped
usage guides, see [Hugging Face Integration](../user-guide/huggingface/index.md).

## Overview

The `opaque.transformers` namespace re-exports the trainer surface:

```python
from opaque.transformers import Trainer, TrainingArguments
from opaque.transformers.trainer.types import EvaluationResult, TrainOutput
from opaque.transformers.trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

# Global HF runtime shims (only needed when using HF primitives without Trainer):
from opaque.patches import apply_runtime_patches
```

| Symbol | Purpose |
|---|---|
| `Trainer` | DP-SGD trainer mirroring the Hugging Face `Trainer` interface. |
| `TrainingArguments` | Standalone dataclass — full HF parity for the subset Trainer honors, plus DP-specific fields. |
| `opaque.transformers.trainer.types.EvaluationResult` | Return type for `evaluation_loop` / `evaluate` / `predict`. |
| `opaque.transformers.trainer.types.TrainOutput` | NamedTuple returned by `train()` — `(global_step, training_loss, metrics)`. |
| `opaque.transformers.trl` | TRL-style configs/trainers: `SFTConfig`, `SFTTrainer`, `DPOConfig`, `DPOTrainer`. |
| `opaque.patches.apply_runtime_patches` | Install the global HF runtime shims (only needed when using HF primitives without `Trainer`). |

## `Trainer`

### Construction

```python
Trainer(
    model: PreTrainedModel | None = None,
    args: TrainingArguments | None = None,
    data_collator: Callable | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    processing_class: PreTrainedTokenizerBase | None = None,
    compute_loss_func: Callable | None = None,
    compute_metrics: Callable | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[Any | None, Any | None] = (None, None),
    optimizer_cls_and_kwargs: tuple[Callable, dict[str, Any]] | None = None,
    preprocess_logits_for_metrics: Callable | None = None,
)
```

| Argument | Type | Notes |
|---|---|---|
| `model` | `PreTrainedModel` | Required.  Moved to `args.device` immediately; PEFT wrappers detected and cached on `self._is_peft`. |
| `args` | `TrainingArguments` | Defaults to `TrainingArguments(output_dir="tmp_trainer")` when omitted. |
| `data_collator` | `Callable \| None` | Defaults to `DataCollatorWithPadding(tokenizer)` when `processing_class` is a tokenizer / sequence-feature extractor, else `default_data_collator`. |
| `train_dataset` | `Dataset \| None` | May be `None` only when `train()` is not called. |
| `eval_dataset` | `Dataset \| None` | Required when `args.eval_strategy != "no"`. |
| `processing_class` | `PreTrainedTokenizerBase \| SequenceFeatureExtractor \| None` | Used for the default collator selection and token-count metrics. |
| `compute_loss_func` | `Callable[[outputs, labels], Tensor] \| None` | Loss override; called under vmap per example during training and per-example-loss evaluation, and once on batched outputs/labels on the default reduced eval path. It takes precedence over label smoothing. Not HF's `(outputs, labels, num_items_in_batch) -> scalar` signature. |
| `compute_metrics` | `Callable[[EvalPrediction], dict] \| None` | Standard HF callback over concatenated predictions / label_ids / inputs / losses. |
| `callbacks` | `list[TrainerCallback] \| None` | User callbacks; `DefaultFlowCallback` is auto-prepended. |
| `optimizers` | `tuple[Any \| None, Any \| None]` | **Not supported.** Passing non-`None` raises `ConfigurationError`: Trainer owns the functional torchopt optimizer. |
| `optimizer_cls_and_kwargs` | `tuple[Callable, dict] \| None` | Trainer-specific.  Override the default torchopt factory.  Validated against the functional contract at construction. |
| `preprocess_logits_for_metrics` | `Callable \| None` | Vmap-batched.  Lets `compute_metrics` consume a reduced representation of logits. |

The constructor seeds Python / NumPy / torch global RNGs from
`args.seed`.  The DP RNG chain is keyed off `args.seed`
independently via `key(args.seed)`.

### Training and evaluation

```python
trainer.train(resume_from_checkpoint=None, ignore_keys_for_eval=None) -> TrainOutput
trainer.evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict[str, float]
trainer.predict(test_dataset, ignore_keys=None, metric_key_prefix="test") -> EvaluationResult
```

`resume_from_checkpoint` accepts a path string, `True` (automatically find the
latest `checkpoint-*/` under `output_dir`), `False`, or `None`.

`evaluate` returns the metrics dict only. Side effects on each call:

- Installs the cached-accountant barrier on the active training
  context.
- Appends a metrics row to `state.log_history` via `log()`.
- Fires `on_evaluate`.
- Updates `state.best_metric` / `state.best_global_step` when
  configured.
- Feeds the metric into `ReduceLROnPlateau` if configured.

`evaluate` does **not** accept a `dict[str, Dataset]`; multi-task eval
is the caller's loop.

`predict` returns the full `EvaluationResult` and fires `on_predict`.

Prediction payloads are offloaded in bounded chunks. On CUDA, a dedicated
transfer stream and at most two pinned staging chunks overlap device-to-host
copies with subsequent model work; completed chunks move to pageable host
memory. Completed CPU chunks merge as a balanced tree, and each merge/final
assembly allocates and pads its destination once instead of rebuilding a growing
prefix for every batch.

Evaluation metrics include aggregate `{prefix}_model_time_sec`,
`{prefix}_gather_time_sec`, `{prefix}_transfer_time_sec`,
`{prefix}_finalization_time_sec`, and `{prefix}_metric_time_sec` phase timings.
They also report `{prefix}_transfer_bytes`, `{prefix}_transfer_overlap_sec`, and
`{prefix}_transfer_overlap_ratio`; overlap is zero on non-CUDA devices.

### End-of-train metrics

`train()` returns `TrainOutput(global_step, training_loss, metrics)`.
`metrics` always includes:

- `train_loss` (running mean across the run)
- `train_steps`
- `train_runtime`, `train_samples_per_second`, `train_steps_per_second`
- `privacy_epsilon`, `privacy_delta`, `privacy_noise_multiplier`
- `num_input_tokens_seen` (only when
  `args.include_num_input_tokens_seen != "no"`)

### Logging and persistence

| Method | Purpose |
|---|---|
| `log(logs, start_time=None)` | Append a row to `state.log_history`, fire `on_log`. |
| `log_metrics(split, metrics)` | Pretty-print metrics for a split (`"train"`, `"eval"`, `"test"`). |
| `save_metrics(split, metrics, combined=True)` | Write `metrics.json` / `all_results.json` under `output_dir`. |
| `save_state()` | Write `trainer_state.json` under the effective output directory. |
| `save_model(output_dir=None)` | Write model weights + tokenizer + `training_args.bin` + `accountant.json`.  Does NOT write the DP runtime bundle (sampler / optimizer / RNG); resume requires `_save_checkpoint`'s output (driven by `save_strategy`). |
| `save_accountant(output_dir=None)` | Write only `accountant.json` and return its path.  Writes no model weights, so it is safe to call after an external saver has already mutated the model. |

### Distributed flags

```python
trainer.is_world_process_zero()  # True on global rank-0
trainer.is_local_process_zero()  # True on each node's rank-0
```

`TrainerState` mirrors these flags
(`state.is_world_process_zero`, `state.is_local_process_zero`) so
callbacks can rank-gate side effects.

### Callbacks

| Method | Effect |
|---|---|
| `add_callback(cb_or_cls)` | Add to handler and to `_base_callbacks` (survives `_reset_state_for_batch_size_retry`). |
| `remove_callback(cb_or_cls)` | Remove from handler. |
| `pop_callback(cb_or_cls)` | Remove and return. |

Auto-registered callbacks:

- `DefaultFlowCallback` — drives the should_log / should_evaluate /
  should_save flags.
- `BestModelSaveCallback` — auto-injected when
  `save_strategy="best"`.
- `ProgressCallback` / `PrinterCallback` per `disable_tqdm`.
- Reporting callbacks selected by `args.report_to` — wrapped via
  `wrap_reporting_callback_class` so privacy metrics
  (`privacy_epsilon`, `privacy_clip_rate`, …) become hierarchical
  paths (`privacy/epsilon`, `privacy/clip_rate`) for backends that
  support metric trees.

`on_substep_end` is **not fired** — each Poisson round is one atomic
clip-noise-step.

### Overridable methods

| Method | Override scope |
|---|---|
| `compute_per_example_loss(fmodel, params, inputs, *, return_logits=False)` | Per-example loss; called under vmap.  Primary extension point for custom training objectives. |
| `prediction_step(model, inputs, prediction_loss_only, ignore_keys=None)` | Override only if the default ModelOutput-shaped eval is wrong for the model. |
| `evaluation_loop(dataloader, *, description, prediction_loss_only, ignore_keys, metric_key_prefix)` | Override only if the full eval loop needs custom orchestration. |
| `create_optimizer()` | Override only to swap the functional torchopt optimizer; prefer `optimizer_cls_and_kwargs`. |
| `create_scheduler(num_training_steps)` | Override only for LR schedules outside the built-in factory. |
| `get_train_dataloader()` | Override only to swap the sampler family.  DP-correctness requires the sampler produces each example with independent probability `ctx.sample_rate`. |
| `get_eval_dataloader(eval_dataset=None)` | Standard `DataLoader` — usually no override needed. |

## `TrainingArguments`

Dataclass surface.  Every field listed here exists on
`opaque.transformers.TrainingArguments`.

### Privacy and DP-SGD

| Field | Type | Default | Purpose |
|---|---|---|---|
| `privacy_target_epsilon` | `float \| None` | `None` | User target ε. When set alone, calibration searches for the smallest noise multiplier that achieves this. With fixed σ, privacy early stopping is available only for independent DP-SGD and is rejected for horizon mechanisms. |
| `privacy_target_delta` | `float \| None` | `None` | Computed as `1 / dataset_size**1.1` when unset. |
| `clipping_mode` | `str` | `"fixed"` | One of `{"fixed", "adaptive", "auto"}`. |
| `clipping_norm` | `float \| dict[str, Any] \| str` | `1.0` | Scalar for global clipping; JSON dict with `"fallback"` key for per-group (keys are regex patterns over parameter names). |
| `clipping_kwargs` | `dict[str, Any] \| str` | `{}` | Mode-specific clipping kwargs. Adaptive accepts `target_quantile` and `clipping_norm_max`; auto accepts `gamma`; fixed accepts no keys. Also accepts JSON or HF-style comma-separated strings. |
| `sampling_mode` | `str` | `"auto"` | Resolved from `privacy_noise_mechanism`; see the complete mechanism-to-sampler table in the [TrainingArguments guide](../user-guide/huggingface/training-arguments.md#dp-ftrl-mechanisms). |
| `sampling_kwargs` | `dict[str, Any] \| str` | `{}` | Mode-specific sampler kwargs. `truncated_batch_size` caps Poisson draws; k-out-of-t accepts `k` and `allocation`; other modes accept no keys. |
| `privacy_noise_mechanism` | `str` | `"gaussian"` | One of `{"gaussian", "mf_band", "mf_blt", "mf_bisr", "mf_bsr", "mf_lambda_cgd", "mf_identity"}`. |
| `privacy_noise_multiplier` | `float \| None` | `None` | Fixed σ. When unset (and `privacy_target_epsilon` is set), calibration searches. Horizon mechanisms report the conservative declared full-horizon ε throughout training. |
| `privacy_noise_mechanism_kwargs` | `dict[str, Any] \| str` | `{}` | Gaussian accepts `compute_dtype` as a native `torch.dtype` or a string name (`float16`, `bfloat16`, `float32`, `float64`); MF mechanisms accept strategy-specific kwargs. |
| `noise_calibration_kwargs` | `dict[str, Any] \| str` | `{}` | Calibration search bounds.  When empty, `__post_init__` injects `{"min": 0.11, "max": 10.0, "tolerance": 1e-3}`. |

### Patches and kernels

| Field | Type | Default | Purpose |
|---|---|---|---|
| `use_compat_patches` | `bool` | `True` | vmap-safety patches (eager-attention, batchify, vmap-safe masking / collator / checkpoint hooks). |
| `use_performance_kernels` | `bool` | `False` | CUDA + Triton kernel group.  Auto-`False` on hosts without CUDA + Triton. |
| `performance_kernels_config` | `dict \| str \| None` | `None` | Flat patch overrides. Keys include `rope`, `rms_norm`, `activation`, `cross_entropy`, `fused_linear_cross_entropy`, `chunked_linear_cross_entropy`, `kv_cache`, `eager_attention`, `batchify`, `vmap_masking`, `empty_batches`, `vmap_checkpointing`. `chunked_linear_cross_entropy` accepts `True` for automatic 2-D tiling, a positive maximum vocabulary tile width, or `False`/`0` to disable it. |

### Compute and precision

| Field | Type | Default | Purpose |
|---|---|---|---|
| `use_cpu` | `bool` | `False` | Pin to CPU even if CUDA is available. |
| `use_mps_device` | `bool` | `False` | Use MPS (Apple Silicon). |
| `bf16` | `bool` | `False` | bf16 autocast on the loss closure. |
| `bf16_full_eval` | `bool` | `False` | Cast model to bf16 for eval scope only. |
| `tf32` | `bool \| None` | `None` | Toggle TF32 on Ampere+. |
| `gradient_checkpointing` | `bool` | `False` | Off by default: vmap recomputes activations per microbatch, so GC adds overhead without memory benefit on models that fit. Opaque automatically uses the vmap-safe non-reentrant path. Incompatible with `torch_compile`. |
| `gradient_checkpointing_kwargs` | `dict \| str \| None` | `None` | Forwarded to `model.gradient_checkpointing_enable(...)`. |
| `torch_compile` | `bool` | `False` | Compile each tensor-only microbatch gradient/clipping kernel with `fullgraph=True`. |
| `torch_compile_backend` | `str \| None` | `None` | Defaults to `"inductor"` when compile is on. |
| `torch_compile_mode` | `str \| None` | `None` | One of `{"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}`. |

### Batches and microbatching

| Field | Type | Default | Purpose |
|---|---|---|---|
| `per_device_train_batch_size` | `int` | `8` | Per-rank logical Poisson batch size. |
| `per_device_eval_batch_size` | `int \| None` | `None` | Eval batch size (fixed, not Poisson). When `None`, defaults to `microbatch_size` when configured, otherwise `per_device_train_batch_size`. |
| `microbatch_size` | `int \| None` | `None` | Physical vmap chunk size. Defaults to the logical per-device train batch; smaller values reduce peak memory without changing privacy accounting. |
| `eval_accumulation_steps` | `int \| None` | `None` | Move eval tensors to CPU every N batches. `None` offloads every batch, minimizing device prediction memory; larger windows reduce transfer calls but retain up to N batches on device. CUDA copies use a bounded asynchronous pinned staging queue; complete predictions remain in pageable CPU memory for final metrics. |
| `eval_delay` | `float` | `0.0` | Skip eval for the first N steps / epochs. |
| `auto_find_microbatch_size` | `bool` | `False` | On train OOM, halve the microbatch and retry; on eval/predict OOM, halve `per_device_eval_batch_size`. Logical batch and privacy unchanged. |

### Optimizer and LR

| Field | Type | Default |
|---|---|---|
| `learning_rate` | `float` | `5e-5` |
| `weight_decay` | `float` | `0.0` |
| `adam_beta1` / `adam_beta2` / `adam_epsilon` | `float` | `0.9` / `0.999` / `1e-8` |
| `optim` | `str` | `"adamw"` |
| `optim_args` | `dict \| str \| None` | `None` |
| `lr_scheduler` | `SchedulerType \| str \| Schedule` | `"linear"` |
| `lr_scheduler_kwargs` | `dict \| str \| None` | `{}` |
| `warmup_steps` | `int \| float` | `0` |

`optim` supports `{"adam", "adamw", "sgd", "rmsprop", "adagrad",
"adafactor", "ademamix", "lion", "radam", "adadelta",
"schedule_free"}`.  HF aliases (`adamw_torch`, `adamw_torch_fused`,
`lion_32bit`, …) are mapped to the matching opaque factory.  HF
names without a DP-aware mapping (8-bit, paged, GaLore, fused-CUDA,
NPU, XLA) are rejected with a redirect message.

For `schedule_free`, evaluation, checkpoints, and `save_model()` use the
published averaged weights. Resumable checkpoints retain the complete optimizer
state and reconstruct the training iterate when loaded.

### Training duration

| Field | Type | Default | Purpose |
|---|---|---|---|
| `num_train_epochs` | `float` | `3.0` | Epoch count. Fractional values run `ceil(num_train_epochs * steps_per_epoch)` steps; `state.epoch` is fractional. |
| `max_steps` | `int` | `-1` | When `> 0`, overrides `num_train_epochs`. |

### Logging

| Field | Type | Default | Purpose |
|---|---|---|---|
| `log_level` | `str` | `"passive"` | Rank-0 log level. |
| `log_level_replica` | `str` | `"warning"` | Non-zero rank log level. |
| `log_on_each_node` | `bool` | `True` | If `False`, only world-rank-0 logs. |
| `logging_dir` | `str \| None` | `None` | Defaults to `{output_dir}/runs`. |
| `logging_strategy` | `str` | `"steps"` | One of `{"no", "steps", "epoch"}`. |
| `logging_first_step` | `bool` | `False` | Emit a log row at step 0. |
| `logging_steps` | `float` | `500` | `< 1` interpreted as fraction of `max_steps`. |
| `disable_tqdm` | `bool \| None` | `None` | Auto-inferred from log level. |
| `report_to` | `str \| list[str] \| None` | `None` | `"wandb"`, `"tensorboard"`, `"mlflow"`, …; `"all"` expands; `None`/`"none"`/`[]` disables. |
| `run_name` / `project` | `str \| None` | `None` | W&B / TB / MLflow run name and project. |
| `trackio_space_id` | `str \| None` | `None` | Trackio Space identifier. |
| `trackio_bucket_id` | `str \| None` | `None` | Trackio storage bucket identifier. |
| `trackio_static_space_id` | `str \| bool \| None` | `None` | Trackio static Space selection. |

### Saving

| Field | Type | Default | Purpose |
|---|---|---|---|
| `save_strategy` | `str` | `"steps"` | `{"no", "steps", "epoch", "best"}`.  `"best"` auto-injects `BestModelSaveCallback`. |
| `save_steps` | `float` | `500` | `< 1` interpreted as fraction. |
| `save_total_limit` | `int \| None` | `None` | Older checkpoints rotated; best-checkpoint protected. |
| `save_safetensors` | `bool` | `True` | Safetensors vs `.bin`.  Load supports both. |
| `save_on_each_node` | `bool` | `False` | Every node's rank-0 writes a copy. |
| `save_only_model` | `bool` | `False` | Write a **weights-only export** (skips `dp_state.pt` / `dp_optimizer.pt`).  Not resumable — `resume_from_checkpoint` requires a complete DP checkpoint; load weights-only exports via `model=` for a fresh run. |
| `restore_callback_states_from_checkpoint` | `bool` | `False` | Restore per-callback state on resume. |
| `output_dir` | `str \| None` | `None` | Defaults to `"trainer_output"`. |
| `overwrite_output_dir` | `bool` | `False` | If `False`, warn when `output_dir` already contains checkpoints. |
| `resume_from_checkpoint` | `str \| None` | `None` | Path to a checkpoint directory.  `True` passed to `train()` auto-finds latest. |
| `enable_jit_checkpoint` | `bool` | `False` | Enable HF's SIGTERM-triggered just-in-time checkpoint callback; the saved checkpoint includes Opaque's DP runtime state. |

### Hub publishing

| Field | Type | Default | Purpose |
|---|---|---|---|
| `push_to_hub` | `bool` | `False` | Push the final model and DP provenance card after training. |
| `hub_model_id` | `str \| None` | `None` | Target Hub repository identifier. |
| `hub_token` | `str \| None` | `None` | Authentication token for Hub operations. |
| `hub_private_repo` | `bool \| None` | `None` | Create a private repository when the target does not exist. |
| `hub_revision` | `str \| None` | `None` | Target Hub branch or revision. |

### Evaluation

| Field | Type | Default | Purpose |
|---|---|---|---|
| `eval_strategy` | `str` | `"no"` | `{"no", "steps", "epoch"}`. |
| `eval_steps` | `float \| None` | `None` | Falls back to `logging_steps` when `eval_strategy="steps"` and unset. |
| `eval_on_start` | `bool` | `False` | Run one eval pass before any training steps. |
| `eval_do_concat_batches` | `bool` | `True` | Concat per-batch predictions / labels at finalize. |
| `prediction_loss_only` | `bool` | `False` | Skip logits / labels materialisation. |
| `include_for_metrics` | `list[str]` | `[]` | Subset of `{"inputs", "loss"}`.  `"loss"` routes eval through the vmap'd per-example closure. |
| `metric_for_best_model` | `str \| None` | `None` | Auto-prefixed with `eval_`.  Required for `save_strategy="best"` / `load_best_model_at_end=True`. |
| `greater_is_better` | `bool \| None` | `None` | Inferred from metric name (`loss` suffix → `False`). |
| `load_best_model_at_end` | `bool` | `False` | After training, restore the best-eval checkpoint.  Raises if no improving step recorded. |
| `ignore_data_skip` | `bool` | `False` | Skip sampler-state restore. Poisson resumes use a distinct stream. |

### Distributed

| Field | Type | Default | Purpose |
|---|---|---|---|
| `local_rank` | `int` | `-1` | Read from `LOCAL_RANK` env var. |
| `ddp_backend` | `str \| None` | `None` | One of `{"nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"}`. |
| `ddp_timeout` | `int` | `1800` | Seconds passed to `init_process_group(timeout=...)`. |
| `average_tokens_across_devices` | `bool` | `True` | Under DDP, average per-rank token counts for `num_input_tokens_seen`. |

### DataLoader

| Field | Type | Default |
|---|---|---|
| `dataloader_num_workers` | `int` | `0` |
| `dataloader_persistent_workers` | `bool` | `False` |
| `dataloader_pin_memory` | `bool` | `True` |
| `dataloader_prefetch_factor` | `int \| None` | `None` |
| `dataloader_multiprocessing_context` | `str \| None` | `None` |
| `dataloader_in_order` | `bool` | `True` |
| `dataloader_drop_last` | `bool` | `False` |
| `remove_unused_columns` | `bool` | `True` |
| `torch_empty_cache_steps` | `int \| None` | `None` |

### Labels and reproducibility

| Field | Type | Default | Purpose |
|---|---|---|---|
| `label_names` | `list[str] \| None` | `None` | Auto-discovered via `find_labels(model.__class__)`; defaults to `["labels"]`. |
| `label_smoothing_factor` | `float` | `0.0` | When `> 0`, applies `F.cross_entropy(..., label_smoothing=...)` on the exposed logits. |
| `seed` | `int` | `42` | Master seed (Python / NumPy / torch globals + DP RNG chain root). |
| `data_seed` | `int \| None` | `None` | Sampler seed.  Falls back to `seed` when unset. |
| `full_determinism` | `bool` | `False` | Enable deterministic algorithms. |

### Token counting and misc

| Field | Type | Default | Purpose |
|---|---|---|---|
| `include_tokens_per_second` | `bool` | `False` | Emit `train_tokens_per_second` in end-of-train metrics. |
| `include_num_input_tokens_seen` | `bool \| str` | `False` | `{"no", "all", "non_padding"}`.  `"non_padding"` uses `attention_mask` or `pad_token_id`. |
| `skip_memory_metrics` | `bool` | `True` | Skip HF-borrowed `TrainerMemoryTracker` snapshots. |
| `activation_offloading` | `bool` | `False` | Offload activations to CPU between forward and backward. |
| `debug` | `list \| str \| None` | `""` | HF debug flags.  Supports `"underflow_overflow"`. |

### Validation

`__post_init__` runs cross-field validation idempotently:

- Strategy strings validated against allowed sets.
- `warmup_steps` must be a non-negative number: a step count when `>= 1`,
  a fraction of the total training steps when in `(0, 1)`.
- `save_strategy="best"` requires `eval_strategy != "no"`.
- `load_best_model_at_end=True` requires both `save_strategy != "no"`
  and `eval_strategy != "no"`.
- `save_steps > 0` enforced when `save_strategy != "no"`.
- `torch_compile_mode` checked against the allowed set.
- `report_to="all"` expands via HF's `get_available_reporting_integrations()`.

Configuration errors raise `ConfigurationError` at construction.

Dict-shaped fields (`clipping_kwargs`, `sampling_kwargs`,
`noise_calibration_kwargs`, `privacy_noise_mechanism_kwargs`,
`optim_args`, `lr_scheduler_kwargs`,
`performance_kernels_config`, `gradient_checkpointing_kwargs`)
accept any of:

- `Mapping` (including OmegaConf `DictConfig`)
- JSON object string: `'{"a": 1, "b": 2}'`
- HF-style comma string: `"a=1,b=2"`
- `None`

Normalized to `dict[str, Any] | None` at construction.

## `EvaluationResult`

Return type for `evaluation_loop`, `evaluate`, and `predict` —
replaces HF's split `EvalLoopOutput` / `PredictionOutput`.

```python
@dataclass
class EvaluationResult:
    predictions: Any | None
    label_ids: Any | None
    metrics: dict[str, float]
    num_samples: int
```

`predictions` and `label_ids` are `None` when
`prediction_loss_only=True`.  Otherwise they're numpy arrays after
the opaque-distributed gather + truncation to the dataset's true
sample count.

## `TrainOutput`

```python
class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: dict[str, float]
```

Returned by `train()`.  Mirrors HF's `TrainOutput`.

## `opaque.transformers.trl` — SFT/DPO trainers

TRL-style class trainers built on `Trainer`.  Import the stable façade:

```python
from opaque.transformers.trl import (
    SFTConfig,
    SFTTrainer,
    DPOConfig,
    DPOTrainer,
)
```

| Symbol | Purpose |
|---|---|
| `SFTTrainer` | DP supervised fine-tuning trainer; mirrors `trl.SFTTrainer`. |
| `SFTConfig` | SFT training arguments — **extends `TrainingArguments`**. |
| `DPOTrainer` | DP Direct Preference Optimization trainer; mirrors `trl.DPOTrainer`. |
| `DPOConfig` | DPO training arguments — **extends `TrainingArguments`**. |

`SFTConfig` / `DPOConfig` subclass
[`TrainingArguments`](#trainingarguments), so **every** inherited field
(privacy / clipping / sampling, optimizer / LR / schedule, eval / save,
`activation_offloading`, …) is settable directly on them.  Only the
SFT/DPO-specific fields are listed below; for task-shaped usage see the
[SFT & DPO trainers guide](../alignment/trainers.md).

!!! note "`gradient_accumulation_steps`"
    Inherited from the base config as a read-only property pinned to `1`
    (no field).  Passing it to a config raises `TypeError`: under Poisson
    per-example DP, one round is one optimizer step.

### `SFTTrainer`

```python
SFTTrainer(
    model: PreTrainedModel | str | None = None,
    args: SFTConfig | None = None,
    data_collator: Callable | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    processing_class: PreTrainedTokenizerBase | None = None,
    compute_loss_func: Callable | None = None,
    compute_metrics: Callable | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[Any | None, Any | None] = (None, None),
    optimizer_cls_and_kwargs: tuple[Callable, dict] | None = None,
    preprocess_logits_for_metrics: Callable | None = None,
    peft_config: Any = None,
    formatting_func: Callable[[dict], str] | None = None,
)
```

| Argument | Notes |
|---|---|
| `model` | Module, or a **string** name/path loaded via `AutoModelForCausalLM.from_pretrained(model, **args.model_init_kwargs)`. |
| `args` | A plain `TrainingArguments` is upcast to `SFTConfig` field-by-field; `None` ⇒ `SFTConfig(output_dir="trainer_output")`. |
| `processing_class` | Tokenizer; loaded from the model's `_name_or_path` when omitted.  Pad token falls back to EOS. |
| `compute_loss_func` | Per-example `(outputs, labels) -> scalar`, run **under vmap**.  Honoured only on `loss_type="nll"`; rejected for `dft` / `chunked_nll`. |
| `peft_config` | When set, wraps the model with `get_peft_model`; chat-template-added tokens are marked trainable. |
| `formatting_func` | `example -> str`, rendered into `dataset_text_field` before tokenization. |

`optimizers` (non-`None`) is rejected — `Trainer` owns the functional
optimizer.

### `SFTConfig`

SFT-specific fields on top of `TrainingArguments`.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `learning_rate` | `float` | `2e-5` | TRL default (overrides the base `5e-5`). |
| `model_init_kwargs` | `dict \| None` | `None` | Forwarded to `from_pretrained` when `model` is a string; ignored for a module. |
| `trust_remote_code` | `bool` | `False` | Forwarded when loading a string model or tokenizer that ships custom Hub code. |
| `dataset_text_field` | `str` | `"text"` | Column holding raw text on a language-modeling dataset. |
| `max_length` | `int \| None` | `1024` | Tokenized sequence length cap; `None` disables truncation (keep-start). |
| `completion_only_loss` | `bool \| None` | `None` | Score only completion tokens. `None` enables this for prompt-completion data, but not language-modeling or chat data. |
| `assistant_only_loss` | `bool` | `False` | Score only assistant turns of chat data (installs the `{% generation %}` template + mask). |
| `eos_token` | `str \| None` | `None` | EOS appended to plain-text examples; overrides `tokenizer.eos_token` when set, else the tokenizer's own EOS is used. |
| `pad_to_multiple_of` | `int \| None` | `None` | Round the padded batch length up to a multiple. |
| `dataset_num_proc` | `int \| None` | `None` | Processes for `datasets.map` preprocessing. |
| `chat_template_path` | `str \| None` | `None` | Tokenizer dir / Jinja file whose chat template + special tokens are cloned onto `processing_class` (resizes embeddings). |
| `loss_type` | `str` | `"nll"` | `"nll"` (CE), `"dft"` (Dynamic Fine-Tuning), or `"chunked_nll"` (fused logits-free CE).  Unknown values fail at the loss dispatch. |
| `log_completion_metrics` | `bool` | `True` | Log per-step `entropy` / `mean_token_accuracy`; `False` skips them.  Also enables fused logits-free loss paths when `False`. |
| `logging_steps` | `float` | `10` | TRL default (overrides base `500`). |
| `gradient_checkpointing` | `bool` | `False` | Off by default: vmap recomputes activations per microbatch, so GC adds overhead without memory benefit on models that fit. |
| `use_performance_kernels` | `bool` | `True` | Opaque-specific: enable model-level Triton kernels (`rope`, `rms_norm`, `activation`, `cross_entropy`) by default.  CUDA + Triton only; no-op on CPU/MPS.  Overrides base `False`. |

`__post_init__` pins `remove_unused_columns=False` (the collator consumes
raw columns) and auto-enables `bf16` when the hardware supports it and no
precision was explicitly chosen.

### `DPOTrainer`

```python
DPOTrainer(
    model: PreTrainedModel | str | None = None,
    ref_model: PreTrainedModel | str | None = None,
    args: DPOConfig | None = None,
    data_collator: Callable | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    processing_class: PreTrainedTokenizerBase | None = None,
    compute_metrics: Callable | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[Any | None, Any | None] = (None, None),
    optimizer_cls_and_kwargs: tuple[Callable, dict] | None = None,
    preprocess_logits_for_metrics: Callable | None = None,
    peft_config: Any = None,
)
```

| Argument | Notes |
|---|---|
| `model` | Module or string name/path (loaded with `model_init_kwargs`). |
| `ref_model` | Module, a **string** name/path (loaded with `model_init_kwargs`), or `None`.  `None` ⇒ PEFT null-ref (adapter disabled), an auto-loaded copy for a string/path policy, or skipped for a reference-free `loss_type`.  Must be a different object from `model`. |
| `args` | A plain `TrainingArguments` is upcast to `DPOConfig`; `None` ⇒ `DPOConfig(output_dir="trainer_output")`. |
| `peft_config` | Wraps the model with `get_peft_model`.  TR-DPO (`sync_ref_model`) is rejected under PEFT. |

There is **no `compute_loss_func`** (the DPO loss is built from the
configured heads) and **no `reference_free` flag** — reference-need is
derived from `loss_type`.

### `DPOConfig`

DPO-specific fields on top of `TrainingArguments`.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `learning_rate` | `float` | `1e-6` | TRL default. |
| `model_init_kwargs` | `dict \| None` | `None` | Forwarded to `from_pretrained` for a string `model` **and** a string `ref_model`. |
| `trust_remote_code` | `bool` | `False` | Forwarded when loading string policy, reference-model, or tokenizer identifiers that ship custom Hub code. |
| `loss_type` | `list[str] \| str` | `["sigmoid"]` | One or more head names (a list ⇒ MPO).  A bare string is coerced to a one-element list.  See the [head menu](../alignment/trainers.md#the-loss_type-menu). |
| `loss_weights` | `list[float] \| None` | `None` | Per-loss MPO weights; `None` ⇒ all-ones.  Must match `len(loss_type)`. |
| `beta` | `float` | `0.1` | Policy–reference KL strength (τ for IPO). |
| `label_smoothing` | `float` | `0.0` | Robust-DPO flip prob `[0, 0.5)`; ε for EXO. |
| `f_divergence_type` | `str` | `"reverse_kl"` | `reverse_kl` / `forward_kl` / `js_divergence` / `alpha_divergence`. |
| `f_alpha_divergence_coef` | `float` | `0.5` | α coefficient for `alpha_divergence`. |
| `ld_alpha` | `float \| None` | `None` | LD-DPO verbose-token weight `[0, 1]`; `None` ⇒ standard DPO. |
| `use_weighting` | `bool` | `False` | WPO length-normalized probability weighting. |
| `discopop_tau` | `float` | `0.05` | DiscoPOP temperature. |
| `simpo_gamma` | `float` | `0.5` | SimPO target reward margin γ. |
| `cpo_alpha` | `float` | `1.0` | CPO chosen-NLL regularizer weight. |
| `orpo_lambda` | `float` | `1.0` | ORPO odds-ratio term weight. |
| `precompute_ref_batch_size` | `int \| None` | `None` | Reference-precompute batch size; defaults to the train batch size. |
| `disable_dropout` | `bool` | `True` | Zero dropout in policy + reference before training. |
| `sync_ref_model` | `bool` | `False` | TR-DPO: EMA-sync the reference toward the policy (full FT only; reference-using `loss_type`). |
| `ref_model_mixup_alpha` | `float` | `0.6` | TR-DPO EMA mixup α: `ref ← (1-α)·ref + α·policy`. |
| `ref_model_sync_steps` | `int` | `512` | TR-DPO sync cadence (steps). |
| `max_length` | `int \| None` | `1024` | Tokenized sequence length cap (keep-start). |
| `pad_to_multiple_of` | `int \| None` | `None` | Round the padded batch length up to a multiple. |
| `dataset_num_proc` | `int \| None` | `None` | Processes for `datasets.map` preprocessing. |
| `log_completion_metrics` | `bool` | `True` | Log per-step `logits/*` / `entropy` / `mean_token_accuracy`; rewards + `logps/*` are always logged. |
| `logging_steps` | `float` | `10` | TRL default. |
| `gradient_checkpointing` | `bool` | `False` | Off by default: vmap recomputes activations per microbatch, so GC adds overhead without memory benefit on models that fit. |
| `use_performance_kernels` | `bool` | `True` | Opaque-specific: enable model-level Triton kernels (`rope`, `rms_norm`, `activation`, `cross_entropy`) by default.  CUDA + Triton only; no-op on CPU/MPS.  Overrides base `False`. |

The reference-free heads are `{"chosen_nll", "simpo", "cpo", "orpo"}` (TRL's
`sft` is `chosen_nll` here); a run is reference-free iff *every* configured
head is in that set.
`__post_init__` coerces `loss_type` to a list, defaults `loss_weights`,
pins `remove_unused_columns=False`, validates label-smoothing bounds /
weight lengths / duplicate heads / TR-DPO reference-need, and auto-enables
`bf16` on supporting hardware.

## Runtime patches

```python
from opaque.patches import apply_runtime_patches

apply_runtime_patches(compat=True)  # install the global HF shims once
```

`Trainer` applies these (and the per-model patches) during construction, so
you only need this when driving DP-SGD over HF models **without** `Trainer`.

For per-model patches and the kernel surface, see
[Model Patches and Kernels](../user-guide/huggingface/model-patches.md).

## API documentation

::: opaque.transformers
    options:
      show_source: true
      heading_level: 3

::: opaque.transformers.trainer
    options:
      show_source: true
      heading_level: 3

::: opaque.transformers.trl
    options:
      show_source: true
      heading_level: 3
