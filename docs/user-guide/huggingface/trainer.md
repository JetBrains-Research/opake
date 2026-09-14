# Trainer

`Trainer` mirrors Hugging Face's `Trainer` interface: construct with
`(model, args, datasets, …)`, call `train()` / `evaluate()` /
`predict()`, and query the result. Under the surface, the loop runs DP-SGD
with `vmap(grad(...))` over per-example losses; the public API is
designed so a user familiar with `Trainer` doesn't have to think about
that.

This page covers the common usage patterns. For the full constructor
signature, every method's parameters / return type, and the public
state objects (`EvaluationResult`, `TrainerState`, `TrainOutput`),
see [API reference — transformers](../../reference/transformers.md).

## Minimal training loop

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from opaque.transformers import Trainer, TrainingArguments

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

args = TrainingArguments(
    output_dir="run-0",
    per_device_train_batch_size=8,
    num_train_epochs=1.0,
    learning_rate=5e-4,
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,
)
out = trainer.train()
print(out.global_step, out.training_loss, out.metrics["privacy_epsilon"])
```

Construction immediately moves the model to the resolved device and
auto-patches it (vmap-safety + optional Triton kernels).  The noise
multiplier is calibrated from `privacy_target_epsilon` at the start of
`train()` (once the dataset size and step count are known), not at
construction. No extra setup calls are needed before `train()`.

At least one of `privacy_noise_multiplier` / `privacy_target_epsilon`
must be set — neither has a silent default; see the
[Privacy targets](./training-arguments.md#privacy-targets) section of
the `TrainingArguments` reference for calibration, fixed-noise accounting,
and the independent-DP-SGD stop-at-ε contract.

## Training, evaluation, prediction

```python
trainer.train()                                  # full DP-SGD loop
trainer.train(resume_from_checkpoint=True)       # auto-find latest checkpoint
metrics = trainer.evaluate()                     # returns dict[str, float]
result = trainer.predict(test_ds)                # returns EvaluationResult
```

End-of-train metrics always include `privacy_epsilon`,
`privacy_delta`, and `privacy_noise_multiplier` alongside the usual
loss / step / runtime fields. Evaluation and prediction route through
the same loop; `predict()` returns the full `EvaluationResult`
(predictions + labels + metrics) while `evaluate()` returns only the
metrics dict.

For horizon mechanisms, `privacy_epsilon` is always the conservative epsilon
for the complete declared `n_steps`, even before the final step. Prefix privacy
queries and privacy-based early stopping are unsupported, so combining a fixed
`privacy_noise_multiplier` with `privacy_target_epsilon` is rejected.

`resume_from_checkpoint` accepts a path string, `True` to auto-find
the latest `checkpoint-*/` under `output_dir`, or `None` for a fresh
run.

Sampler checkpoints track trainer-consumed batches, excluding worker prefetch.
On restore, a saved cursor greater than `global_step` is clamped to
`global_step` with a warning before replaying the sampler stream. Smaller
cursors are preserved, including those from an `ignore_data_skip` restart.

## Per-example eval losses

Setting `args.include_for_metrics=["loss"]` switches the eval path to
the `vmap`-based per-example closure: `EvalPrediction.losses` carries real
per-example losses instead of the batch-mean repeated.  Useful for
threshold calibration, percentile metrics, or member / non-member
analyses driven by `compute_metrics`.

## Logging and saving

The standard surface from `Trainer` is preserved:

- `log(logs)` — append to `state.log_history` and fire `on_log`
  callbacks.
- `save_state()` — write `trainer_state.json` under `output_dir`.
- `save_model(output_dir=None)` — write weights + tokenizer +
  `training_args.bin` + `accountant.json`.  The accountant always
  travels with the saved model.
- `save_accountant(output_dir=None)` — write just `accountant.json` and
  return its path, for building your own final artifact without
  re-saving the model.

Checkpointing during training is driven by `save_strategy` /
`save_steps` on `TrainingArguments`; resume from a checkpoint
preserves the privacy accountant. Independent DP-SGD restores the executed
composition and calibrates against the remaining steps; horizon mechanisms
retain their single declared full-horizon process.

## Callbacks

User callbacks are passed via the `callbacks=` constructor arg or
added later:

```python
trainer.add_callback(MyCallback())
trainer.remove_callback(MyCallback)
```

The trainer auto-registers `DefaultFlowCallback`, the progress
callback pair, and `BestModelSaveCallback` (only when
`save_strategy="best"`).  Reporting integrations selected by
`args.report_to` (W&B, TensorBoard, MLflow, …) are wrapped so the
privacy metric keys (`privacy/epsilon`, `privacy/clip_rate`, …) land
as hierarchical paths in backends that support metric trees.

`on_substep_end` is **not** fired — each DP-SGD step is one atomic
clip-noise-step over one Poisson sample.

## Custom losses

`compute_loss_func` on the constructor accepts a callable
`(outputs, labels) -> scalar` for one-off losses without subclassing.
Training invokes it per example under `vmap`; default evaluation invokes it
once per batch. There is no `num_items_in_batch` argument, and a custom loss
takes precedence over `label_smoothing_factor`.

## PEFT and LoRA

PEFT-wrapped models work transparently: wrap the model with
`peft.get_peft_model(...)` before constructing the trainer.  The
trainer detects the `PeftModel` and routes the functional path through
`make_functional(model, partition_trainable=True)` internally — only
LoRA adapters receive per-example gradients (essential for memory at
multi-billion-parameter scale).  Fused LoRA Triton kernels engage
automatically when adapters have `bias="none"` — see
[Model patches — Fused LoRA operations](model-patches.md#fused-lora-operations).

## Non-private baseline

To measure the utility cost of differential privacy, run the trainer with
noise turned off — `privacy_noise_multiplier=0.0` — keeping everything else
(mechanism, sampler, optimizer, accounting) identical:

```python
import math

from opaque.transformers.trainer import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        privacy_noise_multiplier=0.0,   # σ = 0 → no noise added
        clipping_norm=math.inf,         # optional: disable clipping for plain SGD
        max_steps=100,
    ),
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
out = trainer.train()
assert out.metrics["privacy_epsilon"] == float("inf")  # no guarantee, reported faithfully
```

The accountant composes a non-private step, so `privacy_epsilon` is `inf`
rather than a finite budget — calibration is skipped and the run proceeds
normally. Clipping stays on by default (set `clipping_norm=math.inf` to
disable it as well). See
[TrainingArguments — Non-private baseline](training-arguments.md#non-private-baseline)
for the full contract.

## See also

- [TrainingArguments](training-arguments.md) — the most-tuned
  DP-specific knobs and the batch-size contract.
- [Model patches](model-patches.md) — what the trainer auto-applies
  and how to opt parts in or out.
- [API reference — transformers](../../reference/transformers.md) —
  full parameter inventory for `Trainer`, `TrainingArguments`, and
  the public state objects.
