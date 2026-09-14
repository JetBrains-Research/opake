# Hugging Face Integration

Opaque ships a Hugging Face–style trainer (`opaque.transformers.Trainer`)
plus a per-model patch surface (`opaque.patches`) so existing
`transformers` models can be trained under DP-SGD with the familiar
`Trainer.train()` / `evaluate()` / `predict()` interface.

This section is the routing guide. Pick the page that matches what
you're trying to do.

## What's covered

- **[Trainer](trainer.md)** — common training / eval / predict
  usage. Start here if you have a model and a dataset and want to
  train.
- **[TrainingArguments](training-arguments.md)** — the most-used DP
  knobs, the batch-size contract, and the save / resume claims. The
  exhaustive field listing lives in the API reference.
- **[Model patches](model-patches.md)** — what `apply_runtime_patches`
  and `apply_model_patches` do, why vmap needs them, the model
  compatibility matrix, the Triton kernel surface (including fused
  LoRA), and how to bring your own model.
- **[API reference — transformers](../../reference/transformers.md)** —
  full parameter inventory for `Trainer`, `TrainingArguments`, and
  the public state objects.

LoRA and other PEFT setups use the standard `peft` library workflow;
the only Opaque-specific piece is
`make_functional(model, partition_trainable=True)` — documented under
[Utilities reference](../../reference/utilities.md#trainable-frozen-partition-for-peft-and-lora).

## Which page do I start on?

```
Have a model + dataset, want to train?
│
├─ Yes ── start with trainer.md
│         then training-arguments.md for tuning
│
└─ No ─── Something else
          ├─ Model not on compatibility matrix → model-patches.md
          └─ Need exact field types / defaults → reference/transformers.md
```

## Quick start

A minimal Trainer run looks like a Hugging Face `Trainer` run with a
few DP-specific fields on `TrainingArguments`:

```python
from transformers import AutoModelForCausalLM
from opaque.transformers import Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
args = TrainingArguments(
    output_dir="run-0",
    per_device_train_batch_size=8,
    num_train_epochs=3.0,
    # Trainer-specific privacy knobs:
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
trainer.train()
metrics = trainer.evaluate()
```

The trainer auto-patches the model on construction (vmap-safety,
optional Triton kernels) and calibrates the noise multiplier from
`privacy_target_epsilon`.  Every detail lives in the topic pages
linked above.

## Scope

The HF integration prioritizes **decoder-only text models**
(`*ForCausalLM` and shared text modules) tested against
`transformers==4.57.1`.  Vision-language stacks (e.g.
`*ForConditionalGeneration`) are not in the default patch set; see
[Model patches — model compatibility](model-patches.md#model-compatibility)
for the curated matrix.
