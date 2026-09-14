# SFT & DPO trainers end-to-end

The [SFT](sft.md) and [DPO](dpo.md) guides build a DP run manually from
`opaque.alignment` primitives. This guide covers the **class-based trainers** that
wrap that pipeline: `opaque.transformers.trl.SFTTrainer` and `DPOTrainer`.
They mirror `trl.SFTTrainer` / `trl.DPOTrainer` in structure and method names,
but route the gradient through Opaque's per-example DP
[`Trainer`](../user-guide/huggingface/trainer.md) — one Poisson round is
one clip-noise-step. The alignment collators, losses, and reference helpers are
the same primitives the by-hand guides use; the trainer is the orchestration
layer.

The supported `loss_type` values map onto the corresponding alignment papers:
[DPO](https://arxiv.org/abs/2305.18290), [IPO](https://arxiv.org/abs/2310.12036),
[DiscoPOP](https://arxiv.org/abs/2406.08414), [SimPO](https://arxiv.org/abs/2405.14734),
[ORPO](https://arxiv.org/abs/2403.07691), [WPO](https://arxiv.org/abs/2406.11827),
[LD-DPO](https://arxiv.org/abs/2409.06411), [APO](https://arxiv.org/abs/2408.06266),
[NCA](https://arxiv.org/abs/2402.05369),
[BCO](https://arxiv.org/abs/2404.04656), [SPPO](https://arxiv.org/abs/2405.00675),
and [Dynamic Fine-Tuning](https://arxiv.org/abs/2508.05629).

```python
from opaque.transformers.trl import SFTConfig, SFTTrainer
from opaque.transformers.trl import DPOConfig, DPOTrainer
```

Both configs **extend** the base
[`TrainingArguments`](../reference/transformers.md#trainingarguments), so every
DP knob — `privacy_target_epsilon`, `clipping_norm`, `clipping_mode`,
`sampling_kwargs`, `privacy_noise_multiplier`, the optimizer / LR / schedule
fields, and every eval / save field — is settable directly on `SFTConfig` /
`DPOConfig`. This guide covers only the SFT/DPO-specific fields; for the
inherited DP/clipping/sampling surface see
[TrainingArguments](../user-guide/huggingface/training-arguments.md) and the
[transformers reference](../reference/transformers.md#trainingarguments).

!!! note "`gradient_accumulation_steps` is not a usable knob"
    Under Poisson per-example DP, one round **is** one optimizer step, so there
    is no gradient accumulation. `gradient_accumulation_steps` is a read-only
    property pinned to `1` (it has no field), so passing it to a config raises
    `TypeError`. Grow the effective batch with `per_device_train_batch_size`
    instead; the physical vmap chunk is decoupled (`microbatch_size` /
    `auto_find_microbatch_size`) and privacy-neutral.

## SFTTrainer

The minimal call mirrors TRL: a model (or model name), an `SFTConfig`, a
dataset, and a tokenizer.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from opaque.transformers.trl import SFTConfig, SFTTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
dataset = load_dataset("roneneldan/TinyStories", split="train")

args = SFTConfig(
    output_dir="trainer_output/sft",
    dataset_text_field="text",
    loss_type="nll",                 # or "dft" / "chunked_nll"
    max_length=1024,
    per_device_train_batch_size=8,
    max_steps=50,
    # DP knobs (inherited from TrainingArguments)
    clipping_norm=1.0,
    privacy_noise_multiplier=0.8,    # or privacy_target_epsilon=8.0 to calibrate
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Model loading

Pass `model` as an already-instantiated module, or as a **string** name/path —
then `SFTTrainer` calls `AutoModelForCausalLM.from_pretrained(model,
**model_init_kwargs)`. Use `model_init_kwargs` (a dict on `SFTConfig`) to
forward load-time kwargs such as `torch_dtype` or `attn_implementation`; it is
ignored when `model` is already a module.

```python
args = SFTConfig(
    output_dir="out",
    model_init_kwargs={"torch_dtype": "bfloat16"},
)
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B", args=args, ...)
```

The tokenizer (`processing_class`) is loaded from the model's `_name_or_path`
when omitted; if it has no pad token, the EOS token is reused. Set
`eos_token` on the config to override the EOS appended to plain-text examples.

### Data formats and the loss

`SFTTrainer` tokenizes the dataset for you, dispatching on format: a plain-text
column (`dataset_text_field`, default `"text"`), a `prompt`/`completion` pair,
or a chat-message column (`messages` / `conversations` / `chat`). Pass a
`formatting_func(example) -> str` to render arbitrary rows into the text field
first. Already-tokenized datasets (with an `input_ids` column) pass through
untouched.

`loss_type` selects `"nll"` (standard cross-entropy), `"dft"` (Dynamic
Fine-Tuning), or `"chunked_nll"` (fused logits-free cross-entropy). All use a
DP-safe per-example token-count divisor — see
[SFT end-to-end](sft.md#2-per-example-loss). A custom
`compute_loss_func(outputs, labels) -> scalar` is honoured **only** on the
`"nll"` path (it runs inside vmap, so it must be a pure per-example op — no
`num_items_in_batch`); `"dft"` and `"chunked_nll"` compute their own loss and
reject a custom function.

Set `completion_only_loss=True` to score only the completion tokens of
prompt-completion data, or `assistant_only_loss=True` to score only assistant
turns of chat data (the trainer installs the `{% generation %}`-marked training
chat template and recovers the assistant-token mask). Left as the default
`None`, `completion_only_loss` auto-detects: `True` for prompt-completion data
and `False` for language-modeling data, including chat. Ordinary chat therefore
scores the full rendered conversation; enable `assistant_only_loss` to mask its
non-assistant tokens. Use `chat_template_path` to clone a chat template (and its
special tokens) from another tokenizer/Jinja file onto the processing class
before tokenizing — this resizes the model's embeddings, and under PEFT the new
token rows are marked trainable.

### Telemetry and memory

`log_completion_metrics` (default `True`) gates the per-step logits-derived
diagnostics — `entropy` and `mean_token_accuracy` over the supervised
(non-`-100`) tokens. Set it to `False` to skip materializing them when you
don't need them.

`activation_offloading` (inherited from the base config, shared by SFT and DPO)
offloads activations to CPU between the forward and backward to trade host
bandwidth for GPU memory.

!!! note "TRL-parity defaults"
    `SFTConfig` overrides a few base defaults to match TRL: `learning_rate`
    `2e-5`, `logging_steps` `10`, `gradient_checkpointing` `True`, and bf16
    auto-enabled when the hardware supports it (and you didn't pick a
    precision). `remove_unused_columns` is pinned `False` (the collator
    consumes raw dataset columns).

## DPOTrainer

`DPOTrainer` adds a frozen reference policy and preference data. The minimal
call adds a `ref_model` (or relies on auto-load / PEFT null-ref / a
reference-free `loss_type`):

```python
from opaque.transformers.trl import DPOConfig, DPOTrainer

args = DPOConfig(
    output_dir="trainer_output/dpo",
    loss_type="sigmoid",
    beta=0.1,
    max_length=1024,
    per_device_train_batch_size=8,
    max_steps=50,
    clipping_norm=1.0,
    privacy_noise_multiplier=0.8,
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    ref_model="Qwen/Qwen2.5-0.5B-Instruct",   # str, module, or None
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

The trainer tokenizes preference rows (`prompt`/`chosen`/`rejected`, with the
shared prompt auto-extracted when only `chosen`/`rejected` are given) and, for
reference-using losses, runs a **one-shot reference precompute** before
training that attaches per-example `ref_chosen_logps` / `ref_rejected_logps`
columns. The per-example loss reads those as constants, so no second model runs
inside vmap (see [DPO end-to-end](dpo.md#1-reference-log-probabilities)). The
reference is always precomputed *summed*; length-normalized heads divide by the
completion length at loss time, so one precompute serves both summed and
normalized variants.

### Reference loading

How the reference is resolved (no `reference_free` flag exists — reference-need
is derived from `loss_type`, below):

| `ref_model` | Behavior |
|---|---|
| a module | Used as-is; left in the device and mode in which it arrived. |
| a **string** | Loaded via `AutoModelForCausalLM.from_pretrained(ref_model, **model_init_kwargs)` — `model_init_kwargs` is threaded into the reference load too. |
| `None`, PEFT policy | The base model is the reference; the adapter is disabled around the reference forward (`null_ref_context`). No second model. |
| `None`, string/path policy | A reference **copy is auto-loaded** from the policy path. |
| `None`, in-memory policy, not PEFT | No reference resolvable → raises early (pass `ref_model=`, use PEFT, or a reference-free `loss_type`). |

`precompute_ref_batch_size` sets the batch size for the precompute pass
(defaults to the train batch size). `disable_dropout` (default `True`) zeros
dropout in the policy and reference before training.

### The `loss_type` menu

`loss_type` is one name or a **list** of names (a list ⇒ MPO; see below). The
trainer dispatches each name to an `opaque.alignment.dpo` head; an unknown name
fails with a `KeyError` at the dispatch table. The supported names:

| `loss_type` | Method | Reference? |
|---|---|---|
| `sigmoid` | Standard DPO logistic loss (the default). | yes |
| `sigmoid_norm` | Length-normalized sigmoid. | yes |
| `hinge` | DPO hinge loss. | yes |
| `ipo` | IPO (length-normalized). | yes |
| `robust` | Robust / label-smoothed (cDPO). | yes |
| `exo_pair` | EXO pairwise (needs `label_smoothing > 0`). | yes |
| `nca_pair` | NCA pairwise. | yes |
| `bco_pair` | BCO pairwise. | yes |
| `sppo_hard` | SPPO hard-label. | yes |
| `apo_zero` / `apo_down` | APO-zero / APO-down. | yes |
| `discopop` | DiscoPOP (temperature `discopop_tau`). | yes |
| `simpo` | SimPO: length-normalized sigmoid with a target margin. | **no** |
| `cpo` | CPO: reference-free sigmoid + `cpo_alpha`·chosen-NLL. | **no** |
| `orpo` | ORPO: odds-ratio + `orpo_lambda`·chosen-NLL. | **no** |
| `chosen_nll` | Chosen-completion NLL only (TRL calls this `sft`). | **no** |

**Reference-free heads.** The four reference-free names —
`simpo`, `cpo`, `orpo`, `chosen_nll` — score the policy's own (length-normalized)
log-prob, so the reference precompute is **skipped entirely**. Reference-need
is intrinsic to the configured heads: a run needs a reference iff *any*
configured head is reference-using. Their flat parameters live on `DPOConfig`:

| Parameter | Default | Head | Meaning |
|---|---|---|---|
| `simpo_gamma` | `0.5` | `simpo` | Target reward margin γ subtracted inside the sigmoid. |
| `cpo_alpha` | `1.0` | `cpo` | Weight on the chosen-completion NLL term. |
| `orpo_lambda` | `1.0` | `orpo` | Weight on the odds-ratio term. |

`cpo` and `orpo` are reference-free **composites** assembled by the trainer (a
preference / odds-ratio term plus a per-token-mean chosen NLL); they are not
single exported heads. SimPO and ORPO use the length-normalized per-token
reward; CPO's sigmoid uses the summed policy log-probs.

### BCO baseline

`DPOTrainer` uses BCO's zero baseline (`delta=0.0`); it does not implement
TRL's cross-batch running reward mean. A manual nonzero `delta` must be
public, separately DP, or derived from prior DP outputs; detaching a raw
current- or cross-batch statistic is not sufficient.

### MPO and `loss_weights`

Passing `loss_type` as a list combines the heads into one per-example loss
(MPO — Mix of Preference Objectives — via `mpo_combine`). `loss_weights` (a
list of floats, defaulting to all ones) weights each term and **must match the
length of `loss_type`**; duplicate loss names are rejected. A list may freely
mix length-normalized and summed variants — the reference is always precomputed
summed and normalized per head at loss time.

```python
# MPO: weighted DPO sigmoid + a chosen-NLL (RPO-style) regularizer.
args = DPOConfig(
    output_dir="out",
    loss_type=["sigmoid", "chosen_nll"],
    loss_weights=[1.0, 0.1],
    beta=0.1,
)
```

### TR-DPO (reference sync)

`sync_ref_model=True` periodically moves the reference toward the policy by an
EMA step — `ref ← (1 - α)·ref + α·policy` — recomputed per step **outside**
vmap. `ref_model_mixup_alpha` (default `0.6`) is α; `ref_model_sync_steps`
(default `512`) is the cadence. TR-DPO requires a reference-using `loss_type`
(nothing to sync toward otherwise) and full fine-tuning (not PEFT). Under
TR-DPO the per-step reference logps are recomputed from the evolving reference,
overwriting the seeded columns, and eval scores against the same reference.

### WPO, LD-DPO, and f-divergence

These are flat fields on `DPOConfig`, composing on top of the head(s):

- **WPO** — `use_weighting=True` reweights each example by the policy's
  detached average completion probability on both sides (arXiv:2406.11827).
  The weight is detached and per-example, so per-example DP is preserved.
- **LD-DPO** — `ld_alpha` (a float in `[0, 1]`, `None` ⇒ standard DPO)
  length-desensitizes the sequence log-prob by damping the verbose tail of each
  completion beyond the paper's public length (the shorter completion length;
  arXiv:2409.06411).
- **f-divergence** — `f_divergence_type` selects the regularizer:
  `"reverse_kl"` (default, standard DPO), `"forward_kl"`, `"js_divergence"`, or
  `"alpha_divergence"`. For the α-divergence, `f_alpha_divergence_coef`
  (default `0.5`) is the α coefficient. A non-reverse-KL type remaps the
  log-ratios before the reference-using heads.

### Telemetry

`log_completion_metrics` (default `True`) gates the logits-consuming
per-example diagnostics: `logits/chosen`, `logits/rejected`, `entropy`, and
`mean_token_accuracy`. The reward telemetry — `rewards/chosen`,
`rewards/rejected`, `rewards/accuracies`, `rewards/margins` — and
`logps/chosen` / `logps/rejected` are always logged (they are free byproducts
of the loss). Every telemetry tensor is detached, riding the clipped-grad aux
channel without leaking gradient; the eval loop aggregates the same dict under
`eval_*`.

`activation_offloading` (inherited base field) applies to DPO too.

## Converting from HF / TRL

Pipelines that already construct an HF `TrainingArguments` or a
`trl.SFTConfig` / `trl.DPOConfig` can pass the existing config to the
Opaque equivalents through three class methods:

- `TrainingArguments.from_hf(hf_args, ...)` — base HF translation.
- `SFTConfig.from_trl(trl_sft_cfg, ...)` — TRL SFT translation.
  Requires the optional `trl` extra: `pip install opaque[trl]`.
- `DPOConfig.from_trl(trl_dpo_cfg, ...)` — TRL DPO translation. Same
  extra.

Each class method accepts the same DP-knob kwargs as a regular Opaque
config — at least one of `privacy_noise_multiplier` or
`privacy_target_epsilon` is required, the rest have sensible defaults.

```python
from trl import SFTConfig as TrlSFTConfig
from opaque.transformers.trl import SFTConfig

trl_cfg = TrlSFTConfig(
    output_dir="trainer_output/sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_steps=50,
    bf16=True,
    dataset_text_field="text",
    loss_type="nll",
)

opaque_cfg = SFTConfig.from_trl(
    trl_cfg,
    privacy_noise_multiplier=0.8,
    clipping_norm=1.0,
)
# opaque_cfg.per_device_train_batch_size == 8   (2 × 4: privacy-relevant batch)
# opaque_cfg.microbatch_size             == 2   (HF microbatch → vmap chunk)
```

### Batch semantics

DP-SGD's sample-rate denominator (and therefore $\epsilon$) is the
**logical batch** — the unit over which one gradient + noise step
applies. In HF terms that's `per_device_train_batch_size ×
gradient_accumulation_steps`; in opaque terms it's
`per_device_train_batch_size` alone. The converter collapses HF's
two-field expression into opaque's one-field expression by multiplying,
and folds the HF microbatch into opaque's `microbatch_size` (the vmap
chunk inside the per-example clipped-gradient pass). Dropping
`gradient_accumulation_steps` on the floor would under-account the
sampling amplification and emit a too-optimistic $\epsilon$.

### Rejected fields (raise `ValueError`)

The converter raises with a per-field rationale when the source config
sets any of these:

| HF / TRL field | Why rejected |
|---|---|
| `fp16=True`, `fp16_*` | DP-SGD is bf16-only; use `bf16=True`. |
| `fsdp`, `fsdp_config`, `fsdp_*` | FSDP is not on the per-example DP path. |
| `deepspeed` | DeepSpeed is not on the per-example DP path. |
| `accelerator_config` | Accelerate-driven config is not used. |
| `neftune_noise_alpha` | NEFTune would interact with the privacy accountant. |
| `max_grad_norm` (non-default) | Pre-step global norm clipping has no opaque analogue; use `clipping_norm` for per-example DP clipping instead. |
| `optim="paged_adamw_*"`, `*_8bit`, `*_apex_fused` | Quantized / Apex-fused optimizers are not in opaque-engine's torchopt path. |
| `use_liger_kernel`, `liger_kernel_config` | Liger fused kernels are not on the DP-SGD path. |
| TRL `packing=True`, `padding_free=True`, `eval_packing=True` | Sequence packing / unpadded forwards break the fixed per-example batch shape DP-SGD's vmap requires. |
| TRL `shuffle_dataset=True` | Opaque's Poisson sampler controls ordering. |
| TRL `truncation_mode="keep_end"` | Opaque only supports `keep_start`. |
| TRL `pad_token=...` | Set `tokenizer.pad_token` directly. |
| DPO `loss_type=["aot", ...]` | TRL 1.x added Adversarial Optimal Transport heads opaque doesn't implement. |

### Dropped fields (silent + `RuntimeWarning` when non-default)

HF fields that have no opaque effect get silently dropped — the
converter emits a `RuntimeWarning` if the user set them to a
non-baseline value. Examples: `do_train`, `do_eval`, `tpu_*`,
`ray_scope`, `optim_target_modules`, `batch_eval_metrics`,
`eval_use_gather_object`, `hub_strategy`, `accelerator_config`.

Pass `strict=False` to suppress the warnings entirely.

### Round-trip is one-way

The class methods only translate HF/TRL → Opaque. Opaque-only fields
(every DP knob, `microbatch_size > per_device_eval_batch_size`,
opaque-specific clipping / sampling / noise-mechanism families) cannot
be expressed in HF or TRL terms, so the reverse conversion is not
supported and not implemented.

## Runnable references

- [`examples/train_sft_trainer.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_sft_trainer.py)
  — DP SFT via the class-based `SFTTrainer` (LoRA policy, `nll`/`dft`).
- [`examples/train_dpo_trainer.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpo_trainer.py)
  — DP DPO via the class-based `DPOTrainer` (reference precompute, LoRA
  null-ref).

## See also

- [SFT end-to-end](sft.md) — the by-hand DP-SGD SFT pipeline the trainer wraps.
- [DPO end-to-end](dpo.md) — the by-hand reference-precompute / head-selection
  pipeline.
- [Transformers reference](../reference/transformers.md#opaquetransformerstrl-sftdpo-trainers)
  — the `SFTConfig` / `DPOConfig` / `SFTTrainer` / `DPOTrainer` field-by-field
  reference.
- [TrainingArguments](../user-guide/huggingface/training-arguments.md) — the
  inherited DP / clipping / sampling / save / eval knobs.
- [Trainer](../user-guide/huggingface/trainer.md) — the per-example DP
  trainer both classes subclass.
