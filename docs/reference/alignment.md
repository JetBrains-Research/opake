# Alignment

Functional, mechanism-agnostic primitives for DP-safe preference learning
and supervised fine-tuning: per-example SFT and DPO losses, per-sequence
log-probability helpers, batch collators, dataset transforms,
reference-model handling, and reward metrics. Every public symbol is a
pure function or a factory returning a callable, so each composes under
`vmap(grad(...))` — the per-example differentiation DP-SGD and DP-FTRL
clipping is built on. Import everything from the `opaque.alignment.*`
façades; the mechanism and optimizer are chosen at the call site, never
inside this package. For end-to-end walkthroughs see the
[SFT](../alignment/sft.md) and [DPO](../alignment/dpo.md) guides.

## Overview

### Losses

Per-example loss functions. SFT losses return the per-example mean over an
example's own non-ignored tokens (a DP-safe divisor, not a batch
aggregate). DPO per-pair heads take `(chosen_logratio, rejected_logratio,
*, beta, ...)` and return a per-example scalar; a DPO loss is
`head(sequence_logp(...) - ref_logp, ...)`. Eager losses take `logits`;
the `fused_*` twins take `hidden_states` + the `lm_head` weight and never
materialize `(T, V)` logits (CUDA + half-precision fused kernel with
`opaque-alignment[patches]`, eager fallback otherwise).

SFT:

- **`nll_loss()`** ([`opaque.alignment.sft.loss`](#api-documentation)) —
  per-example causal-LM cross-entropy with a per-example token-count divisor.
- **`dft_loss()`** ([`opaque.alignment.sft.loss`](#api-documentation)) —
  Dynamic Fine-Tuning: NLL weighted by the detached softmax probability of
  each target token.
- **`fused_nll_loss()`** ([`opaque.alignment.sft.loss`](#api-documentation))
  — memory-efficient `nll_loss` over hidden states + `lm_head` weight.
- **`fused_dft_loss()`** ([`opaque.alignment.sft.loss`](#api-documentation))
  — memory-efficient `dft_loss` over hidden states + `lm_head` weight.

DPO per-pair heads ([`opaque.alignment.dpo.loss`](#api-documentation)):

- **`sigmoid_loss()`** — standard DPO sigmoid (logistic) loss; the default.
- **`hinge_loss()`** — DPO hinge loss.
- **`robust_loss()`** — robust (label-smoothed) DPO loss.
- **`ipo_loss()`** — IPO loss (Azar 2024).
- **`discopop_loss()`** — DiscoPOP loss.
- **`chosen_nll_loss()`** — chosen-completion NLL regularizer for MPO/RPO blends.
- **`apo_zero_loss()`** / **`apo_down_loss()`** — APO-zero / APO-down (arXiv:2408.06266).
- **`exo_loss()`** — EXO pairwise loss ([*Towards Efficient Exact Optimization of Language Model Alignment*](https://arxiv.org/abs/2402.00856)).
- **`nca_loss()`** — NCA pairwise loss.
- **`bco_loss()`** — BCO pairwise loss.
- **`sppo_loss()`** — SPPO hard-label loss.

Reference-free heads ([`opaque.alignment.dpo.loss`](#api-documentation)) — no
reference model; score the policy log-prob directly:

- **`simpo_loss()`** — SimPO: length-normalized sigmoid with a target margin γ (Meng 2024).
- **`odds_ratio_loss()`** — ORPO odds-ratio loss on length-normalized log-probs (Hong 2024); pair with `chosen_nll_loss` via `mpo_combine`.

DPO log-ratio combinators ([`opaque.alignment.dpo.loss`](#api-documentation))
for composite objectives:

- **`f_divergence_remap()`** — remap a per-example log-ratio under the chosen f-divergence.
- **`f_divergence_logits()`** — form the remapped preference logits under an f-divergence.
- **`mpo_combine()`** — weighted sum of per-example loss tensors (MPO / TRL `loss_type=list`).
- **`wpo_weights()`** — per-example WPO weight (arXiv:2406.11827).
- **`ld_dpo_split()`** — length-desensitized sequence-logp split (LD-DPO, arXiv:2409.06411).

Primary papers for the implemented alignment objectives:

- DPO — Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290).
- IPO — Azar et al., [*A General Theoretical Paradigm to Understand Learning from Human Preferences*](https://arxiv.org/abs/2310.12036).
- DiscoPOP — Lu et al., [*Discovering Preference Optimization Algorithms with and for Large Language Models*](https://arxiv.org/abs/2406.08414).
- SimPO — Meng et al., [*SimPO: Simple Preference Optimization with a Reference-Free Reward*](https://arxiv.org/abs/2405.14734).
- ORPO — Hong et al., [*ORPO: Monolithic Preference Optimization without Reference Model*](https://arxiv.org/abs/2403.07691).
- WPO — Zhou et al., [*WPO: Enhancing RLHF with Weighted Preference Optimization*](https://arxiv.org/abs/2406.11827).
- LD-DPO — Liu et al., [*Length Desensitization in Direct Preference Optimization*](https://arxiv.org/abs/2409.06411).
- APO — D'Oosterlinck et al., [*Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment*](https://arxiv.org/abs/2408.06266).
- NCA — Chen et al., [*Noise Contrastive Alignment of Language Models with Explicit Rewards*](https://arxiv.org/abs/2402.05369).
- BCO — Jhunjhunwala et al., [*Binary Classifier Optimization for Large Language Model Alignment*](https://arxiv.org/abs/2404.04656).
- SPPO — Wu et al., [*Self-Play Preference Optimization for Language Model Alignment*](https://arxiv.org/abs/2405.00675).

### Log-probabilities

- **`sequence_logp()`** ([`opaque.alignment.dpo.loss`](#api-documentation))
  — sum of completion-token log-probabilities per sequence; applies the
  causal-LM shift, masks to the completion span, and sums. Works
  per-example or on a batch axis. Pass `length_normalized=True` for the
  per-token mean reward `log π(y)/|y|` (SimPO / ORPO), or `ld_alpha` for the
  LD-DPO length-desensitized split. Its compatibility argument
  `shared_prefix_len` is the paper's public length (the shorter completion
  length), not a token-identical prefix.
- **`fused_sequence_logp()`** ([`opaque.alignment.dpo.loss`](#api-documentation))
  — memory-efficient `sequence_logp` over hidden states + `lm_head` weight
  (per-example only; same `length_normalized` option).

### Collators

Factory functions returning a `collate(examples)` callable.

- **`language_modeling_collator()`** ([`opaque.alignment.sft.collator`](#api-documentation))
  — SFT/causal-LM collation: `input_ids`/`attention_mask`/`labels` `(B, L)`
  with pad and (optionally) prompt tokens masked to `-100`, optional
  `completion_mask`, keep-start truncation to `max_length`. Output schema
  `LMBatch` (`opaque.alignment.sft.collator.types`).
- **`preference_collator()`** ([`opaque.alignment.dpo.collator`](#api-documentation))
  — DPO collation in a `(B, ...)` layout: the chosen and rejected
  `input_ids`/`attention_mask`/`completion_mask` trios, plus optional
  precomputed `ref_chosen_logps` / `ref_rejected_logps` `(B,)`. Chosen and
  rejected stay separate (not TRL's concatenated `(2B, L)`) so one pair maps
  to one clipped gradient.
- **`extract_prompt()`** ([`opaque.alignment.dpo.data`](#api-documentation))
  — split an implicit shared prompt out of a preference example before
  tokenizing.

### Data

Shared chat-template data prep ([`opaque.alignment.data`](#api-documentation)).
Install a training chat template, then tokenize chat turns into `input_ids` +
a `completion_mask` for completion-only loss.

- **`clone_chat_template()`** — copy a chat template (and its special tokens)
  from a source tokenizer onto a destination tokenizer and resize the model's
  embedding matrix to match. Returns the 3-tuple `(model, tokenizer,
  added_token_ids)` — the (possibly resized) model, the updated tokenizer, and
  the ids of any newly added special tokens (mark these trainable when the rest
  of the model is frozen). Call it **before** `make_functional` (the embedding
  resize must be captured in the functional snapshot).
- **`get_training_chat_template()`** — return a chat-template string carrying
  validated assistant-turn `{% generation %}` markers so the assistant-token
  mask is recoverable at tokenization time. Unsupported templates raise instead
  of falling back to a potentially incorrect mask.
- **`apply_chat_template_with_mask()`** — tokenize a chat conversation into
  `input_ids` + `completion_mask` (`1` on assistant tokens), the mask consumed
  by `language_modeling_collator(..., completion_only_loss=True)`.

### Reference handling

Reference-model helpers for DPO. These run **outside** the per-example
`vmap(grad(...))` region — a separate forward pass, or PEFT adapter
toggles.

- **`compute_ref_logprobs_for_dataset()`** ([`opaque.alignment.dpo.reference`](#api-documentation))
  — run the reference once over a dataset, attach per-example logp columns,
  and cache to a content-addressed `.safetensors` file keyed by dataset
  identity, `cache_identity`, and `output_columns` (a cache hit skips the
  forward). Cache directories and archives are owner-only; callers remove
  selected cache directories when their private per-example values are no
  longer needed. Set `use_cache=False` for non-reusable one-shot values;
  TR-DPO uses this mode for its seed columns. Under a live process group each
  rank scores its own shard and the columns are gathered back into dataset
  order, so the result matches a single-process run — see
  [Across ranks](../alignment/dpo.md#across-ranks).
- **`null_ref_context()`** ([`opaque.alignment.dpo.reference`](#api-documentation))
  — context manager that turns a model into its own reference, dispatching
  over the separate-model / `"ref"`-adapter / disabled-adapter / no-op
  configurations.
- **`with_disabled_adapter()`** ([`opaque.alignment.dpo.reference`](#api-documentation))
  — context manager that disables a PEFT adapter so the base model serves as
  the reference.
- **`ema_update_reference()`** ([`opaque.alignment.dpo.reference`](#api-documentation))
  — TR-DPO: leafwise EMA step `(1 - alpha) * ref + alpha * policy` to move
  the reference toward the policy.

### Metrics

These metrics are un-noised and outside Opaque's DP accounting. See
[Telemetry outside the guarantee](../limitations.md#telemetry-outside-the-guarantee).
Shared token metrics
([`opaque.alignment.metric`](#api-documentation)):

- **`mean_token_accuracy()`** — mean next-token argmax accuracy over the
  supervised (non-ignored) positions.
- **`entropy_from_logits()`** — mean per-token predictive entropy.

Preference reward telemetry ([`opaque.alignment.dpo.metric`](#api-documentation)):

- **`reward_metrics()`** — `rewards/chosen`, `rewards/rejected`,
  `rewards/accuracies`, `rewards/margins` from per-example log-ratios.

**See also**: [Alignment overview](../alignment/index.md),
[SFT end-to-end](../alignment/sft.md), [DPO end-to-end](../alignment/dpo.md).

## API Documentation

::: opaque.alignment.sft.loss
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.sft.collator
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.sft.collator.types
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.data
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.dpo.loss
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.dpo.collator
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.dpo.data
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.dpo.reference
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.dpo.metric
    options:
      show_source: true
      heading_level: 3

::: opaque.alignment.metric
    options:
      show_source: true
      heading_level: 3
