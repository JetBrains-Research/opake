# SFT end-to-end

This guide builds a DP-SGD supervised fine-tuning (SFT) run: collate a
causal-LM batch, compute a per-example loss, then clip, noise, and optimize it.

## Why SFT here

SFT adds an alignment collator and per-example loss to the standard
[DP-SGD pipeline](../user-guide/dp-sgd.md). The same loss closure also works
with [DP-FTRL](../user-guide/dp-ftrl.md).

The one DP-relevant subtlety is the loss divisor. Both SFT losses divide
by *this example's* non-ignored token count, not a batch-level aggregate
like TRL's `num_items_in_batch`. A batch aggregate would couple
per-example gradients, breaking the per-record sensitivity bound that
clipping relies on; the per-example divisor keeps sensitivity `O(C)`
after clipping.

## 1. Collator

`opake.alignment.sft.collator.language_modeling_collator` is a factory:
it returns a `collate(examples)` callable that pads a list of
tokenized rows into a batch.

```python
from opake.alignment.sft.collator import language_modeling_collator

collate = language_modeling_collator(tokenizer.pad_token_id, max_length)
batch = collate([{"input_ids": ids}, ...])  # LMBatch
```

The output is an `LMBatch` (`opake.alignment.sft.collator.types`) with
`input_ids`/`attention_mask`/`labels` of shape `(B, L)`. Padding and
(when `completion_only_loss=True`) prompt tokens are set to `-100` in
`labels` so the loss ignores them; an optional `completion_mask` is
emitted too. Sequences longer than `max_length` are truncated with keep-start
(matching TRL's `SFTTrainer`); no example is dropped. Pass
`pad_to_multiple_of=` to round the padded length up.

### Completion-only loss from chat data

To train only on assistant turns, the collator needs a `completion_mask`
(`1` on assistant tokens, `0` on the prompt). Produce it from a chat
dataset with the `opake.alignment.data` helpers: install
`{% generation %}` markers on the tokenizer's chat template with
`get_training_chat_template`, then tokenize each conversation with
`apply_chat_template_with_mask`.

```python
from opake.alignment.data import (
    get_training_chat_template,
    apply_chat_template_with_mask,
)

tokenizer.chat_template = get_training_chat_template(tokenizer)
row = apply_chat_template_with_mask(tokenizer, conversation)
# row -> {"input_ids", "completion_mask", "attention_mask"}

collate = language_modeling_collator(
    tokenizer.pad_token_id, max_length, completion_only_loss=True
)
batch = collate([row, ...])  # labels are -100 on prompt + pad tokens
```

The collator reads `completion_mask` and sets `labels` to `-100`
everywhere the mask is `0`, so the loss only sees assistant tokens.
`get_training_chat_template` recognizes validated explicit assistant branches
and selected shared-render forms, then validates that generated spans exclude
user content. It raises `ValueError` for an unsupported template instead of
guessing. The markers are mandatory: `apply_chat_template_with_mask` raises if
the active template lacks a Transformers-recognized generation tag (including
Jinja whitespace-control variants; HF cannot recover the mask otherwise).

## 2. Per-example loss

The loss runs *inside* `vmap(grad(...))`, so it is written for a single
example and returns a per-example scalar (no mean over the batch — the
clipper sums). Pick `nll_loss` (standard cross-entropy) or `dft_loss`
(Dynamic Fine-Tuning: NLL weighted by the detached softmax probability of
each target token):

```python
from opake.functional import make_functional
from opake.alignment.sft.loss import nll_loss  # or dft_loss

fmodel, trainable, frozen = make_functional(
    model, disable_autograd_tracking=True, partition_trainable=True
)

def per_example_loss(trainable_params, input_ids, attention_mask, labels):
    out = fmodel(
        {**frozen, **trainable_params},
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return nll_loss(out.logits, labels)
```

Both `nll_loss(logits, labels)` and `dft_loss(logits, labels)` apply the
causal-LM shift, ignore `-100` positions, and divide by the per-example token
count.

### Fused vs eager

`nll_loss`/`dft_loss` take `logits` `(..., T, V)`. On large vocabularies
the `(T, V)` logits dominate memory. The fused twins `fused_nll_loss` /
`fused_dft_loss` take `hidden_states` `(T, H)` and the `lm_head` weight
`(V, H)` instead, and never materialize logits — on CUDA + half precision
with `opake-alignment[patches]` installed, they call a fused linear-CE
kernel and recompute the LSE in the backward; otherwise they fall back to
the eager projection. They are numerically equivalent to the eager form.
Unlike the eager losses, the fused ones are strictly per-example (one
`(T, H)` sequence, batched only by the outer `vmap`); do not call them on
a batch axis directly. Use the fused path when activation memory is the
bottleneck and you are on a CUDA half-precision setup; otherwise, use the
eager path.

## 3. `vmap(grad)`, clipping, noise, and optimization

`opake.dpsgd.clipping.clipped_grad` wraps the per-example loss in
`vmap(grad(...))`, clips each per-example gradient to `clipping_norm`, and
sums. The remaining steps are exactly the DP-SGD pipeline:

```python
import torchopt
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler
from opake.optimizers import adamw
from opake.random import key, fold_in

grad_fn, clip_state = clipped_grad(
    per_example_loss,
    argnums=0,
    batch_argnums=(1, 2, 3),  # input_ids, attention_mask, labels
    clipping_norm=1.0,
    normalize_by=batch_size,
    return_aux=True,
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=nm, key=key(seed))
opt = adamw(lr=1e-4)
opt_state = opt.init(trainable)
```

## 4. End-to-end loop

```python
sampler = PoissonSampler(
    examples,
    sample_rate=batch_size / len(examples),
    n_steps=num_steps,
    key=fold_in(key(seed), 0, 0),
)

for indices in sampler:
    batch = collate_to_device([examples[i] for i in indices])  # (ids, mask, labels)
    if batch[0].shape[0] == 0:  # empty Poisson draw
        continue
    (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = opt.update(noisy_grads, opt_state, params=trainable)
    trainable = torchopt.apply_updates(trainable, updates)
```

`return_aux=True` surfaces per-example diagnostics on `aux` (e.g.
`aux.loss_values.mean()` for the step loss). To run under a target
privacy budget, calibrate `noise_multiplier` first — see
[DP-SGD calibration](../user-guide/dp-sgd.md#1-calibration).

## Runnable references

- [`examples/train_sft.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_sft.py)
  — full DP-SGD SFT script (collator, `nll`/`dft` losses, the per-example
  `vmap(grad)` loop, and a `--smoke` CPU run on a tiny random model).

## See also

- [Alignment overview](index.md) — the package, its design, and the
  module map.
- [DPO end-to-end](dpo.md) — the preference-learning companion guide.
- [Alignment API reference](../reference/alignment.md) — every public
  function with its import path.
- [DP-SGD end-to-end](../user-guide/dp-sgd.md) — calibration, clipping,
  noise, sampling, and optimizer details the loop above plugs into.
- [DP-FTRL end-to-end](../user-guide/dp-ftrl.md) — the correlated-noise
  mechanism swap.
