# Alignment (`opake.alignment`)

`opake-alignment` provides functional, mechanism-agnostic primitives for
DP-safe preference learning and supervised fine-tuning: per-example SFT and
DPO losses, log-probability helpers, collators, data transforms,
reference-model helpers, and reward metrics. They compose under `vmap(grad(...))`
with either DP-SGD or DP-FTRL.

## Design

- **Functional.** Losses are pure functions of per-example tensors; collators
  and reference helpers return callables.
- **Per-example by construction.** Each loss output for example `i` depends
  only on example `i`'s data, so per-example sensitivity stays `O(C)` after
  clipping. SFT divisors are per-example token counts (not a batch aggregate),
  and the DPO collator keeps chosen/rejected as separate `(B, ...)` tensors so
  one preference pair maps to one clipped gradient.
- **Mechanism-agnostic.** Choose the mechanism and optimizer at the call site.

## Module map

| Module | Contents |
|---|---|
| `opake.alignment.sft.loss` | `nll_loss`, `dft_loss`, and the fused twins `fused_nll_loss`, `fused_dft_loss` |
| `opake.alignment.sft.collator` | `language_modeling_collator` (output schema `LMBatch` in `…collator.types`) |
| `opake.alignment.dpo.loss` | per-sequence logp, DPO loss heads, and log-ratio combinators |
| `opake.alignment.dpo.collator` | `preference_collator` |
| `opake.alignment.dpo.reference` | `compute_ref_logprobs_for_dataset`, `null_ref_context`, `with_disabled_adapter`, `ema_update_reference` |
| `opake.alignment.dpo.metric` | `reward_metrics` |
| `opake.alignment.dpo.data` | `extract_prompt` |

`DPOTrainer` maps its supported `loss_type` values to these functions; see the
[trainer guide](trainers.md#the-loss_type-menu) for the available objectives.

## See also

- [SFT end-to-end](sft.md) — build a DP-SGD supervised fine-tuning run from
  the language-modeling collator and the per-example NLL/DFT losses.
- [DPO end-to-end](dpo.md) — reference-logp precompute, the preference
  collator, choosing a per-pair head, and reward-metric eval.
- [Alignment API reference](../reference/alignment.md) — every public
  function with its import path and a one-line description.
- [DP-SGD end-to-end](../user-guide/dp-sgd.md) — the clipping/noise/optimizer
  pipeline these losses plug into.
