# opake-alignment

Functional, mechanism-agnostic primitives for **DP-safe preference learning**:
per-example loss functions (DPO / SFT families), log-probability helpers, preference
collators, dataset transforms, reference-model helpers, alignment metrics, and
memory-efficient fused-linear twins (`fused_nll_loss` / `fused_dft_loss` /
`fused_sequence_logp`) that project hidden states through the `lm_head` via the
optional `opake-patches` linear-CE kernel.

Built on `opake-engine` (clipping, functional, distributed) and `opake-base`
(serialization). Consumed by both functional training scripts
(`examples/train_dpo.py`-style) and the TRL-style class trainers in
`opake.transformers.trl`.

## Design

- **Functional, no hidden state.** Every public symbol is a pure function, a
  factory returning a callable, or an inert dataclass — no `nn.Module`
  subclasses, no user-instantiated classes. `vmap(grad(...))` composes cleanly
  only over pure functions.
- **Mechanism-agnostic.** Depends only on the substrate packages
  (`opake-engine`, `opake-base`); never on `opake-dpsgd`, `opake-dpftrl`,
  or `opake-optimizers`. The DP mechanism and optimizer are chosen at the call
  site.
- **Per-example API.** Public losses map each record to its own scalar and
  compose with `vmap(grad(...))`; this does not imply family-wide locality
  verification. Batch-coupled Tier 3 variants are not shipped.
- **Direct functions, no registry.** Each method exposes its loss functions
  directly (`dpo_sigmoid`, `nll_loss`, …) — there is no string registry,
  resolver, or variant enum. A config-string consumer (trainer / CLI) builds
  its own name→function mapping at the call site (see `examples/train_dpo.py`).

## Import layout

Method-first, mirroring `opake.dpsgd` / `opake.dpftrl`: each method owns its
primitives under its own namespace.

```
opake.alignment                         <- top-level façade (dpo, sft, data, metric)
opake.api.alignment                     <- implementation namespace
opake.alignment.dpo                     <- DPO method façade (aggregates the below)
opake.alignment.dpo.loss                <- logp + 14 per-pair heads + log-ratio combinators
opake.alignment.dpo.collator            <- preference (DPO) collator factory
opake.alignment.dpo.reference           <- ref-logp precompute, null_ref_context, EMA
opake.alignment.dpo.metric              <- preference reward telemetry
opake.alignment.dpo.data                <- preference prompt extraction
opake.alignment.sft.loss                <- nll_loss, dft_loss + fused_nll_loss, fused_dft_loss
opake.alignment.sft.collator            <- language-modeling (SFT) collator
opake.alignment.data                    <- chat-template prep + completion-mask tokenization
opake.alignment.metric                  <- shared token metrics (accuracy, entropy)
```

`opake.alignment.dpo.loss` is the DPO loss-construction toolkit: the per-sequence
log-prob primitives `sequence_logp` / `fused_sequence_logp`, the 14 per-pair
heads, and the log-ratio combinators all live there (a `per_example_loss` is
`head(sequence_logp(...) - ref_logp, …)`). The lower-level `selective_log_softmax`
building block, `entropy_from_logits` / `mean_token_accuracy` (token metrics), and
chat-template helpers stay internal impl under `opake.api.alignment.*`, surfaced
through the method that consumes them, following the shared-impl re-import pattern of
`opake.dpsgd.clipping`.
