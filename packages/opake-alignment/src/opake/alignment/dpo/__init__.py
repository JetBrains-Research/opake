"""opake.alignment.dpo — functional DPO primitives (method namespace).

Mirrors ``opake.dpsgd`` / ``opake.dpftrl``: the method's primitives live in
sub-concern subpackages, reached directly (e.g. ``dpo.loss.sigmoid_loss``,
``dpo.reference.compute_ref_logprobs_for_dataset``):

- ``loss``      — the DPO loss-construction toolkit: per-sequence logp
  (``sequence_logp`` / ``fused_sequence_logp``), the 14 per-pair heads, and the
  log-ratio combinators.
- ``collator``  — preference (paired) collator factory.
- ``reference`` — reference-logp precompute, adapter dispatch, EMA sync.
- ``metric``    — preference reward telemetry.
- ``data``      — preference prompt extraction.
"""

from opake.alignment.dpo import collator, data, loss, metric, reference

__all__ = ["collator", "data", "loss", "metric", "reference"]
