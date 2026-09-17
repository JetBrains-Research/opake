"""opake.alignment.dpo.reference — reference-model handling for DPO.

DPO scores each response against a frozen reference policy ``π_ref``. These
helpers produce ``π_ref``'s log-probabilities and keep that work *outside* the
per-example ``vmap(grad(...))`` region (it runs a separate forward and/or mutates
``nn.Module`` adapter state):

- ``compute_ref_logprobs_for_dataset(dataset, ref, ...)`` — the common path: run
  the reference once over the dataset (outer loop), caching per-example
  chosen/rejected sequence logps to a content-addressed ``.safetensors`` cache
  keyed by dataset identity, a structured model/tokenizer/preprocessing
  ``cache_identity``, and ``output_columns``. Feed the
  cached ``ref_chosen_logps`` / ``ref_rejected_logps`` into the per-pair head
  as the log-ratio baseline ``policy_logp - ref_logp``.
- ``null_ref_context(model)`` / ``with_disabled_adapter(model)`` — when the policy
  is a PEFT/LoRA adapter, use the *base* model as the reference: enter the context
  (which disables the adapter), run the reference forward, no second model needed.
- ``ema_update_reference(...)`` — TR-DPO: periodically move the reference toward
  the policy by an EMA step between training steps.

End-to-end wiring is in ``examples/train_dpo.py``.
"""

from opake.api.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
    with_disabled_adapter,
)

__all__ = [
    "compute_ref_logprobs_for_dataset",
    "ema_update_reference",
    "null_ref_context",
    "with_disabled_adapter",
]
