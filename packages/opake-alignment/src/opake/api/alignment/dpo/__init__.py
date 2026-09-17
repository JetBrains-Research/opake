"""opake-alignment implementation namespace: DPO method.

Method-first layout (mirrors ``opake.api.dpsgd`` / ``opake.api.dpftrl`` and
the sibling ``opake.api.alignment.sft``): DPO owns its loss family
(``dpo/loss``), preference collator (``dpo/collator``), reference-model handling
(``dpo/reference``), reward telemetry (``dpo/metric``), and preference prompt
extraction (``dpo/data``). The fused DPO path is ``fused_sequence_logp`` (a
memory-efficient drop-in for ``sequence_logp``) in the shared
``opake.api.alignment.logprob`` concern, composed with the ``dpo/loss``
per-pair heads. Shared primitives (logprob, chat-template data, general token
metrics) are reimported from the shared ``opake.api.alignment.*`` concerns at
their use sites.
"""

__all__: list[str] = []
