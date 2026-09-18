"""opake.alignment.sft.collator façade — the language-modeling collator.

Output schema (:class:`LMBatch`) lives in
:mod:`opake.alignment.sft.collator.types`.
"""

from opake.api.alignment.sft.collator import language_modeling_collator

__all__ = ["language_modeling_collator"]
