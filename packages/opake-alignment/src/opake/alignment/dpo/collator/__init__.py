"""opake.alignment.dpo.collator façade — the preference (DPO) collator.

The language-modeling (SFT) collator lives under
:mod:`opake.alignment.sft.collator`.
"""

from opake.api.alignment.dpo.collator import preference_collator

__all__ = ["preference_collator"]
