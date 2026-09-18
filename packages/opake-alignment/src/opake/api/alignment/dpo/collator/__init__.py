"""Collator factories impl — preference (DPO).

The language-modeling (SFT) collator lives under
:mod:`opake.api.alignment.sft.collator`.
"""

from opake.api.alignment.dpo.collator._preference import preference_collator

__all__ = ["preference_collator"]
