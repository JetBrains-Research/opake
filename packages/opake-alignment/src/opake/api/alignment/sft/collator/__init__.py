"""SFT collator impl — the language-modeling collator factory.

``language_modeling_collator`` is a factory returning a stateless collate
callable, following the repo convention that public builders prefer factory
functions over user-instantiated classes. Its output schema (:class:`LMBatch`) lives in
:mod:`opake.api.alignment.sft.collator.types`.
"""

from opake.api.alignment.sft.collator._language_modeling import (
    language_modeling_collator,
)

__all__ = ["language_modeling_collator"]
