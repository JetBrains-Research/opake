"""opake.alignment.data — shared chat-template data-prep (method namespace).

Method-agnostic data prep for chat datasets: install a training chat template,
then tokenize chat turns into ``input_ids`` + a ``completion_mask`` for
completion-only loss.

- ``clone_chat_template`` — copy a chat template (and its special tokens) from
  a source tokenizer onto a destination tokenizer and resize the model's
  embedding matrix to match.
- ``get_training_chat_template`` — return a chat-template string carrying
  assistant-turn ``{% generation %}`` markers so the assistant-token mask can
  be recovered at tokenization time.
- ``apply_chat_template_with_mask`` — tokenize a chat conversation into
  ``input_ids`` + ``completion_mask`` (``1`` on assistant tokens), the mask
  consumed by ``sft.collator.language_modeling_collator(..., completion_only_loss=True)``.
"""

from opake.api.alignment.data import (
    apply_chat_template_with_mask,
    clone_chat_template,
    get_training_chat_template,
)

__all__ = [
    "apply_chat_template_with_mask",
    "clone_chat_template",
    "get_training_chat_template",
]
