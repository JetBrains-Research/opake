"""Shared dataset transforms impl — chat-template helpers.

General, method-agnostic formatting helpers (shared). Preference prompt
extraction (``extract_prompt``) is DPO-specific and lives in
:mod:`opake.api.alignment.dpo.data`.
"""

from opake.api.alignment.data._chat_template import (
    clone_chat_template,
    get_training_chat_template,
)
from opake.api.alignment.data._completion_mask import apply_chat_template_with_mask

__all__ = [
    "apply_chat_template_with_mask",
    "clone_chat_template",
    "get_training_chat_template",
]
