"""Completion-mask extraction from a chat template.

:func:`apply_chat_template_with_mask` tokenizes a chat ``conversation`` and
returns, alongside the ``input_ids``, a per-token ``completion_mask`` that is
``1`` on assistant-turn tokens and ``0`` on the system/user prompt tokens.  It
is the data-prep counterpart to the ``completion_only_loss`` path of
:func:`opake.alignment.sft.collator.language_modeling_collator`: the mask it
produces is exactly the ``completion_mask`` the collator consumes.

The mask is computed by Hugging Face's
``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``,
which only works when the tokenizer's chat template carries
``{% generation %}`` / ``{% endgeneration %}`` markers around the assistant
content.  Install those markers first with
:func:`opake.api.alignment.data._chat_template.get_training_chat_template`;
without them HF only logs a warning and returns an all-zero assistant mask, so
this function checks the template up front and raises a :class:`ValueError`.
"""

from __future__ import annotations

from opake.api.alignment.data._chat_template import (
    _has_generation_marker,
    _resolve_chat_template,
)
from opake.exceptions import ConfigurationError

__all__ = ["apply_chat_template_with_mask"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def apply_chat_template_with_mask(
    tokenizer: PreTrainedTokenizerBase,
    conversation: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Tokenize a chat *conversation* and return the assistant-completion mask.

    Wraps ``tokenizer.apply_chat_template(..., tokenize=True,
    return_assistant_tokens_mask=True, return_dict=True, **kwargs)`` and renames
    Hugging Face's ``assistant_masks`` to ``completion_mask`` (``1`` on
    assistant-turn tokens, ``0`` elsewhere) so the result drops straight into
    :func:`opake.alignment.sft.collator.language_modeling_collator` with
    ``completion_only_loss=True``.

    The active template (including a selected named template) **must** carry a
    Transformers-recognized generation tag around assistant content; install
    canonical markers with
    :func:`opake.api.alignment.data._chat_template.get_training_chat_template`.
    Without the marker HF returns no assistant mask and this function raises a
    :class:`ConfigurationError`.

    Args:
        tokenizer: A tokenizer whose active ``chat_template`` carries a
            Transformers-recognized generation tag (see
            :func:`get_training_chat_template`).
        conversation: A list of chat-message dicts (each with ``"role"`` and
            ``"content"``) for a single conversation.
        **kwargs: Forwarded verbatim to
            ``tokenizer.apply_chat_template`` (e.g. ``max_length``,
            ``truncation``, ``add_generation_prompt``).  The ``tokenize``,
            ``return_assistant_tokens_mask``, and ``return_dict`` arguments are
            fixed by this function and must not be overridden.

    Returns:
        A dict with ``"input_ids"`` and ``"completion_mask"`` (and
        ``"attention_mask"`` when HF returns one).  Both lists have the same
        length.

    Raises:
        ValueError: If the chat template does not carry ``{% generation %}``
            markers, or if the tokenized result carries no (or an empty)
            ``assistant_masks``.  Without the markers Hugging Face only logs a
            warning and returns an all-zero mask (no assistant tokens flagged),
            so this function checks the template up front and raises instead.
            Install the markers with :func:`get_training_chat_template`.
    """
    # Hugging Face only populates assistant_masks when the active chat template
    # contains the '{% generation %}' keyword; otherwise it merely logs a
    # warning and returns an all-zero mask.  Guard up front so the caller gets a
    # clear error (pointing at get_training_chat_template) instead of a silently
    # empty mask.  The template passed via kwargs (if any) takes precedence over
    # the tokenizer's own chat_template, matching apply_chat_template.
    active_template = _resolve_chat_template(
        tokenizer,
        chat_template=kwargs.get("chat_template"),
        tools=kwargs.get("tools"),
    )
    if not _has_generation_marker(active_template):
        raise ConfigurationError(
            *(
                "apply_chat_template_with_mask: the active chat template does not "
                "carry the '{% generation %}' / '{% endgeneration %}' markers that "
                "return_assistant_tokens_mask=True relies on, so no "
                "assistant-token mask can be produced.  Install them first with "
                "opake.alignment.data.get_training_chat_template:\n\n"
                "    tokenizer.chat_template = get_training_chat_template(tokenizer)\n",
            )
        )

    encoded = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        **kwargs,
    )

    assistant_masks = encoded.get("assistant_masks")
    if not _has_assistant_tokens(assistant_masks):
        raise ConfigurationError(
            *(
                "apply_chat_template_with_mask: the tokenizer returned no "
                "assistant-token mask.  This means the chat template does not "
                "carry the '{% generation %}' / '{% endgeneration %}' markers that "
                "return_assistant_tokens_mask=True relies on.  Install them first "
                "with opake.alignment.data.get_training_chat_template:\n\n"
                "    tokenizer.chat_template = get_training_chat_template(tokenizer)\n",
            )
        )

    result: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "completion_mask": assistant_masks,
    }
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        result["attention_mask"] = attention_mask
    return result


def _has_assistant_tokens(assistant_masks: Any) -> bool:
    """Return whether a list, tensor, or array mask flags any assistant token."""
    if assistant_masks is None:
        return False
    if hasattr(assistant_masks, "any"):
        return bool(assistant_masks.any())
    return any(assistant_masks)
