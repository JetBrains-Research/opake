"""Chat-template helpers for alignment training.

- :func:`clone_chat_template` copies a chat template (and associated special
  tokens, e.g. ``<|im_start|>``) from a source tokenizer onto a destination
  tokenizer, then resizes the model's input-embedding matrix to the new vocab.
- :func:`get_training_chat_template` returns a Jinja2 chat template that wraps
  the assistant turn with ``{% generation %}`` / ``{% endgeneration %}`` so
  ``apply_chat_template(..., return_assistant_tokens_mask=True)`` yields a
  boolean mask usable for ``assistant_only_loss``.

Ordering invariant: :func:`clone_chat_template` mutates the embedding matrix
via ``resize_token_embeddings``. A functional snapshot
(``make_functional`` / ``torch.func.functional_call``) captures the embedding
tensor at snapshot time, so clone the template **before** snapshotting;
otherwise the snapshot keeps the old, smaller matrix and the functional forward
errors at runtime (index-out-of-range in the embedding lookup) the first time a
newly-added token id is seen.
"""

from __future__ import annotations

from opake.exceptions import ConfigurationError, InputTypeError

__all__ = ["clone_chat_template", "get_training_chat_template"]

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Sentinel strings used by Jinja2 chat-template generation markers.
_GEN_START = "{% generation %}"
_GEN_END = "{% endgeneration %}"
_GENERATION_TAG_PATTERN = re.compile(r"\{%-?\s*generation\s*-?%\}")


def clone_chat_template(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_tokenizer_or_path: PreTrainedTokenizerBase | str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, list[int]]:
    """Copy a chat template and special tokens from *source* onto *tokenizer*.

    The function:

    1. Loads the *source* tokenizer (if a path/name string is given, it is
       loaded via ``AutoTokenizer.from_pretrained`` with
       ``trust_remote_code=False``).
    2. Copies ``source.chat_template`` onto ``tokenizer.chat_template``.
    3. Collects every token that is marked ``special=True`` in the source
       tokenizer's ``added_tokens_decoder`` and adds any that are **missing**
       from *tokenizer* via ``tokenizer.add_special_tokens``.
    4. Calls ``model.resize_token_embeddings(len(tokenizer))`` to extend (or
       confirm the size of) the model's input-embedding matrix.

    Ordering invariant: this mutates the model's embedding matrix and must be
    called before any ``make_functional`` / ``torch.func.functional_call``
    snapshot, otherwise the snapshot keeps the old (smaller) matrix and the
    functional forward errors at runtime (embedding index-out-of-range) when a
    newly-added token id is first seen.

    Args:
        model: The :class:`~transformers.PreTrainedModel` whose embedding
            matrix will be resized to accommodate the cloned special tokens.
        tokenizer: The destination :class:`~transformers.PreTrainedTokenizerBase`
            that will receive the chat template and any new special tokens.
        source_tokenizer_or_path: Either a loaded
            :class:`~transformers.PreTrainedTokenizerBase` whose ``chat_template``
            and special tokens are to be copied, **or** a string (local path or
            Hugging Face Hub model ID) that will be passed to
            ``AutoTokenizer.from_pretrained``.

    Returns:
        A ``(model, tokenizer, added_token_ids)`` tuple.  ``model`` and
        ``tokenizer`` are mutated in-place and also returned for convenient
        chaining.  ``added_token_ids`` is the list of vocabulary indices of the
        special tokens newly added to *tokenizer* (empty when none were added).
        Callers fine-tuning with PEFT use it to mark the new embedding rows
        trainable, since a frozen base would never learn an embedding for a
        token that did not exist at pre-training time.

    Raises:
        ValueError: If *source_tokenizer_or_path* does not have a
            ``chat_template`` attribute set (i.e. the attribute is ``None``).
        TypeError: If *source_tokenizer_or_path* is neither a string nor a
            :class:`~transformers.PreTrainedTokenizerBase`.
    """
    # Resolve source tokenizer.
    if isinstance(source_tokenizer_or_path, str):
        from transformers import AutoTokenizer  # lazy import

        source_tokenizer = AutoTokenizer.from_pretrained(
            source_tokenizer_or_path,
            trust_remote_code=False,
        )
    else:
        # Duck-type check: must look like a tokenizer.
        if not hasattr(source_tokenizer_or_path, "chat_template"):
            raise InputTypeError(
                *(
                    "source_tokenizer_or_path must be a string path or a "
                    "PreTrainedTokenizerBase instance; "
                    f"got {type(source_tokenizer_or_path)!r}",
                )
            )
        source_tokenizer = source_tokenizer_or_path

    # Copy chat template.
    chat_template = getattr(source_tokenizer, "chat_template", None)
    if chat_template is None:
        raise ConfigurationError(
            *(
                "The source tokenizer does not have a chat_template set "
                "(chat_template is None).  Set it before calling "
                "clone_chat_template.",
            )
        )
    tokenizer.chat_template = chat_template

    # Preserve named roles and AddedToken metadata, rather than treating every
    # special token as an anonymous additional special token.
    _register_model_specific_special_tokens(tokenizer, source_tokenizer)
    source_special_tokens = _source_special_tokens(source_tokenizer)
    existing_vocab = set(tokenizer.get_vocab())
    special_token_strings = {
        str(token)
        for value in source_special_tokens.values()
        for token in (value if isinstance(value, list) else [value])
    }

    added_token_ids: list[int] = []
    if source_special_tokens:
        n_added = tokenizer.add_special_tokens(source_special_tokens)
        # Resolve the vocabulary indices of the tokens that did not exist before
        # cloning so PEFT callers can mark exactly those embedding rows trainable.
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_str in special_token_strings - existing_vocab:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != unk_id:
                added_token_ids.append(int(tid))
        logger.info(
            "clone_chat_template: added %d new special token(s) to tokenizer: %s",
            n_added,
            sorted(special_token_strings - existing_vocab),
        )
    else:
        logger.debug("clone_chat_template: no new special tokens to add.")

    # Resize model embeddings. Called unconditionally so the invariant
    # ``get_input_embeddings().weight.shape[0] == len(tokenizer)`` holds on
    # exit even when no new tokens were added (resize is a no-op then).
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    else:
        logger.warning(
            "clone_chat_template: model has no resize_token_embeddings method; "
            "skipping embedding resize.  The embedding size may not match the "
            "tokenizer vocabulary size."
        )

    return model, tokenizer, added_token_ids


def get_training_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a chat-template string with assistant-turn generation markers.

    Wraps the assistant-turn content of *tokenizer*'s chat template with
    ``{% generation %}`` / ``{% endgeneration %}`` Jinja2 markers so that
    ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
    returns a per-token boolean mask suitable for ``assistant_only_loss``.

    The function is **idempotent**: a template that already contains a
    Transformers-recognized generation tag is returned unchanged after its
    rendered spans are validated as assistant-only.

    Strategy (close to TRL ``chat_template_utils.py``):

    - If the tokenizer's current template already contains a generation marker
      (including Jinja whitespace-control variants), validate and return it.
    - Otherwise, locate the assistant content expression in an explicit
      assistant branch, a shared Llama/Qwen-style role expression, or a
      Gemma-style shared render expression that maps the ``assistant`` role to
      ``model``.
    - Render each candidate with distinct system, user, and assistant probe
      text. Accept it only when the generation spans contain assistant content
      and exclude non-assistant content.
    - Raise :class:`ValueError` when the template cannot be transformed without
      guessing.

    Args:
        tokenizer: A tokenizer whose ``chat_template`` attribute must be a
            non-empty Jinja2 string.

    Returns:
        A Jinja2 template string guaranteed to contain
        ``{% generation %}`` ... ``{% endgeneration %}``.

    Raises:
        ValueError: If ``tokenizer.chat_template`` is ``None`` or empty, or if
            its assistant render path cannot be identified and validated.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ConfigurationError(
            *(
                "tokenizer.chat_template is not set.  Assign a Jinja2 template "
                "string to tokenizer.chat_template before calling "
                "get_training_chat_template.",
            )
        )
    template = _resolve_chat_template(tokenizer)

    # Idempotency: already has the marker.
    if _has_generation_marker(template):
        if _generation_block_marks_only_assistant(template, tokenizer):
            return template
        raise ConfigurationError(
            *(
                "get_training_chat_template: existing generation markers do not "
                "identify a validated assistant-only render path in the tokenizer's "
                "chat template. Ensure they exclude system and user content.",
            )
        )

    # Strategy 1: wrap {{ generation_token }} — TRL v2 canonical placeholder.
    _GENERATION_TOKEN_EXPR = "{{ generation_token }}"
    if _GENERATION_TOKEN_EXPR in template:
        candidate = template.replace(
            _GENERATION_TOKEN_EXPR,
            f"{_GEN_START}{_GENERATION_TOKEN_EXPR}{_GEN_END}",
            1,
        )
        if _generation_block_marks_only_assistant(candidate, tokenizer):
            return candidate

    # Strategy 2: wrap the assistant content expression inside the assistant
    # conditional block. Handles unmarked templates with explicit assistant
    # branches, such as ChatML.
    template_out = _wrap_assistant_content(template)
    if template_out is not None and _generation_block_marks_only_assistant(
        template_out, tokenizer
    ):
        return template_out

    # Strategy 3: Llama 3 constructs every turn in a shared ``content`` variable
    # and renders it outside a role branch.
    template_out = _wrap_shared_content_expression(template)
    if template_out is not None and _generation_block_marks_only_assistant(
        template_out, tokenizer
    ):
        return template_out

    # Strategy 4: Gemma maps assistant to the rendered "model" role, then emits
    # every role through one expression outside the role-selection branch.
    template_out = _wrap_gemma_shared_expression(template)
    if template_out is not None and _generation_block_marks_only_assistant(
        template_out, tokenizer
    ):
        return template_out

    # Strategy 5: shared OR-clause rendering (Qwen2.5-Instruct pattern). When
    # user/system/assistant turns share one expression inside a multi-condition
    # ``or`` clause, the last assistant mention is the tool-call ``elif`` and
    # Strategy 2 would wrap an unreachable expression (empty
    # ``generation_indices``). Rewrite the shared expression with an inner
    # role-guard so only the assistant case renders inside the mask.
    template_out = _wrap_shared_or_clause(template)
    if template_out is not None and _generation_block_marks_only_assistant(
        template_out, tokenizer
    ):
        return template_out

    raise ConfigurationError(
        *(
            "get_training_chat_template: could not identify and validate an "
            "assistant-only render path in the tokenizer's chat template. "
            "Provide a template with explicit '{% generation %}' / "
            "'{% endgeneration %}' markers.",
        )
    )


# Private helpers


def _wrap_assistant_content(template: str) -> str | None:
    """Attempt to insert generation markers around assistant content.

    Searches for the last Jinja2 ``{{ ... content ... }}`` expression that
    occurs inside an ``assistant`` conditional branch (identified by the
    presence of ``'assistant'`` in the preceding ``if`` or ``elif`` tag).

    Returns the modified template string on success, or ``None`` if no
    suitable expression is found.
    """
    import re

    # Find all assistant branch tags ({% if/elif ... 'assistant' ... %}).
    assistant_block_pat = re.compile(
        r"\{%-?\s*(?:if|elif)\b[^%]*['\"]assistant['\"][^%]*-?%\}"
    )
    matches = list(assistant_block_pat.finditer(template))
    if not matches:
        return None

    # Take the last assistant branch start.
    last_branch_start = matches[-1].end()

    # Find the first content expression before the assistant branch ends. A
    # shared expression after the branch would render every role and must be
    # handled by a role-aware strategy below instead.
    substring = template[last_branch_start:]
    next_control = re.search(r"\{%-?\s*(?:elif|else|endif|endfor)\b", substring)
    if next_control is not None:
        substring = substring[: next_control.start()]
    content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
    m = content_pat.search(substring)
    if m is None:
        return None

    abs_start = last_branch_start + m.start()
    abs_end = last_branch_start + m.end()
    return (
        template[:abs_start]
        + _GEN_START
        + template[abs_start:abs_end]
        + _GEN_END
        + template[abs_end:]
    )


def _wrap_shared_or_clause(template: str) -> str | None:
    """Handle templates that render user/system/assistant via one shared expression.

    Some chat templates (notably Qwen2.5-Instruct) place an OR clause like::

        {%- if (message.role == "user") or (message.role == "system" and not loop.first)
              or (message.role == "assistant" and not message.tool_calls) %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}

    inside the message loop.  Wrapping the shared expression with
    ``{% generation %}`` markers would (incorrectly) flag user and system
    tokens as assistant tokens.  This helper rewrites the shared expression
    with an inner ``{%- if message.role == 'assistant' %}`` guard so the
    assistant case renders through a ``{% generation %}``-wrapped branch and
    user/system tokens fall through unflagged.

    Returns ``None`` when the template doesn't match the shared-OR-clause
    pattern.
    """
    import re

    or_clause_pat = re.compile(
        r"\{%-?\s*(?:if|elif)\b[^%]*['\"]assistant['\"][^%]*\bor\b[^%]*-?%\}"
        r"|\{%-?\s*(?:if|elif)\b[^%]*\bor\b[^%]*['\"]assistant['\"][^%]*-?%\}"
    )
    match = or_clause_pat.search(template)
    if match is None:
        return None

    # Find the FIRST content expression after the OR clause and before the
    # next Jinja control tag — that's the shared expression we need to split.
    after = template[match.end() :]
    next_ctrl = re.search(r"\{%-?\s*(?:elif|else|endif|endfor)\b", after)
    bound = next_ctrl.start() if next_ctrl else len(after)
    inner = after[:bound]

    content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
    expr_match = content_pat.search(inner)
    if expr_match is None:
        return None

    expr = expr_match.group()
    abs_start = match.end() + expr_match.start()
    abs_end = match.end() + expr_match.end()

    # Role-guarded variant: wrap the whole expression for the assistant case so
    # every token it emits (content + im_end + newline) is flagged. Flagging
    # im_end is desirable — the model learns to stop.
    guarded = (
        "{%- if message.role == 'assistant' %}"
        + _GEN_START
        + expr
        + _GEN_END
        + "{%- else %}"
        + expr
        + "{%- endif %}"
    )
    return template[:abs_start] + guarded + template[abs_end:]


def _wrap_shared_content_expression(template: str) -> str | None:
    """Wrap Llama-style shared ``content`` rendering for assistant messages."""
    role_ref = r"message(?:\.role|\[['\"]role['\"]\])"
    content_assignment = re.compile(
        rf"\{{%-?\s*set\s+content\s*=\s*[^%]*{role_ref}[^%]*\bcontent\b[^%]*-?%\}}"
    )
    if content_assignment.search(template) is None:
        return None

    content_refs = list(re.finditer(r"\{\{-?\s*content\s*-?\}\}", template))
    if len(content_refs) != 1:
        return None

    match = content_refs[0]
    expression = match.group()
    guarded = (
        "{%- if message['role'] == 'assistant' %}"
        + _GEN_START
        + expression
        + _GEN_END
        + "{%- else %}"
        + expression
        + "{%- endif %}"
    )
    return template[: match.start()] + guarded + template[match.end() :]


def _wrap_gemma_shared_expression(template: str) -> str | None:
    """Wrap a Gemma-style shared turn expression for assistant messages only.

    Gemma-style templates first map ``assistant`` to the rendered role ``model``
    and then render every message with one expression outside that conditional.
    Wrapping the expression directly would therefore mark user turns too.  This
    helper duplicates that expression behind an assistant-role guard.

    The transformation is deliberately limited to templates with exactly one
    content-rendering expression and a recognizable assistant-to-model mapping.
    More complex templates must provide generation tags explicitly.
    """
    role_ref = r"message(?:\.role|\[['\"]role['\"]\])"
    branch_mapping = re.compile(
        rf"\{{%-?\s*if\b[^%]*{role_ref}[^%]*==\s*['\"]assistant['\"][^%]*"
        rf"-?%\}}.*?\{{%-?\s*set\s+role\s*=\s*['\"]model['\"]\s*-?%\}}",
        re.DOTALL,
    )
    inline_mapping = re.compile(
        rf"\{{%-?\s*set\s+role\s*=\s*['\"]model['\"]\s+if\s+{role_ref}"
        rf"\s*==\s*['\"]assistant['\"]\s+else\b[^%]*-?%\}}"
    )
    if (
        branch_mapping.search(template) is None
        and inline_mapping.search(template) is None
    ):
        return None

    content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
    content_matches = list(content_pat.finditer(template))
    if len(content_matches) != 1:
        return None

    match = content_matches[0]
    expr = match.group()
    guarded = (
        "{%- if message['role'] == 'assistant' %}"
        + _GEN_START
        + expr
        + _GEN_END
        + "{%- else %}"
        + expr
        + "{%- endif %}"
    )
    return template[: match.start()] + guarded + template[match.end() :]


def _generation_block_marks_only_assistant(
    template: str, tokenizer: PreTrainedTokenizerBase | None = None
) -> bool:
    """Return whether generation spans contain assistant text and no prompt text."""
    from transformers.utils.chat_template_utils import (
        _compile_jinja_template,
        _render_with_assistant_indices,
    )

    try:
        compiled = _compile_jinja_template(template)
        user_probe = "opake_user_probe"
        assistant_probe = "opake_assistant_probe"
        messages = [
            {"role": "user", "content": user_probe},
            {"role": "assistant", "content": assistant_probe},
        ]
        rendered, indices = _render_with_assistant_indices(
            compiled,
            messages,
            None,
            None,
            False,
            **(tokenizer.special_tokens_map if tokenizer is not None else {}),
        )
    except Exception:
        # Template won't compile or render — treat as failure of this strategy.
        return False

    generated = "".join(rendered[start:end] for start, end in indices)
    if assistant_probe not in generated or user_probe in generated:
        return False

    system_probe = "opake_system_probe"
    try:
        rendered, indices = _render_with_assistant_indices(
            compiled,
            [
                {"role": "system", "content": system_probe},
                *messages,
            ],
            None,
            None,
            False,
            **(tokenizer.special_tokens_map if tokenizer is not None else {}),
        )
    except Exception:
        # Some templates intentionally reject a system role. The primary
        # user/assistant probe already establishes their supported behavior.
        return True
    generated = "".join(rendered[start:end] for start, end in indices)
    return system_probe not in generated


def _has_generation_marker(template: str) -> bool:
    """Return whether *template* has a Transformers-recognized generation tag."""
    return _GENERATION_TAG_PATTERN.search(template) is not None


def _resolve_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    *,
    chat_template: str | None = None,
    tools: list[dict] | None = None,
) -> str:
    """Resolve a template through Transformers' named-template selection rules."""
    return tokenizer.get_chat_template(chat_template, tools)


def _source_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, object]:
    """Return the source's named special tokens with AddedToken metadata intact."""
    added_tokens = getattr(tokenizer, "added_tokens_decoder", {})

    def preserved_token(token: object) -> object:
        token_id = tokenizer.convert_tokens_to_ids(str(token))
        return added_tokens.get(token_id, token)

    special_tokens = {
        name: preserved_token(token)
        for name, token in getattr(tokenizer, "_special_tokens_map", {}).items()
        if token is not None
    }
    extra_tokens = getattr(tokenizer, "extra_special_tokens", None)
    if extra_tokens is None:
        extra_tokens = getattr(tokenizer, "additional_special_tokens", ())
    if extra_tokens:
        special_tokens["extra_special_tokens"] = [
            preserved_token(token) for token in extra_tokens
        ]

    assigned = {
        str(token)
        for value in special_tokens.values()
        for token in (value if isinstance(value, list) else [value])
    }
    unassigned = [
        token
        for token in added_tokens.values()
        if getattr(token, "special", False) and str(token) not in assigned
    ]
    if unassigned:
        special_tokens.setdefault("extra_special_tokens", []).extend(unassigned)
    return special_tokens


def _register_model_specific_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    source_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Teach the destination about source-specific roles before cloning them."""
    destination_attributes = set(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    model_specific_tokens = {
        name: token
        for name, token in getattr(source_tokenizer, "_special_tokens_map", {}).items()
        if name not in destination_attributes and token is not None
    }
    if model_specific_tokens:
        tokenizer._set_model_specific_special_tokens(model_specific_tokens)
