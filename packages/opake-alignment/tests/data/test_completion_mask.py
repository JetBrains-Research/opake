"""Tests for :mod:`opake.api.alignment.data._completion_mask`.

Coverage of :func:`apply_chat_template_with_mask`:

- Full round-trip: install a ``{% generation %}`` template via
  :func:`get_training_chat_template`, tokenize a system+user+assistant
  conversation, and assert the returned ``completion_mask`` is the same length
  as ``input_ids``, is ``1`` exactly on the assistant-response tokens, and ``0``
  on the system/user prompt tokens.
- ``input_ids`` / ``completion_mask`` keys are present; ``attention_mask`` is
  surfaced when HF emits one.
- A tokenizer whose template lacks ``{% generation %}`` markers raises the
  documented ``ValueError`` (HF returns no assistant mask).

All tests are CPU-only, network-free, and build the smallest possible tokenizer
with a tiny hand-built BPE vocab.  ``transformers`` is required; tests are
skipped when it is not installed via ``pytest.importorskip``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from tokenizers import AddedToken, Tokenizer  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from opake.api.alignment.data._chat_template import (  # noqa: E402
    get_training_chat_template,
)
from opake.api.alignment.data._completion_mask import (  # noqa: E402
    apply_chat_template_with_mask,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Every word that appears in the role markers and conversation must be its own
# vocab token so the prompt/assistant token boundaries are deterministic.
_VOCAB_WORDS = [
    "system",
    "user",
    "assistant",
    "model",
    "colon",
    "You",
    "are",
    "helpful",
    "What",
    "is",
    "two",
    "plus",
    "It",
    "equals",
    "four",
]

# A role-conditional template (system / user / assistant branches).  The
# generation markers are deliberately absent; install them with
# get_training_chat_template before tokenizing.
_ROLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "system colon {{ message['content'] }} "
    "{% elif message['role'] == 'user' %}"
    "user colon {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}"
    "assistant colon {{ message['content'] }} "
    "{% endif %}"
    "{% endfor %}"
)

_GEMMA_TEMPLATE = (
    "{% for message in messages %}"
    "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim "
    "+ '<end_of_turn>\\n' }}"
    "{% endfor %}"
)

_WHITESPACE_CONTROLLED_ROLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "system colon {{ message['content'] }} "
    "{% elif message['role'] == 'user' %}"
    "user colon {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}"
    "assistant colon {%- generation -%}{{ message['content'] }}{%- endgeneration -%} "
    "{% endif %}"
    "{% endfor %}"
)

_CONVERSATION = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is two plus two"},
    {"role": "assistant", "content": "It equals four"},
]

# The assistant content tokens — exactly the positions the mask must flag with 1.
_ASSISTANT_TOKENS = ["It", "equals", "four"]


def _make_chat_tokenizer() -> PreTrainedTokenizerFast:
    """Return a word-level :class:`PreTrainedTokenizerFast` for chat tests."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(
        [
            AddedToken("<unk>", special=True),
            AddedToken("<start_of_turn>", special=True),
            AddedToken("<end_of_turn>", special=True),
        ]
    )
    tok.add_tokens(list(_VOCAB_WORDS))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


# ---------------------------------------------------------------------------
# Tests: apply_chat_template_with_mask — round-trip
# ---------------------------------------------------------------------------


class TestApplyChatTemplateWithMask:
    def test_mask_length_matches_input_ids(self) -> None:
        """completion_mask has the same length as input_ids."""
        tok = _make_chat_tokenizer()
        tok.chat_template = get_training_chat_template_for(tok)

        result = apply_chat_template_with_mask(tok, _CONVERSATION)

        assert len(result["completion_mask"]) == len(result["input_ids"])

    def test_returns_expected_keys(self) -> None:
        """Result carries input_ids, completion_mask, and attention_mask."""
        tok = _make_chat_tokenizer()
        tok.chat_template = get_training_chat_template_for(tok)

        result = apply_chat_template_with_mask(tok, _CONVERSATION)

        assert set(result) == {"input_ids", "completion_mask", "attention_mask"}

    def test_mask_one_on_assistant_zero_on_prompt(self) -> None:
        """Mask is 1 exactly on assistant tokens, 0 on system/user prompt tokens."""
        tok = _make_chat_tokenizer()
        tok.chat_template = get_training_chat_template_for(tok)

        result = apply_chat_template_with_mask(tok, _CONVERSATION)
        ids = result["input_ids"]
        mask = result["completion_mask"]
        tokens = tok.convert_ids_to_tokens(ids)

        # Every token flagged 1 must be an assistant-response token, and every
        # assistant-response token (and only those) must be flagged 1.
        unmasked = [t for t, m in zip(tokens, mask, strict=False) if m == 1]
        assert unmasked == _ASSISTANT_TOKENS, (
            f"completion_mask should be 1 only on {_ASSISTANT_TOKENS}; "
            f"tokens/mask = {list(zip(tokens, mask, strict=False))}"
        )

        # The system/user prompt tokens must all be masked to 0.
        for token, m in zip(tokens, mask, strict=False):
            if token not in _ASSISTANT_TOKENS:
                assert m == 0, f"prompt token {token!r} should be masked 0, got {m}"

    def test_raises_without_generation_markers(self) -> None:
        """A template lacking {% generation %} markers raises ValueError."""
        tok = _make_chat_tokenizer()
        # Use the plain role template (no {% generation %} markers).
        tok.chat_template = _ROLE_TEMPLATE
        assert "{% generation %}" not in tok.chat_template

        with pytest.raises(ValueError, match="generation"):
            apply_chat_template_with_mask(tok, _CONVERSATION)

    def test_accepts_whitespace_controlled_generation_marker(self) -> None:
        """The guard accepts the same marker spelling as Transformers."""
        tok = _make_chat_tokenizer()
        tok.chat_template = _WHITESPACE_CONTROLLED_ROLE_TEMPLATE

        result = apply_chat_template_with_mask(tok, _CONVERSATION)

        assert any(result["completion_mask"])

    @pytest.mark.parametrize(
        ("return_tensors", "expected_type"),
        [("pt", torch.Tensor), ("np", np.ndarray)],
    )
    def test_returns_nonempty_tensor_or_array_mask(
        self, return_tensors: str, expected_type: type
    ) -> None:
        """Tensor-backed masks are accepted without ambiguous truth-value errors."""
        tok = _make_chat_tokenizer()
        tok.chat_template = get_training_chat_template_for(tok)

        result = apply_chat_template_with_mask(
            tok, _CONVERSATION, return_tensors=return_tensors
        )

        assert isinstance(result["completion_mask"], expected_type)
        assert bool(result["completion_mask"].any())

    def test_rejects_all_zero_assistant_mask(self) -> None:
        """A recognized marker without rendered assistant spans still fails."""
        tok = _make_chat_tokenizer()
        tok.chat_template = (
            "{% for message in messages %}"
            "{% if false %}"
            "{%- generation -%}{{ message['content'] }}{%- endgeneration -%}"
            "{% endif %}"
            "{% endfor %}"
        )

        with pytest.raises(ValueError, match="returned no assistant-token mask"):
            apply_chat_template_with_mask(tok, _CONVERSATION)

    def test_resolves_named_template_override(self) -> None:
        """A named override follows Transformers' template-resolution behavior."""
        tok = _make_chat_tokenizer()
        tok.chat_template = {"default": _ROLE_TEMPLATE, "masked": _CHATML_WITH_GEN}

        result = apply_chat_template_with_mask(
            tok, _CONVERSATION, chat_template="masked"
        )

        assert any(result["completion_mask"])

    def test_gemma_mask_excludes_system_and_user_content(self) -> None:
        """Gemma's shared render expression masks only assistant content."""
        tok = _make_chat_tokenizer()
        tok.chat_template = _GEMMA_TEMPLATE
        tok.chat_template = get_training_chat_template(tok)

        result = apply_chat_template_with_mask(tok, _CONVERSATION)
        tokens = tok.convert_ids_to_tokens(result["input_ids"])
        prompt_tokens = {"You", "are", "helpful", "What", "is", "two", "plus"}

        for token, mask in zip(tokens, result["completion_mask"], strict=True):
            if token in prompt_tokens:
                assert mask == 0
            elif token in _ASSISTANT_TOKENS:
                assert mask == 1


def get_training_chat_template_for(tok: PreTrainedTokenizerFast) -> str:
    """Install the plain role template, then return its generation-marker form."""
    tok.chat_template = _ROLE_TEMPLATE
    return get_training_chat_template(tok)


_CHATML_WITH_GEN = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "user colon {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}"
    "assistant colon {% generation %}{{ message['content'] }}{% endgeneration %} "
    "{% endif %}"
    "{% endfor %}"
)
