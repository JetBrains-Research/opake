"""Tests for :mod:`opake.api.alignment.data._chat_template`.

Covers :func:`get_training_chat_template` (inserting / preserving the
``{% generation %}`` markers, error cases) and :func:`clone_chat_template`
(copying the template and special tokens, and the resize invariant
``embedding rows == len(tokenizer)``).

All tests are CPU-only and network-free with the smallest possible tokenizer
and model configs. ``transformers`` is required (``pytest.importorskip``).
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from tokenizers import AddedToken, Tokenizer  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import (  # noqa: E402
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from opake.api.alignment.data._chat_template import (  # noqa: E402
    clone_chat_template,
    get_training_chat_template,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GEN_START = "{% generation %}"
_GEN_END = "{% endgeneration %}"

# A minimal ChatML-style template with an explicit assistant branch.
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "user: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
)

# Same template but already containing the generation markers.
_CHATML_WITH_GEN = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "user: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "assistant: {% generation %}{{ message['content'] }}{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
)

_GEMMA_BRANCH_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{% set role = 'model' %}"
    "{% else %}"
    "{% set role = message['role'] %}"
    "{% endif %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim "
    "+ '<end_of_turn>\\n' }}"
    "{% endfor %}"
)

_GEMMA_INLINE_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim "
    "+ '<end_of_turn>\\n' }}"
    "{% endfor %}"
)

_LLAMA_3_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] "
    "+ '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

_LLAMA_3_EXPLICIT_BRANCH_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
)


def _make_fast_tokenizer(
    extra_specials: list[str] | None = None,
    n_regular: int = 3,
) -> PreTrainedTokenizerFast:
    """Return a :class:`PreTrainedTokenizerFast` with a trivial BPE vocab.

    The vocabulary always contains ``<unk>`` (special) plus *n_regular* normal
    tokens ``w0, w1, …``.  Optionally, additional special tokens are added.
    No network access is performed.
    """
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    specials = [AddedToken("<unk>", special=True)]
    if extra_specials:
        specials += [AddedToken(s, special=True) for s in extra_specials]
    tok.add_special_tokens(specials)
    tok.add_tokens([f"w{i}" for i in range(n_regular)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


def _make_tiny_model(vocab_size: int) -> LlamaForCausalLM:
    """Return the smallest possible LlamaForCausalLM on CPU."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
    )
    return LlamaForCausalLM(cfg)


# ---------------------------------------------------------------------------
# Tests: get_training_chat_template
# ---------------------------------------------------------------------------


class TestGetTrainingChatTemplate:
    def test_generation_marker_inserted(self) -> None:
        """Template without markers should gain {% generation %} / {% endgeneration %}."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        result = get_training_chat_template(tokenizer)
        assert _GEN_START in result
        assert _GEN_END in result

    def test_idempotent_when_markers_present(self) -> None:
        """Template that already has {% generation %} must be returned unchanged."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_WITH_GEN
        result = get_training_chat_template(tokenizer)
        assert result == _CHATML_WITH_GEN

    def test_idempotent_with_whitespace_controlled_markers(self) -> None:
        """Transformers-recognized whitespace-control tags are preserved."""
        tokenizer = _make_fast_tokenizer()
        template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'assistant' %}"
            "{%- generation -%}{{ message['content'] }}{%- endgeneration -%}"
            "{% else %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = template
        assert get_training_chat_template(tokenizer) == template

    def test_rejects_premarked_user_spans(self) -> None:
        """Existing markers must be validated before accepting the template."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{%- generation -%}{{ message['content'] }}{%- endgeneration -%}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )
        with pytest.raises(ValueError, match="existing generation markers"):
            get_training_chat_template(tokenizer)

    def test_rejects_premarked_system_spans(self) -> None:
        """Existing markers must not include system prompt content."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{%- generation -%}{{ message['content'] }}{%- endgeneration -%}"
            "{% elif message['role'] == 'assistant' %}"
            "{%- generation -%}{{ message['content'] }}{%- endgeneration -%}"
            "{% endif %}"
            "{% endfor %}"
        )
        with pytest.raises(ValueError, match="existing generation markers"):
            get_training_chat_template(tokenizer)

    def test_idempotent_double_call(self) -> None:
        """Calling the function twice on a plain template yields the same output both times."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        first = get_training_chat_template(tokenizer)
        # Simulate a second call by setting chat_template to the result.
        tokenizer2 = _make_fast_tokenizer()
        tokenizer2.chat_template = first
        second = get_training_chat_template(tokenizer2)
        assert first == second

    def test_unsupported_template_raises_instead_of_guessing(self) -> None:
        """A template with no identifiable assistant path fails explicitly."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        with pytest.raises(ValueError, match="assistant-only render path"):
            get_training_chat_template(tokenizer)

    def test_raises_on_none_template(self) -> None:
        """ValueError is raised when chat_template is None."""
        tokenizer = _make_fast_tokenizer()
        # PreTrainedTokenizerFast defaults chat_template to None.
        tokenizer.chat_template = None
        with pytest.raises(ValueError, match="chat_template"):
            get_training_chat_template(tokenizer)

    def test_raises_on_empty_template(self) -> None:
        """ValueError is raised when chat_template is an empty string."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = ""
        with pytest.raises(ValueError, match="chat_template"):
            get_training_chat_template(tokenizer)

    def test_original_template_unchanged(self) -> None:
        """The tokenizer's chat_template attribute must not be mutated in-place."""
        tokenizer = _make_fast_tokenizer()
        original = _CHATML_TEMPLATE
        tokenizer.chat_template = original
        _ = get_training_chat_template(tokenizer)
        # The tokenizer should still hold the original (unmodified) template.
        assert tokenizer.chat_template == original

    def test_markers_placed_around_content_not_header(self) -> None:
        """The generation markers must wrap the *content* expression, not surrounding text."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        result = get_training_chat_template(tokenizer)
        # The user-turn content expression must NOT be wrapped.
        # Find the assistant branch in the result and confirm markers are there.
        assert "assistant: " in result
        # The generation marker must appear inside the assistant branch, not around user branch.
        assistant_idx = result.find("assistant:")
        gen_idx = result.find(_GEN_START)
        assert gen_idx > assistant_idx, (
            "{% generation %} should appear after the 'assistant:' literal"
        )

    def test_shared_or_clause_qwen_pattern(self) -> None:
        """OR-clause shared rendering (Qwen2.5-Instruct) wraps assistant only.

        Templates that emit user/system/assistant turns through one shared
        ``{{ ... message.role ... message.content ... }}`` inside a
        multi-condition ``{%- if (role == "user") or ... or (role ==
        "assistant" and not tool_calls) %}`` must NOT have user/system tokens
        flagged by ``return_assistant_tokens_mask=True``.  Strategy 2b in
        ``get_training_chat_template`` rewrites the shared expression with an
        inner role-guard so only the assistant render path goes through the
        ``{% generation %}`` block.
        """
        # Minimal Qwen-style template that triggers Strategy 2b: the OR clause
        # combines user/system/assistant-without-tool-calls through one shared
        # expression, while a second elif handles the assistant-with-tool-calls
        # case (which Strategy 2's "last assistant branch" heuristic would pick
        # by default, producing an unreachable generation block).
        qwen_style = (
            "{% for message in messages %}"
            '{%- if (message.role == "user") or (message.role == "system" and not loop.first) '
            'or (message.role == "assistant" and not message.tool_calls) %}'
            "{{- message.role + ': ' + message.content + '\\n' }}"
            '{%- elif message.role == "assistant" %}'
            "{{- message.role + ': ' + message.content + ' (tool-call)\\n' }}"
            "{%- endif %}"
            "{% endfor %}"
        )
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = qwen_style
        result = get_training_chat_template(tokenizer)
        assert _GEN_START in result
        assert _GEN_END in result

        # The wrap must include a role-guard so user/system don't enter the
        # generation block.
        assert "message.role == 'assistant'" in result, (
            "Strategy 2b should add an inner role-guard around the shared expression"
        )

        # Validate the wrapped template actually flags assistant content via
        # HF's tracker when rendered.
        from transformers.utils.chat_template_utils import (
            _compile_jinja_template,
            _render_with_assistant_indices,
        )

        compiled = _compile_jinja_template(result)
        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
        rendered, indices = _render_with_assistant_indices(
            compiled, msgs, None, None, False
        )
        assert indices, "generation block must fire for assistant turns"
        # Every span must be non-degenerate.
        assert all(end > start for start, end in indices)
        # And the spans must cover assistant text, NOT user text.
        for start, end in indices:
            span = rendered[start:end]
            assert "first question" not in span, (
                f"user content leaked into generation span: {span!r}"
            )
            assert "second question" not in span, (
                f"user content leaked into generation span: {span!r}"
            )
        joined = "".join(rendered[s:e] for s, e in indices)
        assert "first answer" in joined
        assert "second answer" in joined

    @pytest.mark.parametrize(
        "template",
        [_GEMMA_BRANCH_TEMPLATE, _GEMMA_INLINE_TEMPLATE],
        ids=["branch-role-mapping", "inline-role-mapping"],
    )
    def test_gemma_shared_render_marks_only_assistant(self, template: str) -> None:
        """Gemma's shared turn expression excludes system and user spans."""
        from transformers.utils.chat_template_utils import (
            _compile_jinja_template,
            _render_with_assistant_indices,
        )

        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = template
        result = get_training_chat_template(tokenizer)

        messages = [
            {"role": "system", "content": "system probe"},
            {"role": "user", "content": "first user probe"},
            {"role": "assistant", "content": "first assistant probe"},
            {"role": "user", "content": "second user probe"},
            {"role": "assistant", "content": "second assistant probe"},
        ]
        original_rendered, original_indices = _render_with_assistant_indices(
            _compile_jinja_template(template), messages, None, None, False
        )
        rendered, indices = _render_with_assistant_indices(
            _compile_jinja_template(result), messages, None, None, False
        )

        assert rendered == original_rendered
        assert original_indices == []
        generated = "".join(rendered[start:end] for start, end in indices)
        assert "first assistant probe" in generated
        assert "second assistant probe" in generated
        assert "system probe" not in generated
        assert "first user probe" not in generated
        assert "second user probe" not in generated

    @pytest.mark.parametrize(
        "template",
        [_LLAMA_3_TEMPLATE, _GEMMA_INLINE_TEMPLATE, _CHATML_TEMPLATE],
        ids=["meta-llama-3-8b-instruct", "gemma-3-1b-it", "shared-chatml"],
    )
    def test_representative_templates_render_assistant_only(
        self, template: str
    ) -> None:
        """Llama 3, Gemma 3, and ChatML templates produce assistant-only spans."""
        from transformers.utils.chat_template_utils import (
            _compile_jinja_template,
            _render_with_assistant_indices,
        )

        tokenizer = _make_fast_tokenizer()
        if template == _LLAMA_3_TEMPLATE:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})
        tokenizer.chat_template = template
        transformed = get_training_chat_template(tokenizer)
        rendered, indices = _render_with_assistant_indices(
            _compile_jinja_template(transformed),
            [
                {"role": "user", "content": "user probe"},
                {"role": "assistant", "content": "assistant probe"},
            ],
            None,
            None,
            False,
            **tokenizer.special_tokens_map,
        )

        generated = "".join(rendered[start:end] for start, end in indices)
        assert "assistant probe" in generated
        assert "user probe" not in generated

    def test_resolves_default_template_from_template_mapping(self) -> None:
        """A template mapping follows Transformers' default-template selection."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = {"default": _CHATML_WITH_GEN}
        assert get_training_chat_template(tokenizer) == _CHATML_WITH_GEN

    def test_llama_explicit_branch_template_remains_supported(self) -> None:
        """Explicit-branch Llama templates retain their prior transform path."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _LLAMA_3_EXPLICIT_BRANCH_TEMPLATE
        assert _GEN_START in get_training_chat_template(tokenizer)


# ---------------------------------------------------------------------------
# Tests: clone_chat_template
# ---------------------------------------------------------------------------


class TestCloneChatTemplate:
    def test_chat_template_copied(self) -> None:
        """The destination tokenizer's chat_template is set to the source's value."""
        src = _make_fast_tokenizer()
        src.chat_template = "my_template"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        clone_chat_template(model, dst, src)

        assert dst.chat_template == "my_template"

    def test_new_special_tokens_added(self) -> None:
        """After cloning, destination tokenizer contains all source special tokens."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()  # no im_start/im_end
        model = _make_tiny_model(len(dst))

        clone_chat_template(model, dst, src)

        dst_vocab = dst.get_vocab()
        assert "<|im_start|>" in dst_vocab
        assert "<|im_end|>" in dst_vocab

    def test_vocab_grew(self) -> None:
        """After cloning a source that adds two new special tokens, len(tokenizer) increases."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()
        len_before = len(dst)
        model = _make_tiny_model(len_before)

        clone_chat_template(model, dst, src)

        assert len(dst) == len_before + 2

    def test_resize_invariant(self) -> None:
        """CRITICAL (risk α10): embedding matrix rows == len(tokenizer) after clone."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        model, dst, _added = clone_chat_template(model, dst, src)

        embed_rows = model.get_input_embeddings().weight.shape[0]
        assert embed_rows == len(dst), (
            f"Embedding rows ({embed_rows}) must equal tokenizer vocab size ({len(dst)})"
        )

    def test_noop_when_tokens_already_present(self) -> None:
        """When destination already has all source special tokens, vocab does not grow."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        len_before = len(dst)
        model = _make_tiny_model(len_before)

        model, dst, _added = clone_chat_template(model, dst, src)

        assert len(dst) == len_before
        assert model.get_input_embeddings().weight.shape[0] == len_before

    def test_resize_invariant_noop_case(self) -> None:
        """Resize invariant holds even when no new tokens were added (no-op case)."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer(extra_specials=["<|im_start|>"])
        model = _make_tiny_model(len(dst))

        model, dst, _added = clone_chat_template(model, dst, src)

        assert model.get_input_embeddings().weight.shape[0] == len(dst)

    def test_returns_model_tokenizer_and_added_tokens(self) -> None:
        """Return value is a (model, tokenizer, added_token_ids) triple."""
        src = _make_fast_tokenizer()
        src.chat_template = "t"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        ret_model, ret_tok, added = clone_chat_template(model, dst, src)

        assert ret_model is model
        assert ret_tok is dst
        assert added == []  # no new special tokens in this case

    def test_added_token_ids_reported(self) -> None:
        """The added-token ids index exactly the newly added special tokens."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        _model, dst, added = clone_chat_template(model, dst, src)

        assert sorted(added) == sorted(
            dst.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
        )
        # Every reported id is a real vocab row inside the resized embedding.
        n_rows = _model.get_input_embeddings().weight.shape[0]
        assert all(0 <= tid < n_rows for tid in added)

    def test_preserves_special_token_roles_and_metadata(self) -> None:
        """Cloning retains named roles and AddedToken whitespace behavior."""
        src = _make_fast_tokenizer()
        src.chat_template = "template"
        src.add_special_tokens(
            {
                "bos_token": AddedToken("<bos>", lstrip=True, special=True),
                "eos_token": AddedToken("<eos>", rstrip=True, special=True),
                "pad_token": "<pad>",
                "additional_special_tokens": [
                    AddedToken("<extra>", single_word=True, special=True)
                ],
            }
        )
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        _model, dst, added = clone_chat_template(model, dst, src)

        assert dst.bos_token == "<bos>"
        assert dst.eos_token == "<eos>"
        assert dst.pad_token == "<pad>"
        assert "<extra>" in dst.extra_special_tokens
        assert dst.added_tokens_decoder[dst.bos_token_id].lstrip is True
        assert dst.added_tokens_decoder[dst.eos_token_id].rstrip is True
        assert dst.added_tokens_decoder[
            dst.convert_tokens_to_ids("<extra>")
        ].single_word
        assert len(added) == 4

    def test_preserves_model_specific_special_token_roles(self) -> None:
        """Cloning registers source-specific special-token roles on the target."""
        src = _make_fast_tokenizer()
        src.chat_template = "template"
        image_token = AddedToken("<image>", special=True)
        src._set_model_specific_special_tokens({"image_token": image_token})
        src.add_special_tokens({"image_token": image_token})
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        _model, dst, added = clone_chat_template(model, dst, src)

        assert str(dst._special_tokens_map["image_token"]) == "<image>"
        assert dst.convert_tokens_to_ids("<image>") in added

    def test_raises_on_source_without_chat_template(self) -> None:
        """ValueError is raised when the source tokenizer has no chat_template."""
        src = _make_fast_tokenizer()
        src.chat_template = None  # unset
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        with pytest.raises(ValueError, match="chat_template"):
            clone_chat_template(model, dst, src)
