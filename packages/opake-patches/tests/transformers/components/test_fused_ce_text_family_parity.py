# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""HF-golden parity checks for fused-CE wrapper on text families (CPU path)."""

from __future__ import annotations

import copy
import importlib
import types

import pytest
import torch

from opake.api.patches.transformers.components.cross_entropy import (
    _make_fused_ce_causal_lm_forward,
)

_TRANSFORM_FAMILIES = ("cohere", "cohere2", "granite", "gemma2", "gemma3")

_FAMILY_SPECS = {
    "cohere": {
        "module": "transformers.models.cohere.modeling_cohere",
        "config_module": "transformers.models.cohere.configuration_cohere",
        "config_cls": "CohereConfig",
        "model_cls": "CohereForCausalLM",
        "extra": {"logit_scale": 0.25},
    },
    "cohere2": {
        "module": "transformers.models.cohere2.modeling_cohere2",
        "config_module": "transformers.models.cohere2.configuration_cohere2",
        "config_cls": "Cohere2Config",
        "model_cls": "Cohere2ForCausalLM",
        "extra": {"logit_scale": 0.25},
    },
    "granite": {
        "module": "transformers.models.granite.modeling_granite",
        "config_module": "transformers.models.granite.configuration_granite",
        "config_cls": "GraniteConfig",
        "model_cls": "GraniteForCausalLM",
        "extra": {"logits_scaling": 8.0},
    },
    "gemma2": {
        "module": "transformers.models.gemma2.modeling_gemma2",
        "config_module": "transformers.models.gemma2.configuration_gemma2",
        "config_cls": "Gemma2Config",
        "model_cls": "Gemma2ForCausalLM",
        "extra": {"head_dim": 16, "final_logit_softcapping": 0.5},
    },
    "gemma3": {
        "module": "transformers.models.gemma3.modeling_gemma3",
        "config_module": "transformers.models.gemma3.configuration_gemma3",
        "config_cls": "Gemma3TextConfig",
        "model_cls": "Gemma3ForCausalLM",
        "extra": {
            "head_dim": 16,
            "num_hidden_layers": 2,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "final_logit_softcapping": 0.5,
        },
    },
    "olmo2": {
        "module": "transformers.models.olmo2.modeling_olmo2",
        "config_module": "transformers.models.olmo2.configuration_olmo2",
        "config_cls": "Olmo2Config",
        "model_cls": "Olmo2ForCausalLM",
        "extra": {},
    },
    "olmo3": {
        "module": "transformers.models.olmo3.modeling_olmo3",
        "config_module": "transformers.models.olmo3.configuration_olmo3",
        "config_cls": "Olmo3Config",
        "model_cls": "Olmo3ForCausalLM",
        "extra": {},
    },
    "smollm3": {
        "module": "transformers.models.smollm3.modeling_smollm3",
        "config_module": "transformers.models.smollm3.configuration_smollm3",
        "config_cls": "SmolLM3Config",
        "model_cls": "SmolLM3ForCausalLM",
        "extra": {},
    },
    "ministral": {
        "module": "transformers.models.ministral.modeling_ministral",
        "config_module": "transformers.models.ministral.configuration_ministral",
        "config_cls": "MinistralConfig",
        "model_cls": "MinistralForCausalLM",
        "extra": {"head_dim": 16},
    },
    "glm4": {
        "module": "transformers.models.glm4.modeling_glm4",
        "config_module": "transformers.models.glm4.configuration_glm4",
        "config_cls": "Glm4Config",
        "model_cls": "Glm4ForCausalLM",
        "extra": {},
    },
}

_UPSTREAM_FORWARDS = {
    family: getattr(importlib.import_module(spec["module"]), spec["model_cls"]).forward
    for family, spec in _FAMILY_SPECS.items()
}


def _tiny_model(family: str):
    spec = _FAMILY_SPECS[family]
    cfg_mod = importlib.import_module(spec["config_module"])
    model_mod = importlib.import_module(spec["module"])
    cfg_cls = getattr(cfg_mod, spec["config_cls"])
    model_cls = getattr(model_mod, spec["model_cls"])
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_theta": 10000.0,
    }
    kwargs.update(spec["extra"])
    cfg = cfg_cls(**kwargs)
    cfg._attn_implementation = "eager"
    return model_cls(cfg)


def _bind_upstream_forward(model, family):
    model.forward = types.MethodType(_UPSTREAM_FORWARDS[family], model)


def _bind_fused_forward(model, family):
    fused = _make_fused_ce_causal_lm_forward(_UPSTREAM_FORWARDS[family])
    model.forward = types.MethodType(fused, model)


def _assert_matching_gradients(base, wrapped):
    for (name, left), (wrapped_name, right) in zip(
        base.named_parameters(), wrapped.named_parameters(), strict=True
    ):
        assert name == wrapped_name
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", sorted(_FAMILY_SPECS))
def test_fused_ce_wrapper_matches_hf_loss_on_cpu(family):
    """The fused path matches upstream loss without materializing logits."""
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_upstream_forward(base, family)
    _bind_fused_forward(wrapped, family)

    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        out_base = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        out_wrapped = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_only=True,
            return_dict=True,
        )

    assert out_wrapped.logits is None  # fused path skips lm_head materialization
    assert torch.allclose(out_base.loss, out_wrapped.loss, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", _TRANSFORM_FAMILIES)
@pytest.mark.parametrize("fallback", [False, True])
def test_transformed_family_loss_and_gradients_match_upstream(family, fallback):
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_upstream_forward(base, family)
    _bind_fused_forward(wrapped, family)

    if family in ("cohere", "cohere2"):
        base.logit_scale = wrapped.logit_scale = 0.5

    input_ids = torch.randint(3, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    kwargs = {"logits_to_keep": input_ids.shape[-1]} if fallback else {}

    out_base = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        **kwargs,
    )
    backbone_calls = []
    handle = wrapped.model.register_forward_hook(lambda *_: backbone_calls.append(None))
    try:
        out_wrapped = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            loss_only=True,
            **kwargs,
        )
    finally:
        handle.remove()

    assert len(backbone_calls) == 1
    if fallback:
        torch.testing.assert_close(out_wrapped.logits, out_base.logits)
    else:
        assert out_wrapped.logits is None
    torch.testing.assert_close(out_wrapped.loss, out_base.loss, atol=1e-4, rtol=1e-4)

    out_base.loss.backward()
    out_wrapped.loss.backward()
    _assert_matching_gradients(base, wrapped)


@pytest.mark.cuda
@pytest.mark.parametrize("family", _TRANSFORM_FAMILIES)
def test_transformed_family_cuda_float_fallback_matches_upstream(family):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    torch.manual_seed(0)
    base = _tiny_model(family).cuda()
    wrapped = copy.deepcopy(base)
    _bind_upstream_forward(base, family)
    _bind_fused_forward(wrapped, family)
    if family in ("cohere", "cohere2"):
        base.logit_scale = wrapped.logit_scale = 0.5

    input_ids = torch.randint(3, base.config.vocab_size, (2, 9), device="cuda")
    labels = input_ids.clone()
    out_base = base(input_ids=input_ids, labels=labels, use_cache=False)
    backbone_calls = []
    handle = wrapped.model.register_forward_hook(lambda *_: backbone_calls.append(None))
    try:
        out_wrapped = wrapped(
            input_ids=input_ids, labels=labels, use_cache=False, loss_only=True
        )
    finally:
        handle.remove()

    assert len(backbone_calls) == 1
    torch.testing.assert_close(out_wrapped.logits, out_base.logits)
    torch.testing.assert_close(out_wrapped.loss, out_base.loss)
    out_base.loss.backward()
    out_wrapped.loss.backward()
    _assert_matching_gradients(base, wrapped)


def test_fused_ce_explicitly_preserves_logits_for_metrics():
    """The SFT metrics path can opt out without changing legacy callers."""
    model = _tiny_model("olmo2")
    _bind_fused_forward(model, "olmo2")
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            labels=input_ids,
            loss_only=False,
            return_dict=True,
        )

    assert output.logits is not None


def test_fused_ce_preserves_router_auxiliary_loss_contract():
    sentinel = object()
    calls = []

    def original(_self, **kwargs):
        calls.append(kwargs)
        return sentinel

    config = types.SimpleNamespace(output_router_logits=False)
    model = types.SimpleNamespace(config=config)
    forward = _make_fused_ce_causal_lm_forward(original)
    labels = torch.ones(1, 3, dtype=torch.long)

    output = forward(
        model,
        labels=labels,
        output_router_logits=True,
        loss_only=True,
    )

    assert output is sentinel
    assert calls == [
        {
            "input_ids": None,
            "attention_mask": None,
            "position_ids": None,
            "past_key_values": None,
            "inputs_embeds": None,
            "labels": labels,
            "use_cache": None,
            "output_attentions": None,
            "output_hidden_states": None,
            "return_dict": None,
            "cache_position": None,
            "logits_to_keep": 0,
            "output_router_logits": True,
        }
    ]


def test_marker_false_delegates_to_original_forward():
    """Logits-consuming calls retain the exact model-native output contract."""
    sentinel = object()
    calls = []

    def original(_self, **kwargs):
        calls.append(kwargs)
        return sentinel

    forward = _make_fused_ce_causal_lm_forward(original)
    labels = torch.ones(1, 3, dtype=torch.long)

    output = forward(object(), labels=labels, loss_only=False)

    assert output is sentinel
    assert calls[0]["labels"] is labels
    assert "loss_only" not in calls[0]


@pytest.mark.parametrize(
    ("config_values", "expected_scale"),
    [
        ({"logit_scale": 0.25}, 0.25),
        ({"logits_scaling": 16.0}, 1.0 / 16.0),
        ({"logit_scale": 0.25, "logits_scaling": 2.0}, 0.125),
    ],
)
def test_fused_ce_passes_family_scaling_without_copy(
    monkeypatch, config_values, expected_scale
):
    hidden = torch.randn(1, 5, 8)
    weight = torch.nn.Parameter(torch.randn(32, 8))
    labels = torch.randint(0, 32, (1, 5))
    captured = {}

    class Dummy:
        config = types.SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=False,
            final_logit_softcapping=0.0,
            **config_values,
        )
        vocab_size = 32
        lm_head = types.SimpleNamespace(weight=weight)

        def model(self, **kwargs):
            del kwargs
            return (hidden,)

        def loss_function(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("fallback should not run")

    def fake_chunked(
        hidden_states, kernel_weight, kernel_labels, *args, chunk_vocab=None
    ):
        captured["chunk_vocab"] = chunk_vocab
        captured["hidden"] = hidden_states
        captured["weight"] = kernel_weight
        captured["labels"] = kernel_labels
        captured["scale"] = args[-1]
        return hidden_states.new_tensor(4.0)

    monkeypatch.setattr(
        "opake.api.patches.kernels._linear_ce_chunked.linear_nll_sum_chunked",
        fake_chunked,
    )
    output = _make_fused_ce_causal_lm_forward(lambda *args, **kwargs: None)(
        Dummy(),
        input_ids=torch.ones_like(labels),
        labels=labels,
        loss_only=True,
        return_dict=False,
    )

    assert output[1] is None
    assert captured["chunk_vocab"] is None
    assert captured["hidden"] is hidden
    assert captured["weight"] is weight
    assert captured["labels"] is labels
    assert captured["scale"] == pytest.approx(expected_scale)


def test_scaled_fused_ce_preserves_tied_weight_chain_rule():
    class Backbone(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding

        def forward(self, input_ids=None, **kwargs):
            del kwargs
            return (self.embedding(input_ids),)

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            embedding = torch.nn.Embedding(32, 8)
            self.model = Backbone(embedding)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)
            self.lm_head.weight = embedding.weight
            self.vocab_size = 32
            self.config = types.SimpleNamespace(
                output_attentions=False,
                output_hidden_states=False,
                use_return_dict=False,
                final_logit_softcapping=0.0,
                logit_scale=0.25,
                logits_scaling=1.0,
            )

        def loss_function(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("fallback should not run")

    torch.manual_seed(3)
    eager = TiedModel()
    fused = copy.deepcopy(eager)
    input_ids = torch.randint(0, 32, (2, 7))
    labels = input_ids.clone()

    hidden = eager.model(input_ids=input_ids)[0]
    logits = eager.lm_head(hidden) * eager.config.logit_scale
    eager_loss = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].reshape(-1, eager.vocab_size),
        labels[..., 1:].reshape(-1),
    )
    eager_loss.backward()

    fused_forward = _make_fused_ce_causal_lm_forward(lambda *args, **kwargs: None)
    fused_loss = fused_forward(
        fused,
        input_ids=input_ids,
        labels=labels,
        loss_only=True,
        return_dict=False,
    )[0]
    fused_loss.backward()

    assert torch.allclose(fused_loss, eager_loss, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        fused.lm_head.weight.grad,
        eager.lm_head.weight.grad,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("family", sorted(_FAMILY_SPECS))
def test_fused_ce_wrapper_preserves_logits_to_keep(family):
    """T1: wrapper must preserve HF logits slicing semantics."""
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_upstream_forward(base, family)
    _bind_fused_forward(wrapped, family)

    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_base = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )
        out_wrapped = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )

    assert (
        out_base.logits.shape
        == out_wrapped.logits.shape
        == (
            2,
            2,
            base.config.vocab_size,
        )
    )
    assert torch.allclose(out_base.logits, out_wrapped.logits, atol=1e-6, rtol=1e-5)
