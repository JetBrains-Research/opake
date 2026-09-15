# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""HF-golden parity checks for fused-CE wrapper on text families (CPU path)."""

from __future__ import annotations

import copy
import importlib
import types

import pytest
import torch

from opaque.api.patches.transformers.components.cross_entropy import (
    _make_fused_ce_causal_lm_forward,
)

_FAMILY_SPECS = {
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


def _bind_fused_forward(model):
    original = type(model).forward
    fused = _make_fused_ce_causal_lm_forward(original)
    model.forward = types.MethodType(fused, model)


@pytest.mark.parametrize("family", sorted(_FAMILY_SPECS))
def test_fused_ce_wrapper_matches_hf_loss_on_cpu(family):
    """T1: with labels, the fused-CE wrapper matches stock HF *loss* and skips
    logits. Off-CUDA the chunked kernel powers the fused path (it streams the LSE
    in fp32, so the loss matches HF's fp32 CE); ``logits`` is ``None`` because the
    fast path never materializes ``lm_head`` — the opt-in ``logits=None`` contract.
    The fallback-with-logits path is covered by ``preserves_logits_to_keep``."""
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_fused_forward(wrapped)

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


def test_fused_ce_explicitly_preserves_logits_for_metrics():
    """The SFT metrics path can opt out without changing legacy callers."""
    model = _tiny_model("olmo2")
    _bind_fused_forward(model)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            labels=input_ids,
            loss_only=False,
            return_dict=True,
        )

    assert output.logits is not None


def test_router_auxiliary_loss_contract_kept_without_loss_only():
    """Without ``loss_only`` a router-logits request defers to HF's forward."""
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
        loss_only=False,
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
        "opaque.api.patches.kernels._linear_ce_chunked.linear_nll_sum_chunked",
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
    _bind_fused_forward(wrapped)

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
