# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Integration check: fused ForCausalLM path forwards label_smoothing to CCE."""

from __future__ import annotations

import importlib.util
import types

import pytest
import torch

from opaque.api.patches.transformers.components.cross_entropy import (
    _make_fused_ce_causal_lm_forward,
)

pytest.importorskip("triton")
from opaque.api.patches.kernels.linear_cross_entropy import (
    Opaque_LinearCrossEntropyLoss,
)

_KERNELS_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)
pytestmark = pytest.mark.skipif(
    not _KERNELS_AVAILABLE,
    reason="Fused CE label-smoothing tests require CUDA + Triton",
)


class _DummyForCausalLM:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.config = types.SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=False,
            final_logit_softcapping=0.0,
            logit_scale=1.0,
            logits_scaling=1.0,
        )
        self.vocab_size = vocab_size
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False).to(
            device="cuda", dtype=torch.bfloat16
        )

    def model(self, input_ids=None, **kwargs):
        del kwargs
        batch, seq = input_ids.shape
        hidden = self.lm_head.weight.new_zeros((batch, seq, self.lm_head.in_features))
        return (hidden, "dummy-cache")

    def loss_function(self, *args, **kwargs):
        raise AssertionError("fallback loss_function should not run in fused path")


def test_fused_for_causal_lm_forwards_label_smoothing(monkeypatch):
    model = _DummyForCausalLM(hidden_size=32, vocab_size=64)
    model.config.logit_scale = 0.25
    model.config.logits_scaling = 2.0
    labels = torch.randint(0, model.vocab_size, (2, 8), device="cuda")
    input_ids = torch.randint(0, model.vocab_size, (2, 8), device="cuda")

    called = {"original": False}
    captured = {}

    def original_forward(*args, **kwargs):
        del args, kwargs
        called["original"] = True
        raise AssertionError("original forward should not run")

    def fake_apply(
        hidden_states,
        weight,
        labels_,
        ignore_index,
        logit_softcapping,
        label_smoothing,
        use_token_scaling=False,
        logit_scale=1.0,
    ):
        captured["hidden_dtype"] = hidden_states.dtype
        captured["weight_dtype"] = weight.dtype
        captured["ignore_index"] = ignore_index
        captured["softcap"] = logit_softcapping
        captured["label_smoothing"] = label_smoothing
        captured["logit_scale"] = logit_scale
        captured["weight"] = weight
        captured["label_shape"] = tuple(labels_.shape)
        # The kernel's four outputs: (nll_sum, lse, valids, token_weight).
        empty = hidden_states.new_empty(0)
        return hidden_states.new_tensor(12.0, dtype=torch.float32), empty, empty, empty

    monkeypatch.setattr(
        Opaque_LinearCrossEntropyLoss, "apply", staticmethod(fake_apply)
    )

    fused_forward = _make_fused_ce_causal_lm_forward(original_forward)
    output = fused_forward(
        model,
        input_ids=input_ids,
        labels=labels,
        loss_only=True,
        return_dict=False,
        label_smoothing=0.1,
        ignore_index=-100,
    )

    assert called["original"] is False
    assert captured["label_smoothing"] == pytest.approx(0.1)
    assert captured["logit_scale"] == pytest.approx(0.125)
    assert captured["weight"] is model.lm_head.weight
    assert captured["ignore_index"] == -100
    assert captured["softcap"] == 0
    assert captured["hidden_dtype"] == torch.bfloat16
    assert captured["weight_dtype"] == torch.bfloat16
    assert captured["label_shape"] == tuple(labels.shape)
    # return_dict=False path => (loss, logits, ...)
    assert torch.is_tensor(output[0])
    assert output[1] is None
