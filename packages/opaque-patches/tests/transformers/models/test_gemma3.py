# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for gemma3 model."""

import pytest

pytest.importorskip("transformers")

import sys
from pathlib import Path

import torch
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
)

from opaque.exceptions import ConfigurationError
from opaque.patches import apply_model_patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
    get_tiny_config_kwargs,
)


def _tiny_config(**overrides):
    kwargs = get_tiny_config_kwargs()
    kwargs.update(
        {
            "head_dim": 16,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "num_hidden_layers": 2,
        }
    )
    kwargs.update(overrides)
    config = Gemma3TextConfig(**kwargs)
    config._attn_implementation = "sdpa"
    return config


@pytest.fixture
def tiny_model(device):
    model = Gemma3ForCausalLM(_tiny_config()).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_gemma3_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_gemma3_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_gemma3_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_gemma3_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)


def test_gemma3_bidirectional_attention_fails_closed(device):
    model = Gemma3ForCausalLM(_tiny_config(use_bidirectional_attention=True)).to(device)
    apply_model_patches(model, eager_attention=True)

    with pytest.raises(
        ConfigurationError,
        match=r"`or_mask_function` cannot be combined with vmap masking",
    ):
        model(input_ids=torch.tensor([[1, 2, 3, 4]], device=device))
