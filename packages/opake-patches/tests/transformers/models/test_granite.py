# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for granite model."""

import pytest

pytest.importorskip("transformers")

import sys
from pathlib import Path

from transformers.models.granite.modeling_granite import (
    GraniteConfig,
    GraniteForCausalLM,
)

from opake.patches import apply_model_patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
    get_tiny_config_kwargs,
)


@pytest.fixture
def tiny_model(device):
    kwargs = get_tiny_config_kwargs()

    config = GraniteConfig(**kwargs)
    config._attn_implementation = "sdpa"
    model = GraniteForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_granite_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_granite_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_granite_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_granite_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
