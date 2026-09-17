# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for glm4 model."""

import pytest

pytest.importorskip("transformers")

import sys
from pathlib import Path

from transformers.models.glm4.modeling_glm4 import Glm4Config, Glm4ForCausalLM

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

    config = Glm4Config(**kwargs)
    config._attn_implementation = "sdpa"
    model = Glm4ForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_glm4_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_glm4_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_glm4_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_glm4_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
