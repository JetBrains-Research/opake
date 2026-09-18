# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the gpt2 family (compat-only: batchify + kv_cache, no kernels)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

from opake.patches import apply_model_patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
)


@pytest.fixture
def tiny_model(device):
    # GPT-2 keeps its default 0.1 dropout on purpose: the compat ``disable_dropout``
    # patch zeros it during model-prep, which is what lets gpt2 run on sdpa under
    # vmap (active dropout's RNG otherwise trips vmap's "same randomness"). This
    # fixture is thus also the integration proof that the patch works.
    config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=4,
    )
    config._attn_implementation = "sdpa"
    model = GPT2LMHeadModel(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_gpt2_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_gpt2_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_gpt2_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_gpt2_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
