# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for gemma2 model."""

import pytest

pytest.importorskip("transformers")

import sys
from pathlib import Path

from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2ForCausalLM

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
    kwargs.update({"head_dim": 16})
    config = Gemma2Config(**kwargs)
    config._attn_implementation = "sdpa"
    model = Gemma2ForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_gemma2_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_gemma2_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_gemma2_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_gemma2_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)


def test_gemma2_family_preserves_softcap_attention_backends():
    """Install the specialized eager and SDPA backends for Gemma2."""
    from transformers.models.gemma2 import modeling_gemma2

    from opake.api.patches.transformers.components.attention import (
        vmap_eager_attention_forward_gemma2,
        vmap_sdpa_attention_forward_gemma2,
    )
    from opake.api.patches.transformers.models.gemma2 import (
        apply_gemma2_family_patches,
    )

    apply_gemma2_family_patches(eager_attention=True)

    assert (
        modeling_gemma2.eager_attention_forward is vmap_eager_attention_forward_gemma2
    )
    assert (
        modeling_gemma2.ALL_ATTENTION_FUNCTIONS["sdpa"]
        is vmap_sdpa_attention_forward_gemma2
    )
