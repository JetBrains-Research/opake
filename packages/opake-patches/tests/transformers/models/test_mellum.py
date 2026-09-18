# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Mellum 2.0 family (MoE), plus the original dense Mellum."""

import inspect
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from opake.api.patches.transformers.models.mellum import (
    apply_mellum_family_patches,
    apply_mellum_patches,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
    build_moe_model,
    experts_forward_patched,
    get_tiny_config_kwargs,
)


@pytest.fixture
def mellum_modeling():
    return pytest.importorskip("transformers.models.mellum.modeling_mellum")


@pytest.fixture
def tiny(device):
    return build_moe_model(
        "mellum", device, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64
    )


def test_mellum_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_mellum_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_mellum_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_mellum_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_mellum_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)


def test_original_mellum_routes_via_llama(device):
    """Original dense Mellum (``Mellum-4b``, model_type='llama') is served by the
    llama family — no Mellum-specific patch needed."""
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from opake.api.patches.transformers._family import family_name
    from opake.patches import apply_model_patches

    config = LlamaConfig(**get_tiny_config_kwargs())
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(device)
    assert family_name(model) == "llama"
    apply_model_patches(model, eager_attention=True)
    assert_forward_backward(model, device)


def test_mellum_keeps_transformers_rotary_embedding(mellum_modeling):
    original = mellum_modeling.apply_rotary_pos_emb

    apply_mellum_family_patches(performance=True, rope=True)

    assert mellum_modeling.apply_rotary_pos_emb is original


def test_mellum_keeps_transformers_rms_norm(mellum_modeling):
    original = mellum_modeling.MellumRMSNorm.forward

    apply_mellum_patches(
        mellum_modeling.nn.Module(),
        performance=False,
        compat=False,
        rms_norm=True,
    )

    assert mellum_modeling.MellumRMSNorm.forward is original


def test_mellum_installs_chunked_linear_cross_entropy(monkeypatch, mellum_modeling):
    patched = []
    monkeypatch.setattr(
        "opake.api.patches.transformers._factory._patch_forward",
        lambda *args, **kwargs: patched.append((args, kwargs)),
    )
    apply_mellum_patches(
        object(),
        performance=False,
        compat=False,
        fused_linear_cross_entropy=True,
    )

    assert len(patched) == 1
    factory = patched[0][0][1]
    forward = factory(lambda *args, **kwargs: None)
    assert inspect.getclosurevars(forward).nonlocals["force_chunked"] == 2048
