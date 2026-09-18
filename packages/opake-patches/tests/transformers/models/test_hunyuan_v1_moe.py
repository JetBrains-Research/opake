# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hunyuan_v1_moe family (MoE)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
    build_moe_model,
    experts_forward_patched,
)


@pytest.fixture
def tiny(device):
    return build_moe_model(
        "hunyuan_v1_moe",
        device,
        num_experts=8,
        moe_topk=2,
        num_experts_per_tok=2,
        head_dim=16,
    )


def test_hunyuan_v1_moe_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_hunyuan_v1_moe_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_hunyuan_v1_moe_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_hunyuan_v1_moe_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_hunyuan_v1_moe_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)
