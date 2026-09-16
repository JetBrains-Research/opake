# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_next family (hybrid linear-attention MoE).

Experts are patched (vmap-safe), and the recurrent padding-mask patch makes the
GatedDeltaNet path traceable by DP-SGD's per-example gradient transform.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_grad,
    build_moe_model,
    experts_forward_patched,
)


@pytest.fixture
def tiny(device):
    return build_moe_model(
        "qwen3_next",
        device,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        num_hidden_layers=4,
    )


def test_qwen3_next_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_qwen3_next_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_qwen3_next_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_qwen3_next_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)
