# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mixtral family (MoE)."""

import sys
from pathlib import Path

import pytest
import torch

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
        "mixtral", device, num_local_experts=8, num_experts_per_tok=2
    )


def test_mixtral_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_mixtral_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_mixtral_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_mixtral_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_mixtral_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixtral_vmap_grad_bf16_fused(tiny, device):
    """bf16 on CUDA engages the sparse fused-MoE Triton path through the real
    clipped_grad (DP-SGD) pipeline end-to-end."""
    assert_vmap_grad(tiny[0], device, dtype=torch.bfloat16)
