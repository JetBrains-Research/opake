# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the gpt_oss family (MoE with custom expert activation).

Experts are intentionally left to HF (custom clamped-SwiGLU / MXFP4 activation),
so Opaque patches only RMSNorm/RoPE/CE. HF's own experts forward is built on
``torch._grouped_mm`` + ``torch.histc``, which ARE vmap-traceable — so DP-SGD
``vmap(grad)`` works through the unpatched experts, and there's no need for an
Opaque expert kernel here.
"""

import sys
from pathlib import Path

import pytest
import torch

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
        "gpt_oss", device, num_local_experts=8, num_experts_per_tok=2
    )


def test_gpt_oss_experts_not_patched(tiny):
    # Custom experts are intentionally left to HF (mirrors Liger swiglu=False);
    # HF's grouped-GEMM forward is already fast and vmap-safe.
    assert not experts_forward_patched(tiny[1])


def test_gpt_oss_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_gpt_oss_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)
    sink_params = {
        name: parameter
        for name, parameter in tiny[0].named_parameters()
        if name.endswith(".sinks")
    }
    assert sink_params
    assert all(parameter.grad is not None for parameter in sink_params.values())


def test_gpt_oss_vmap_grad(tiny, device):
    """DP-SGD per-sample gradients run through HF's (vmap-safe) experts forward."""
    grads = assert_vmap_grad(tiny[0], device)
    sink_grads = {
        name: gradient
        for name, gradient in grads.pytree.items()
        if name.endswith(".sinks")
    }
    assert sink_grads
    assert all(torch.isfinite(gradient).all() for gradient in sink_grads.values())
