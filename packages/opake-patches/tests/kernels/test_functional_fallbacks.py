"""Tests for functional kernel fallbacks when Triton is unavailable.

These tests validate function APIs from opake.patches.kernels on CPU/MPS
in environments where Triton is not installed.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is not None,
    reason="Fallback tests only apply when Triton is unavailable",
)
def test_opake_swiglu_fallback_matches_torch():
    from opake.patches.kernels import opake_swiglu

    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)

    out = opake_swiglu(gate, up)
    expected = F.silu(gate) * up

    assert torch.allclose(out, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is not None,
    reason="Fallback tests only apply when Triton is unavailable",
)
def test_opake_cross_entropy_loss_fallback_shape_and_values():
    from opake.patches.kernels import opake_cross_entropy_loss

    logits = torch.randn(2, 3, 7)
    labels = torch.randint(0, 7, (2, 3))

    out = opake_cross_entropy_loss(logits, labels)
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    assert out.shape == labels.shape
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is not None,
    reason="Fallback tests only apply when Triton is unavailable",
)
def test_opake_lora_qkv_fallback_rejects_trainable_biases():
    from opake.exceptions import ConfigurationError
    from opake.patches.kernels import opake_lora_qkv

    x = torch.randn(2, 5, 6)
    projections = [torch.randn(4, 6), torch.randn(6, 3), torch.randn(3, 4), 0.5] * 3

    with pytest.raises(ConfigurationError, match="frozen Q/K/V base biases"):
        opake_lora_qkv(
            x,
            *projections,
            bq=torch.randn(4, requires_grad=True),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is not None,
    reason="Fallback tests only apply when Triton is unavailable",
)
def test_opake_lora_w_fallback_matches_reference():
    from opake.patches.kernels import opake_lora_w

    x = torch.randn(2, 5, 6)
    w = torch.randn(4, 6)
    a = torch.randn(6, 3)
    b = torch.randn(3, 4)
    scaling = 0.5

    out = opake_lora_w(x, w, a, b, scaling)
    expected = x @ w.transpose(-1, -2) + (x @ a @ b) * scaling

    assert torch.allclose(out, expected, atol=1e-6)
