# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit and vmap-safety tests for the WPO reweighting helper.

Covers :func:`wpo_weights` — the WPO per-example reweighting (arXiv:2406.11827).
A hand-computed weight-alignment case, the **detach** invariant, and the
all-zero-mask ``clamp(min=1)`` div-by-zero guard.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from opake.api.alignment.dpo.loss._wpo import wpo_weights

# ---------------------------------------------------------------------------
# wpo_weights — per-example reweighting (arXiv:2406.11827)
# ---------------------------------------------------------------------------


def test_wpo_weights_hand_computed() -> None:
    """Equation 2 aligns each token by the policy's collision probability."""
    probabilities = torch.tensor(
        [
            [[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    logits = probabilities.log()
    # The realised token is class 0. Row 0's aligned probabilities are
    # [0.5 / 0.5, 0.25 / 0.625] = [1, 0.4]; the masked geometric mean is sqrt(0.4).
    logps = logits[..., 0]
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask, logits)
    expected = torch.tensor([0.4**0.5, 1.0])
    assert out.shape == (2,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_wpo_weights_is_detached() -> None:
    """The weight is detached even with grad-tracking inputs (DP Tier 1, §3.3)."""
    logps = torch.tensor([[-1.0, -2.0, 0.0]], requires_grad=True)
    logits = torch.randn(1, 3, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask, logits)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_wpo_weights_all_zero_mask_no_div0() -> None:
    """An all-zero mask row uses clamp(min=1): avg_logp = 0 → weight = 1."""
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]])
    logits = torch.randn(2, 3, 4)
    mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask, logits)
    assert torch.isfinite(out).all()
    # Row 0: numerator is 0 (all masked), denom clamped to 1 → exp(0) = 1.
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-6)


def test_wpo_weights_divisor_is_exact_token_count_under_bf16() -> None:
    """The detached WPO average must not round 257 completion tokens to 256."""

    def _uniform_weight(n_valid: int) -> float:
        logps = torch.full((n_valid,), -1.0, dtype=torch.bfloat16)
        logits = torch.zeros(n_valid, 2, dtype=torch.bfloat16)
        mask = torch.ones(n_valid, dtype=torch.bool)
        return wpo_weights(logps, mask, logits).item()

    assert _uniform_weight(256) == pytest.approx(_uniform_weight(257), rel=1e-4)


def test_wpo_weights_vmap_safe() -> None:
    """wpo_weights runs under vmap over a batch axis and stays detached."""
    logps = torch.randn(4, 5)
    logits = torch.randn(4, 5, 7)
    mask = torch.ones(4, 5)
    out = vmap(wpo_weights)(logps, mask, logits)
    assert out.shape == (4,)
    assert out.requires_grad is False
