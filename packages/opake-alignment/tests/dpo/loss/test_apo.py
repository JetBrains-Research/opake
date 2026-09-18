# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO APO loss variants (arXiv:2408.06266).

Shape, the degenerate points where both terms sum to 1, and vmap-safety for
:func:`apo_zero_loss`. Closed-form reference values live in
``tests/test_reference_values.py`` and parity against TRL's ``apo_zero`` /
``apo_down`` in ``tests/test_trl_parity.py``.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._apo import apo_down_loss, apo_zero_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# apo_zero_loss
# ===========================================================================


class TestDpoApoZero:
    """Hand-computed reference cases for :func:`apo_zero_loss`."""

    def test_symmetric_zero_inputs(self) -> None:
        """c=0, r=0 → (1-sig(0)) + sig(0) = 0.5 + 0.5 = 1.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = apo_zero_loss(c, r, beta=0.1)
        assert out.shape == ()
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_c2_r2_beta1_is_one(self) -> None:
        """c=2, r=2, β=1 → (1-sig(2)) + sig(2) = 1.0 always."""
        c = _T(2.0)
        r = _T(2.0)
        out = apo_zero_loss(c, r, beta=1.0)
        # For any x: (1-sig(x)) + sig(x) = 1.
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_batched_shape_preserved(self) -> None:
        """(B,) inputs → (B,) output."""
        c = torch.randn(8)
        r = torch.randn(8)
        out = apo_zero_loss(c, r, beta=0.1)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(apo_zero_loss-sum)) over (4,) batch yields finite grads."""
        b = 4
        torch.manual_seed(42)
        c = torch.randn(b, requires_grad=False)
        r = torch.randn(b, requires_grad=False)

        def per_example(ci: torch.Tensor, ri: torch.Tensor) -> torch.Tensor:
            return apo_zero_loss(ci, ri, beta=0.1).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(c, r)
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()


# ===========================================================================
# apo_down_loss
# ===========================================================================


class TestDpoApoDown:
    """Hand-computed reference cases for :func:`apo_down_loss`."""

    def test_symmetric_zero_inputs(self) -> None:
        """c=0, r=0 → sig(0) + (1-sig(0)) = 0.5 + 0.5 = 1.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = apo_down_loss(c, r, beta=0.1)
        assert out.shape == ()
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_c1_r0_beta05_equals_one(self) -> None:
        """c=1, r=0, β=0.5 → sig(0.5) + (1-sig(0.5)) = 1.0."""
        c = _T(1.0)
        r = _T(0.0)
        out = apo_down_loss(c, r, beta=0.5)
        # b*c = b*(c-r) so lc + lr = sig(x) + (1-sig(x)) = 1
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_batched_shape_preserved(self) -> None:
        """(B,) inputs → (B,) output."""
        c = torch.randn(6)
        r = torch.randn(6)
        out = apo_down_loss(c, r, beta=0.2)
        assert out.shape == (6,)
        assert torch.isfinite(out).all()

    def test_loss_nonnegative(self) -> None:
        """Both sigmoid terms are in [0,1] so their sum is in [0,2]."""
        torch.manual_seed(0)
        c = torch.randn(32)
        r = torch.randn(32)
        out = apo_down_loss(c, r, beta=0.1)
        assert (out >= 0).all()
        assert (out <= 2.0 + 1e-6).all()
