# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO SPPO hard-label loss variant (work-unit γ.2).

Covers :func:`sppo_loss` (SPPO hard-label loss):
- ≥3 hand-computed reference cases (small, analytically tractable inputs).
- vmap-safety (``torch.func.vmap(torch.func.grad(...))``).
- Imports target the concrete implementation paths (public façade not wired
  for this work-unit; façade wiring is γ.W).
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._sppo import sppo_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# sppo_loss
# ===========================================================================


class TestDpoSppoHard:
    """Hand-computed reference cases for :func:`sppo_loss`."""

    def test_zero_inputs_beta05(self) -> None:
        """c=0, r=0, β=0.5 → target = 1.0; (0-1)^2 + (0+1)^2 = 2.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = sppo_loss(c, r, beta=0.5)
        assert out.shape == ()
        assert torch.allclose(out, _T(2.0), atol=1e-6)

    def test_at_nash_equilibrium_is_zero(self) -> None:
        """c = 0.5/β, r = -0.5/β → both squared terms are 0 → loss = 0."""
        beta = 0.1
        target = 0.5 / beta  # = 5.0 exactly (float, not int)
        c = _T(target)
        r = _T(-target)
        out = sppo_loss(c, r, beta=beta)
        assert torch.allclose(out, _T(0.0), atol=1e-6)

    def test_asymmetric_inputs_beta01(self) -> None:
        """c=1, r=-1, β=0.1 → target=5; (1-5)^2 + (-1+5)^2 = 16+16 = 32."""
        c = _T(1.0)
        r = _T(-1.0)
        out = sppo_loss(c, r, beta=0.1)
        assert torch.allclose(out, _T(32.0), atol=1e-6)

    def test_float_division_not_integer_division(self) -> None:
        """0.5/beta must be floating-point: for beta=3, target=1/6, not 0."""
        beta = 3.0
        target = 0.5 / beta
        # If integer division were used: 0 // 3 = 0 → loss = c^2 + r^2 for c=0, r=0.
        # With float division: target ≈ 0.1667.
        assert abs(target - 1.0 / 6.0) < 1e-9
        c = _T(target)  # exactly at target
        r = _T(-target)
        out = sppo_loss(c, r, beta=beta)
        assert torch.allclose(out, _T(0.0), atol=1e-6)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(6)
        c = torch.randn(8)
        r = torch.randn(8)
        out = sppo_loss(c, r, beta=0.5)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(sppo_loss-sum)) over (4,) batch yields finite grads."""
        b = 4
        torch.manual_seed(7)
        c = torch.randn(b)
        r = torch.randn(b)

        def per_example(ci: torch.Tensor, ri: torch.Tensor) -> torch.Tensor:
            return sppo_loss(ci, ri, beta=0.1).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(c, r)
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()

    def test_loss_nonnegative(self) -> None:
        """Squared terms are always ≥ 0."""
        torch.manual_seed(8)
        c = torch.randn(32)
        r = torch.randn(32)
        out = sppo_loss(c, r, beta=0.2)
        assert (out >= 0).all()
