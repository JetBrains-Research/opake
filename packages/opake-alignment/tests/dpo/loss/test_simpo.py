# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SimPO loss (length-normalized sigmoid with a target margin)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._sigmoid import sigmoid_loss
from opake.api.alignment.dpo.loss._simpo import simpo_loss


class TestSimpo:
    def test_delta_zero_no_margin(self) -> None:
        """Δ = 0, β = 1, γ = 0, ε = 0: -log σ(0) = log 2."""
        out = simpo_loss(torch.tensor(0.3), torch.tensor(0.3), beta=1.0)
        assert math.isclose(out.item(), math.log(2.0), rel_tol=1e-6)

    def test_margin_shifts_loss(self) -> None:
        """At Δ = 0 the margin γ gives -log σ(-γ) = softplus(γ)."""
        gamma = 0.7
        out = simpo_loss(torch.tensor(0.0), torch.tensor(0.0), beta=2.0, gamma=gamma)
        assert math.isclose(
            out.item(), F.softplus(torch.tensor(gamma)).item(), rel_tol=1e-6
        )

    def test_reduces_to_sigmoid_at_gamma_zero(self) -> None:
        """γ = 0 recovers the standard DPO sigmoid loss on the same log-ratios."""
        c, r = torch.tensor([0.5, -0.2, 1.1]), torch.tensor([-0.3, 0.4, 0.0])
        torch.testing.assert_close(
            simpo_loss(c, r, beta=0.1), sigmoid_loss(c, r, beta=0.1)
        )

    def test_margin_increases_loss(self) -> None:
        """A positive margin makes the objective stricter (loss not lower)."""
        c, r = torch.tensor(0.8), torch.tensor(0.2)
        base = simpo_loss(c, r, beta=1.0, gamma=0.0)
        with_margin = simpo_loss(c, r, beta=1.0, gamma=0.5)
        assert with_margin.item() > base.item()

    def test_vmap_grad_finite(self) -> None:
        torch.manual_seed(0)
        c, r = torch.randn(5), torch.randn(5)
        gc, gr = vmap(
            grad(lambda a, b: simpo_loss(a, b, beta=0.1, gamma=0.3), argnums=(0, 1))
        )(c, r)
        assert torch.isfinite(gc).all()
        assert torch.isfinite(gr).all()
