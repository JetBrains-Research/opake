# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DPO hinge loss variant (plan §7.1, §11.3).

Covers :func:`hinge_loss` (Liu 2023) — §7.1 — with hand-computed reference
cases, a relu-subgradient test, a vmap-safety contract test (§3.4), and the
NaN-injection Tier-1 DP-purity contract (§11.3): setting one example's
chosen_logratio to NaN must only corrupt that example's output, not its
neighbours.

Imports target the concrete implementation paths because the public façade
wiring is handled by unit γ.W.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._hinge import hinge_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _t(x: float) -> torch.Tensor:
    """Create a 0-dim float32 tensor."""
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------------
# hinge_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_hinge_delta0_beta1() -> None:
    """delta=0, beta=1 → relu(1 - 0) = 1.0."""
    loss = hinge_loss(_t(0.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(1.0), atol=_ATOL)


def test_hinge_delta2_beta1_zero() -> None:
    """delta=2, beta=1 → relu(1 - 2) = 0.0 (margin satisfied)."""
    loss = hinge_loss(_t(2.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(0.0), atol=_ATOL)


def test_hinge_delta05_beta1() -> None:
    """delta=0.5, beta=1 → relu(1 - 0.5) = 0.5."""
    loss = hinge_loss(_t(0.5), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(0.5), atol=_ATOL)


def test_hinge_delta2_beta03() -> None:
    """delta=2, beta=0.3 → relu(1 - 0.6) = 0.4."""
    loss = hinge_loss(_t(2.0), _t(0.0), beta=0.3)
    assert torch.allclose(loss, _t(0.4), atol=_ATOL)


def test_hinge_delta_neg1_beta1() -> None:
    """delta=-1, beta=1 → relu(1 - (-1)) = 2.0 (chosen < rejected)."""
    loss = hinge_loss(_t(-1.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(2.0), atol=_ATOL)


def test_hinge_uses_relu_not_python_branch() -> None:
    """Differentiable at 0: the subgradient is 0 exactly at the hinge boundary."""
    # At the hinge boundary (beta*delta == 1), the relu subgradient selects 0.
    # torch.relu is differentiable a.e.; autograd picks subgrad=0 at 0.
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.zeros(1)
    loss = hinge_loss(x, y, beta=1.0)
    loss.backward()
    # At delta=1 the relu input is 0; its subgradient = 0 → grad on x is 0
    assert x.grad is not None
    assert x.grad.item() == 0.0


# ---------------------------------------------------------------------------
# vmap-safety
# ---------------------------------------------------------------------------


def test_hinge_vmap_finite_grads() -> None:
    """vmap(grad(hinge_loss)) over (4,) batch produces finite per-example grads."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return hinge_loss(c, r, beta=0.1).sum()

    # grad(argnums=(0,1)) returns a 2-tuple; vmap maps over the batch axis
    graded = grad(loss_fn, argnums=(0, 1))
    vmapped = vmap(graded)

    chosen = torch.linspace(-0.5, 1.5, 4)
    rejected = torch.zeros(4)
    gc, _gr = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    assert torch.all(torch.isfinite(gc))


# ---------------------------------------------------------------------------
# NaN-injection Tier-1 DP-purity contract (§11.3)
# ---------------------------------------------------------------------------


def test_hinge_nan_injection_local() -> None:
    """NaN at index 2 of chosen_logratio must corrupt only output[2]."""
    chosen = torch.tensor([0.5, 1.0, float("nan"), -0.5])
    rejected = torch.zeros(4)
    out = hinge_loss(chosen, rejected, beta=1.0)

    assert torch.isfinite(out[0])
    assert torch.isfinite(out[1])
    assert torch.isnan(out[2])
    assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# 0-dim scalar compatibility (vmap maps over 0-dim within batch)
# ---------------------------------------------------------------------------


def test_hinge_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(0.5)
    r = torch.tensor(0.0)
    out = hinge_loss(c, r, beta=1.0)
    assert out.ndim == 0
    assert torch.allclose(out, torch.tensor(0.5), atol=_ATOL)
