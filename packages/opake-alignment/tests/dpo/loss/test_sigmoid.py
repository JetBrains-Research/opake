# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DPO sigmoid loss variant (plan §7.1, §11.3).

Covers :func:`sigmoid_loss` (Rafailov 2023) — §7.1 — with hand-computed
reference cases, label-smoothing effect tests, vmap-safety contract tests
(§3.4), and the NaN-injection Tier-1 DP-purity contract (§11.3): setting one
example's chosen_logratio to NaN must only corrupt that example's output, not
its neighbours.

Imports target the concrete implementation paths because the public façade
wiring is handled by unit γ.W.
"""

from __future__ import annotations

import math

import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._sigmoid import sigmoid_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _t(x: float) -> torch.Tensor:
    """Create a 0-dim float32 tensor."""
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------------
# sigmoid_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_sigmoid_delta0_beta1_ls0() -> None:
    """delta=0, beta=1, ls=0 → -log(sigma(0)) = log(2) ≈ 0.6931."""
    loss = sigmoid_loss(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, _t(math.log(2.0)), atol=_ATOL)


def test_sigmoid_delta1_beta1_ls0() -> None:
    """delta=1, beta=1, ls=0 → -log(sigma(1)) ≈ 0.3133."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = sigmoid_loss(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_delta_neg1_beta1_ls0() -> None:
    """delta=-1, beta=1, ls=0 → -log(sigma(-1)) ≈ 1.3133 (penalises reversal)."""
    expected = -math.log(1.0 / (1.0 + math.exp(1.0)))
    loss = sigmoid_loss(_t(-1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_delta2_beta05_ls0() -> None:
    """delta=2, beta=0.5 → same as delta=1, beta=1 (scaling equivalence)."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = sigmoid_loss(_t(2.0), _t(0.0), beta=0.5, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# sigmoid_loss — label_smoothing effect


def test_sigmoid_ls_increases_loss_for_positive_delta() -> None:
    """Label smoothing blends in the reversed term → higher loss for delta > 0."""
    loss_no_ls = sigmoid_loss(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    loss_ls = sigmoid_loss(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert loss_ls > loss_no_ls


def test_sigmoid_ls_reference_value() -> None:
    """delta=2, beta=1, ls=0.1 → hand-computed blend ≈ 0.3269."""
    sig2 = 1.0 / (1.0 + math.exp(-2.0))
    sig_neg2 = 1.0 / (1.0 + math.exp(2.0))
    expected = -math.log(sig2) * 0.9 - math.log(sig_neg2) * 0.1
    loss = sigmoid_loss(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_ls0_matches_ls_formula_explicitly() -> None:
    """At ls=0, the formula collapses to -logsigmoid(beta*delta)."""
    c, r = _t(1.5), _t(0.5)
    delta = 1.0
    expected = -math.log(1.0 / (1.0 + math.exp(-delta)))
    loss = sigmoid_loss(c, r, beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# ---------------------------------------------------------------------------
# vmap-safety
# ---------------------------------------------------------------------------


def test_sigmoid_vmap_finite_grads() -> None:
    """vmap(grad(sigmoid_loss)) over (4,) batch produces finite per-example grads."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return sigmoid_loss(c, r, beta=0.1).sum()

    # grad(argnums=(0,1)) returns a 2-tuple; vmap maps over the batch axis
    graded = grad(loss_fn, argnums=(0, 1))
    vmapped = vmap(graded)

    chosen = torch.linspace(-1.0, 1.0, 4)
    rejected = torch.zeros(4)
    gc, _gr = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    assert torch.all(torch.isfinite(gc))


def test_sigmoid_vmap_correct_gradient_direction() -> None:
    """Grad of sigmoid loss w.r.t. chosen_logratio is negative (loss decreases as chosen increases)."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return sigmoid_loss(c, r, beta=1.0).sum()

    # Default argnums=0: returns single tensor (grad w.r.t. chosen_logratio)
    graded = grad(loss_fn)
    vmapped = vmap(graded)

    chosen = torch.tensor([0.0, 1.0, -0.5, 2.0])
    rejected = torch.zeros(4)
    gc = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    # Increasing chosen_logratio decreases the loss → gradient is negative
    assert torch.all(gc < 0)


# ---------------------------------------------------------------------------
# NaN-injection Tier-1 DP-purity contract (§11.3)
# ---------------------------------------------------------------------------


def test_sigmoid_nan_injection_local() -> None:
    """NaN at index 2 of chosen_logratio must corrupt only output[2]."""
    chosen = torch.tensor([0.5, 1.0, float("nan"), -0.5])
    rejected = torch.zeros(4)
    out = sigmoid_loss(chosen, rejected, beta=1.0)

    assert torch.isfinite(out[0])
    assert torch.isfinite(out[1])
    assert torch.isnan(out[2])
    assert torch.isfinite(out[3])


def test_sigmoid_nan_injection_rejected() -> None:
    """NaN injected into rejected_logratio at index 0 corrupts only output[0]."""
    chosen = torch.ones(4)
    rejected = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    out = sigmoid_loss(chosen, rejected, beta=1.0)

    assert torch.isnan(out[0])
    assert torch.isfinite(out[1])
    assert torch.isfinite(out[2])
    assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# 0-dim scalar compatibility (vmap maps over 0-dim within batch)
# ---------------------------------------------------------------------------


def test_sigmoid_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(1.0)
    r = torch.tensor(0.0)
    out = sigmoid_loss(c, r, beta=1.0)
    assert out.ndim == 0
    assert torch.isfinite(out)
