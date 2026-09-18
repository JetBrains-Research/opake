# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DPO robust loss variant (plan §7.1, §11.3).

Covers :func:`robust_loss` (label-smoothed Rafailov) — §7.1 — with
hand-computed reference cases, label-smoothing effect tests (including the
``(1 - 2*ls)`` normalisation that distinguishes it from ``sigmoid_loss``), the
NaN-injection Tier-1 DP-purity contract (§11.3), and a 0-dim scalar
compatibility check.

Imports target the concrete implementation paths because the public façade
wiring is handled by unit γ.W.
"""

from __future__ import annotations

import math

import torch

from opake.api.alignment.dpo.loss._robust import robust_loss
from opake.api.alignment.dpo.loss._sigmoid import sigmoid_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _t(x: float) -> torch.Tensor:
    """Create a 0-dim float32 tensor."""
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------------
# robust_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_robust_delta0_beta1_ls0() -> None:
    """delta=0, ls=0 → same as sigmoid: log(2) ≈ 0.6931."""
    loss = robust_loss(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(math.log(2.0)), atol=_ATOL)


def test_robust_delta1_beta1_ls0() -> None:
    """delta=1, ls=0 → reduces to -log(sigma(1)) ≈ 0.3133 (same as sigmoid)."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = robust_loss(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_delta2_beta1_ls0() -> None:
    """delta=2, ls=0 → -log(sigma(2)) ≈ 0.1269."""
    sig2 = 1.0 / (1.0 + math.exp(-2.0))
    expected = -math.log(sig2)
    loss = robust_loss(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# robust_loss — label_smoothing effect


def test_robust_ls_reference_value_delta0() -> None:
    """delta=0, ls=0.1 → normalised blend; denominator (1-0.2)=0.8."""
    # logsigmoid(0) == logsigmoid(-0) == log(0.5)
    # numerator: -log(0.5)*0.9 + log(0.5)*0.1 = log(2)*(0.9 - 0.1) = log(2)*0.8
    # divided by 0.8 → log(2), same as ls=0 for this symmetric point.
    expected = math.log(2.0)
    loss = robust_loss(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_ls_reference_value_delta1() -> None:
    """delta=1, ls=0.1 → hand-computed: (-log(sig(1))*0.9 + log(sig(-1))*0.1) / 0.8."""
    sig1 = 1.0 / (1.0 + math.exp(-1.0))
    sig_neg1 = 1.0 / (1.0 + math.exp(1.0))
    expected = (-math.log(sig1) * 0.9 + math.log(sig_neg1) * 0.1) / 0.8
    loss = robust_loss(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_ls_changes_loss_vs_sigmoid() -> None:
    """At ls>0 robust and sigmoid differ: robust normalises by (1-2*ls)."""
    delta_c, delta_r = _t(1.5), _t(0.5)
    ls = 0.15
    loss_sig = sigmoid_loss(delta_c, delta_r, beta=1.0, label_smoothing=ls)
    loss_rob = robust_loss(delta_c, delta_r, beta=1.0, label_smoothing=ls)
    # They should differ (robust divides by 0.7, sigmoid does not normalise)
    assert not torch.allclose(loss_sig, loss_rob, atol=_ATOL)


def test_robust_ls0_equals_sigmoid_ls0() -> None:
    """At ls=0 the robust loss must match the sigmoid loss exactly."""
    c, r = _t(1.3), _t(0.2)
    loss_sig = sigmoid_loss(c, r, beta=0.1, label_smoothing=0.0)
    loss_rob = robust_loss(c, r, beta=0.1, label_smoothing=0.0)
    assert torch.allclose(loss_sig, loss_rob, atol=_ATOL)


def test_robust_ls_near_05_is_large() -> None:
    """As ls → 0.5 the denominator → 0; loss magnitude grows without bound."""
    # At ls=0.49 the denominator is 0.02; loss should be large.
    loss = robust_loss(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.49)
    assert torch.isfinite(loss)  # still finite
    loss_ls0 = robust_loss(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert loss.abs() > loss_ls0.abs()


# ---------------------------------------------------------------------------
# NaN-injection Tier-1 DP-purity contract (§11.3)
# ---------------------------------------------------------------------------


def test_robust_nan_injection_local() -> None:
    """NaN at index 1 of rejected_logratio must corrupt only output[1]."""
    chosen = torch.zeros(4)
    rejected = torch.tensor([0.5, float("nan"), 1.0, -0.5])
    out = robust_loss(chosen, rejected, beta=1.0, label_smoothing=0.1)

    assert torch.isfinite(out[0])
    assert torch.isnan(out[1])
    assert torch.isfinite(out[2])
    assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# 0-dim scalar compatibility (vmap maps over 0-dim within batch)
# ---------------------------------------------------------------------------


def test_robust_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(1.0)
    r = torch.tensor(0.0)
    out = robust_loss(c, r, beta=1.0, label_smoothing=0.0)
    assert out.ndim == 0
    assert torch.isfinite(out)
