# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ORPO odds-ratio loss (reference-free, on length-normalized logp)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._orpo import odds_ratio_loss


def _ref(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Naive reference: -log σ(log_odds_c - log_odds_r), stable for moderate logp."""
    log_odds = lambda x: x - torch.log(1.0 - torch.exp(x))  # noqa: E731
    return -F.logsigmoid(log_odds(c) - log_odds(r))


class TestOddsRatio:
    def test_equal_logp_is_log2(self) -> None:
        """Equal odds ⇒ Δ log-odds = 0 ⇒ -log σ(0) = log 2."""
        out = odds_ratio_loss(torch.tensor(-0.5), torch.tensor(-0.5))
        assert math.isclose(out.item(), math.log(2.0), rel_tol=1e-6)

    def test_matches_naive_reference(self) -> None:
        """Stable log1mexp matches the naive formula on moderate log-probs."""
        c = torch.tensor([-0.1, -0.7, -2.0, -1.3])
        r = torch.tensor([-0.9, -0.2, -1.1, -3.0])
        torch.testing.assert_close(odds_ratio_loss(c, r), _ref(c, r))

    def test_higher_chosen_lowers_loss(self) -> None:
        """Raising chosen log-prob (higher odds) decreases the loss."""
        r = torch.tensor(-1.0)
        hi = odds_ratio_loss(torch.tensor(-0.2), r)
        lo = odds_ratio_loss(torch.tensor(-2.0), r)
        assert hi.item() < lo.item()

    def test_log1mexp_numerically_stable(self) -> None:
        """No NaN/Inf for log-probs near 0 (p→1) or very negative (p→0)."""
        c = torch.tensor([-1e-7, -50.0, -0.6931, -10.0])
        r = torch.tensor([-3.0, -1e-6, -20.0, -0.3])
        out = odds_ratio_loss(c, r)
        assert torch.isfinite(out).all()

    def test_vmap_grad_finite(self) -> None:
        torch.manual_seed(0)
        c = -torch.rand(6) - 0.05  # strictly negative log-probs
        r = -torch.rand(6) - 0.05
        gc, gr = vmap(grad(odds_ratio_loss, argnums=(0, 1)))(c, r)
        assert torch.isfinite(gc).all()
        assert torch.isfinite(gr).all()
