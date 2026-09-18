# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO BCO pairwise loss variant.

Shape, the ``delta`` baseline's effect, and the analytic value at the origin.
Closed-form reference values live in ``tests/test_reference_values.py`` and
parity against TRL's ``bco_pair`` in ``tests/test_trl_parity.py``.
"""

from __future__ import annotations

import math

import torch

from opake.api.alignment.dpo.loss._bco import bco_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# bco_loss
# ===========================================================================


class TestDpoBcoPair:
    """Hand-computed reference cases + delta-shift for :func:`bco_loss`."""

    def test_zero_inputs_delta0(self) -> None:
        """c=0, r=0, β=0.1, δ=0 → -logsig(0) - logsig(-0) = 2*log(2)."""
        c = _T(0.0)
        r = _T(0.0)
        out = bco_loss(c, r, beta=0.1, delta=0.0)
        expected = _T(2.0 * math.log(2.0))
        assert out.shape == ()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_delta_zero_vs_nonzero_are_different(self) -> None:
        """δ=0 and δ=0.3 must give strictly different losses for non-trivial input."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        out_d0 = bco_loss(c, r, beta=beta, delta=0.0)
        out_d3 = bco_loss(c, r, beta=beta, delta=0.3)
        assert not torch.allclose(out_d0, out_d3, atol=1e-4)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(5)
        c = torch.randn(8)
        r = torch.randn(8)
        out = bco_loss(c, r, beta=0.2, delta=0.1)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()
