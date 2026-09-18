# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO NCA pairwise loss variant.

Shape, the analytic value at the origin, and NaN locality. Closed-form
reference values live in ``tests/test_reference_values.py`` and parity against
TRL's ``nca_pair`` in ``tests/test_trl_parity.py``.
"""

from __future__ import annotations

import math

import torch

from opake.api.alignment.dpo.loss._nca import nca_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# nca_loss
# ===========================================================================


class TestDpoNcaPair:
    """Hand-computed reference cases + NaN-injection for :func:`nca_loss`."""

    def test_zero_inputs_beta01(self) -> None:
        """c=0, r=0, β=0.1 → -logsig(0) - 0.5*logsig(-0) - 0.5*logsig(-0).

        All three logsigmoid calls evaluate to log(0.5) = -log(2), so
        the total is -(-log2) - 0.5*(-log2) - 0.5*(-log2) = 2*log(2).
        """
        c = _T(0.0)
        r = _T(0.0)
        out = nca_loss(c, r, beta=0.1)
        expected = _T(2.0 * math.log(2))
        assert out.shape == ()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(3)
        c = torch.randn(8)
        r = torch.randn(8)
        out = nca_loss(c, r, beta=0.3)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_nan_injection_locality(self) -> None:
        """Tier-1 NaN-injection contract: NaN in example 2 → only index 2 is NaN.

        Replacing one example's ``chosen_logratio`` with NaN propagates to
        that example's loss only. All other losses remain finite.
        """
        b = 5
        torch.manual_seed(4)
        c = torch.randn(b)
        r = torch.randn(b)
        c_nan = c.clone()
        c_nan[2] = float("nan")

        out = nca_loss(c_nan, r, beta=0.5)
        assert out.shape == (b,)
        # Only index 2 is NaN.
        assert torch.isnan(out[2]), "NaN example should produce NaN loss"
        for i in range(b):
            if i != 2:
                assert torch.isfinite(out[i]), f"Example {i} should be finite"
