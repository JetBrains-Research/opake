# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO EXO pairwise loss variant.

Shape, the label-smoothing floor, and non-negativity (EXO is a KL divergence).
Closed-form reference values live in ``tests/test_reference_values.py`` and
parity against TRL's ``exo_pair`` in ``tests/test_trl_parity.py``.
"""

from __future__ import annotations

import torch

from opake.api.alignment.dpo.loss._exo import exo_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# exo_loss
# ===========================================================================


class TestDpoExoPair:
    """Hand-computed reference cases for :func:`exo_loss`."""

    def test_label_smoothing_zero_clamped_to_1e3(self) -> None:
        """label_smoothing=0 is silently clamped to 1e-3 (same as default)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_default = exo_loss(c, r, beta=1.0, label_smoothing=1e-3)
        out_zero_ls = exo_loss(c, r, beta=1.0, label_smoothing=0.0)
        # Both should produce the same output: ls=0 is clamped to 1e-3.
        assert torch.allclose(out_default, out_zero_ls, atol=1e-7)

    def test_negative_label_smoothing_clamped(self) -> None:
        """label_smoothing<0 is also clamped to 1e-3 (no log-of-negative)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_neg = exo_loss(c, r, beta=1.0, label_smoothing=-0.5)
        out_ref = exo_loss(c, r, beta=1.0, label_smoothing=1e-3)
        assert torch.allclose(out_neg, out_ref, atol=1e-7)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(1)
        c = torch.randn(8)
        r = torch.randn(8)
        out = exo_loss(c, r, beta=0.2)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_loss_positive(self) -> None:
        """EXO loss is a KL divergence: always ≥ 0."""
        torch.manual_seed(2)
        c = torch.randn(32)
        r = torch.randn(32)
        out = exo_loss(c, r, beta=0.1)
        assert (out >= -1e-5).all(), "EXO loss should be non-negative"
