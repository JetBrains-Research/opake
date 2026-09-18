"""Tests for the DPO IPO squared loss variant.

Covers work-unit γ.3 of the opake-alignment plan (§10, §11.2, §11.3).

Hand-computed reference cases plus vmap-safety contract tests (§11.2).

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._ipo import ipo_loss

# ---------------------------------------------------------------------------
# ipo_loss
# ---------------------------------------------------------------------------


class TestDpoIpo:
    """Hand-computed reference cases for the IPO squared loss."""

    def test_ipo_delta_zero_beta_one(self) -> None:
        """Δ = 0, β = 1: (0 − 1/2)² = 0.25."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = ipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.25), atol=1e-6)

    def test_ipo_delta_one_half_beta_one(self) -> None:
        """Δ = 0.5, β = 1: (0.5 − 0.5)² = 0.0."""
        chosen = torch.tensor(0.5)
        rejected = torch.tensor(0.0)
        out = ipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)

    def test_ipo_delta_one_beta_two(self) -> None:
        """Δ = 1, β = 2: (1 − 1/4)² = 0.5625."""
        chosen = torch.tensor(1.5)
        rejected = torch.tensor(0.5)
        out = ipo_loss(chosen, rejected, beta=2.0)
        expected = (1.0 - 1.0 / (2.0 * 2.0)) ** 2
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_ipo_negative_delta(self) -> None:
        """Δ = -1, β = 1: (-1 − 0.5)² = 2.25."""
        chosen = torch.tensor(-1.0)
        rejected = torch.tensor(0.0)
        out = ipo_loss(chosen, rejected, beta=1.0)
        expected = (-1.0 - 0.5) ** 2  # 2.25
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_ipo_batched(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.tensor([0.0, 0.5, 1.0])
        rejected = torch.tensor([0.0, 0.0, 0.0])
        out = ipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == (3,)
        # check against per-example results
        for i in range(3):
            expected = ipo_loss(chosen[i], rejected[i], beta=1.0)
            assert torch.allclose(out[i], expected, atol=1e-6)

    def test_ipo_vmap_grad_finite(self) -> None:
        """vmap(grad(ipo_loss)) over a (4,) batch yields finite gradients."""
        torch.manual_seed(0)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return ipo_loss(c, r, beta=0.1)

        grads_c = vmap(grad(per_example))(chosen, rejected)
        assert grads_c.shape == (4,)
        assert torch.isfinite(grads_c).all()


# ---------------------------------------------------------------------------
# Additional vmap(grad) safety over a (4,) synthetic batch
# ---------------------------------------------------------------------------


class TestVmapGradComposed:
    """Additional vmap(grad) safety tests over a (4,) synthetic batch."""

    def test_ipo_vmap_grad_finite_larger_batch(self) -> None:
        """vmap(grad(ipo_loss)) over a (4,) batch: grads w.r.t. chosen finite."""
        torch.manual_seed(7)
        chosen = torch.randn(4, requires_grad=False)
        rejected = torch.randn(4, requires_grad=False)

        def fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return ipo_loss(c, r, beta=0.5)

        grads = vmap(grad(fn))(chosen, rejected)
        assert grads.shape == (4,)
        assert torch.isfinite(grads).all()
