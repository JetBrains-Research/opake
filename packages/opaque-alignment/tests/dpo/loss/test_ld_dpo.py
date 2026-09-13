# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the LD-DPO logp-split helper (plan §7.1, §3.4).

Covers :func:`ld_dpo_split` — the LD-DPO length-desensitised logp split
(arXiv:2409.06411). Full-prefix coverage reducing to the plain masked-sum, the
``alpha=0`` prefix-only case, and a ``torch.func.vmap(torch.func.grad)``
finite-gradient contract test (§3.4) with a tensor ``shared_prefix_len``.

Imports target the concrete impl paths because the public façade is wired by a
later unit (γ.W).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.alignment.dpo.loss._ld_dpo import ld_dpo_split

# ---------------------------------------------------------------------------
# ld_dpo_split — length-desensitised logp split (arXiv:2409.06411)
# ---------------------------------------------------------------------------


def test_ld_dpo_full_prefix_equals_masked_sum() -> None:
    """shared_prefix_len covering the whole sequence → plain masked-sum."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=4, alpha=0.3)
    plain = (logps * mask).sum(dim=-1)
    assert torch.allclose(out, plain, atol=1e-6)


def test_ld_dpo_alpha_one_equals_masked_sum() -> None:
    """alpha=1 recovers the plain masked-sum regardless of prefix length."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=1, alpha=1.0)
    plain = (logps * mask).sum(dim=-1)
    assert torch.allclose(out, plain, atol=1e-6)


def test_ld_dpo_alpha_zero_prefix_only() -> None:
    """alpha=0 → only the prefix tokens contribute."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=2, alpha=0.0)
    # Only positions 0 and 1 count: -1 + -2 = -3.
    assert torch.allclose(out, torch.tensor([-3.0]), atol=1e-6)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_ld_dpo_rejects_alpha_outside_paper_domain(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        ld_dpo_split(
            torch.ones(4),
            torch.ones(4),
            shared_prefix_len=2,
            alpha=alpha,
        )


def test_ld_dpo_tail_weighting() -> None:
    """Completion tokens after the shared prefix are weighted by alpha."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=2, alpha=0.5)
    # prefix: -1 + -2 = -3 ; tail: 0.5 * (-3 + -4) = -3.5 ; total -6.5.
    assert torch.allclose(out, torch.tensor([-6.5]), atol=1e-6)


def test_ld_dpo_positions_are_completion_relative() -> None:
    """Prompt offset does not consume the shared completion-token count."""
    logps = torch.tensor([[-10.0, -20.0, -1.0, -2.0, -3.0]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=2, alpha=0.5)
    # First two completion tokens are shared; only the third is damped.
    assert torch.allclose(out, torch.tensor([-4.5]), atol=1e-6)


def test_ld_dpo_per_example_tensor_prefix() -> None:
    """A per-example tensor shared_prefix_len broadcasts against the pos axis."""
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
    mask = torch.ones(2, 3)
    prefix = torch.tensor([[1], [3]])  # (B, 1) broadcasts against (T,)
    out = ld_dpo_split(logps, mask, shared_prefix_len=prefix, alpha=0.0)
    # Row 0: prefix len 1 → only pos 0 → -1.
    # Row 1: prefix len 3 → whole seq → -6.
    assert torch.allclose(out, torch.tensor([-1.0, -6.0]), atol=1e-6)


def test_ld_dpo_vmap_grad_finite_tensor_prefix() -> None:
    """vmap(grad(ld_dpo_split-sum)) over (4, T) → finite (§3.4 vmap-safety)."""
    seq_len = 5
    lp = torch.randn(4, seq_len)
    mask = torch.ones(4, seq_len)

    def per_example(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return ld_dpo_split(x, m, shared_prefix_len=2, alpha=0.5).sum()

    g = vmap(grad(per_example))(lp, mask)
    assert g.shape == (4, seq_len)
    assert torch.isfinite(g).all()
    # Per-token grad equals the position weight: 1.0 for pos<2, else 0.5.
    expected_row = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5])
    assert torch.allclose(g, expected_row.expand(4, seq_len), atol=1e-6)


def test_ld_dpo_vmap_grad_per_example_tensor_prefix() -> None:
    """vmap(grad) with a vmapped per-example tensor prefix stays finite (§3.4).

    Exercises the broadcast of a per-example ``shared_prefix_len`` tensor
    against the position axis *inside* the vmap region (adversarial check that
    the comparison/where path is genuinely vmap-safe with a tensor prefix).
    """
    seq_len = 4
    lp = torch.randn(4, seq_len)
    mask = torch.ones(4, seq_len)
    prefix = torch.tensor([[1], [2], [3], [4]])  # (B, 1)

    def per_example(x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return ld_dpo_split(x, m, shared_prefix_len=p, alpha=0.5).sum()

    g = vmap(grad(per_example))(lp, mask, prefix)
    assert g.shape == (4, seq_len)
    assert torch.isfinite(g).all()
