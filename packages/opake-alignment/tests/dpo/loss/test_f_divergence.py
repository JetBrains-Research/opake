# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the DPO f-divergence remaps (plan §7.1, §3.4).

Covers the two public helpers in ``loss/dpo/_f_divergence.py``:

- :func:`f_divergence_remap` — the per-side log-ratio remap ``g`` for each of
  the four supported f-divergences (``reverse_kl``, ``forward_kl``,
  ``js_divergence``, ``alpha_divergence``).
- :func:`f_divergence_logits` — the combiner
  ``g(chosen) - g(rejected)``.

Each divergence has a hand-computed reference case on tiny inputs; the combiner
is checked against an independent ``remap(chosen) - remap(rejected)`` on a
non-trivial example. Tests also cover the ``alpha == 1`` reverse-KL limit,
``torch.func.vmap(torch.func.grad(...))`` safety, dtype-aware exponent clamps,
and parity with TRL's public remaps and final logits.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._f_divergence import (
    f_divergence_logits,
    f_divergence_remap,
)

# ---------------------------------------------------------------------------
# f_divergence_remap — per-divergence hand-computed cases
# ---------------------------------------------------------------------------


def test_remap_reverse_kl_is_identity() -> None:
    """reverse_kl returns the log-ratio unchanged (the DPO default)."""
    logratio = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.5])
    out = f_divergence_remap(logratio, f_divergence_type="reverse_kl")
    assert torch.equal(out, logratio)


def test_remap_reverse_kl_is_default() -> None:
    """The default f_divergence_type is reverse_kl (identity)."""
    logratio = torch.tensor([0.3, -1.2, 2.0])
    assert torch.equal(f_divergence_remap(logratio), logratio)


def test_remap_forward_kl_at_zero() -> None:
    """forward_kl at logratio=0 → 1 - exp(-0) = 0."""
    out = f_divergence_remap(torch.tensor(0.0), f_divergence_type="forward_kl")
    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_remap_forward_kl_hand_vector() -> None:
    """forward_kl matches 1 - exp(-x) elementwise on a small vector."""
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    out = f_divergence_remap(x, f_divergence_type="forward_kl")
    expected = 1 - torch.exp(-x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_remap_js_divergence_at_zero() -> None:
    """js_divergence at logratio=0 → log(2) + logsigmoid(0) = 0."""
    out = f_divergence_remap(torch.tensor(0.0), f_divergence_type="js_divergence")
    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_remap_js_divergence_hand_vector() -> None:
    """js_divergence matches log(2) + F.logsigmoid elementwise."""
    x = torch.tensor([-3.0, -0.5, 0.0, 1.0, 4.0])
    out = f_divergence_remap(x, f_divergence_type="js_divergence")
    assert torch.allclose(out, math.log(2.0) + F.logsigmoid(x), atol=1e-6)


def test_remap_alpha_divergence_alpha2_at_zero() -> None:
    """alpha=2 at logratio=0 → (exp((2-1)*0) - 1)/(2-1) = 0."""
    out = f_divergence_remap(
        torch.tensor(0.0), f_divergence_type="alpha_divergence", alpha=2.0
    )
    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_remap_alpha_divergence_hand_vector() -> None:
    """alpha_divergence matches (exp((alpha-1)*x)-1)/(alpha-1).

    Uses alpha=0.5 to exercise a negative ``(alpha - 1)`` denominator.
    """
    alpha = 0.5
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    out = f_divergence_remap(x, f_divergence_type="alpha_divergence", alpha=alpha)
    expected = (torch.exp((alpha - 1) * x) - 1) / (alpha - 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_remap_alpha_divergence_alpha_one_is_identity() -> None:
    """alpha == 1 is the reverse_kl limit → returns identity (not an error)."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    out = f_divergence_remap(
        x,
        f_divergence_type="alpha_divergence",
        alpha=1.0,
    )
    assert torch.allclose(out, x, atol=1e-6)


def test_remap_alpha_divergence_near_one_is_identity() -> None:
    """Alpha within TRL's 1e-6 tolerance uses the reverse-KL limit."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    out = f_divergence_remap(
        x,
        f_divergence_type="alpha_divergence",
        alpha=1.0 + 5e-7,
    )
    assert torch.equal(out, x)


def test_remap_unknown_type_raises() -> None:
    """An unrecognised divergence name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown f_divergence_type"):
        f_divergence_remap(torch.tensor(0.0), f_divergence_type="totally_bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# f_divergence_logits — combiner
# ---------------------------------------------------------------------------


def test_logits_reverse_kl_is_plain_difference() -> None:
    """Under reverse_kl the combiner reduces to chosen - rejected."""
    chosen = torch.tensor([1.5, -0.5, 2.0])
    rejected = torch.tensor([0.5, 0.5, -1.0])
    out = f_divergence_logits(chosen, rejected, f_divergence_type="reverse_kl")
    assert torch.allclose(out, chosen - rejected, atol=1e-6)


def test_logits_equals_remap_difference_nontrivial() -> None:
    """f_divergence_logits = remap(chosen) - remap(rejected) (alpha_divergence)."""
    chosen = torch.tensor([0.7, -1.3, 2.5, 0.0])
    rejected = torch.tensor([-0.4, 0.9, 1.1, -2.0])
    out = f_divergence_logits(
        chosen, rejected, f_divergence_type="alpha_divergence", alpha=2.0
    )
    expected = f_divergence_remap(
        chosen, f_divergence_type="alpha_divergence", alpha=2.0
    ) - f_divergence_remap(rejected, f_divergence_type="alpha_divergence", alpha=2.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_logits_js_hand_at_zero() -> None:
    """js combiner at chosen=rejected=0 → logsigmoid(0) - logsigmoid(0) = 0."""
    out = f_divergence_logits(
        torch.tensor(0.0), torch.tensor(0.0), f_divergence_type="js_divergence"
    )
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_logits_alpha_one_is_plain_difference() -> None:
    """The combiner at alpha==1 falls through to reverse_kl (identity)."""
    chosen = torch.tensor([0.5, -1.5])
    rejected = torch.tensor([-0.3, 0.8])
    out = f_divergence_logits(
        chosen,
        rejected,
        f_divergence_type="alpha_divergence",
        alpha=1.0,
    )
    assert torch.allclose(out, chosen - rejected, atol=1e-6)


# ---------------------------------------------------------------------------
# vmap(grad(...)) contract — §3.4 vmap-safety
# ---------------------------------------------------------------------------


def test_logits_vmap_grad_finite_alpha_divergence() -> None:
    """vmap(grad(f_divergence_logits-sum)) over (4,) yields finite grads.

    Exercises the alpha_divergence ``_cap_exp`` path under composed
    ``torch.func.vmap`` / ``torch.func.grad`` (§3.4 vmap-safety contract).
    """
    chosen = torch.tensor([0.3, -1.1, 2.0, 0.7])
    rejected = torch.tensor([-0.2, 0.4, 1.0, -1.5])

    def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return f_divergence_logits(
            c, r, f_divergence_type="alpha_divergence", alpha=2.0
        ).sum()

    g_chosen = vmap(grad(per_example))(chosen, rejected)
    assert g_chosen.shape == (4,)
    assert torch.isfinite(g_chosen).all()


@pytest.mark.parametrize(
    "f_divergence_type",
    ["reverse_kl", "forward_kl", "js_divergence"],
)
def test_remap_vmap_grad_finite_all_types(f_divergence_type: str) -> None:
    """vmap(grad(remap-sum)) is finite for every non-alpha divergence."""
    x = torch.tensor([-2.0, -0.3, 0.0, 1.5])

    def per_example(v: torch.Tensor) -> torch.Tensor:
        return f_divergence_remap(v, f_divergence_type=f_divergence_type).sum()  # type: ignore[arg-type]

    grads = vmap(grad(per_example))(x)
    assert grads.shape == (4,)
    assert torch.isfinite(grads).all()


# ---------------------------------------------------------------------------
# Low-precision overflow clamp — _cap_exp
# ---------------------------------------------------------------------------


def test_forward_kl_fp16_large_negative_no_overflow() -> None:
    """forward_kl computes 1-exp(-x); a large negative x must stay finite in fp16.

    fp16 max finite is ~65,504. exp(20) ~ 4.85e8 overflows fp16.
    The dtype-aware cap of 11.0 keeps fp16 finite.
    """
    x = torch.tensor([-1000.0, -50.0, 0.0], dtype=torch.float16)
    out = f_divergence_remap(x, f_divergence_type="forward_kl")
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_forward_kl_bf16_large_negative_no_overflow() -> None:
    """forward_kl computes 1-exp(-x); a large negative x must stay finite in bf16.

    bf16 saturates to ``inf`` around exp(89); without the ``_cap_exp`` clamp on
    the exponent ``-x``, a large negative log-ratio would blow up. With the
    clamp the output stays finite.
    """
    x = torch.tensor([-1000.0, -50.0, 0.0], dtype=torch.bfloat16)
    out = f_divergence_remap(x, f_divergence_type="forward_kl")
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_alpha_divergence_bf16_large_logratio_no_overflow() -> None:
    """alpha_divergence exp((alpha-1)*x) must not overflow bf16 for large x."""
    x = torch.tensor([1000.0, 100.0, 0.0], dtype=torch.bfloat16)
    out = f_divergence_remap(x, f_divergence_type="alpha_divergence", alpha=2.0)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_logits_bf16_large_logratio_no_overflow() -> None:
    """The combiner stays finite in bf16 for large log-ratios on both sides."""
    chosen = torch.tensor([900.0, 0.0], dtype=torch.bfloat16)
    rejected = torch.tensor([-900.0, 5.0], dtype=torch.bfloat16)
    out = f_divergence_logits(
        chosen, rejected, f_divergence_type="alpha_divergence", alpha=3.0
    )
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# TRL parity for public remaps and final loss logits.
# ---------------------------------------------------------------------------


def test_trl_parity_forward_kl_logits() -> None:
    """Forward-KL scores and their chosen-rejected delta match TRL."""
    chosen = torch.tensor([0.5, -1.0, 2.0])
    rejected = torch.tensor([-0.3, 0.8, 0.0])

    # TRL version: g_trl(x) = 1 - exp(-x)
    trl_chosen = 1 - torch.exp(-chosen)
    trl_rejected = 1 - torch.exp(-rejected)
    trl_delta = trl_chosen - trl_rejected

    assert torch.allclose(
        f_divergence_remap(chosen, f_divergence_type="forward_kl"),
        trl_chosen,
        atol=1e-6,
    )
    out = f_divergence_logits(chosen, rejected, f_divergence_type="forward_kl")
    assert torch.allclose(out, trl_delta, atol=1e-6)


def test_trl_parity_js_divergence_logits() -> None:
    """Jensen-Shannon scores and their chosen-rejected delta match TRL."""
    chosen = torch.tensor([0.7, -0.5, 3.0])
    rejected = torch.tensor([-0.2, 1.1, 0.0])

    trl_chosen = math.log(2.0) + F.logsigmoid(chosen)
    trl_rejected = math.log(2.0) + F.logsigmoid(rejected)
    trl_delta = trl_chosen - trl_rejected

    assert torch.allclose(
        f_divergence_remap(chosen, f_divergence_type="js_divergence"),
        trl_chosen,
        atol=1e-6,
    )
    out = f_divergence_logits(chosen, rejected, f_divergence_type="js_divergence")
    assert torch.allclose(out, trl_delta, atol=1e-6)


def test_trl_parity_alpha_divergence_logits() -> None:
    """Alpha-divergence scores and their chosen-rejected delta match TRL."""
    alpha = 2.5
    chosen = torch.tensor([0.3, -0.8, 1.5])
    rejected = torch.tensor([-0.4, 0.6, -1.0])

    trl_chosen = (torch.exp((alpha - 1) * chosen) - 1) / (alpha - 1)
    trl_rejected = (torch.exp((alpha - 1) * rejected) - 1) / (alpha - 1)
    trl_delta = trl_chosen - trl_rejected

    assert torch.allclose(
        f_divergence_remap(chosen, f_divergence_type="alpha_divergence", alpha=alpha),
        trl_chosen,
        atol=1e-6,
    )
    out = f_divergence_logits(
        chosen, rejected, f_divergence_type="alpha_divergence", alpha=alpha
    )
    assert torch.allclose(out, trl_delta, atol=1e-6)


def test_trl_parity_alpha_one_logits() -> None:
    """TRL falls through alpha==1 to reverse_kl; opake now does the same."""
    chosen = torch.tensor([0.5, -1.0])
    rejected = torch.tensor([-0.3, 0.2])

    # TRL: alpha==1 → reverse_kl → identity → chosen - rejected
    trl_delta = chosen - rejected

    out = f_divergence_logits(
        chosen, rejected, f_divergence_type="alpha_divergence", alpha=1.0
    )
    assert torch.allclose(out, trl_delta, atol=1e-6)
