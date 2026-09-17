# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + contract tests for the SFT DFT losses ``dft_loss`` / ``fused_dft_loss``.

Covers the DFT (Dynamic Fine-Tuning) loss and its fused-linear twin (plan §7.3,
§11.2, §11.3; work-units ε.1 and the fused-linear ε.* unit).

DFT arXiv:2508.05629 — per-token loss = -detach(softmax_prob) * logp.

Verified properties for :func:`dft_loss`
-----------------------------------------
* Hand-computed case with detached-prob weighting; gradient detach invariant
  (§11.3 corollary); DP-corrected divisor assertion.
* **DP-corrected divisor** (§3.3, §8.2): 2-example batch with different
  completion lengths → per-example divisor differs from a batch-level divisor.
* **NaN-injection** (Tier 1, §11.3): NaN in one example's logits produces NaN
  only for that example; neighbours are finite.
* **Vmap-safety** (§3.4, §11.2): ``vmap(grad(dft_loss.sum))`` over a
  ``(4, T, V)`` batch with all-valid masks yields finite gradients.

Verified properties for :func:`fused_dft_loss`
----------------------------------------------
``fused_dft_loss`` is a **per-example** drop-in for ``dft_loss`` that takes
hidden states + the ``lm_head`` weight instead of logits, driven by
``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path).

- **CPU** (float64): asserts the per-example contract, shapes, and
  ``vmap(grad)`` composability against the eager reference ``dft_loss``.
- **GPU** (bf16, ``[patches]``): the same parity, exercising the fused
  opake-patches linear-CE kernel path (DFT via ``use_token_scaling``).

Imports target concrete implementation paths because the public façade
``__init__.py`` is wired in the separate ε.W unit.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import grad, vmap

from opake.api.alignment.logprob._gather import selective_log_softmax
from opake.api.alignment.sft.loss import dft_loss, fused_dft_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5

_B, _T, _H, _V = 4, 9, 6, 17


def _logits_uniform(t: int, v: int) -> torch.Tensor:
    """Return all-zero (uniform distribution) logits of shape ``(t, v)``."""
    return torch.zeros(t, v)


def _make_inputs(seed: int, *, dtype=torch.float64, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_B, _T, _H, generator=gen, dtype=dtype)
    weight = torch.randn(_V, _H, generator=gen, dtype=dtype)
    labels = torch.randint(0, _V, (_B, _T), generator=gen)
    labels[:, :2] = -100  # a prompt span ignored per example
    return hidden.to(device), weight.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# dft_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


class TestDftLoss:
    """Verified reference cases for the DFT (Dynamic Fine-Tuning) loss.

    DFT arXiv:2508.05629 — per-token loss = -detach(softmax_prob) * logp.
    """

    def test_hand_computed_detached_weighting(self) -> None:
        """Small case: -p.detach() * logp per-token, averaged over 2 valid tokens.

        logits = [[2.0, 0.0], [0.0, 1.0]], labels = [0, 1, 1]
        shifted logits: [[2.0, 0.0], [0.0, 1.0]]
        shifted labels: [1, 1] (both valid)

        Row 0: lse0 = log(e^2 + 1); logp0 = -lse0; p0 = 1/exp(lse0)
        Row 1: lse1 = log(1 + e^1); logp1 = 1 - lse1; p1 = e^1/exp(lse1)
        DFT mean = (-p0*logp0 + -p1*logp1) / 2
        """
        lse0 = math.log(math.exp(2.0) + math.exp(0.0))
        lse1 = math.log(math.exp(0.0) + math.exp(1.0))
        logp0 = 0.0 - lse0  # log-prob of token 1 from row [2, 0]
        logp1 = 1.0 - lse1  # log-prob of token 1 from row [0, 1]
        p0 = math.exp(0.0) / math.exp(lse0)  # softmax prob of token 1 from row [2, 0]
        p1 = math.exp(1.0) / math.exp(lse1)  # softmax prob of token 1 from row [0, 1]
        expected = (-p0 * logp0 + -p1 * logp1) / 2.0

        # Build (1, T=3, V=2) shaped input; last logit row is irrelevant (shifted away)
        logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        labels = torch.tensor([[0, 1, 1]])

        out = dft_loss(logits, labels)
        assert out.shape == (1,)
        assert torch.allclose(out, torch.tensor([expected]), atol=_ATOL)

    def test_all_ignored_returns_zero(self) -> None:
        """All-ignored row: div-0 guard → 0.0 (same invariant as nll_loss)."""
        T = 3
        V = 4
        logits = _logits_uniform(T, V)
        labels = torch.full((T,), -100, dtype=torch.long)
        out = dft_loss(logits, labels)
        assert out.shape == ()
        assert torch.isfinite(out)
        assert torch.allclose(out, torch.zeros(()), atol=_ATOL)

    def test_batched_shape(self) -> None:
        """Batched (B, T, V) input returns shape (B,)."""
        B, T, V = 3, 5, 8
        torch.manual_seed(5)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = dft_loss(logits, labels)
        assert out.shape == (B,)

    def test_per_example_matches_batched(self) -> None:
        """Stacking per-example calls must equal the batched result."""
        B, T, V = 3, 6, 5
        torch.manual_seed(6)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[1, 2] = -100  # exercise ignore-index path

        batched = dft_loss(logits, labels)
        per_example = torch.stack([dft_loss(logits[i], labels[i]) for i in range(B)])
        assert torch.allclose(batched, per_example, atol=_ATOL)

    # ------------------------------------------------------------------
    # Gradient detach invariant — the weighting probability p must be detached
    # ------------------------------------------------------------------

    def test_gradient_flows_only_through_logp_not_p(self) -> None:
        """Gradient of dft_loss equals gradient of -p.detach() * logp.

        If ``p`` were not detached the gradient w.r.t. the logits would include
        a second term from ∂p/∂logits, changing the gradient entirely.  This
        test confirms that:
          1. grad(dft_loss) == grad(-p.detach()*logp)   [detached reference]
          2. grad(dft_loss) != grad(-p_no_detach*logp)  [non-detached differs]
        """
        torch.manual_seed(7)
        logits_base = torch.randn(1, 3, 4)
        labels = torch.randint(0, 4, (1, 3))

        # --- (1) grad of dft_loss ---
        lg1 = logits_base.clone().requires_grad_(True)
        dft_loss(lg1, labels).sum().backward()
        grad_dft = lg1.grad.clone()

        # --- (2) reference: -p.detach() * logp ---
        lg2 = logits_base.clone().requires_grad_(True)
        sl2 = lg2[:, :-1, :]  # shifted logits
        shifted_labels = labels[:, 1:]  # shifted labels
        clamped = shifted_labels.clamp(min=0)
        logp_ref = selective_log_softmax(sl2, clamped)
        p_det = (
            torch.softmax(sl2, dim=-1)
            .gather(-1, clamped.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        mask_ref = (shifted_labels != -100).float()
        divisor_ref = mask_ref.sum(-1).clamp(min=1)
        ((-p_det * logp_ref * mask_ref).sum(-1) / divisor_ref).sum().backward()
        grad_ref = lg2.grad.clone()

        # --- (3) non-detached p (must differ from dft_loss gradient) ---
        lg3 = logits_base.clone().requires_grad_(True)
        sl3 = lg3[:, :-1, :]
        logp_nodetch = selective_log_softmax(sl3, clamped)
        p_nodetch = (
            torch.softmax(sl3, dim=-1).gather(-1, clamped.unsqueeze(-1)).squeeze(-1)
        )
        mask_nd = (labels[:, 1:] != -100).float()
        divisor_nd = mask_nd.sum(-1).clamp(min=1)
        ((-p_nodetch * logp_nodetch * mask_nd).sum(-1) / divisor_nd).sum().backward()
        grad_nodetch = lg3.grad.clone()

        # Assertions
        assert torch.allclose(grad_dft, grad_ref, atol=1e-6), (
            "dft_loss gradient must equal -p.detach()*logp gradient"
        )
        assert not torch.allclose(grad_dft, grad_nodetch, atol=1e-7), (
            "a non-detached p changes the gradient — dft_loss must detach p"
        )

    # ------------------------------------------------------------------
    # DP-corrected divisor
    # ------------------------------------------------------------------

    def test_dp_corrected_divisor(self) -> None:
        """Per-example divisor (own token count) differs from batch-level total.

        2-example batch: example 0 has 3 valid tokens, example 1 has 1.
        The per-example implementation divides by 3 and 1 respectively.
        A batch-level divisor of 4 would produce different values.
        """
        torch.manual_seed(8)
        V = 4
        T = 6
        logits = torch.randn(2, T, V)

        labels_0 = torch.tensor([0, 1, 2, 3, -100, -100])  # 3 valid shifted
        labels_1 = torch.tensor([0, 1, -100, -100, -100, -100])  # 1 valid shifted
        labels = torch.stack([labels_0, labels_1])

        out = dft_loss(logits, labels)
        assert out.shape == (2,)

        # Reference: hand-compute per-example DFT losses
        sl0 = labels_0[1:]
        sl1 = labels_1[1:]
        sl0_logits = logits[0, :-1, :]
        sl1_logits = logits[1, :-1, :]

        mask0 = (sl0 != -100).float()
        mask1 = (sl1 != -100).float()
        c0 = sl0.clamp(min=0)
        c1 = sl1.clamp(min=0)
        logp0 = selective_log_softmax(sl0_logits, c0)
        logp1 = selective_log_softmax(sl1_logits, c1)
        p0 = (
            torch.softmax(sl0_logits, dim=-1)
            .gather(-1, c0.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        p1 = (
            torch.softmax(sl1_logits, dim=-1)
            .gather(-1, c1.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        expected_0 = (-p0 * logp0 * mask0).sum() / mask0.sum().clamp(min=1)
        expected_1 = (-p1 * logp1 * mask1).sum() / mask1.sum().clamp(min=1)

        assert torch.allclose(out[0], expected_0, atol=_ATOL)
        assert torch.allclose(out[1], expected_1, atol=_ATOL)

        # Confirm that a batch-level divisor of 4 would yield DIFFERENT values
        total_valid = mask0.sum() + mask1.sum()  # 4.0
        batch_0 = (-p0 * logp0 * mask0).sum() / total_valid
        batch_1 = (-p1 * logp1 * mask1).sum() / total_valid
        assert not torch.allclose(out[0], batch_0, atol=1e-7), (
            "per-example and batch divisors must differ for example 0"
        )
        assert not torch.allclose(out[1], batch_1, atol=1e-7), (
            "per-example and batch divisors must differ for example 1"
        )

    # ------------------------------------------------------------------
    # NaN-injection (Tier 1, §11.3)
    # ------------------------------------------------------------------

    def test_nan_injection_only_corrupt_one_example(self) -> None:
        """NaN logits in one example → only that example's loss is NaN.

        Tier-1 NaN-injection contract (plan §11.3).  Example 2 of a 4-example
        batch has all NaN logits; examples 0, 1, 3 must remain finite.
        """
        B, T, V = 4, 5, 6
        torch.manual_seed(9)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        logits[2] = float("nan")

        out = dft_loss(logits, labels)
        assert out.shape == (B,)
        assert torch.isfinite(out[0]), "example 0 must be finite"
        assert torch.isfinite(out[1]), "example 1 must be finite"
        assert torch.isnan(out[2]), "example 2 (NaN logits) must be NaN"
        assert torch.isfinite(out[3]), "example 3 must be finite"

    # ------------------------------------------------------------------
    # Vmap-safety (§3.4, §11.2)
    # ------------------------------------------------------------------

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(dft_loss.sum)) over (4, T, V) with all-valid labels is finite."""
        B, T, V = 4, 6, 8
        torch.manual_seed(10)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        def per_example_fn(lg: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
            return dft_loss(lg, lab).sum()

        grads = vmap(grad(per_example_fn))(logits, labels)
        assert grads.shape == (B, T, V)
        assert torch.isfinite(grads).all(), (
            "vmap(grad(dft_loss)) must yield finite grads"
        )


# ---------------------------------------------------------------------------
# fused_dft_loss — per-example fused-linear twin of dft_loss
# ---------------------------------------------------------------------------


def test_fused_dft_matches_eager_forward_cpu() -> None:
    """Per-example forward (under vmap) matches eager ``dft_loss(hidden @ W.T, …)``."""
    hidden, weight, labels = _make_inputs(seed=1)

    got = vmap(lambda h, lab: fused_dft_loss(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: dft_loss(h @ weight.T, lab))(hidden, labels)

    assert got.shape == (_B,)
    assert torch.allclose(got, want, atol=1e-10)


def test_fused_dft_vmap_grad_matches_eager_cpu() -> None:
    """``vmap(grad(...))`` w.r.t. hidden and weight matches the eager reference."""
    hidden, weight, labels = _make_inputs(seed=2)

    g_h_fused = vmap(grad(lambda h, lab: fused_dft_loss(h, weight, lab)))(
        hidden, labels
    )
    g_h_eager = vmap(grad(lambda h, lab: dft_loss(h @ weight.T, lab)))(hidden, labels)
    assert g_h_fused.shape == hidden.shape
    assert torch.allclose(g_h_fused, g_h_eager, atol=1e-10)

    # Grad w.r.t. the shared weight, summed over the per-example batch.
    g_w_fused = grad(
        lambda w: vmap(lambda h, lab: fused_dft_loss(h, w, lab))(hidden, labels).sum()
    )(weight)
    g_w_eager = grad(
        lambda w: vmap(lambda h, lab: dft_loss(h @ w.T, lab))(hidden, labels).sum()
    )(weight)
    assert torch.allclose(g_w_fused, g_w_eager, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_dft_lce_path_matches_eager_gpu() -> None:
    """The fused kernel path (CUDA + bf16) matches the eager reference.

    Exercises the opake-patches linear-CE kernel (DFT via
    ``use_token_scaling``); bf16 matmul is coarse, so tolerances are loose.
    """
    hidden, weight, labels = _make_inputs(seed=3, dtype=torch.bfloat16, device="cuda")

    got = vmap(lambda h, lab: fused_dft_loss(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: dft_loss(h @ weight.T, lab))(hidden, labels)
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=0.0)

    g_fused = vmap(grad(lambda h, lab: fused_dft_loss(h, weight, lab)))(hidden, labels)
    g_eager = vmap(grad(lambda h, lab: dft_loss(h @ weight.T, lab)))(hidden, labels)
    assert torch.isfinite(g_fused).all()
    assert torch.allclose(g_fused.float(), g_eager.float(), atol=1e-2, rtol=0.0)


def test_dft_divisor_is_exact_token_count_under_bf16() -> None:
    """The eager divisor must not round 257 valid tokens down to 256."""

    def _uniform_mean(n_valid: int) -> float:
        logits = torch.zeros(n_valid + 1, 8, dtype=torch.bfloat16)
        labels = torch.zeros(n_valid + 1, dtype=torch.long)
        return dft_loss(logits, labels).item()

    assert _uniform_mean(256) == pytest.approx(_uniform_mean(257), rel=1e-4)
