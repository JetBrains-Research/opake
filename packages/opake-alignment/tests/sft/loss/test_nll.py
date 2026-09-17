# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + contract tests for the SFT NLL losses ``nll_loss`` / ``fused_nll_loss``.

Covers the standard causal-LM NLL loss and its fused-linear twin (plan §7.3,
§11.2, §11.3; work-units ε.1 and the fused-linear ε.* unit).

Verified properties for :func:`nll_loss`
-----------------------------------------
* ≥3 hand-computed reference cases (uniform logits → log(V), ignore-index
  exclusion, all-ignored div-0 guard).
* **DP-corrected divisor** (§3.3, §8.2): 2-example batch with different
  completion lengths → per-example divisor differs from a batch-level divisor.
* **NaN-injection** (Tier 1, §11.3): NaN in one example's logits produces NaN
  only for that example; neighbours are finite.
* **Vmap-safety** (§3.4, §11.2): ``vmap(grad(nll_loss.sum))`` over a
  ``(4, T, V)`` batch with all-valid masks yields finite gradients.

Verified properties for :func:`fused_nll_loss`
----------------------------------------------
``fused_nll_loss`` is a **per-example** drop-in for ``nll_loss`` that takes
hidden states + the ``lm_head`` weight instead of logits, driven by
``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path).

- **CPU** (float64): asserts the per-example contract, shapes, and
  ``vmap(grad)`` composability against the eager reference ``nll_loss``.
- **GPU** (bf16, ``[patches]``): the same parity, exercising the fused
  opake-patches linear-CE kernel path (NLL plain).

Imports target concrete implementation paths because the public façade
``__init__.py`` is wired in the separate ε.W unit.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import grad, vmap

from opake.api.alignment.logprob._gather import selective_log_softmax
from opake.api.alignment.sft.loss import fused_nll_loss, nll_loss

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
# nll_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


class TestNllLoss:
    """Verified reference cases for the standard causal-LM NLL loss."""

    def test_uniform_logits_all_valid(self) -> None:
        """Uniform logits → per-token NLL = log(V); mean = log(V).

        With all-zero logits over V=4 tokens the log-softmax at every position
        is log(1/4) = -log(4).  Three shifted tokens, all valid → mean NLL = log(4).
        """
        V = 4
        T = 4  # seq len; shifted to T-1=3 predictions
        logits = _logits_uniform(T, V)
        labels = torch.tensor([0, 1, 2, 3])  # shifted labels: [1, 2, 3], all valid
        out = nll_loss(logits, labels)
        expected = math.log(V)
        assert out.shape == (), "per-example call must return a scalar"
        assert torch.allclose(out, torch.tensor(expected), atol=_ATOL)

    def test_ignore_index_excluded(self) -> None:
        """Tokens with label -100 are excluded from the mean.

        With V=3 uniform logits and two of three shifted positions valid,
        the mean NLL is log(3), regardless of the ignored position's logit row.
        """
        V = 3
        T = 4
        logits = _logits_uniform(T, V)
        labels = torch.tensor([0, 1, -100, 2])  # shifted: [1, -100, 2]; 2 valid
        out = nll_loss(logits, labels)
        expected = math.log(V)  # log(3) — only the 2 valid tokens count
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=_ATOL)

    def test_all_ignored_returns_zero(self) -> None:
        """All-ignored row: divisor clamped to 1 → loss = 0 (no div-by-zero).

        When every shifted label is -100 the mask sum is 0.  After
        ``clamp(min=1)`` the divisor is 1 and the numerator is 0 → 0.0.
        This guards the edge case that would otherwise produce NaN.
        """
        T = 3
        V = 4
        logits = _logits_uniform(T, V)
        labels = torch.full((T,), -100, dtype=torch.long)
        out = nll_loss(logits, labels)
        assert out.shape == ()
        assert torch.isfinite(out), "all-ignored row must not produce NaN/inf"
        assert torch.allclose(out, torch.zeros(()), atol=_ATOL)

    def test_batched_shape(self) -> None:
        """Batched (B, T, V) input returns shape (B,)."""
        B, T, V = 3, 5, 8
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = nll_loss(logits, labels)
        assert out.shape == (B,)

    def test_per_example_matches_batched(self) -> None:
        """Stacking per-example calls must equal the batched result."""
        B, T, V = 3, 6, 5
        torch.manual_seed(1)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[0, 3] = -100  # exercise ignore-index path

        batched = nll_loss(logits, labels)
        per_example = torch.stack([nll_loss(logits[i], labels[i]) for i in range(B)])
        assert torch.allclose(batched, per_example, atol=_ATOL)

    # ------------------------------------------------------------------
    # DP-corrected divisor: per-example token count, not batch token count
    # ------------------------------------------------------------------

    def test_dp_corrected_divisor(self) -> None:
        """Per-example divisor differs from a batch-level total-token divisor.

        Constructs a 2-example batch where example 0 has 3 valid shifted tokens
        and example 1 has 1 valid shifted token.  The per-example divisors (3
        and 1) differ from the batch-level divisor (4), so the two approaches
        produce different per-example losses.  This test asserts the per-example
        result — verifying that the implementation uses ``mask.sum(-1)`` per
        example, not a cross-example aggregate (plan §3.3, §8.2).
        """
        torch.manual_seed(2)
        V = 4
        T = 6  # seq len
        logits = torch.randn(2, T, V)

        # Example 0: shifted labels [1, 2, 3, -100, -100] → 3 valid tokens
        # Example 1: shifted labels [1, -100, -100, -100, -100] → 1 valid token
        labels_0 = torch.tensor([0, 1, 2, 3, -100, -100])
        labels_1 = torch.tensor([0, 1, -100, -100, -100, -100])
        labels = torch.stack([labels_0, labels_1])  # (2, 6)

        out = nll_loss(logits, labels)
        assert out.shape == (2,)

        # Reference: hand-compute per-example losses
        sl0 = labels_0[1:]  # shifted labels for example 0
        sl1 = labels_1[1:]  # shifted labels for example 1
        sl0_logits = logits[0, :-1, :]  # shifted logits (T-1, V)
        sl1_logits = logits[1, :-1, :]

        mask0 = (sl0 != -100).float()
        mask1 = (sl1 != -100).float()
        c0 = sl0.clamp(min=0)
        c1 = sl1.clamp(min=0)
        logp0 = selective_log_softmax(sl0_logits, c0)
        logp1 = selective_log_softmax(sl1_logits, c1)
        expected_0 = (-logp0 * mask0).sum() / mask0.sum().clamp(min=1)
        expected_1 = (-logp1 * mask1).sum() / mask1.sum().clamp(min=1)

        assert torch.allclose(out[0], expected_0, atol=_ATOL)
        assert torch.allclose(out[1], expected_1, atol=_ATOL)

        # Confirm that a batch-level divisor would give a DIFFERENT answer,
        # proving the test distinguishes the two approaches.
        total_valid = mask0.sum() + mask1.sum()  # 4.0
        batch_0 = (-logp0 * mask0).sum() / total_valid
        batch_1 = (-logp1 * mask1).sum() / total_valid
        assert not torch.allclose(out[0], batch_0, atol=1e-7), (
            "per-example and batch divisors must differ for this construction"
        )
        assert not torch.allclose(out[1], batch_1, atol=1e-7), (
            "per-example and batch divisors must differ for this construction"
        )

    # ------------------------------------------------------------------
    # NaN-injection (Tier 1, §11.3)
    # ------------------------------------------------------------------

    def test_nan_injection_only_corrupt_one_example(self) -> None:
        """NaN logits in one example → only that example's loss is NaN.

        This is the Tier-1 NaN-injection contract (plan §11.3): replacing one
        example's input with NaN must not corrupt the gradients/outputs of any
        other example in the batch.
        """
        B, T, V = 4, 5, 6
        torch.manual_seed(3)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        # Inject NaN into example 1's logits (all positions)
        logits[1] = float("nan")

        out = nll_loss(logits, labels)
        assert out.shape == (B,)
        assert torch.isfinite(out[0]), "example 0 must be finite"
        assert torch.isnan(out[1]), "example 1 (NaN logits) must be NaN"
        assert torch.isfinite(out[2]), "example 2 must be finite"
        assert torch.isfinite(out[3]), "example 3 must be finite"

    # ------------------------------------------------------------------
    # Vmap-safety (§3.4, §11.2)
    # ------------------------------------------------------------------

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(nll_loss.sum)) over (4, T, V) with all-valid labels is finite.

        Uses ``torch.func.vmap(torch.func.grad(...))`` to verify that the
        function is free of Python control flow on tensor values, ``.item()``
        calls, and module-state side-effects — all of which would break vmap.
        """
        B, T, V = 4, 6, 8
        torch.manual_seed(4)
        logits = torch.randn(B, T, V)
        # All-valid labels: integers in [0, V), no -100 ignore tokens.
        labels = torch.randint(0, V, (B, T))

        def per_example_fn(lg: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
            return nll_loss(lg, lab).sum()

        grads = vmap(grad(per_example_fn))(logits, labels)
        assert grads.shape == (B, T, V)
        assert torch.isfinite(grads).all(), (
            "vmap(grad(nll_loss)) must yield finite grads"
        )


# ---------------------------------------------------------------------------
# fused_nll_loss — per-example fused-linear twin of nll_loss
# ---------------------------------------------------------------------------


def test_fused_nll_matches_eager_forward_cpu() -> None:
    """Per-example forward (under vmap) matches eager ``nll_loss(hidden @ W.T, …)``."""
    hidden, weight, labels = _make_inputs(seed=1)

    got = vmap(lambda h, lab: fused_nll_loss(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: nll_loss(h @ weight.T, lab))(hidden, labels)

    assert got.shape == (_B,)
    assert torch.allclose(got, want, atol=1e-10)


def test_fused_nll_vmap_grad_matches_eager_cpu() -> None:
    """``vmap(grad(...))`` w.r.t. hidden and weight matches the eager reference."""
    hidden, weight, labels = _make_inputs(seed=2)

    g_h_fused = vmap(grad(lambda h, lab: fused_nll_loss(h, weight, lab)))(
        hidden, labels
    )
    g_h_eager = vmap(grad(lambda h, lab: nll_loss(h @ weight.T, lab)))(hidden, labels)
    assert g_h_fused.shape == hidden.shape
    assert torch.allclose(g_h_fused, g_h_eager, atol=1e-10)

    # Grad w.r.t. the shared weight, summed over the per-example batch.
    g_w_fused = grad(
        lambda w: vmap(lambda h, lab: fused_nll_loss(h, w, lab))(hidden, labels).sum()
    )(weight)
    g_w_eager = grad(
        lambda w: vmap(lambda h, lab: nll_loss(h @ w.T, lab))(hidden, labels).sum()
    )(weight)
    assert torch.allclose(g_w_fused, g_w_eager, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_nll_lce_path_matches_eager_gpu() -> None:
    """The fused kernel path (CUDA + bf16) matches the eager reference.

    Exercises the opake-patches linear-CE kernel (NLL plain); bf16 matmul is
    coarse, so tolerances are loose.
    """
    hidden, weight, labels = _make_inputs(seed=3, dtype=torch.bfloat16, device="cuda")

    got = vmap(lambda h, lab: fused_nll_loss(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: nll_loss(h @ weight.T, lab))(hidden, labels)
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=0.0)

    g_fused = vmap(grad(lambda h, lab: fused_nll_loss(h, weight, lab)))(hidden, labels)
    g_eager = vmap(grad(lambda h, lab: nll_loss(h @ weight.T, lab)))(hidden, labels)
    assert torch.isfinite(g_fused).all()
    assert torch.allclose(g_fused.float(), g_eager.float(), atol=1e-2, rtol=0.0)


def test_nll_divisor_is_exact_token_count_under_bf16() -> None:
    """The eager divisor must not round 257 valid tokens down to 256."""

    def _uniform_mean(n_valid: int) -> float:
        logits = torch.zeros(n_valid + 1, 8, dtype=torch.bfloat16)
        labels = torch.zeros(n_valid + 1, dtype=torch.long)
        return nll_loss(logits, labels).item()

    assert _uniform_mean(256) == pytest.approx(_uniform_mean(257), rel=1e-4)
