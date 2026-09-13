# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the logprob primitives (plan §7.5, §11.2/§11.3).

Covers the three pure logprob helpers:

- :func:`selective_log_softmax` — gather of ``log_softmax`` at indices.
- :func:`sequence_logp` — DPO/causal-LM per-sequence completion logp.

Each function has >=3 hand-computed reference cases (tiny tensors where the
expected logp is computed by hand or via a reference ``F.log_softmax`` loop),
plus a ``torch.func.vmap(torch.func.grad(...))`` finite-gradient contract test
(mirrors ``opaque-engine/tests/functional/test_functional.py``). Imports target
the concrete impl paths because the public façade is not wired yet (unit β.W).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.logprob._sequence import (
    fused_sequence_logp,
    sequence_logp,
)

# ---------------------------------------------------------------------------
# selective_log_softmax
# ---------------------------------------------------------------------------


def test_selective_log_softmax_hand_uniform() -> None:
    """Uniform logits → log p = log(1/V) for every selected token."""
    v = 4
    logits = torch.zeros(3, v)  # all-equal logits → uniform distribution
    indices = torch.tensor([0, 1, 3])
    out = selective_log_softmax(logits, indices)
    expected = torch.full((3,), math.log(1.0 / v))
    assert out.shape == (3,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_selective_log_softmax_hand_two_class() -> None:
    """Two-logit case computed by hand: logp = z_i - logsumexp(z)."""
    logits = torch.tensor([[1.0, 3.0]])
    lse = math.log(math.exp(1.0) + math.exp(3.0))
    # pick index 1 (the value 3.0)
    out = selective_log_softmax(logits, torch.tensor([1]))
    assert torch.allclose(out, torch.tensor([3.0 - lse]), atol=1e-6)
    # pick index 0 (the value 1.0)
    out0 = selective_log_softmax(logits, torch.tensor([0]))
    assert torch.allclose(out0, torch.tensor([1.0 - lse]), atol=1e-6)


def test_selective_log_softmax_hand_2d_grid() -> None:
    """A (T,) input (no batch axis) returns a scalar-per-position vector."""
    logits = torch.tensor([2.0, 0.0, -1.0])
    lse = torch.logsumexp(logits, dim=-1)
    out = selective_log_softmax(logits, torch.tensor(0))  # 0-dim index
    assert out.shape == ()
    assert torch.allclose(out, logits[0] - lse, atol=1e-6)


def test_selective_log_softmax_matches_reference() -> None:
    """Matches the explicit ``F.log_softmax(...).gather(...).squeeze(-1)``."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    indices = torch.randint(0, 7, (2, 5))
    out = selective_log_softmax(logits, indices)
    ref = F.log_softmax(logits, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    assert out.shape == (2, 5)
    assert torch.allclose(out, ref, atol=1e-6)


def test_selective_log_softmax_large_logits_stable() -> None:
    """Large logits must not overflow (log-sum-exp trick, never exp(logits))."""
    logits = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 2.0]])
    out = selective_log_softmax(logits, torch.tensor([1]))
    assert torch.isfinite(out).all()
    # Shift-invariance: subtracting a constant leaves log-softmax unchanged.
    shifted = selective_log_softmax(logits - 1e4, torch.tensor([1]))
    assert torch.allclose(out, shifted, atol=1e-4)


# ---------------------------------------------------------------------------
# sequence_logp
# ---------------------------------------------------------------------------


def _reference_sequence_logp(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Independent reference: shift, per-token logp loop, mask, sum."""
    shifted = logits[..., :-1, :]
    targets = input_ids[..., 1:]
    m = mask[..., 1:].to(logits.dtype)
    logp = F.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1))
    logp = logp.squeeze(-1)
    return (logp * m).sum(dim=-1)


def test_sequence_logp_hand_single_completion_token() -> None:
    """One completion token: logp equals that single token's logp."""
    # T=2, V=3. Only the prediction for position 0 (target = input_ids[1]) counts.
    logits = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])  # (1, 2, 3)
    input_ids = torch.tensor([[2, 1]])  # target at shift is input_ids[1] = 1
    # mask shifted is completion_mask[1:] = [1] → the single prediction counts.
    completion_mask = torch.tensor([[0, 1]])
    out = sequence_logp(logits, input_ids, completion_mask)
    expected = torch.tensor([math.log(1.0 / 3)])  # uniform logits row at pos 0
    assert out.shape == (1,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sequence_logp_hand_masked_out_is_zero() -> None:
    """An all-zero (shifted) mask yields a zero summed logp."""
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 6)
    input_ids = torch.randint(0, 6, (2, 4))
    completion_mask = torch.zeros(2, 4, dtype=torch.long)
    out = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out, torch.zeros(2), atol=1e-7)


def test_sequence_logp_hand_two_tokens_sum() -> None:
    """Two completion tokens: result is the sum of their per-token logps."""
    # T=3, V=2. shifts produce 2 predictions; both masked in.
    logits = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]]  # (1, 3, 2)
    )
    input_ids = torch.tensor([[0, 1, 0]])  # targets after shift: [1, 0]
    completion_mask = torch.tensor([[1, 1, 1]])  # shifted → [1, 1]
    # pos0 predicts token 1: logp = 0 - logsumexp([1,0])
    lse0 = math.log(math.exp(1.0) + math.exp(0.0))
    # pos1 predicts token 0: logp = 0 - logsumexp([0,2])
    lse1 = math.log(math.exp(0.0) + math.exp(2.0))
    expected = torch.tensor([(0.0 - lse0) + (0.0 - lse1)])
    out = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sequence_logp_length_normalized_divisor_is_exact_under_bf16() -> None:
    """Length normalization must not round 257 completion tokens down to 256."""

    def _uniform_mean(n_valid: int) -> float:
        logits = torch.zeros(n_valid + 1, 8, dtype=torch.bfloat16)
        input_ids = torch.zeros(n_valid + 1, dtype=torch.long)
        completion_mask = torch.ones(n_valid + 1, dtype=torch.bool)
        return sequence_logp(
            logits, input_ids, completion_mask, length_normalized=True
        ).item()

    assert _uniform_mean(256) == pytest.approx(_uniform_mean(257), rel=1e-4)


def test_sequence_logp_matches_reference_batched() -> None:
    """Batched (B, T, V) matches the independent reference implementation."""
    torch.manual_seed(2)
    logits = torch.randn(3, 5, 8)
    input_ids = torch.randint(0, 8, (3, 5))
    completion_mask = torch.randint(0, 2, (3, 5))
    out = sequence_logp(logits, input_ids, completion_mask)
    ref = _reference_sequence_logp(logits, input_ids, completion_mask)
    assert out.shape == (3,)
    assert torch.allclose(out, ref, atol=1e-6)


def test_sequence_logp_per_example_matches_batched() -> None:
    """Per-example (T, V) call equals the corresponding batched row.

    Guards against silently dropping the batch axis: a per-example call must
    return a 0-dim scalar, and stacking these must equal the batched result.
    """
    torch.manual_seed(3)
    logits = torch.randn(4, 6, 5)
    input_ids = torch.randint(0, 5, (4, 6))
    completion_mask = torch.randint(0, 2, (4, 6))

    batched = sequence_logp(logits, input_ids, completion_mask)
    per_example = torch.stack(
        [sequence_logp(logits[i], input_ids[i], completion_mask[i]) for i in range(4)]
    )
    # Each per-example result must be a true scalar (no leftover batch axis).
    single = sequence_logp(logits[0], input_ids[0], completion_mask[0])
    assert single.shape == ()
    assert torch.allclose(batched, per_example, atol=1e-6)


def test_sequence_logp_shared_prefix_len_ignored_when_alpha_none() -> None:
    """shared_prefix_len is accepted and ignored when ld_alpha is None."""
    torch.manual_seed(4)
    logits = torch.randn(2, 4, 6)
    input_ids = torch.randint(0, 6, (2, 4))
    completion_mask = torch.ones(2, 4, dtype=torch.long)
    out_with = sequence_logp(
        logits,
        input_ids,
        completion_mask,
        ld_alpha=None,
        shared_prefix_len=torch.tensor([1, 2]),
    )
    out_without = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out_with, out_without, atol=1e-7)


# ---------------------------------------------------------------------------
# vmap(grad(...)) contract tests (plan §11.2)
# ---------------------------------------------------------------------------


def test_sequence_logp_vmap_grad_finite() -> None:
    """vmap(grad(sequence_logp-sum)) over a (B, T, V) batch yields finite grads.

    Mirrors the ``torch.func.vmap`` / ``torch.func.grad`` style in
    opaque-engine's functional tests. grad is taken over the float ``logits``
    argument (argnum 0); ``input_ids`` and ``completion_mask`` ride along.
    """
    b, t, v = 4, 6, 8
    torch.manual_seed(8)
    logits = torch.randn(b, t, v)
    input_ids = torch.randint(0, v, (b, t))
    completion_mask = torch.randint(0, 2, (b, t))

    def per_example(lg: torch.Tensor, ids: torch.Tensor, m: torch.Tensor):
        return sequence_logp(lg, ids, m).sum()

    grads = vmap(grad(per_example))(logits, input_ids, completion_mask)
    assert grads.shape == (b, t, v)
    assert torch.isfinite(grads).all()


def test_selective_log_softmax_vmap_grad_finite() -> None:
    """vmap(grad(selective_log_softmax-sum)) over a (B, T, V) batch is finite."""
    b, t, v = 4, 6, 8
    torch.manual_seed(10)
    logits = torch.randn(b, t, v)
    indices = torch.randint(0, v, (b, t))

    def per_example(lg: torch.Tensor, idx: torch.Tensor):
        return selective_log_softmax(lg, idx).sum()

    grads = vmap(grad(per_example))(logits, indices)
    assert grads.shape == (b, t, v)
    assert torch.isfinite(grads).all()


# ---------------------------------------------------------------------------
# fused_sequence_logp — memory-efficient drop-in over hidden states
# ---------------------------------------------------------------------------
#
# Per-example drop-in for ``sequence_logp`` (hidden + lm_head weight instead of
# logits), driven by ``vmap(grad)``. On CPU it takes its eager fallback
# (``sequence_logp(hidden @ W.T, …)``), so these assert the contract / shapes /
# vmap(grad) composability; the GPU test exercises the fused linear-CE kernel.

_FB, _FT, _FH, _FV = 4, 9, 6, 17


def _fused_logp_inputs(seed: int, *, dtype=torch.float64, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_FB, _FT, _FH, generator=gen, dtype=dtype)
    weight = torch.randn(_FV, _FH, generator=gen, dtype=dtype)
    input_ids = torch.randint(0, _FV, (_FB, _FT), generator=gen)
    completion_mask = torch.zeros(_FB, _FT, dtype=dtype)
    completion_mask[:, 3:] = 1.0  # completion span = tokens [3:]
    return (
        hidden.to(device),
        weight.to(device),
        input_ids.to(device),
        completion_mask.to(device),
    )


def test_fused_sequence_logp_matches_eager_cpu() -> None:
    """Per-example forward + vmap(grad) match eager ``sequence_logp(hidden @ W.T, …)``."""
    hidden, weight, ids, cmask = _fused_logp_inputs(seed=1)

    got = vmap(lambda h, i, c: fused_sequence_logp(h, weight, i, c))(hidden, ids, cmask)
    want = vmap(lambda h, i, c: sequence_logp(h @ weight.T, i, c))(hidden, ids, cmask)
    assert got.shape == (_FB,)
    assert torch.allclose(got, want, atol=1e-10)

    g_fused = vmap(grad(lambda h, i, c: fused_sequence_logp(h, weight, i, c)))(
        hidden, ids, cmask
    )
    g_eager = vmap(grad(lambda h, i, c: sequence_logp(h @ weight.T, i, c)))(
        hidden, ids, cmask
    )
    assert g_fused.shape == hidden.shape
    assert torch.allclose(g_fused, g_eager, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sequence_logp_lce_path_gpu() -> None:
    """The fused kernel path (CUDA + bf16) matches eager ``sequence_logp``."""
    hidden, weight, ids, cmask = _fused_logp_inputs(
        seed=2, dtype=torch.bfloat16, device="cuda"
    )

    got = vmap(lambda h, i, c: fused_sequence_logp(h, weight, i, c))(hidden, ids, cmask)
    want = vmap(lambda h, i, c: sequence_logp(h @ weight.T, i, c))(hidden, ids, cmask)
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=0.0)

    g_fused = vmap(grad(lambda h, i, c: fused_sequence_logp(h, weight, i, c)))(
        hidden, ids, cmask
    )
    g_eager = vmap(grad(lambda h, i, c: sequence_logp(h @ weight.T, i, c)))(
        hidden, ids, cmask
    )
    assert torch.isfinite(g_fused).all()
    assert torch.allclose(g_fused.float(), g_eager.float(), atol=1e-2, rtol=0.0)


# ---------------------------------------------------------------------------
# sequence_logp — length_normalized (per-token mean reward for SimPO / ORPO)
# ---------------------------------------------------------------------------


def test_length_normalized_is_per_token_mean() -> None:
    """length_normalized=True divides the summed logp by the completion count."""
    torch.manual_seed(0)
    logits = torch.randn(7, 11)
    ids = torch.randint(0, 11, (7,))
    cmask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    summed = sequence_logp(logits, ids, cmask)
    mean = sequence_logp(logits, ids, cmask, length_normalized=True)
    count = cmask[1:].sum()  # shifted completion-token count
    torch.testing.assert_close(mean, summed / count)


def test_length_normalized_clamps_empty_completion() -> None:
    """An all-prompt mask clamps the divisor to 1 (no division by zero)."""
    torch.manual_seed(1)
    logits = torch.randn(4, 6)
    ids = torch.randint(0, 6, (4,))
    cmask = torch.zeros(4)
    out = sequence_logp(logits, ids, cmask, length_normalized=True)
    torch.testing.assert_close(out, torch.tensor(0.0))


def test_length_normalized_fused_matches_eager() -> None:
    """fused_sequence_logp(length_normalized) matches the eager mean (CPU fallback)."""
    torch.manual_seed(2)
    hidden = torch.randn(7, 5, dtype=torch.float64)
    weight = torch.randn(11, 5, dtype=torch.float64)
    ids = torch.randint(0, 11, (7,))
    cmask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    got = fused_sequence_logp(hidden, weight, ids, cmask, length_normalized=True)
    want = sequence_logp(hidden @ weight.T, ids, cmask, length_normalized=True)
    torch.testing.assert_close(got, want)


# ---------------------------------------------------------------------------
# sequence_logp — LD-DPO length-desensitization (ld_alpha)
# ---------------------------------------------------------------------------


def test_ld_alpha_one_recovers_plain_sum() -> None:
    """ld_alpha=1.0 weights every token by 1.0 == the plain masked sum."""
    torch.manual_seed(0)
    logits = torch.randn(8, 13)
    ids = torch.randint(0, 13, (8,))
    cmask = torch.ones(8)
    cmask[:3] = 0.0
    plain = sequence_logp(logits, ids, cmask)
    ld = sequence_logp(logits, ids, cmask, ld_alpha=1.0, shared_prefix_len=2)
    torch.testing.assert_close(ld, plain)


def test_ld_alpha_zero_keeps_only_prefix() -> None:
    """ld_alpha=0.0 keeps the first shared_prefix_len completion tokens."""
    torch.manual_seed(1)
    logits = torch.randn(8, 13)
    ids = torch.randint(0, 13, (8,))
    cmask = torch.ones(8)
    prefix = 4
    ld = sequence_logp(logits, ids, cmask, ld_alpha=0.0, shared_prefix_len=prefix)
    # Reference: first `prefix` completion-token logps, independent of prompt.
    per_token = selective_log_softmax(logits[:-1], ids[1:])
    completion_pos = cmask[1:].cumsum(-1)
    want = (per_token * cmask[1:] * (completion_pos <= prefix)).sum(-1)
    torch.testing.assert_close(ld, want)


def test_ld_alpha_positions_are_completion_relative() -> None:
    """Prompt positions do not consume the shared completion-token count."""
    torch.manual_seed(2)
    logits = torch.randn(8, 13)
    ids = torch.randint(0, 13, (8,))
    cmask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])
    per_token = selective_log_softmax(logits[:-1], ids[1:])
    target_mask = cmask[1:]
    completion_pos = target_mask.cumsum(-1)
    weight = torch.where(completion_pos <= 2, 1.0, 0.25) * target_mask
    want = (per_token * weight).sum()

    got = sequence_logp(
        logits,
        ids,
        cmask,
        ld_alpha=0.25,
        shared_prefix_len=2,
    )
    torch.testing.assert_close(got, want)


def test_ld_alpha_requires_shared_prefix_len() -> None:
    logits = torch.randn(5, 7)
    ids = torch.randint(0, 7, (5,))
    cmask = torch.ones(5)
    with pytest.raises(ValueError, match="shared_prefix_len"):
        sequence_logp(logits, ids, cmask, ld_alpha=0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_ld_alpha_rejects_values_outside_paper_domain(alpha: float) -> None:
    logits = torch.randn(5, 7)
    ids = torch.randint(0, 7, (5,))
    cmask = torch.ones(5)
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        sequence_logp(logits, ids, cmask, ld_alpha=alpha, shared_prefix_len=2)


def test_ld_alpha_vmap_grad_finite() -> None:
    torch.manual_seed(2)
    logits = torch.randn(4, 6, 9)
    ids = torch.randint(0, 9, (4, 6))
    cmask = torch.ones(4, 6)

    def per_example(lg, i, c):
        return sequence_logp(lg, i, c, ld_alpha=0.5, shared_prefix_len=2)

    g = vmap(grad(per_example))(logits, ids, cmask)
    assert g.shape == logits.shape
    assert torch.isfinite(g).all()


# ------------------------------------------------------------------------
# Dispatcher gate
# ------------------------------------------------------------------------
def test_selective_log_softmax_available_false_on_cpu() -> None:
    """The chunked-CE Triton kernel can't run on CPU; gate must say so."""
    from opaque.api.alignment._fused_lce import selective_log_softmax_available

    logits = torch.randn(3, 4, 8)
    assert selective_log_softmax_available(logits) is False


def test_selective_log_softmax_eager_path_matches_reference_on_cpu() -> None:
    """When the gate is False the dispatcher falls back to eager log_softmax+gather."""
    logits = torch.randn(2, 5, 13)
    indices = torch.randint(0, 13, (2, 5))
    out = selective_log_softmax(logits, indices)
    ref = (
        torch.log_softmax(logits, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(out, ref)
