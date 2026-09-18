"""Selective log-softmax kernel tests.

Mirrors the chunked CE kernel's vmap-grad coverage: forward/backward parity
against ``F.log_softmax + gather``, vmap parity, and the single-chunk vs chunked
vocab regimes. Skips when CUDA + Triton aren't available.
"""

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.cross_entropy import (
    opake_selective_log_softmax,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Same tolerances as test_cross_entropy.py — accumulation over large vocab.
RTOL_FWD = 1e-4
ATOL_FWD = 1e-6
RTOL_BWD = 1e-4
ATOL_BWD = 1e-6

# Single-chunk (<= MAX_FUSED_SIZE=65536) and chunked (> 65536) vocab regimes.
VOCAB_SIZES = [32768, 128256]


def pytorch_selective_log_softmax(logits, indices):
    """Reference: log_softmax(logits, -1).gather(-1, indices[..., None]).squeeze(-1)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_forward_matches_reference(vocab_size):
    """``opake_selective_log_softmax`` matches eager ``log_softmax + gather``."""
    torch.manual_seed(0)
    logits = torch.randn(2, 16, vocab_size, device="cuda", dtype=torch.float32)
    indices = torch.randint(0, vocab_size, (2, 16), device="cuda")

    out = opake_selective_log_softmax(logits, indices)
    ref = pytorch_selective_log_softmax(logits, indices)

    torch.testing.assert_close(out, ref, rtol=RTOL_FWD, atol=ATOL_FWD)


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_backward_matches_reference(vocab_size):
    """Gradient w.r.t. logits matches the eager reference."""
    torch.manual_seed(1)
    logits_a = torch.randn(
        2, 16, vocab_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    logits_b = logits_a.detach().clone().requires_grad_(True)
    indices = torch.randint(0, vocab_size, (2, 16), device="cuda")

    out = opake_selective_log_softmax(logits_a, indices)
    out.sum().backward()

    ref = pytorch_selective_log_softmax(logits_b, indices)
    ref.sum().backward()

    torch.testing.assert_close(
        logits_a.grad, logits_b.grad, rtol=RTOL_BWD, atol=ATOL_BWD
    )


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_vmap_grad_matches_reference(vocab_size):
    """``vmap(grad(opake_selective_log_softmax-sum))`` matches the eager reference."""
    torch.manual_seed(2)
    logits = torch.randn(4, 16, vocab_size, device="cuda", dtype=torch.float32)
    indices = torch.randint(0, vocab_size, (4, 16), device="cuda")

    def opake_sum(lg, idx):
        return opake_selective_log_softmax(lg, idx).sum()

    def eager_sum(lg, idx):
        return pytorch_selective_log_softmax(lg, idx).sum()

    grad_opake = vmap(grad(opake_sum))(logits, indices)
    grad_eager = vmap(grad(eager_sum))(logits, indices)

    torch.testing.assert_close(grad_opake, grad_eager, rtol=RTOL_BWD, atol=ATOL_BWD)


def test_ignored_index_returns_zero():
    """Indices at the kernel's ``-100`` sentinel return ``0`` (CE-kernel convention)."""
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 1024, device="cuda", dtype=torch.float32)
    indices = torch.full((2, 4), -100, device="cuda", dtype=torch.long)

    out = opake_selective_log_softmax(logits, indices)
    assert torch.all(out == 0.0), out


def test_zero_dim_index_broadcasts():
    """0-dim scalar index broadcasts across leading dims, matching the eager contract."""
    torch.manual_seed(4)
    logits = torch.randn(2, 4, 32, device="cuda", dtype=torch.float32)
    # Expand the 0-dim index to per-row indices for the kernel's contract.
    idx0 = torch.tensor(7, device="cuda")
    expanded = idx0.expand(2, 4).contiguous()

    out = opake_selective_log_softmax(logits, expanded)
    ref = pytorch_selective_log_softmax(logits, expanded)
    torch.testing.assert_close(out, ref, rtol=RTOL_FWD, atol=ATOL_FWD)
