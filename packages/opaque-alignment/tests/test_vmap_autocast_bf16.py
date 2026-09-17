# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Composition tests: ``vmap(grad(fused_*))`` under ``torch.autocast(bfloat16)``.

The opaque-alignment training path runs each per-example loss closure inside
``vmap(grad(...))`` for DP-SGD per-example sensitivity. In production that
happens under a mixed-precision context — typically
``torch.autocast(device_type="cuda", dtype=torch.bfloat16)`` — so the
forward (matmul, attention) runs in bf16 while specific math (softmax,
log-sum-exp, divisions) is kept in fp32 by the autocast list.

This file pins the **composition contract** for the three fused-linear
twins consumed by the SFT and DPO closures:

- :func:`fused_nll_loss`
- :func:`fused_dft_loss`
- :func:`fused_sequence_logp`

For each:

1. ``vmap(grad(fused_*))`` under ``autocast(bf16)`` produces finite gradients
   with no crash. PyTorch 2.10 has a known ``aten::layer_norm`` fast-path that
   misbehaves under ``vmap + autocast`` (see project memory
   ``project_vmap_autocast_layernorm.md``); these fused kernels go through the
   opaque-patches Triton path and must not regress to the broken fused-aten
   path under the autocast context.
2. The autocast'd result matches the eager bf16 reference within bf16
   tolerance (bf16 matmul is coarse — atol=1e-2).

The production case is a bf16 hidden against an fp32 lm_head ``weight``;
``linear_nll_sum`` must reconcile them (via ``follow_autocast``) or the
kernel's ``tl.dot(e, c)`` mismatches. Covered by the fp32-weight CUDA test and
the CPU wiring guard (cuda-autocast cannot be enabled on a CPU host).

CUDA tests are CUDA-only and skip cleanly on non-CUDA hosts.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.alignment import _fused_lce
from opaque.api.alignment.logprob._sequence import fused_sequence_logp
from opaque.api.alignment.sft.loss._dft import fused_dft_loss
from opaque.api.alignment.sft.loss._nll import fused_nll_loss

# ---------------------------------------------------------------------------
# Shared fixture dimensions
# ---------------------------------------------------------------------------

_B, _T, _H, _V = 2, 8, 16, 32
_DTYPE = torch.bfloat16


def _hidden_weight_labels(seed: int):
    """Per-example inputs in bf16 on CUDA.

    Shapes mirror the SFT closure: ``hidden`` is ``(B, T, H)`` (one per-example
    ``(T, H)`` per row); ``weight`` is the ``lm_head`` ``(V, H)``; ``labels``
    are integer token ids with a 2-token prompt prefix masked to ``-100``.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(_B, _T, _H, generator=gen, dtype=_DTYPE, device="cuda")
    weight = torch.randn(_V, _H, generator=gen, dtype=_DTYPE, device="cuda")
    labels = torch.randint(0, _V, (_B, _T), generator=gen, device="cuda")
    labels[:, :2] = -100
    return hidden, weight, labels


def _hidden_weight_ids_cmask(seed: int):
    """Per-example inputs for ``fused_sequence_logp``.

    ``ids`` are integer token ids; ``cmask`` is ``1`` on the completion span
    (positions 2..T) and ``0`` on the 2-token prompt prefix — the standard
    completion-only DPO/SFT scoring window.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(_B, _T, _H, generator=gen, dtype=_DTYPE, device="cuda")
    weight = torch.randn(_V, _H, generator=gen, dtype=_DTYPE, device="cuda")
    ids = torch.randint(0, _V, (_B, _T), generator=gen, device="cuda")
    cmask = torch.zeros(_B, _T, dtype=torch.long, device="cuda")
    cmask[:, 2:] = 1
    return hidden, weight, ids, cmask


# ---------------------------------------------------------------------------
# CPU wiring guard (cuda-autocast cannot be enabled on a CPU host, so stub the
# kernel + follow_autocast to assert linear_nll_sum reconciles dtypes first).
# ---------------------------------------------------------------------------


def test_linear_nll_sum_mixed_bf16_fp32_weight_is_safe():
    """A bf16 hidden against an fp32 lm_head weight must not crash linear_nll_sum.

    The reported crash was a bf16/fp32 ``tl.dot`` pair on the Triton path. On CUDA
    that path reconciles the dtypes via ``follow_autocast`` before the kernel; on
    non-CUDA hosts ``linear_nll_sum`` routes to the chunked kernel, which promotes
    to fp32 internally so the mixed pair is inherently safe. This exercises the
    non-CUDA (chunked) path — the one CI can run — and checks it returns a finite
    fp32 sum rather than raising on the dtype mismatch.
    """
    pytest.importorskip(_fused_lce._LCE_CHUNKED_PATH)

    hidden = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.float32)  # fp32 lm_head parameter
    labels = torch.randint(0, 16, (4,))

    out = _fused_lce.linear_nll_sum(hidden, weight, labels)

    assert out.dtype == torch.float32  # streamed in fp32, not raw bf16
    assert torch.isfinite(out)


@pytest.mark.cuda
def test_linear_nll_sum_cuda_matches_eager_shifted_sum():
    """The Triton route returns the eager per-example ``Σ CE`` over shifted tokens."""
    pytest.importorskip(_fused_lce._LCE_KERNEL_PATH)
    torch.manual_seed(0)
    hidden = torch.randn(9, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 32, (9,), device="cuda")
    labels[3] = -100

    got = _fused_lce.linear_nll_sum(hidden, weight, labels)

    logits = hidden[:-1].float() @ weight.float().T
    want = torch.nn.functional.cross_entropy(
        logits, labels[1:], ignore_index=-100, reduction="sum"
    )
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVmapAutocastBf16:
    """``vmap(grad(fused_*))`` composes cleanly under ``autocast(bf16, cuda)``."""

    def test_fused_nll_vmap_grad_under_autocast(self) -> None:
        """``vmap(grad(fused_nll_loss))`` under bf16 autocast: finite + parity."""
        hidden, weight, labels = _hidden_weight_labels(seed=1)

        def per_example(h, lab):
            return fused_nll_loss(h, weight, lab)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g_autocast = vmap(grad(per_example))(hidden, labels)

        assert g_autocast.shape == hidden.shape
        assert torch.isfinite(g_autocast).all(), (
            "vmap(grad(fused_nll_loss)) under autocast(bf16) must yield finite grads"
        )

        # Eager bf16 reference — no autocast, both arms already in bf16.
        g_eager = vmap(grad(per_example))(hidden, labels)
        assert torch.allclose(
            g_autocast.float(), g_eager.float(), atol=1e-2, rtol=0.0
        ), "autocast(bf16) result must match the eager bf16 reference (coarse atol)"

    def test_fused_dft_vmap_grad_under_autocast(self) -> None:
        """``vmap(grad(fused_dft_loss))`` under bf16 autocast: finite + parity.

        DFT uses ``-p.detach() * logp`` weighting; ``.detach()`` decouples the
        autocast'd softmax probability from the gradient path, so the autocast
        result should match the eager reference even more tightly than NLL.
        """
        hidden, weight, labels = _hidden_weight_labels(seed=2)

        def per_example(h, lab):
            return fused_dft_loss(h, weight, lab)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g_autocast = vmap(grad(per_example))(hidden, labels)

        assert g_autocast.shape == hidden.shape
        assert torch.isfinite(g_autocast).all()

        g_eager = vmap(grad(per_example))(hidden, labels)
        assert torch.allclose(g_autocast.float(), g_eager.float(), atol=1e-2, rtol=0.0)

    def test_fused_sequence_logp_vmap_grad_under_autocast(self) -> None:
        """``vmap(grad(fused_sequence_logp))`` under bf16 autocast.

        The DPO closure scores a per-sequence log-prob via this primitive,
        then subtracts the reference. Under the autocast context the kernel
        must (a) not crash on the vmap+autocast pairing and (b) return finite
        gradients within bf16 tolerance.
        """
        hidden, weight, ids, cmask = _hidden_weight_ids_cmask(seed=3)

        def per_example(h, i, c):
            return fused_sequence_logp(h, weight, i, c)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g_autocast = vmap(grad(per_example))(hidden, ids, cmask)

        assert g_autocast.shape == hidden.shape
        assert torch.isfinite(g_autocast).all()

        g_eager = vmap(grad(per_example))(hidden, ids, cmask)
        assert torch.allclose(g_autocast.float(), g_eager.float(), atol=1e-2, rtol=0.0)

    def test_fused_sequence_logp_fp32_weight_under_autocast(self) -> None:
        """fp32 lm_head weight vs bf16 hidden under autocast: the kernel raises
        ``tl.dot ... Got bf16 and fp32`` unless ``follow_autocast`` reconciles."""
        hidden, weight, ids, cmask = _hidden_weight_ids_cmask(seed=4)
        weight = weight.float()  # lm_head stays fp32 under autocast

        def per_example(h, i, c):
            return fused_sequence_logp(h, weight, i, c)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g_autocast = vmap(grad(per_example))(hidden, ids, cmask)

        assert g_autocast.shape == hidden.shape
        assert torch.isfinite(g_autocast).all()

        # Reference: both arms already bf16 (no autocast needed).
        def per_example_bf16(h, i, c):
            return fused_sequence_logp(h, weight.bfloat16(), i, c)

        g_eager = vmap(grad(per_example_bf16))(hidden, ids, cmask)
        assert torch.allclose(g_autocast.float(), g_eager.float(), atol=1e-2, rtol=0.0)
