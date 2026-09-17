# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for vmap-compatible save_on_cpu (Patch 5 in _checkpoint_patches).

Verifies that save_on_cpu produces correct clipped gradients and actually
offloads tensors to CPU when used with Opake's clipped_grad pipeline.
"""

import pytest
import torch

from opake.patches import apply_runtime_patches

# Saved-tensor hooks (used by save_on_cpu) are blocked inside torch.func.grad
# upstream; the runtime patch removes that restriction.
apply_runtime_patches(vmap_checkpointing=True)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="save_on_cpu requires CUDA"
    ),
]


class TestSaveOnCpuWithClippedGrad:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_clipped_grad_correctness(self, dtype):
        """save_on_cpu produces identical clipped gradients."""
        from opake.api.engine.clipping import clipped_grad

        W = torch.randn(32, 32, device="cuda", dtype=dtype)

        def loss_fn(w, x):
            return (x @ w).relu().sum()

        grad_fn, state = clipped_grad(
            loss_fn, argnums=0, batch_argnums=(1,), clipping_norm=1.0
        )

        x = torch.randn(4, 32, device="cuda", dtype=dtype)
        grads_vanilla, _ = grad_fn(W, x, state=state)

        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            grads_offload, _ = grad_fn(W, x, state=state)

        torch.testing.assert_close(grads_offload.pytree, grads_vanilla.pytree)

    def test_clipped_grad_with_checkpoint(self):
        """save_on_cpu + checkpoint produces correct clipped gradients."""
        from torch.utils.checkpoint import checkpoint

        from opake.api.engine.clipping import clipped_grad

        W1 = torch.randn(32, 32, device="cuda")
        W2 = torch.randn(32, 32, device="cuda")

        def block(h, w):
            return (h @ w).relu()

        def loss_fn(w1, w2, x):
            h = checkpoint(block, x, w1, use_reentrant=False)
            h = checkpoint(block, h, w2, use_reentrant=False)
            return h.sum()

        grad_fn, state = clipped_grad(
            loss_fn, argnums=(0, 1), batch_argnums=(2,), clipping_norm=1.0
        )

        x = torch.randn(4, 32, device="cuda")
        grads_vanilla, _ = grad_fn(W1, W2, x, state=state)

        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            grads_offload, _ = grad_fn(W1, W2, x, state=state)

        for g_v, g_o in zip(grads_vanilla.pytree, grads_offload.pytree, strict=False):
            torch.testing.assert_close(g_o, g_v)

    def test_no_pin_memory(self):
        """save_on_cpu(pin_memory=False) path works with clipped_grad."""
        from opake.api.engine.clipping import clipped_grad

        W = torch.randn(32, 32, device="cuda")

        def loss_fn(w, x):
            return (x @ w).relu().sum()

        grad_fn, state = clipped_grad(
            loss_fn, argnums=0, batch_argnums=(1,), clipping_norm=1.0
        )

        x = torch.randn(4, 32, device="cuda")
        grads_vanilla, _ = grad_fn(W, x, state=state)

        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            grads_offload, _ = grad_fn(W, x, state=state)

        torch.testing.assert_close(grads_offload.pytree, grads_vanilla.pytree)

    def test_offload_reduces_gpu_memory(self):
        """save_on_cpu moves saved activations to CPU, reducing GPU peak.

        Uses an activation-dominated workload (small weights, long sequences,
        many layers) to isolate the offloading effect. On real models, savings
        depend on the activation-to-weight ratio and microbatch size — see
        benchmarks in docs/user-guide/memory-optimizations.md.
        """
        import gc

        from opake.api.engine.clipping import clipped_grad

        d, layers = 64, 32
        Ws = [torch.randn(d, d, device="cuda") for _ in range(layers)]

        def loss_fn(ws, x):
            h = x
            for w in ws:
                h = (h @ w).relu()
            return h.sum()

        grad_fn, state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1,),
            clipping_norm=1.0,
            microbatch_size=None,
        )
        x = torch.randn(8, 4096, d, device="cuda")

        grad_fn(Ws, x, state=state)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        grad_fn(Ws, x, state=state)
        peak_vanilla = torch.cuda.max_memory_allocated()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            grad_fn(Ws, x, state=state)
        peak_offload = torch.cuda.max_memory_allocated()

        savings = (peak_vanilla - peak_offload) / peak_vanilla
        assert savings > 0.3, (
            f"Expected >30% GPU memory savings, got {savings:.1%} "
            f"(vanilla={peak_vanilla / 1e6:.0f}MB, "
            f"offload={peak_offload / 1e6:.0f}MB)"
        )
