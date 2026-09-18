"""
RoPE (Rotary Position Embedding) Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-4b scale (uses mellum_config from conftest)
"""

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.rope_embedding import Opake_RoPE, Opake_RoPE_QK

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# RoPE: elementwise cos/sin multiply, same math in Triton and PyTorch
RTOL_FORWARD = 8e-3
ATOL_FORWARD = 4e-2
RTOL_BACKWARD = 1e-2
ATOL_BACKWARD = 5e-8


# ============================================================================
# Helpers
# ============================================================================


def rotate_half(x):
    """Standard rotate_half for RoPE."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def generate_cos_sin(seq_len, head_dim, device="cuda", dtype=torch.bfloat16):
    """Generate cos/sin caches for RoPE."""
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device)
    freqs = torch.outer(positions, freqs)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def pytorch_rope(Q, cos, sin):
    """PyTorch reference RoPE implementation."""
    batch, seq_len, n_heads, _head_dim = Q.shape
    cos_expanded = cos[None, :, None, :].expand(batch, seq_len, n_heads, -1)
    sin_expanded = sin[None, :, None, :].expand(batch, seq_len, n_heads, -1)
    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)
    return Q * cos_full + rotate_half(Q) * sin_full


def opake_rope(Q, cos, sin):
    """Opake Triton kernel (Opake_RoPE)."""
    return Opake_RoPE.apply(Q, cos, sin)


def pytorch_rope_qk(Q, K, cos, sin, rope_indices=None):
    """PyTorch reference for fused Q/K RoPE inputs in attention layout."""
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)
    if rope_indices is None:
        cos_full = cos_full[None, None]
        sin_full = sin_full[None, None]
    else:
        cos_full = cos_full[rope_indices].unsqueeze(-3)
        sin_full = sin_full[rope_indices].unsqueeze(-3)
    return (
        Q * cos_full + rotate_half(Q) * sin_full,
        K * cos_full + rotate_half(K) * sin_full,
    )


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestRoPEForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opake vs pytorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        Q = torch.randn(
            batch, seq_len, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)

        out_opake = opake_rope(Q, cos, sin)
        out_pytorch = pytorch_rope(Q, cos, sin)

        print("\nRoPE Forward:")
        assert_precision(
            out_opake,
            out_pytorch,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="output",
        )


# ============================================================================
# Backward Pass Tests
# ============================================================================


class TestRoPEBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opake vs pytorch Q.grad."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        Q_pt = torch.randn(
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)
        out_pt = pytorch_rope(Q_pt, cos, sin)
        out_pt.mean().backward()

        Q_op = Q_pt.detach().clone().requires_grad_(True)
        out_op = opake_rope(Q_op, cos, sin)
        out_op.mean().backward()

        print("\nRoPE Backward:")
        assert_precision(
            Q_op.grad, Q_pt.grad, rtol=RTOL_BACKWARD, atol=ATOL_BACKWARD, label="Q.grad"
        )


class TestRoPEQK:
    """Test fused Q/K RoPE with grouped-query attention layouts."""

    def test_forward_backward_and_vmap_match_pytorch(self):
        torch.manual_seed(7)
        batch, seq_len, n_heads_q, n_heads_k, head_dim, vmap_batch = 2, 16, 8, 2, 32, 3
        cos, sin = generate_cos_sin(seq_len, head_dim, dtype=torch.float32)

        Q_ref = torch.randn(
            batch, n_heads_q, seq_len, head_dim, device="cuda", requires_grad=True
        )
        K_ref = torch.randn(
            batch, n_heads_k, seq_len, head_dim, device="cuda", requires_grad=True
        )
        Q_op = Q_ref.detach().clone().requires_grad_(True)
        K_op = K_ref.detach().clone().requires_grad_(True)
        Q_before = Q_op.detach().clone()
        K_before = K_op.detach().clone()

        out_ref = pytorch_rope_qk(Q_ref, K_ref, cos, sin)
        out_op = Opake_RoPE_QK.apply(Q_op, K_op, cos, sin, None)
        torch.testing.assert_close(out_op[0], out_ref[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_op[1], out_ref[1], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(Q_op, Q_before, rtol=0, atol=0)
        torch.testing.assert_close(K_op, K_before, rtol=0, atol=0)

        (out_ref[0].square().mean() + out_ref[1].square().mean()).backward()
        (out_op[0].square().mean() + out_op[1].square().mean()).backward()
        torch.testing.assert_close(Q_op.grad, Q_ref.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(K_op.grad, K_ref.grad, rtol=1e-6, atol=1e-6)

        Q = torch.randn(vmap_batch, batch, n_heads_q, seq_len, head_dim, device="cuda")
        K = torch.randn(vmap_batch, batch, n_heads_k, seq_len, head_dim, device="cuda")
        out_vmap = vmap(lambda q, k: Opake_RoPE_QK.apply(q, k, cos, sin, None))(Q, K)
        out_reference = vmap(lambda q, k: pytorch_rope_qk(q, k, cos, sin))(Q, K)
        torch.testing.assert_close(out_vmap[0], out_reference[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_vmap[1], out_reference[1], rtol=1e-6, atol=1e-6)

        def loss_opake(q, k):
            q_rot, k_rot = Opake_RoPE_QK.apply(q, k, cos, sin, None)
            return q_rot.square().mean() + k_rot.square().mean()

        def loss_reference(q, k):
            q_rot, k_rot = pytorch_rope_qk(q, k, cos, sin)
            return q_rot.square().mean() + k_rot.square().mean()

        grads_vmap = vmap(grad(loss_opake, argnums=(0, 1)))(Q, K)
        grads_reference = vmap(grad(loss_reference, argnums=(0, 1)))(Q, K)
        torch.testing.assert_close(
            grads_vmap[0], grads_reference[0], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            grads_vmap[1], grads_reference[1], rtol=1e-6, atol=1e-6
        )

    def test_vmap_shared_rope_indices_match_pytorch(self):
        torch.manual_seed(8)
        batch, seq_len, n_heads_q, n_heads_k, head_dim, vmap_batch = 2, 16, 8, 2, 32, 3
        cos, sin = generate_cos_sin(seq_len * 2, head_dim, dtype=torch.float32)
        rope_indices = torch.stack(
            [torch.randperm(seq_len * 2, device="cuda")[:seq_len] for _ in range(batch)]
        )
        Q = torch.randn(vmap_batch, batch, n_heads_q, seq_len, head_dim, device="cuda")
        K = torch.randn(vmap_batch, batch, n_heads_k, seq_len, head_dim, device="cuda")

        out_vmap = vmap(lambda q, k: Opake_RoPE_QK.apply(q, k, cos, sin, rope_indices))(
            Q, K
        )
        out_reference = vmap(
            lambda q, k: pytorch_rope_qk(q, k, cos, sin, rope_indices)
        )(Q, K)
        torch.testing.assert_close(out_vmap[0], out_reference[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_vmap[1], out_reference[1], rtol=1e-6, atol=1e-6)

        def loss_opake(q, k):
            q_rot, k_rot = Opake_RoPE_QK.apply(q, k, cos, sin, rope_indices)
            return q_rot.square().mean() + k_rot.square().mean()

        def loss_reference(q, k):
            q_rot, k_rot = pytorch_rope_qk(q, k, cos, sin, rope_indices)
            return q_rot.square().mean() + k_rot.square().mean()

        grads_vmap = vmap(grad(loss_opake, argnums=(0, 1)))(Q, K)
        grads_reference = vmap(grad(loss_reference, argnums=(0, 1)))(Q, K)
        torch.testing.assert_close(
            grads_vmap[0], grads_reference[0], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            grads_vmap[1], grads_reference[1], rtol=1e-6, atol=1e-6
        )


class TestSlowRoPEPositionIds:
    """Regression for the Opake_SlowRoPE position_ids path: forward unsqueezed
    the indexed caches at the wrong dim and backward ignored position_ids.
    """

    def test_forward_backward_match_reference(self):
        from opake.api.patches.kernels.rope_embedding import opake_slow_rope

        torch.manual_seed(0)
        batch, n_heads, seq_len, head_dim = 2, 4, 16, 32
        Q_ref = torch.randn(
            batch,
            n_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        half_cos, half_sin = generate_cos_sin(
            seq_len * 2, head_dim, dtype=torch.float32
        )
        cos = torch.cat([half_cos, half_cos], dim=-1)  # full-dim HF-style cache
        sin = torch.cat([half_sin, half_sin], dim=-1)
        # Non-trivial positions so cache indexing actually matters
        position_ids = torch.stack(
            [torch.randperm(seq_len * 2, device="cuda")[:seq_len] for _ in range(batch)]
        )

        cos_sel = cos[position_ids].unsqueeze(1)  # (batch, 1, seq, dim)
        sin_sel = sin[position_ids].unsqueeze(1)
        out_ref = Q_ref * cos_sel + rotate_half(Q_ref) * sin_sel
        out_ref.square().mean().backward()

        Q_op = Q_ref.detach().clone().requires_grad_(True)
        out_op = opake_slow_rope(Q_op, cos, sin, position_ids=position_ids)
        out_op.square().mean().backward()

        torch.testing.assert_close(out_op, out_ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(Q_op.grad, Q_ref.grad, rtol=1e-6, atol=1e-6)

    def test_vmap_grad_with_batched_position_ids(self):
        """vmap(grad) with per-example position_ids matches the eager loop."""
        from opake.api.patches.kernels.rope_embedding import opake_slow_rope

        torch.manual_seed(1)
        B, n_heads, seq_len, head_dim = 3, 4, 16, 32
        Q = torch.randn(B, n_heads, seq_len, head_dim, device="cuda")
        half_cos, half_sin = generate_cos_sin(
            seq_len * 2, head_dim, dtype=torch.float32
        )
        cos = torch.cat([half_cos, half_cos], dim=-1)
        sin = torch.cat([half_sin, half_sin], dim=-1)
        position_ids = torch.stack(
            [torch.randperm(seq_len * 2, device="cuda")[:seq_len] for _ in range(B)]
        )

        def loss(q, pos):
            # add the per-example batch dim back: (1, heads, seq, dim)
            out = opake_slow_rope(
                q.unsqueeze(0), cos, sin, position_ids=pos.unsqueeze(0)
            )
            return out.square().mean()

        g_vmap = vmap(grad(loss))(Q, position_ids)
        g_loop = torch.stack([grad(loss)(Q[i], position_ids[i]) for i in range(B)])
        torch.testing.assert_close(g_vmap, g_loop, rtol=1e-6, atol=1e-6)


# ============================================================================
# Vmap Tests
# ============================================================================


class TestRoPEVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opake Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)

        Q = torch.randn(
            VMAP_BATCH,
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

        out_pt = vmap(lambda q: pytorch_rope(q, cos, sin))(Q)
        out_op = vmap(lambda q: opake_rope(q, cos, sin))(Q)

        print("\nRoPE vmap forward:")
        assert_precision(
            out_op, out_pt, rtol=RTOL_FORWARD, atol=ATOL_FORWARD, label="output"
        )

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap forward must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)

        Q = torch.randn(
            VMAP_BATCH,
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

        pt_stats = measure_time_and_memory(
            lambda q: vmap(lambda qi: pytorch_rope(qi, cos, sin))(q), Q
        )
        op_stats = measure_time_and_memory(
            lambda q: vmap(lambda qi: opake_rope(qi, cos, sin))(q), Q
        )

        assert_perf_benefit(pt_stats, op_stats, label="RoPE vmap forward")


class TestRoPEVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opake Triton vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)

        Q = torch.randn(
            VMAP_BATCH,
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def f_pt(q):
            return pytorch_rope(q, cos, sin).mean()

        def f_op(q):
            return opake_rope(q, cos, sin).mean()

        grads_pt = vmap(grad(f_pt))(Q)
        grads_op = vmap(grad(f_op))(Q)

        print("\nRoPE vmap(grad):")
        assert_precision(
            grads_op, grads_pt, rtol=RTOL_BACKWARD, atol=ATOL_BACKWARD, label="Q.grad"
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)

        Q = torch.randn(
            VMAP_BATCH,
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def make_pt_fn():
            def f(q):
                return pytorch_rope(q, cos, sin).mean()

            return vmap(grad(f))

        def make_op_fn():
            def f(q):
                return opake_rope(q, cos, sin).mean()

            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), Q)
        op_stats = measure_time_and_memory(make_op_fn(), Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE vmap(grad)")


# ============================================================================
# Performance Tests
# ============================================================================


class TestRoPEPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward-only: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)
        Q = torch.randn(
            batch, seq_len, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )

        def pytorch_fn(q):
            return pytorch_rope(q, cos, sin)

        def opake_fn(q):
            return opake_rope(q, cos, sin)

        pt_stats = measure_time_and_memory(pytorch_fn, Q)
        op_stats = measure_time_and_memory(opake_fn, Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE forward")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward+backward: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        cos, sin = generate_cos_sin(seq_len, HEAD_DIM)
        Q = torch.randn(
            batch,
            seq_len,
            N_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        def pytorch_fn(q):
            return pytorch_rope(q, cos, sin)

        def opake_fn(q):
            return opake_rope(q, cos, sin)

        pt_stats = measure_time_and_memory(pytorch_fn, Q)
        op_stats = measure_time_and_memory(opake_fn, Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
