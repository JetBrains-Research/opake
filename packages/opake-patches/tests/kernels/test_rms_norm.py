"""RMSNorm Triton kernel tests.

Tests:
1. Forward pass vs PyTorch reference (Llama + Gemma variants)
2. Backward pass vs PyTorch reference (Llama)
3. vmap forward: Triton vmap vs PyTorch vmap
4. vmap(grad): per-example gradients (DP-SGD path)
5. Forward, backward, vmap forward, and vmap(grad) performance benchmarks

Performance: opake must beat reference on time (within tolerance) or peak memory.

Config: Mellum-4b ``hidden_dim`` and sequence length via ``mellum_config``
(``conftest``), comparable to ``train_dpsgd.py --preset mellum-kstack`` geometry.
"""

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.rms_norm import Opake_RMSNorm, opake_rms_norm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RTOL_F = 0.05
ATOL_F = 0.05
RTOL_B = 2e-2
ATOL_B = 1e-3


def ref_llama_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * inv
    return (normed * w.float()).to(x.dtype)


def ref_gemma_rms(
    x: torch.Tensor, w: torch.Tensor, eps: float, offset: float
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * inv
    scale = w.float() + offset
    return (normed * scale).to(x.dtype)


def opake_llama(x, w, eps=1e-6):
    y, _, _ = Opake_RMSNorm.apply(x, w, eps, 0.0, "llama", False, None)
    return y


def opake_gemma(x, w, eps=1e-6, offset=1.0):
    y, _, _ = Opake_RMSNorm.apply(x, w, eps, offset, "gemma", False, None)
    return y


class TestRMSNormForward:
    def test_llama_forward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(0)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5
        y_o = opake_llama(x, w, eps)
        y_r = ref_llama_rms(x, w, eps)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="llama fwd")

    def test_gemma_forward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(1)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        off = 1.0
        y_o = opake_gemma(x, w, eps, off)
        y_r = ref_gemma_rms(x, w, eps, off)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="gemma fwd")

    def test_forward_reuses_rstd_and_contiguous_input(self):
        B, T, H = 2, 4, 64
        eps = 1e-5
        x = torch.randn(
            B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        y, rstd, x_saved = Opake_RMSNorm.apply(x, w, eps, 0.0, "llama", False, None)

        assert y.shape == x.shape
        assert rstd.shape == x.shape[:-1]
        assert not rstd.requires_grad
        assert not x_saved.requires_grad
        assert x_saved.data_ptr() == x.data_ptr()
        expected_rstd = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
        torch.testing.assert_close(rstd, expected_rstd, rtol=RTOL_B, atol=ATOL_B)
        assert isinstance(opake_rms_norm(x, w, eps), torch.Tensor)

    def test_noncontiguous_input_reuses_forward_materialization(self):
        B, T, H = 2, 4, 64
        eps = 1e-5
        x = (
            torch.randn(B, H, T, device="cuda", dtype=torch.bfloat16)
            .transpose(1, 2)
            .detach()
            .requires_grad_()
        )
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        w_ref = w.detach().clone().requires_grad_()

        y, _, x_saved = Opake_RMSNorm.apply(x, w, eps, 0.0, "llama", False, None)
        ref = ref_llama_rms(x_ref, w_ref, eps)
        y.sum().backward()
        ref.sum().backward()

        assert not x.is_contiguous()
        assert x_saved.is_contiguous()
        torch.testing.assert_close(y, ref, rtol=RTOL_F, atol=ATOL_F)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=RTOL_B, atol=ATOL_B)
        torch.testing.assert_close(w.grad, w_ref.grad, rtol=RTOL_B, atol=ATOL_B)

    def test_saved_input_preserves_version_checking(self):
        x = torch.randn(
            2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        y, _, _ = Opake_RMSNorm.apply(x, w, 1e-5, 0.0, "llama", False, None)

        with torch.no_grad():
            x.add_(1)

        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()


class TestRMSNormBackward:
    def test_llama_backward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(2)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        eps = 1e-5

        x0 = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w0 = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y0 = ref_llama_rms(x0, w0, eps)
        y0.mean().backward()

        x1 = x0.detach().clone().requires_grad_(True)
        w1 = w0.detach().clone().requires_grad_(True)
        y1 = opake_llama(x1, w1, eps)
        y1.mean().backward()

        assert_precision(x1.grad, x0.grad, rtol=RTOL_B, atol=ATOL_B, label="dx llama")
        assert_precision(w1.grad, w0.grad, rtol=RTOL_B, atol=ATOL_B, label="dw llama")

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    def test_frozen_weight_skips_dw_and_scalar_extraction(self, casting_mode):
        B, T, H = 2, 4, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0
        x = torch.randn(
            B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
        x_ref = x.detach().clone().requires_grad_()
        reference = ref_gemma_rms if casting_mode == "gemma" else ref_llama_rms
        ref = (
            reference(x_ref, w, eps, offset)
            if casting_mode == "gemma"
            else reference(x_ref, w, eps)
        )

        y, _, _ = Opake_RMSNorm.apply(x, w, eps, offset, casting_mode, False, None)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            torch.autograd.backward(y, torch.ones_like(y))
            torch.cuda.synchronize()
        ref.sum().backward()

        keys = {event.key for event in profiler.key_averages()}
        assert "aten::item" not in keys
        assert "aten::_local_scalar_dense" not in keys
        assert "aten::sum" not in keys
        assert w.grad is None
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=RTOL_B, atol=ATOL_B)


class TestRMSNormVmapForward:
    """vmap over microbatch dim: Triton vmap vs PyTorch vmap (Llama)."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def ref_call(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_call(inp, wt):
            return opake_llama(inp, wt, eps)

        y_pt = vmap(ref_call, in_dims=(0, None))(x, w)
        y_op = vmap(op_call, in_dims=(0, None))(x, w)

        print("\nvmap forward precision check:")
        assert_precision(y_op, y_pt, rtol=RTOL_F, atol=ATOL_F, label="vmap forward")

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def pt_fn(inp, wt):
            return vmap(lambda i, t: ref_llama_rms(i, t, eps), in_dims=(0, None))(
                inp, wt
            )

        def op_fn(inp, wt):
            return vmap(lambda i, t: opake_llama(i, t, eps), in_dims=(0, None))(inp, wt)

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm vmap forward")


class TestRMSNormVmapGradPerExampleDW:
    """Regression: with trainable W, _RMSNormBackward.vmap returned the
    batch-summed dW (out_dim=None) as every example's "per-example" gradient.
    """

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    @pytest.mark.parametrize("in_place", [False, True])
    def test_vmap_grad_matches_eager_loop(self, casting_mode, in_place):
        torch.manual_seed(7)
        B, T, H = 4, 8, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0

        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, wt):
            y, _, _ = Opake_RMSNorm.apply(
                xi, wt, eps, offset, casting_mode, in_place, None
            )
            return y.mean()

        # vmap path (the fix under test)
        gx_vmap, gw_vmap = vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))(x, w)

        # Eager per-sample loop (ground truth)
        gx_eager = torch.stack([grad(f, argnums=0)(x[i], w) for i in range(B)])
        gw_eager = torch.stack([grad(f, argnums=1)(x[i], w) for i in range(B)])

        torch.testing.assert_close(
            gx_vmap,
            gx_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dX mismatch ({casting_mode})",
        )
        torch.testing.assert_close(
            gw_vmap,
            gw_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dW mismatch ({casting_mode}) — "
            "vmap dW must not be the batch-sum",
        )

    def test_frozen_weight_vmap_skips_dw_kernel(self):
        B, T, H = 4, 8, 64
        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi):
            return opake_llama(xi, w).mean()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            gx_vmap = vmap(grad(f))(x)
            torch.cuda.synchronize()
        gx_eager = torch.stack([grad(f)(x[i]) for i in range(B)])

        keys = {event.key for event in profiler.key_averages()}
        assert not any("_rms_norm_weight_grad" in key for key in keys)
        torch.testing.assert_close(gx_vmap, gx_eager, rtol=RTOL_B, atol=ATOL_B)

    def test_trainable_dw_uses_triton_reduction_without_scalar_extraction(self):
        B, T, H = 4, 8, 64
        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, wt):
            y, _, _ = Opake_RMSNorm.apply(xi, wt, 1e-5, 0.0, "llama", False, None)
            return y.mean()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))(x, w)
            torch.cuda.synchronize()

        keys = {event.key for event in profiler.key_averages()}
        assert "aten::item" not in keys
        assert "aten::_local_scalar_dense" not in keys
        assert any("_rms_norm_weight_grad_partial_kernel" in key for key in keys)
        assert any("_rms_norm_weight_grad_reduce_kernel" in key for key in keys)


class TestRMSNormVmapGrad:
    """vmap(grad): per-example gradients (Llama)."""

    def test_vmap_grad_llama(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def f_pt(inp, wt):
            return ref_llama_rms(inp, wt, eps).mean()

        def f_op(inp, wt):
            return opake_llama(inp, wt, eps).mean()

        gx_pt, gw_pt = vmap(grad(f_pt, argnums=(0, 1)), in_dims=(0, None))(x, w)
        gx_op, gw_op = vmap(grad(f_op, argnums=(0, 1)), in_dims=(0, None))(x, w)

        print("\nvmap(grad) precision check:")
        assert_precision(gx_op, gx_pt, rtol=RTOL_B, atol=ATOL_B, label="vmap gx")
        assert_precision(gw_op, gw_pt, rtol=RTOL_B, atol=ATOL_B, label="vmap gw")

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def make_pt_fn():
            def f(inp, wt):
                return ref_llama_rms(inp, wt, eps).mean()

            return vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))

        def make_op_fn():
            def f(inp, wt):
                return opake_llama(inp, wt, eps).mean()

            return vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))

        pt_stats = measure_time_and_memory(make_pt_fn(), x, w)
        op_stats = measure_time_and_memory(make_op_fn(), x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm vmap(grad)")


class TestRMSNormPerformance:
    """Benchmark forward-only and forward+backward vs PyTorch reference."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5

        def pt_fn(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_fn(inp, wt):
            return opake_llama(inp, wt, eps)

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm forward")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        eps = 1e-5

        def pt_fn(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_fn(inp, wt):
            return opake_llama(inp, wt, eps)

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm backward")


class TestRMSNormBlockKernel:
    """Force the block-kernel forward path: small hidden dim (head_dim-like) plus
    many rows (e.g. QK-norm under vmapped per-example gradients). Forward output
    and backward grads must match eager at this shape.
    """

    # n_rows (40960) >= 4096*8 and BLOCK_SIZE (=128) <= 256 -> use_block is True.
    _DIM = 128
    _ROWS = 40 * 1024

    def test_block_llama_forward(self, assert_precision):
        torch.manual_seed(3)
        x = torch.randn(self._ROWS, self._DIM, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(self._DIM, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5
        y_o = opake_llama(x, w, eps)
        y_r = ref_llama_rms(x, w, eps)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="block llama fwd")

    def test_block_gemma_forward(self, assert_precision):
        torch.manual_seed(4)
        x = torch.randn(self._ROWS, self._DIM, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(self._DIM, device="cuda", dtype=torch.bfloat16)
        eps, off = 1e-6, 1.0
        y_o = opake_gemma(x, w, eps, off)
        y_r = ref_gemma_rms(x, w, eps, off)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="block gemma fwd")

    def test_block_backward(self, assert_precision):
        # Forward hits the block kernel; the block backward must still match the
        # eager grads at this shape.
        torch.manual_seed(6)
        eps = 1e-5
        x0 = torch.randn(
            self._ROWS,
            self._DIM,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        w0 = torch.randn(
            self._DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ref_llama_rms(x0, w0, eps).mean().backward()

        x1 = x0.detach().clone().requires_grad_(True)
        w1 = w0.detach().clone().requires_grad_(True)
        opake_llama(x1, w1, eps).mean().backward()

        assert_precision(x1.grad, x0.grad, rtol=RTOL_B, atol=ATOL_B, label="block dx")
        assert_precision(w1.grad, w0.grad, rtol=RTOL_B, atol=ATOL_B, label="block dw")
