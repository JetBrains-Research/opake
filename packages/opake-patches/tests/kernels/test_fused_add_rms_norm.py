"""Fused residual + RMSNorm kernel tests (forward, backward, vmap, vmap(grad)).

Shapes use ``mellum_config`` (Mellum-4b–style tensors from ``conftest``).
"""

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.fused_add_rms_norm import Opake_FusedAddRMSNorm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RTOL_F = 0.05
ATOL_F = 0.05
RTOL_B = 2e-2
ATOL_B = 1e-3


def ref_llama_fused(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    rf = r.float()
    s = xf + rf
    inv = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    normed = s * inv
    y = (normed * w.float()).to(x.dtype)
    s_bf = s.to(x.dtype)
    return y, s_bf


def ref_gemma_fused(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    rf = r.float()
    s = xf + rf
    inv = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    y = (s * inv * (1 + w.float())).to(x.dtype)
    return y, s.to(x.dtype)


def opake_llama(x, r, w, eps=1e-5):
    y, s, _ = Opake_FusedAddRMSNorm.apply(x, r, w, eps, 0.0, "llama", False)
    return y, s


class TestFusedAddRMSNormForward:
    def test_llama_forward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(0)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5
        y_o, s_o = opake_llama(x, r, w, eps)
        y_e, s_e = ref_llama_fused(x, r, w, eps)
        assert_precision(s_o, s_e, rtol=RTOL_F, atol=ATOL_F, label="sum S")
        assert_precision(y_o, y_e, rtol=RTOL_F, atol=ATOL_F, label="fused y")


class TestFusedAddRMSNormBackward:
    def test_llama_backward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(2)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        eps = 1e-5

        x0 = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        r0 = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w0 = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y_ref, s_ref = ref_llama_fused(x0, r0, w0, eps)
        (y_ref.mean() + s_ref.mean()).backward()

        x1 = x0.detach().clone().requires_grad_(True)
        r1 = r0.detach().clone().requires_grad_(True)
        w1 = w0.detach().clone().requires_grad_(True)
        y_op, s_op = opake_llama(x1, r1, w1, eps)
        (y_op.mean() + s_op.mean()).backward()

        assert_precision(x1.grad, x0.grad, rtol=RTOL_B, atol=ATOL_B, label="dx")
        assert_precision(r1.grad, r0.grad, rtol=RTOL_B, atol=ATOL_B, label="dr")
        assert_precision(w1.grad, w0.grad, rtol=RTOL_B, atol=ATOL_B, label="dw")

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    @pytest.mark.parametrize("in_place", [False, True])
    def test_eager_backward_reuses_forward_rstd(self, casting_mode, in_place):
        torch.manual_seed(11)
        B, T, H = 2, 4, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0
        reference = ref_gemma_fused if casting_mode == "gemma" else ref_llama_fused

        x0 = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16).requires_grad_()
        r0 = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16).requires_grad_()
        w0 = torch.randn(H, device="cuda", dtype=torch.bfloat16).requires_grad_()
        y_ref, s_ref = reference(x0, r0, w0, eps)
        (y_ref.mean() + s_ref.mean()).backward()

        x1 = x0.detach().clone().requires_grad_(True)
        r1 = r0.detach().clone().requires_grad_(True)
        w1 = w0.detach().clone().requires_grad_(True)
        y_op, s_op, rstd = Opake_FusedAddRMSNorm.apply(
            x1, r1, w1, eps, offset, casting_mode, in_place
        )
        assert rstd.shape == (B, T)
        assert not rstd.requires_grad
        (y_op.mean() + s_op.mean()).backward()

        torch.testing.assert_close(x1.grad, x0.grad, rtol=RTOL_B, atol=ATOL_B)
        torch.testing.assert_close(r1.grad, r0.grad, rtol=RTOL_B, atol=ATOL_B)
        torch.testing.assert_close(w1.grad, w0.grad, rtol=RTOL_B, atol=ATOL_B)

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    def test_frozen_weight_skips_dw_and_scalar_extraction(self, casting_mode):
        B, T, H = 2, 4, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0
        x = torch.randn(
            B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        r = torch.randn(
            B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
        x_ref = x.detach().clone().requires_grad_()
        r_ref = r.detach().clone().requires_grad_()
        reference = ref_gemma_fused if casting_mode == "gemma" else ref_llama_fused
        y_ref, s_ref = reference(x_ref, r_ref, w, eps)

        y, s, _ = Opake_FusedAddRMSNorm.apply(x, r, w, eps, offset, casting_mode, False)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            torch.autograd.backward((y, s), (torch.ones_like(y), torch.ones_like(s)))
            torch.cuda.synchronize()
        (y_ref + s_ref).sum().backward()

        keys = {event.key for event in profiler.key_averages()}
        assert "aten::item" not in keys
        assert "aten::_local_scalar_dense" not in keys
        assert "aten::sum" not in keys
        assert w.grad is None
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=RTOL_B, atol=ATOL_B)
        torch.testing.assert_close(r.grad, r_ref.grad, rtol=RTOL_B, atol=ATOL_B)


class TestFusedAddRMSNormVmapForward:
    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5
        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def ref_c(a, b_):
            return ref_llama_fused(a, b_, w, eps)

        def op_c(a, b_):
            return Opake_FusedAddRMSNorm.apply(a, b_, w, eps, 0.0, "llama", False)

        y_pt, s_pt = vmap(ref_c)(x, r)
        y_op, s_op, rstd_op = vmap(op_c)(x, r)
        assert rstd_op.shape == x.shape[:-1]
        print("\nvmap forward precision check:")
        assert_precision(s_op, s_pt, rtol=RTOL_F, atol=ATOL_F, label="vmap S")
        assert_precision(y_op, y_pt, rtol=RTOL_F, atol=ATOL_F, label="vmap y")

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5
        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def pt_fn(a, b_):
            return vmap(lambda u, v: ref_llama_fused(u, v, w, eps)[0])(a, b_)

        def op_fn(a, b_):
            return vmap(lambda u, v: opake_llama(u, v, w, eps)[0])(a, b_)

        pt_stats = measure_time_and_memory(pt_fn, x, r)
        op_stats = measure_time_and_memory(op_fn, x, r)
        assert_perf_benefit(pt_stats, op_stats, label="fused add rms vmap forward")


class TestFusedAddRMSNormVmapGradPerExampleDW:
    """Regression: same batch-summed dW bug as plain RMSNorm, fused-add variant."""

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    @pytest.mark.parametrize("in_place", [False, True])
    def test_vmap_grad_matches_eager_loop(self, casting_mode, in_place):
        torch.manual_seed(13)
        B, T, H = 4, 8, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0

        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, ri, wt):
            y, s_, _ = Opake_FusedAddRMSNorm.apply(
                xi, ri, wt, eps, offset, casting_mode, in_place
            )
            return (y + s_).mean()

        # vmap path (the fix under test) — batch over (x, r), w is shared
        gx_vmap, gr_vmap, gw_vmap = vmap(
            grad(f, argnums=(0, 1, 2)), in_dims=(0, 0, None)
        )(x, r, w)

        # Eager per-sample loop (ground truth)
        gx_eager = torch.stack([grad(f, argnums=0)(x[i], r[i], w) for i in range(B)])
        gr_eager = torch.stack([grad(f, argnums=1)(x[i], r[i], w) for i in range(B)])
        gw_eager = torch.stack([grad(f, argnums=2)(x[i], r[i], w) for i in range(B)])

        torch.testing.assert_close(
            gx_vmap,
            gx_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dX mismatch ({casting_mode})",
        )
        torch.testing.assert_close(
            gr_vmap,
            gr_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dR mismatch ({casting_mode})",
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
        r = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, ri):
            y, s = opake_llama(xi, ri, w)
            return (y + s).mean()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            gx_vmap, gr_vmap = vmap(grad(f, argnums=(0, 1)))(x, r)
            torch.cuda.synchronize()
        gx_eager = torch.stack([grad(f, argnums=0)(x[i], r[i]) for i in range(B)])
        gr_eager = torch.stack([grad(f, argnums=1)(x[i], r[i]) for i in range(B)])

        keys = {event.key for event in profiler.key_averages()}
        assert not any("_rms_norm_weight_grad" in key for key in keys)
        torch.testing.assert_close(gx_vmap, gx_eager, rtol=RTOL_B, atol=ATOL_B)
        torch.testing.assert_close(gr_vmap, gr_eager, rtol=RTOL_B, atol=ATOL_B)

    def test_trainable_dw_uses_triton_reduction_without_scalar_extraction(self):
        B, T, H = 4, 8, 64
        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, ri, wt):
            y, s, _ = Opake_FusedAddRMSNorm.apply(xi, ri, wt, 1e-5, 0.0, "llama", False)
            return (y + s).mean()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            vmap(grad(f, argnums=(0, 1, 2)), in_dims=(0, 0, None))(x, r, w)
            torch.cuda.synchronize()

        keys = {event.key for event in profiler.key_averages()}
        assert "aten::item" not in keys
        assert "aten::_local_scalar_dense" not in keys
        assert any("_rms_norm_weight_grad_partial_kernel" in key for key in keys)
        assert any("_rms_norm_weight_grad_reduce_kernel" in key for key in keys)


class TestFusedAddRMSNormVmapGrad:
    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5
        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def f_ref(a, b_):
            y, s_ = ref_llama_fused(a, b_, w, eps)
            return (y + s_).mean()

        def f_op(a, b_):
            y, s_ = opake_llama(a, b_, w, eps)
            return (y + s_).mean()

        gx_pt, gr_pt = vmap(grad(f_ref, argnums=(0, 1)))(x, r)
        gx_op, gr_op = vmap(grad(f_op, argnums=(0, 1)))(x, r)
        print("\nvmap(grad) precision check:")
        assert_precision(gx_op, gx_pt, rtol=RTOL_B, atol=ATOL_B, label="vgx")
        assert_precision(gr_op, gr_pt, rtol=RTOL_B, atol=ATOL_B, label="vgr")

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5
        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def make_pt():
            def f(a, b_):
                y, s_ = ref_llama_fused(a, b_, w, eps)
                return (y + s_).mean()

            return vmap(grad(f, argnums=(0, 1)))

        def make_op():
            def f(a, b_):
                y, s_ = opake_llama(a, b_, w, eps)
                return (y + s_).mean()

            return vmap(grad(f, argnums=(0, 1)))

        pt_stats = measure_time_and_memory(make_pt(), x, r)
        op_stats = measure_time_and_memory(make_op(), x, r)
        assert_perf_benefit(pt_stats, op_stats, label="fused add rms vmap(grad)")


class TestFusedAddRMSNormPerformance:
    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        eps = 1e-5
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def pt_fn(a, b_):
            return ref_llama_fused(a, b_, w, eps)[0]

        def op_fn(a, b_):
            return opake_llama(a, b_, w, eps)[0]

        pt_stats = measure_time_and_memory(pt_fn, x, r)
        op_stats = measure_time_and_memory(op_fn, x, r)
        assert_perf_benefit(pt_stats, op_stats, label="fused add rms forward (y only)")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        eps = 1e-5
        x = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        r = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def pt_fn(a, b_, wt):
            y, s_ = ref_llama_fused(a, b_, wt, eps)
            return y + s_

        def op_fn(a, b_, wt):
            y, s_ = opake_llama(a, b_, wt, eps)
            return y + s_

        pt_stats = measure_time_and_memory(pt_fn, x, r, w)
        op_stats = measure_time_and_memory(op_fn, x, r, w)
        assert_perf_benefit(pt_stats, op_stats, label="fused add rms backward")
