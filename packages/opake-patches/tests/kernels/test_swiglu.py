"""
SwiGLU Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap forward: Triton vmap vs PyTorch vmap
4. vmap(grad): per-example gradients — the DP-SGD path
5. Forward and backward performance benchmarks

Target precision (bfloat16):
- norm_err: max |a - b| / max(|b|, threshold) < rtol
  where threshold filters out near-zero values that inflate relative errors
- Performance: speedup > 1.0x OR memory reduction > 1.0x for vmap

Config: Mellum-4b scale (intermediate_dim=8256)
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.swiglu import Opake_SwiGLU

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RTOL_FORWARD = 1e-5
ATOL_FORWARD = 1e-5
RTOL_BACKWARD = 1e-2
ATOL_BACKWARD = 5e-8


def pytorch_swiglu(gate, up):
    """PyTorch reference: SwiGLU = silu(gate) * up."""
    return F.silu(gate) * up


def opake_swiglu(gate, up):
    """Opake Triton kernel."""
    return Opake_SwiGLU.apply(gate, up)


class TestSwiGLUForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opake vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        out_pytorch = pytorch_swiglu(gate, up)
        out_opake = opake_swiglu(gate, up)

        print("\nForward precision check:")
        assert_precision(
            out_opake,
            out_pytorch,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="forward",
        )


class TestSwiGLUBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opake vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        # PyTorch reference
        gate_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out_pt = pytorch_swiglu(gate_pt, up_pt)
        out_pt.mean().backward()

        # Opake kernel
        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opake_swiglu(gate_op, up_op)
        out_op.mean().backward()

        print("\nBackward precision check:")
        assert_precision(
            gate_op.grad,
            gate_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate.grad",
        )
        assert_precision(
            up_op.grad,
            up_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up.grad",
        )


class TestSwiGLUVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opake Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        out_pt = vmap(pytorch_swiglu)(gate, up)
        out_op = vmap(opake_swiglu)(gate, up)

        print("\nvmap forward precision check:")
        assert_precision(
            out_op, out_pt, rtol=RTOL_FORWARD, atol=ATOL_FORWARD, label="vmap forward"
        )

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap forward must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        pt_stats = measure_time_and_memory(
            lambda g, u: vmap(pytorch_swiglu)(g, u), gate, up
        )
        op_stats = measure_time_and_memory(
            lambda g, u: vmap(opake_swiglu)(g, u), gate, up
        )

        assert_perf_benefit(pt_stats, op_stats, label="vmap forward")


class TestSwiGLUVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path.

    Exercises both Opake_SwiGLU.vmap() (forward) and
    _SwiGLUBackward.vmap() (backward) with Triton kernels.
    """

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opake Triton vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def f_pt(g, u):
            return pytorch_swiglu(g, u).mean()

        def f_op(g, u):
            return opake_swiglu(g, u).mean()

        grads_pt_gate, grads_pt_up = vmap(grad(f_pt, argnums=(0, 1)))(gate, up)
        grads_op_gate, grads_op_up = vmap(grad(f_op, argnums=(0, 1)))(gate, up)

        print("\nvmap(grad) precision check:")
        assert_precision(
            grads_op_gate,
            grads_pt_gate,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="vmap(grad) gate",
        )
        assert_precision(
            grads_op_up,
            grads_pt_up,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="vmap(grad) up",
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def make_pt_fn():
            def f(g, u):
                return pytorch_swiglu(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        def make_op_fn():
            def f(g, u):
                return opake_swiglu(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        pt_stats = measure_time_and_memory(make_pt_fn(), gate, up)
        op_stats = measure_time_and_memory(make_op_fn(), gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="vmap(grad)")


class TestSwiGLUPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward-only: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opake_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="forward")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward+backward: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opake_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="backward")


class TestSwiGLURepeatedBackward:
    """A retained graph must be refused on the second traversal, not silently wrong.

    The Triton backward writes grad_gate/grad_up over the saved gate/up
    buffers, so a second walk would read overwritten activations. Triton
    stores bypass the autograd version counter, so nothing else would notice.
    """

    @pytest.mark.parametrize("fn", [Opake_SwiGLU])
    def test_repeated_backward_raises(self, fn):
        gate = torch.randn(4, 256, device="cuda", requires_grad=True)
        up = torch.randn(4, 256, device="cuda", requires_grad=True)
        grad_out = torch.randn_like(gate)
        out = fn.apply(gate, up)

        torch.autograd.grad(out, (gate, up), grad_out, retain_graph=True)
        with pytest.raises(NotImplementedError, match="Repeated backward"):
            torch.autograd.grad(out, (gate, up), grad_out, retain_graph=True)

    @pytest.mark.parametrize("fn", [Opake_SwiGLU])
    def test_single_backward_unaffected(self, fn):
        gate = torch.randn(4, 256, device="cuda", requires_grad=True)
        up = torch.randn(4, 256, device="cuda", requires_grad=True)
        out = fn.apply(gate, up)
        torch.autograd.grad(out, (gate, up), torch.randn_like(gate))


class TestSwiGLUSecondOrder:
    """Second-order differentiation is refused, not silently wrong.

    Reverse-over-reverse reaches the inner backward's guard; forward-over-reverse
    stops earlier, on the missing forward-mode rule.
    """

    @staticmethod
    def _fn(kernel, up):
        return lambda gate: kernel.apply(gate, up).sum()

    @pytest.mark.parametrize("kernel", [Opake_SwiGLU])
    @pytest.mark.parametrize(
        "transform",
        [
            lambda f, x: torch.func.jacrev(torch.func.jacrev(f))(x),
            lambda f, x: torch.func.hessian(f)(x),
            lambda f, x: torch.func.jacfwd(torch.func.jacrev(f))(x),
            lambda f, x: torch.func.jvp(
                torch.func.grad(f), (x,), (torch.ones_like(x),)
            ),
        ],
        ids=["jacrev_jacrev", "hessian", "jacfwd_jacrev", "hvp"],
    )
    def test_second_order_raises(self, kernel, transform):
        gate = torch.randn(8, device="cuda")
        up = torch.randn(8, device="cuda")
        with pytest.raises(NotImplementedError):
            transform(self._fn(kernel, up), gate)

    @pytest.mark.parametrize("kernel", [Opake_SwiGLU])
    def test_create_graph_raises(self, kernel):
        gate = torch.randn(8, device="cuda", requires_grad=True)
        up = torch.randn(8, device="cuda", requires_grad=True)
        (grad_gate,) = torch.autograd.grad(
            kernel.apply(gate, up).sum(), gate, create_graph=True
        )
        with pytest.raises(NotImplementedError, match=r"[Dd]ouble backward"):
            torch.autograd.grad(grad_gate.sum(), gate)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
