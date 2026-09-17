"""
LoRA Kernel Tests

Tests:
1. Forward pass vs PyTorch reference (non-vmap)
2. Backward pass vs PyTorch reference (non-vmap)
3. vmap (per-sample grad) precision vs PyTorch vmap
4. Performance: forward+backward time and memory vs PyTorch

Covers three LoRA variants:
- LoRA_W: Single linear projection with LoRA
- LoRA_QKV: Fused Q, K, V projections with LoRA
- LoRA_MLP: Fused MLP (gate, up, down) with SwiGLU and LoRA

Config: Mellum-4b scale (hidden=3072, intermediate=8256, rank=64)
"""

import gc
import math
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opaque.api.patches.kernels import lora as lora_kernels
from opaque.api.patches.kernels.lora import (
    ACTIVATION_SWIGLU,
    Opaque_LoRA_MLP,
    Opaque_LoRA_QKV,
    Opaque_LoRA_W,
    _LoRAMLPBackward,
    _LoRAMLPBackwardLite,
    _LoRAMLPInputBackward,
    _LoRAQKVBackward,
    _LoRAQKVBackwardLite,
    _LoRAWBackward,
    _LoRAWBackwardLite,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

SCALING = 0.1

# Tolerances (atol + rtol formula: |a - b| <= atol + rtol * |b|)
# W/QKV: addmm_ fused accumulation differs from separate add+matmul+scale
RTOL_LORA_FWD = 1e-2
ATOL_LORA_FWD = 4e-2
RTOL_LORA_BWD = 2e-2
ATOL_LORA_BWD = 5e-6

# MLP: 3-matmul chain with SwiGLU
RTOL_LORA_MLP_FWD = 2e-2
ATOL_LORA_MLP_FWD = 1e-1
RTOL_LORA_MLP_BWD = 1e-1
ATOL_LORA_MLP_BWD = 5e-6


# ============================================================================
# Helpers
# ============================================================================


def _kaiming_weight(out_features, in_features, **kwargs):
    """Create weight matrix with Kaiming initialization (realistic scale)."""
    return torch.randn(out_features, in_features, **kwargs) * math.sqrt(
        2.0 / in_features
    )


def _lora_weight(dim, rank, **kwargs):
    """Create LoRA weight with proper initialization scale."""
    return torch.randn(dim, rank, **kwargs) * math.sqrt(1.0 / dim)


def _make_qkv_weights(hidden, rank, **kw):
    """Create Q, K, V weight sets with proper initialization."""
    Wq = _kaiming_weight(hidden, hidden, **kw)
    Aq = _lora_weight(hidden, rank, **kw)
    Bq = _lora_weight(rank, hidden, **kw)
    Wk = _kaiming_weight(hidden, hidden, **kw)
    Ak = _lora_weight(hidden, rank, **kw)
    Bk = _lora_weight(rank, hidden, **kw)
    Wv = _kaiming_weight(hidden, hidden, **kw)
    Av = _lora_weight(hidden, rank, **kw)
    Bv = _lora_weight(rank, hidden, **kw)
    return Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv


# ============================================================================
# Reference Implementations
# ============================================================================


def pytorch_lora_linear(X, W, A, B, scaling, bias=None):
    """PyTorch reference LoRA linear: out = X @ W.T + bias + X @ A @ B * scaling."""
    out = F.linear(X, W, bias)
    if A is not None and B is not None:
        out = out + (X @ A) @ B * scaling
    return out


def opaque_lora_linear(X, W, A, B, scaling):
    """Opaque kernel implementation."""
    return Opaque_LoRA_W.apply(X, W, A, B, scaling)


def pytorch_lora_qkv(  # noqa: PLR0913, PLR0917
    X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv, bq=None, bk=None, bv=None
):
    """PyTorch reference for Q, K, V projections."""
    Q = pytorch_lora_linear(X, Wq, Aq, Bq, Sq, bq)
    K = pytorch_lora_linear(X, Wk, Ak, Bk, Sk, bk)
    V = pytorch_lora_linear(X, Wv, Av, Bv, Sv, bv)
    return Q, K, V


def opaque_lora_qkv(  # noqa: PLR0913, PLR0917
    X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv, bq=None, bk=None, bv=None
):
    """Opaque kernel implementation."""
    return Opaque_LoRA_QKV.apply(
        X, Wq, Aq, Bq, Sq, bq, Wk, Ak, Bk, Sk, bk, Wv, Av, Bv, Sv, bv
    )


def pytorch_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """PyTorch reference for MLP with SwiGLU."""
    gate = pytorch_lora_linear(X, Wg, Ag, Bg, Sg)
    up = pytorch_lora_linear(X, Wu, Au, Bu, Su)
    h = F.silu(gate) * up
    out = pytorch_lora_linear(h, Wd, Ad, Bd, Sd)
    return out


def opaque_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """Opaque kernel implementation."""
    result = Opaque_LoRA_MLP.apply(
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, ACTIVATION_SWIGLU
    )
    return result[0]


@pytest.mark.parametrize(
    ("vmap_rule", "in_dims"),
    [
        (Opaque_LoRA_W.vmap, (0, 0, None, None, None)),
        (_LoRAWBackward.vmap, (0, 0, 0, None, None)),
        (_LoRAWBackwardLite.vmap, (0, None, None, 0, None)),
        (
            Opaque_LoRA_QKV.vmap,
            (
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        ),
        (
            _LoRAQKVBackward.vmap,
            (0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
        ),
        (
            _LoRAQKVBackwardLite.vmap,
            (
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        ),
        (
            Opaque_LoRA_MLP.vmap,
            (
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        ),
        (
            _LoRAMLPBackward.vmap,
            (0, 0, 0, None, None, None, None, None, None, None, None, None, 0, 0, None),
        ),
        (
            _LoRAMLPInputBackward.vmap,
            (0, 0, 0, None, None, None, None, None, None, None),
        ),
        (
            _LoRAMLPBackwardLite.vmap,
            (
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
            ),
        ),
    ],
)
def test_vmap_rules_reject_mapped_parameters(vmap_rule, in_dims):
    """Forward and backward vmap rules accept only activation batch dimensions."""
    with pytest.raises(ValueError, match="vmap requires"):
        vmap_rule(None, in_dims, *(None for _ in in_dims))


def test_rectangular_linear_and_qkv_vmap_gradients_match_pytorch(assert_precision):
    """Rectangular LoRA projections preserve eager and vmapped input gradients."""
    torch.manual_seed(42)
    vmap_batch, batch, seq, hidden, output, rank = (2, 2, 3, 8, 5, 2)
    kw = {"device": "cuda", "dtype": torch.float32}
    X = torch.randn(vmap_batch, batch, seq, hidden, **kw)
    Wq, Aq, Bq = (
        _kaiming_weight(output, hidden, **kw),
        _lora_weight(hidden, rank, **kw),
        _lora_weight(rank, output, **kw),
    )
    Wk, Ak, Bk = (
        _kaiming_weight(output, hidden, **kw),
        _lora_weight(hidden, rank, **kw),
        _lora_weight(rank, output, **kw),
    )
    Wv, Av, Bv = (
        _kaiming_weight(output, hidden, **kw),
        _lora_weight(hidden, rank, **kw),
        _lora_weight(rank, output, **kw),
    )

    assert_precision(
        opaque_lora_linear(X[0], Wq, Aq, Bq, SCALING),
        pytorch_lora_linear(X[0], Wq, Aq, Bq, SCALING),
        rtol=RTOL_LORA_FWD,
        atol=ATOL_LORA_FWD,
        label="rectangular linear output",
    )

    def linear_pt(x):
        return pytorch_lora_linear(x, Wq, Aq, Bq, SCALING).square().mean()

    def linear_op(x):
        return opaque_lora_linear(x, Wq, Aq, Bq, SCALING).square().mean()

    assert_precision(
        vmap(grad(linear_op))(X),
        vmap(grad(linear_pt))(X),
        rtol=RTOL_LORA_BWD,
        atol=ATOL_LORA_BWD,
        label="rectangular linear vmapped X.grad",
    )

    assert_precision(
        torch.stack(
            opaque_lora_qkv(
                X[0], Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        ),
        torch.stack(
            pytorch_lora_qkv(
                X[0], Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        ),
        rtol=RTOL_LORA_FWD,
        atol=ATOL_LORA_FWD,
        label="rectangular QKV output",
    )

    def qkv_pt(x):
        return sum(
            projection.square().mean()
            for projection in pytorch_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        )

    def qkv_op(x):
        return sum(
            projection.square().mean()
            for projection in opaque_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        )

    assert_precision(
        vmap(grad(qkv_op))(X),
        vmap(grad(qkv_pt))(X),
        rtol=RTOL_LORA_BWD,
        atol=ATOL_LORA_BWD,
        label="rectangular QKV vmapped X.grad",
    )


# ============================================================================
# LoRA-W Tests (single projection)
# ============================================================================


class TestLoRAWForward:
    """Test LoRA-W forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, bfloat16)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        out_pt = pytorch_lora_linear(X, W, A, B, SCALING)
        out_op = opaque_lora_linear(X, W, A, B, SCALING)

        print("\nLoRA-W Forward:")
        assert_precision(
            out_op, out_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="output"
        )


class TestLoRAWBackward:
    """Test LoRA-W backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap, test X.grad, A.grad, B.grad)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        B_pt = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        out_pt = pytorch_lora_linear(X_pt, W, A_pt, B_pt, SCALING)
        out_pt.mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        X_before = X_op.detach().clone()
        A_op = A_pt.detach().clone().requires_grad_(True)
        B_op = B_pt.detach().clone().requires_grad_(True)

        out_op = opaque_lora_linear(X_op, W, A_op, B_op, SCALING)
        out_op.mean().backward()

        torch.testing.assert_close(X_op, X_before, rtol=0, atol=0)
        print("\nLoRA-W Backward:")
        assert_precision(
            X_op.grad, X_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad"
        )
        assert_precision(
            A_op.grad, A_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="A.grad"
        )
        assert_precision(
            B_op.grad, B_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="B.grad"
        )

    @pytest.mark.parametrize("missing", ["A", "B"])
    def test_incomplete_adapter_matches_base_projection(self, missing):
        """A one-sided adapter is inactive in eager and vmapped backward."""
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.float32}
        X = torch.randn(2, 3, 8, **kw, requires_grad=True)
        X_before = X.detach().clone()
        W = _kaiming_weight(5, 8, **kw)
        A = None if missing == "A" else _lora_weight(8, 2, **kw).requires_grad_(True)
        B = None if missing == "B" else _lora_weight(2, 5, **kw).requires_grad_(True)
        present = B if A is None else A

        grad_out = torch.randn(2, 3, 5, **kw)
        out = opaque_lora_linear(X, W, A, B, SCALING)
        (dX, adapter_grad) = torch.autograd.grad(
            out, (X, present), grad_out, allow_unused=True
        )
        X_ref = X.detach().clone().requires_grad_(True)
        (dX_ref,) = torch.autograd.grad(F.linear(X_ref, W), X_ref, grad_out)

        torch.testing.assert_close(dX, dX_ref)
        torch.testing.assert_close(X, X_before, rtol=0, atol=0)
        assert adapter_grad is None

        X_vmap = torch.randn(3, 2, 8, **kw)
        got = vmap(grad(lambda x: opaque_lora_linear(x, W, A, B, SCALING).sum()))(
            X_vmap
        )
        expected = vmap(grad(lambda x: F.linear(x, W).sum()))(X_vmap)
        torch.testing.assert_close(got, expected)


class TestLoRAWVmapForward:
    """Test LoRA-W vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        out_pt = vmap(lambda x: pytorch_lora_linear(x, W, A, B, SCALING))(X)
        out_op = vmap(lambda x: opaque_lora_linear(x, W, A, B, SCALING))(X)

        print("\nLoRA-W vmap forward:")
        assert_precision(
            out_op, out_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="output"
        )


class TestLoRAWVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            return pytorch_lora_linear(x, W, A, B, SCALING).mean()

        def f_op(x):
            return opaque_lora_linear(x, W, A, B, SCALING).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-W vmap(grad):")
        assert_precision(
            grads_op, grads_pt, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad"
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                return pytorch_lora_linear(x, W, A, B, SCALING).mean()

            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                return opaque_lora_linear(x, W, A, B, SCALING).mean()

            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-W vmap(grad)", max_perf_overhead=0.20
        )


class TestLoRAWPerformance:
    """Test LoRA-W kernel performance (non-vmap)."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        def pytorch_fn(x):
            return pytorch_lora_linear(x, W, A, B, SCALING)

        def opaque_fn(x):
            return opaque_lora_linear(x, W, A, B, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-W forward", max_perf_overhead=0.50
        )

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        B = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        def pytorch_fn(x, a, b):
            return pytorch_lora_linear(x, W, a, b, SCALING)

        def opaque_fn(x, a, b):
            return opaque_lora_linear(x, W, a, b, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X, A, B)
        op_stats = measure_time_and_memory(opaque_fn, X, A, B)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-W backward", max_perf_overhead=0.20
        )


# ============================================================================
# LoRA-QKV Tests (fused Q, K, V projections)
# ============================================================================


class TestLoRAQKVBiasValidation:
    """Validate the frozen-bias contract for fused QKV projections."""

    def test_rejects_trainable_biases_in_eager_and_vmap(self):
        """Trainable biases must use the unfused projection path."""
        from opaque.exceptions import ConfigurationError

        hidden, rank = 8, 2
        kw = {"device": "cuda", "dtype": torch.float32}
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(hidden, rank, **kw)
        bq = torch.randn(hidden, **kw, requires_grad=True)
        x = torch.randn(2, 3, hidden, **kw)

        def qkv_projection(input_):
            return Opaque_LoRA_QKV.apply(
                input_,
                Wq,
                Aq,
                Bq,
                SCALING,
                bq,
                Wk,
                Ak,
                Bk,
                SCALING,
                None,
                Wv,
                Av,
                Bv,
                SCALING,
                None,
            )

        with pytest.raises(ConfigurationError, match="frozen Q/K/V base biases"):
            qkv_projection(x)
        with pytest.raises(ConfigurationError, match="frozen Q/K/V base biases"):
            vmap(qkv_projection)(torch.randn(2, 2, 3, hidden, **kw))


class TestLoRAQKVForward:
    """Test LoRA-QKV forward pass precision."""

    @pytest.mark.parametrize("bias_layout", ["none", "mixed", "all"])
    def test_forward_matches_pytorch(
        self, assert_precision, mellum_config, bias_layout
    ):
        """Forward: opaque vs pytorch (non-vmap, check Q, K, V each)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        biases = {
            "none": (None, None, None),
            "mixed": (torch.randn(HIDDEN, **kw), None, torch.randn(HIDDEN, **kw)),
            "all": (
                torch.randn(HIDDEN, **kw),
                torch.randn(HIDDEN, **kw),
                torch.randn(HIDDEN, **kw),
            ),
        }
        bq, bk, bv = biases[bias_layout]

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X,
            Wq,
            Aq,
            Bq,
            SCALING,
            Wk,
            Ak,
            Bk,
            SCALING,
            Wv,
            Av,
            Bv,
            SCALING,
            bq,
            bk,
            bv,
        )
        Q_op, K_op, V_op = opaque_lora_qkv(
            X,
            Wq,
            Aq,
            Bq,
            SCALING,
            Wk,
            Ak,
            Bk,
            SCALING,
            Wv,
            Av,
            Bv,
            SCALING,
            bq,
            bk,
            bv,
        )

        print("\nLoRA-QKV Forward:")
        assert_precision(Q_op, Q_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="Q")
        assert_precision(K_op, K_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="K")
        assert_precision(V_op, V_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="V")


class TestLoRAQKVBackward:
    """Test LoRA-QKV backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        Wq, Aq_pt, Bq_pt, Wk, Ak_pt, Bk_pt, Wv, Av_pt, Bv_pt = _make_qkv_weights(
            HIDDEN, RANK, **kw
        )
        Aq_pt = Aq_pt.requires_grad_(True)
        Bq_pt = Bq_pt.requires_grad_(True)
        Ak_pt = Ak_pt.requires_grad_(True)
        Bk_pt = Bk_pt.requires_grad_(True)
        Av_pt = Av_pt.requires_grad_(True)
        Bv_pt = Bv_pt.requires_grad_(True)
        bq = torch.randn(HIDDEN, **kw)
        bk = None
        bv = torch.randn(HIDDEN, **kw)

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X_pt,
            Wq,
            Aq_pt,
            Bq_pt,
            SCALING,
            Wk,
            Ak_pt,
            Bk_pt,
            SCALING,
            Wv,
            Av_pt,
            Bv_pt,
            SCALING,
            bq,
            bk,
            bv,
        )
        (Q_pt + K_pt + V_pt).mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        X_before = X_op.detach().clone()
        Aq_op = Aq_pt.detach().clone().requires_grad_(True)
        Bq_op = Bq_pt.detach().clone().requires_grad_(True)
        Ak_op = Ak_pt.detach().clone().requires_grad_(True)
        Bk_op = Bk_pt.detach().clone().requires_grad_(True)
        Av_op = Av_pt.detach().clone().requires_grad_(True)
        Bv_op = Bv_pt.detach().clone().requires_grad_(True)

        Q_op, K_op, V_op = opaque_lora_qkv(
            X_op,
            Wq,
            Aq_op,
            Bq_op,
            SCALING,
            Wk,
            Ak_op,
            Bk_op,
            SCALING,
            Wv,
            Av_op,
            Bv_op,
            SCALING,
            bq,
            bk,
            bv,
        )
        (Q_op + K_op + V_op).mean().backward()

        torch.testing.assert_close(X_op, X_before, rtol=0, atol=0)
        print("\nLoRA-QKV Backward:")
        assert_precision(
            X_op.grad, X_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad"
        )
        assert_precision(
            Aq_op.grad,
            Aq_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Aq.grad",
        )
        assert_precision(
            Bq_op.grad,
            Bq_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Bq.grad",
        )
        assert_precision(
            Ak_op.grad,
            Ak_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Ak.grad",
        )
        assert_precision(
            Bk_op.grad,
            Bk_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Bk.grad",
        )
        assert_precision(
            Av_op.grad,
            Av_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Av.grad",
        )
        assert_precision(
            Bv_op.grad,
            Bv_pt.grad,
            rtol=RTOL_LORA_BWD,
            atol=ATOL_LORA_BWD,
            label="Bv.grad",
        )


class TestLoRAQKVVmapForward:
    """Test LoRA-QKV vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        Q_pt, K_pt, V_pt = vmap(
            lambda x: pytorch_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        )(X)
        Q_op, K_op, V_op = vmap(
            lambda x: opaque_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
        )(X)

        print("\nLoRA-QKV vmap forward:")
        assert_precision(Q_op, Q_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="Q")
        assert_precision(K_op, K_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="K")
        assert_precision(V_op, V_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="V")


class TestLoRAQKVVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        bq = torch.randn(HIDDEN, **kw)
        bk = None
        bv = torch.randn(HIDDEN, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            Q, K, V = pytorch_lora_qkv(
                x,
                Wq,
                Aq,
                Bq,
                SCALING,
                Wk,
                Ak,
                Bk,
                SCALING,
                Wv,
                Av,
                Bv,
                SCALING,
                bq,
                bk,
                bv,
            )
            return (Q + K + V).mean()

        def f_op(x):
            Q, K, V = opaque_lora_qkv(
                x,
                Wq,
                Aq,
                Bq,
                SCALING,
                Wk,
                Ak,
                Bk,
                SCALING,
                Wv,
                Av,
                Bv,
                SCALING,
                bq,
                bk,
                bv,
            )
            return (Q + K + V).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-QKV vmap(grad):")
        assert_precision(
            grads_op, grads_pt, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad"
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                Q, K, V = pytorch_lora_qkv(
                    x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
                )
                return (Q + K + V).mean()

            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                Q, K, V = opaque_lora_qkv(
                    x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
                )
                return (Q + K + V).mean()

            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-QKV vmap(grad)", max_perf_overhead=0.20
        )


class TestLoRAQKVPerformance:
    """Test LoRA-QKV kernel performance (non-vmap)."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)

        def pytorch_fn(x):
            Q, K, V = pytorch_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
            return Q + K + V

        def opaque_fn(x):
            Q, K, V = opaque_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
            return Q + K + V

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-QKV forward", max_perf_overhead=0.20
        )

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        Aq = Aq.requires_grad_(True)
        Bq = Bq.requires_grad_(True)
        Ak = Ak.requires_grad_(True)
        Bk = Bk.requires_grad_(True)
        Av = Av.requires_grad_(True)
        Bv = Bv.requires_grad_(True)

        def pytorch_fn(x, aq, bq, ak, bk, av, bv):
            Q, K, V = pytorch_lora_qkv(
                x, Wq, aq, bq, SCALING, Wk, ak, bk, SCALING, Wv, av, bv, SCALING
            )
            return Q + K + V

        def opaque_fn(x, aq, bq, ak, bk, av, bv):
            Q, K, V = opaque_lora_qkv(
                x, Wq, aq, bq, SCALING, Wk, ak, bk, SCALING, Wv, av, bv, SCALING
            )
            return Q + K + V

        pt_stats = measure_time_and_memory(pytorch_fn, X, Aq, Bq, Ak, Bk, Av, Bv)
        op_stats = measure_time_and_memory(opaque_fn, X, Aq, Bq, Ak, Bk, Av, Bv)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-QKV backward", max_perf_overhead=0.20
        )


# ============================================================================
# LoRA-MLP Tests (fused gate, up, down with SwiGLU)
# ============================================================================


class TestLoRAMLPForward:
    """Test LoRA-MLP forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, bfloat16)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        out_pt = pytorch_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )
        out_op = opaque_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )

        print("\nLoRA-MLP Forward:")
        assert_precision(
            out_op,
            out_pt,
            rtol=RTOL_LORA_MLP_FWD,
            atol=ATOL_LORA_MLP_FWD,
            label="output",
        )


class TestLoRAMLPBackward:
    """Test LoRA-MLP backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bg_pt = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bu_pt = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad_pt = _lora_weight(INTERMEDIATE, RANK, **kw).requires_grad_(True)
        Bd_pt = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        out_pt = pytorch_lora_mlp(
            X_pt,
            Wg,
            Ag_pt,
            Bg_pt,
            SCALING,
            Wu,
            Au_pt,
            Bu_pt,
            SCALING,
            Wd,
            Ad_pt,
            Bd_pt,
            SCALING,
        )
        out_pt.mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        X_before = X_op.detach().clone()
        Ag_op = Ag_pt.detach().clone().requires_grad_(True)
        Bg_op = Bg_pt.detach().clone().requires_grad_(True)
        Au_op = Au_pt.detach().clone().requires_grad_(True)
        Bu_op = Bu_pt.detach().clone().requires_grad_(True)
        Ad_op = Ad_pt.detach().clone().requires_grad_(True)
        Bd_op = Bd_pt.detach().clone().requires_grad_(True)

        out_op = opaque_lora_mlp(
            X_op,
            Wg,
            Ag_op,
            Bg_op,
            SCALING,
            Wu,
            Au_op,
            Bu_op,
            SCALING,
            Wd,
            Ad_op,
            Bd_op,
            SCALING,
        )
        out_op.mean().backward()

        torch.testing.assert_close(X_op, X_before, rtol=0, atol=0)
        print("\nLoRA-MLP Backward:")
        assert_precision(
            X_op.grad,
            X_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="X.grad",
        )
        assert_precision(
            Ag_op.grad,
            Ag_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Ag.grad",
        )
        assert_precision(
            Bg_op.grad,
            Bg_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Bg.grad",
        )
        assert_precision(
            Au_op.grad,
            Au_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Au.grad",
        )
        assert_precision(
            Bu_op.grad,
            Bu_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Bu.grad",
        )
        assert_precision(
            Ad_op.grad,
            Ad_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Ad.grad",
        )
        assert_precision(
            Bd_op.grad,
            Bd_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="Bd.grad",
        )

    def test_long_bfloat16_reduction_preserves_adapter_gradient(self):
        rows = 65_536
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        X = torch.ones(rows, 1, **kw)
        Wg = torch.ones(1, 1, **kw)
        Wu = torch.ones(1, 1, **kw)
        Wd = torch.zeros(1, 1, **kw)
        Ad_pt = torch.ones(1, 1, **kw, requires_grad=True)
        Bd_pt = torch.ones(1, 1, **kw, requires_grad=True)

        out_pt = pytorch_lora_mlp(
            X, Wg, None, None, 1.0, Wu, None, None, 1.0, Wd, Ad_pt, Bd_pt, 1.0
        )
        expected = torch.autograd.grad(out_pt.sum(), (Ad_pt, Bd_pt))

        Ad_op = Ad_pt.detach().clone().requires_grad_(True)
        Bd_op = Bd_pt.detach().clone().requires_grad_(True)
        out_op = opaque_lora_mlp(
            X, Wg, None, None, 1.0, Wu, None, None, 1.0, Wd, Ad_op, Bd_op, 1.0
        )
        actual = torch.autograd.grad(out_op.sum(), (Ad_op, Bd_op))

        for actual_grad, expected_grad in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-2, atol=0)


class TestLoRAMLPVmapForward:
    """Test LoRA-MLP vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        out_pt = vmap(
            lambda x: pytorch_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )
        )(X)
        out_op = vmap(
            lambda x: opaque_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )
        )(X)

        print("\nLoRA-MLP vmap forward:")
        assert_precision(
            out_op,
            out_pt,
            rtol=RTOL_LORA_MLP_FWD,
            atol=ATOL_LORA_MLP_FWD,
            label="output",
        )


class TestLoRAMLPVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_rectangular_projections_match_pytorch(self, assert_precision):
        """Eager and vmapped gradients support distinct MLP input/output widths."""
        torch.manual_seed(42)
        vmap_batch, batch, seq, hidden, intermediate, output, rank = (
            2,
            2,
            3,
            8,
            12,
            5,
            2,
        )
        kw = {"device": "cuda", "dtype": torch.float32}

        Wg = _kaiming_weight(intermediate, hidden, **kw)
        Ag = _lora_weight(hidden, rank, **kw).requires_grad_(True)
        Bg = _lora_weight(rank, intermediate, **kw).requires_grad_(True)
        Wu = _kaiming_weight(intermediate, hidden, **kw)
        Au = _lora_weight(hidden, rank, **kw).requires_grad_(True)
        Bu = _lora_weight(rank, intermediate, **kw).requires_grad_(True)
        Wd = _kaiming_weight(output, intermediate, **kw)
        Ad = _lora_weight(intermediate, rank, **kw).requires_grad_(True)
        Bd = _lora_weight(rank, output, **kw).requires_grad_(True)

        X_pt = torch.randn(batch, seq, hidden, **kw, requires_grad=True)
        out_pt = pytorch_lora_mlp(
            X_pt, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )
        out_pt.square().mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        lora_op = [
            p.detach().clone().requires_grad_(True) for p in (Ag, Bg, Au, Bu, Ad, Bd)
        ]
        out_op = opaque_lora_mlp(
            X_op,
            Wg,
            lora_op[0],
            lora_op[1],
            SCALING,
            Wu,
            lora_op[2],
            lora_op[3],
            SCALING,
            Wd,
            lora_op[4],
            lora_op[5],
            SCALING,
        )
        out_op.square().mean().backward()

        assert_precision(
            out_op,
            out_pt,
            rtol=RTOL_LORA_MLP_FWD,
            atol=ATOL_LORA_MLP_FWD,
            label="output",
        )
        assert_precision(
            X_op.grad,
            X_pt.grad,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="X.grad",
        )

        X_vmap = torch.randn(vmap_batch, batch, seq, hidden, **kw)

        def f_pt(x):
            return (
                pytorch_lora_mlp(
                    x,
                    Wg,
                    Ag.detach(),
                    Bg.detach(),
                    SCALING,
                    Wu,
                    Au.detach(),
                    Bu.detach(),
                    SCALING,
                    Wd,
                    Ad.detach(),
                    Bd.detach(),
                    SCALING,
                )
                .square()
                .mean()
            )

        def f_op(x):
            return (
                opaque_lora_mlp(
                    x,
                    Wg,
                    Ag.detach(),
                    Bg.detach(),
                    SCALING,
                    Wu,
                    Au.detach(),
                    Bu.detach(),
                    SCALING,
                    Wd,
                    Ad.detach(),
                    Bd.detach(),
                    SCALING,
                )
                .square()
                .mean()
            )

        assert_precision(
            vmap(grad(f_op))(X_vmap),
            vmap(grad(f_pt))(X_vmap),
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="vmapped X.grad",
        )

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            return pytorch_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            ).mean()

        def f_op(x):
            return opaque_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            ).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-MLP vmap(grad):")
        assert_precision(
            grads_op,
            grads_pt,
            rtol=RTOL_LORA_MLP_BWD,
            atol=ATOL_LORA_MLP_BWD,
            label="X.grad",
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                return pytorch_lora_mlp(
                    x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
                ).mean()

            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                return opaque_lora_mlp(
                    x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
                ).mean()

            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-MLP vmap(grad)", max_perf_overhead=0.20
        )


class TestLoRAMLPPerformance:
    """Test LoRA-MLP kernel performance (non-vmap)."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        def pytorch_fn(x):
            return pytorch_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )

        def opaque_fn(x):
            return opaque_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-MLP forward", max_perf_overhead=0.50
        )

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = {"device": "cuda", "dtype": torch.bfloat16}

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw).requires_grad_(True)
        Bd = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        def pytorch_fn(x, ag, bg, au, bu, ad, bd):
            return pytorch_lora_mlp(
                x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING
            )

        def opaque_fn(x, ag, bg, au, bu, ad, bd):
            return opaque_lora_mlp(
                x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING
            )

        pt_stats = measure_time_and_memory(pytorch_fn, X, Ag, Bg, Au, Bu, Ad, Bd)
        op_stats = measure_time_and_memory(opaque_fn, X, Ag, Bg, Au, Bu, Ad, Bd)

        assert_perf_benefit(
            pt_stats, op_stats, label="LoRA-MLP backward", max_perf_overhead=0.20
        )


class _EphemeralActivation(torch.autograd.Function):
    @staticmethod
    def forward(seed, value):
        return value.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.sum(), None


def _staged_backward_loss(kind, *, trainable=True, tokens=8, hidden=32, rank=4):
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    seed = torch.zeros((), **kw, requires_grad=True)
    X = _EphemeralActivation.apply(seed, torch.randn(tokens, hidden, **kw))
    X_ref = weakref.ref(X)

    def adapter(dim, out):
        A = _lora_weight(dim, rank, **kw).requires_grad_(trainable)
        B = _lora_weight(rank, out, **kw).requires_grad_(trainable)
        return A, B

    if kind == "w":
        W = _kaiming_weight(hidden, hidden, **kw)
        A, B = adapter(hidden, hidden)
        loss = opaque_lora_linear(X, W, A, B, SCALING).sum()
    elif kind == "qkv":
        projections = []
        for _ in range(3):
            W = _kaiming_weight(hidden, hidden, **kw)
            A, B = adapter(hidden, hidden)
            projections.extend((W, A, B, SCALING, None))
        loss = sum(Opaque_LoRA_QKV.apply(X, *projections)).sum()
    else:
        intermediate = hidden * 2
        Wg = _kaiming_weight(intermediate, hidden, **kw)
        Ag, Bg = adapter(hidden, intermediate)
        Wu = _kaiming_weight(intermediate, hidden, **kw)
        Au, Bu = adapter(hidden, intermediate)
        Wd = _kaiming_weight(hidden, intermediate, **kw)
        Ad, Bd = adapter(intermediate, hidden)
        loss = Opaque_LoRA_MLP.apply(
            X,
            Wg,
            Ag,
            Bg,
            SCALING,
            Wu,
            Au,
            Bu,
            SCALING,
            Wd,
            Ad,
            Bd,
            SCALING,
            ACTIVATION_SWIGLU,
        )[0].sum()

    del X
    return loss, X_ref


@pytest.mark.cuda
class TestLoRABackwardLiveness:
    @pytest.mark.parametrize(
        ("kind", "input_stage"),
        [
            ("w", "_lora_w_backward_lite"),
            ("qkv", "_lora_qkv_backward_lite"),
            ("mlp", "_lora_mlp_input_backward_impl"),
        ],
    )
    def test_saved_input_released_before_input_gradient(
        self, monkeypatch, kind, input_stage
    ):
        """Ordinary backward drops its saved activation before allocating dX."""
        loss, X_ref = _staged_backward_loss(kind)
        assert X_ref() is not None
        original = getattr(lora_kernels, input_stage)
        input_stage_called = False

        def assert_released(*args):
            nonlocal input_stage_called
            input_stage_called = True
            assert X_ref() is None
            return original(*args)

        monkeypatch.setattr(lora_kernels, input_stage, assert_released)
        loss.backward()
        assert input_stage_called

    def test_retain_graph_keeps_saved_input(self, monkeypatch):
        """Early release defers to autograd when a caller retains the graph."""
        loss, X_ref = _staged_backward_loss("w")
        original = lora_kernels._lora_w_backward_lite
        observed_saved_input = False

        def assert_retained(*args):
            nonlocal observed_saved_input
            observed_saved_input = X_ref() is not None
            return original(*args)

        monkeypatch.setattr(lora_kernels, "_lora_w_backward_lite", assert_retained)
        loss.backward(retain_graph=True)
        assert observed_saved_input
        assert X_ref() is not None


@pytest.mark.cuda
class TestLoRABackwardMemory:
    def test_mlp_recomputes_intermediate_activations(self):
        """MLP backward saves the narrow input instead of gate/up activations."""
        tokens, hidden = 64, 128
        saved_shapes = []

        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = _staged_backward_loss(
                "mlp", trainable=True, tokens=tokens, hidden=hidden
            )

        assert (tokens, hidden * 2) not in saved_shapes
        loss.backward()

    @pytest.mark.parametrize("kind", ["w", "qkv", "mlp"])
    def test_trainable_adapter_peak_tracks_frozen_adapter_peak(self, kind):
        """Staging avoids an activation-sized trainable-adapter memory penalty."""
        tokens, hidden = 8192, 1024

        def peak(trainable):
            warmup, _ = _staged_backward_loss(
                kind, trainable=trainable, tokens=tokens, hidden=hidden
            )
            warmup.backward()
            torch.cuda.synchronize()
            del warmup
            gc.collect()
            torch.cuda.empty_cache()

            loss, _ = _staged_backward_loss(
                kind, trainable=trainable, tokens=tokens, hidden=hidden
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            torch.cuda.synchronize()
            result = torch.cuda.max_memory_allocated()
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            return result

        trainable_peak = peak(True)
        frozen_peak = peak(False)
        input_bytes = tokens * hidden * torch.bfloat16.itemsize
        tolerance = max(input_bytes // 2, 8 * 1024**2)
        assert trainable_peak <= frozen_peak + tolerance, (
            f"{kind} trainable adapters retained an activation-sized buffer: "
            f"trainable={trainable_peak / 1024**2:.1f} MiB, "
            f"frozen={frozen_peak / 1024**2:.1f} MiB"
        )


class TestLoRAMLPRepeatedBackward:
    """The fused LoRA MLP backward overwrites its saved gate/up buffers."""

    @staticmethod
    def _inputs():
        X = torch.randn(4, 64, device="cuda", requires_grad=True)

        def w(o, i):
            return torch.randn(o, i, device="cuda", requires_grad=True)

        def lora(o, i, r=8):
            return w(i, r), w(r, o)

        Wg, Wu, Wd = w(128, 64), w(128, 64), w(64, 128)
        Ag, Bg = lora(128, 64)
        Au, Bu = lora(128, 64)
        Ad, Bd = lora(64, 128)
        return X, Wg, Ag, Bg, 1.0, Wu, Au, Bu, 1.0, Wd, Ad, Bd, 1.0

    @pytest.mark.cuda
    def test_repeated_backward_raises(self):
        args = self._inputs()
        out = opaque_lora_mlp(*args)
        grad_out = torch.randn_like(out)

        torch.autograd.grad(out, args[0], grad_out, retain_graph=True)
        with pytest.raises(NotImplementedError, match="Repeated backward"):
            torch.autograd.grad(out, args[0], grad_out, retain_graph=True)

    @pytest.mark.cuda
    def test_single_backward_unaffected(self):
        args = self._inputs()
        out = opaque_lora_mlp(*args)
        torch.autograd.grad(out, args[0], torch.randn_like(out))


class TestLoRAMLPSecondOrder:
    """Second-order differentiation through the fused LoRA MLP is refused."""

    @pytest.mark.cuda
    def test_create_graph_raises(self):
        args = TestLoRAMLPRepeatedBackward._inputs()
        (grad_X,) = torch.autograd.grad(
            opaque_lora_mlp(*args).sum(), args[0], create_graph=True
        )
        with pytest.raises(NotImplementedError, match=r"[Dd]ouble backward"):
            torch.autograd.grad(grad_X.sum(), args[0])

    @pytest.mark.cuda
    def test_jacrev_jacrev_raises(self):
        args = TestLoRAMLPRepeatedBackward._inputs()
        rest = args[1:]

        def fn(x):
            return opaque_lora_mlp(x, *rest).sum()

        with pytest.raises(NotImplementedError):
            torch.func.jacrev(torch.func.jacrev(fn))(args[0].detach())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
