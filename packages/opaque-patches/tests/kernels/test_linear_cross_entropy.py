"""
Linear Cross Entropy Kernel Tests (Fused lm_head + CE)

Tests:
1. Forward pass vs PyTorch reference (matmul + F.cross_entropy)
2. Backward pass vs PyTorch reference (d_hidden, d_weight)
3. vmap (per-sample forward) vs PyTorch vmap — precision + performance
4. vmap(grad) (per-example gradients — DP-SGD path) — precision + performance
5. Standard forward and backward performance benchmarks
6. Softcapping (Gemma2) and logit scaling (Granite)

Uses bf16 throughout — CCE backward requires half precision.
``mellum_config`` (see ``kernels/conftest.py``) uses Mellum-4b-shaped tensors
(seq 1024, hidden 3072, vocab up to 128256): realistic geometry comparable to
``train_dpsgd.py --preset mellum-kstack``, not tiny matrices where launch
and dispatch dominate the timing story.

Parametrized over vocab sizes: 32768 (single-chunk) and 128256 (Mellum-4b, chunked path).
Reference computes in fp32 for comparison baseline.

CCE shift=True is always enabled: position i predicts labels[i+1] (HF label shifting).
Reference uses manual shift: logits[:-1] vs labels[1:].
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

import opaque.api.patches.kernels.linear_cross_entropy as linear_ce_mod
from opaque.api.patches.kernels.linear_cross_entropy import (
    Opaque_LinearCrossEntropyLoss,
    opaque_linear_cross_entropy_loss,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# bf16 tolerances (lower precision than fp32 due to half-precision inputs)
RTOL_FORWARD = 5e-3
ATOL_FORWARD = 1e-4
RTOL_BACKWARD = 1e-2
ATOL_BACKWARD = 2e-3

# Vocab sizes: single-chunk (<= 65536) and chunked (> 65536)
VOCAB_SIZES = [32768, 128256]
MEMORY_TEST_VOCAB = 128256
MEMORY_TEST_BATCH = 2
MEMORY_TEST_SEQ_LEN = 129
MEMORY_TEST_HIDDEN_DIM = 64
VMAP_COPY_TEST_BATCH = 2
VMAP_COPY_TEST_SEQ_LEN = 513
VMAP_COPY_TEST_HIDDEN_DIM = 256
VMAP_COPY_TEST_VOCAB = 256
VMAP_COPY_TEST_VMAP_BATCH = 2


# ============================================================================
# Reference Implementations
# ============================================================================


def pytorch_linear_ce(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    softcap: float | None = None,
    scaling: float = 0.0,
    label_smoothing=0.0,
):
    """PyTorch reference: matmul + shift + F.cross_entropy.

    This is what HuggingFace transformers does without the fused kernel:
    logits = self.lm_head(hidden_states), then loss_function(logits, labels).
    Computes logits in float32 for reference precision.
    """
    logits = hidden_states.float() @ weight.float().T

    if scaling != 0:
        logits = logits / scaling

    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)

    # HF-style label shifting: position i predicts labels[i+1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    V = logits.shape[-1]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss


def opaque_linear_ce(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    softcap: float = 0.0,
    scaling: float = 0.0,
    label_smoothing=0.0,
):
    """Opaque kernel wrapper for functional use in vmap/grad.

    Kernel returns nll_sum (unreduced). We reduce here to match
    the PyTorch reference (F.cross_entropy with reduction='mean').
    Scaling is passed into the tiled kernel without transforming the full
    weight tensor.
    """
    logit_scale = 1.0 if scaling == 0 else 1.0 / scaling
    nll_sum = Opaque_LinearCrossEntropyLoss.apply(
        hidden_states,
        weight,
        labels,
        ignore_index,
        softcap,
        label_smoothing,
        False,  # use_token_scaling
        logit_scale,
    )
    shifted_labels = labels[..., 1:].contiguous().flatten()
    n_valid = (shifted_labels != ignore_index).sum().float().clamp(min=1)
    return nll_sum / n_valid


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestLinearCEForward:
    """Test forward pass precision against PyTorch reference."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_matches_pytorch(self, assert_precision, mellum_config, vocab_size):
        """Forward: fused linear CE vs matmul + F.cross_entropy."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        out_pt = pytorch_linear_ce(hidden, weight, labels)
        out_op = opaque_linear_ce(hidden, weight, labels)

        print(f"\nLinear CE Forward (D={hidden_dim}, V={vocab_size}):")
        assert_precision(
            out_op.float().unsqueeze(0),
            out_pt.float().unsqueeze(0),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="loss",
        )

    def test_forward_uses_next_token_labels(self, assert_precision):
        """The causal shift must read label ``i + 1`` for each hidden state ``i``."""
        torch.manual_seed(48)
        batch, seq_len, hidden_dim, vocab_size = 2, 7, 16, 32
        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.arange(batch * seq_len, device="cuda").reshape(batch, seq_len)
        labels %= vocab_size

        out_pt = pytorch_linear_ce(hidden, weight, labels)
        out_op = opaque_linear_ce(hidden, weight, labels)

        assert_precision(
            out_op.float().unsqueeze(0),
            out_pt.float().unsqueeze(0),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="causally shifted loss",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_with_ignore_index(
        self, assert_precision, mellum_config, vocab_size
    ):
        """Forward with masked labels (-100) at some positions."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")
        labels[:, -20:] = -100

        out_pt = pytorch_linear_ce(hidden, weight, labels)
        out_op = opaque_linear_ce(hidden, weight, labels)

        print(f"\nLinear CE Forward with ignore_index (V={vocab_size}):")
        assert_precision(
            out_op.float().unsqueeze(0),
            out_pt.float().unsqueeze(0),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="loss",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_label_smoothing(self, assert_precision, mellum_config, vocab_size):
        """Forward with label smoothing matches PyTorch."""
        torch.manual_seed(43)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        for ls in (0.05, 0.1):
            out_pt = pytorch_linear_ce(hidden, weight, labels, label_smoothing=ls)
            out_op = opaque_linear_ce(hidden, weight, labels, label_smoothing=ls)
            print(f"\nLinear CE forward label_smoothing={ls} (V={vocab_size}):")
            assert_precision(
                out_op.float().unsqueeze(0),
                out_pt.float().unsqueeze(0),
                rtol=RTOL_FORWARD,
                atol=ATOL_FORWARD,
                label="loss",
            )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_label_smoothing_hidden_grad(
        self, assert_precision, mellum_config, vocab_size
    ):
        ls = 0.1
        torch.manual_seed(44)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden_pt = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden_pt, weight, labels, label_smoothing=ls)
        loss_pt.backward()

        hidden_op = hidden_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden_op, weight, labels, label_smoothing=ls)
        loss_op.backward()

        print(f"\nLinear CE backward d_hidden label_smoothing (V={vocab_size}):")
        assert_precision(
            hidden_op.grad.float(),
            hidden_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="hidden_states.grad",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_label_smoothing_weight_grad(
        self, assert_precision, mellum_config, vocab_size
    ):
        ls = 0.1
        torch.manual_seed(45)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight_pt = torch.randn(
            vocab_size,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden, weight_pt, labels, label_smoothing=ls)
        loss_pt.backward()

        weight_op = weight_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden, weight_op, labels, label_smoothing=ls)
        loss_op.backward()

        print(f"\nLinear CE backward d_weight label_smoothing (V={vocab_size}):")
        assert_precision(
            weight_op.grad.float(),
            weight_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="weight.grad",
        )


# ============================================================================
# Backward Pass Tests
# ============================================================================


class TestLinearCEBackward:
    """Test backward pass gradients against PyTorch reference."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_hidden_states_grad(
        self, assert_precision, mellum_config, vocab_size
    ):
        """Backward: d_hidden_states matches PyTorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden_pt = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden_pt, weight, labels)
        loss_pt.backward()

        hidden_op = hidden_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden_op, weight, labels)
        loss_op.backward()

        print(f"\nLinear CE Backward d_hidden (V={vocab_size}):")
        assert_precision(
            hidden_op.grad.float(),
            hidden_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="hidden_states.grad",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_weight_grad(self, assert_precision, mellum_config, vocab_size):
        """Backward: d_weight matches PyTorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight_pt = torch.randn(
            vocab_size,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden, weight_pt, labels)
        loss_pt.backward()

        weight_op = weight_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden, weight_op, labels)
        loss_op.backward()

        print(f"\nLinear CE Backward d_weight (V={vocab_size}):")
        assert_precision(
            weight_op.grad.float(),
            weight_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="weight.grad",
        )


@pytest.mark.parametrize("vmapped", [False, True], ids=["direct", "vmap"])
@pytest.mark.parametrize("use_token_scaling", [False, True], ids=["plain", "scaled"])
@pytest.mark.parametrize("has_ignored", [False, True], ids=["all-valid", "ignored"])
def test_backward_reuses_forward_statistics(
    monkeypatch, vmapped, use_token_scaling, has_ignored
):
    torch.manual_seed(46)
    b, t, d, vocab = 2, 6, 16, 256
    hidden = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab, d, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (b, t), device="cuda")
    if has_ignored:
        labels[:, 2] = -100

    forward_calls = 0
    valid_calls = 0
    original_forward = linear_ce_mod._forward_impl
    original_valids = linear_ce_mod._build_flat_valids

    def recording_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    def recording_valids(*args, **kwargs):
        nonlocal valid_calls
        valid_calls += 1
        return original_valids(*args, **kwargs)

    monkeypatch.setattr(linear_ce_mod, "_forward_impl", recording_forward)
    monkeypatch.setattr(linear_ce_mod, "_build_flat_valids", recording_valids)

    def loss(h, w, lab):
        return Opaque_LinearCrossEntropyLoss.apply(
            h, w, lab, -100, 0, 0.0, use_token_scaling
        )

    if vmapped:
        vmap(grad(loss, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    else:
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        torch.autograd.grad(loss(hidden, weight, labels), (hidden, weight))

    assert forward_calls == 1
    assert valid_calls == 1


# ============================================================================
# Vmap Forward Tests
# ============================================================================


class TestLinearCEVmapForward:
    """Test vmap forward: batched forward via Triton vmap vs PyTorch."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_forward_precision(self, assert_precision, mellum_config, vocab_size):
        """Batched forward: opaque vmap vs PyTorch vmap."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def f_pt(h, t):
            return pytorch_linear_ce(h, weight, t)

        def f_op(h, t):
            return opaque_linear_ce(h, weight, t)

        out_pt = vmap(f_pt, in_dims=(0, 0))(hidden, labels)
        out_op = vmap(f_op, in_dims=(0, 0))(hidden, labels)

        print(f"\nLinear CE vmap forward (V={vocab_size}):")
        assert_precision(
            out_op.float(),
            out_pt.float(),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="loss",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Fused vmap forward must save memory vs materialized logits."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def f_pt(h, t):
            return pytorch_linear_ce(h, weight, t)

        def f_op(h, t):
            return opaque_linear_ce(h, weight, t)

        pt_stats = measure_time_and_memory(
            lambda h, t: vmap(f_pt, in_dims=(0, 0))(h, t), hidden, labels
        )
        op_stats = measure_time_and_memory(
            lambda h, t: vmap(f_op, in_dims=(0, 0))(h, t), hidden, labels
        )

        assert_perf_benefit(
            pt_stats, op_stats, label=f"Linear CE vmap forward (V={vocab_size})"
        )


# ============================================================================
# Vmap(grad) Tests — the DP-SGD path
# ============================================================================


class TestLinearCEVmapGrad:
    """Test vmap(grad): per-example gradients for DP-SGD."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_grad_hidden_states(self, assert_precision, mellum_config, vocab_size):
        """Per-example gradients w.r.t. hidden_states."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def f_pt(h, t):
            return pytorch_linear_ce(h, weight, t)

        def f_op(h, t):
            return opaque_linear_ce(h, weight, t)

        grads_pt = vmap(grad(f_pt, argnums=0), in_dims=(0, 0))(hidden, labels)
        grads_op = vmap(grad(f_op, argnums=0), in_dims=(0, 0))(hidden, labels)

        print(f"\nLinear CE vmap(grad) d_hidden (V={vocab_size}):")
        assert_precision(
            grads_op.float(),
            grads_pt.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="per-example hidden.grad",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_grad_weight(self, assert_precision, mellum_config, vocab_size):
        """Per-example gradients w.r.t. weight (non-batched param)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def f_pt(h, w, t):
            return pytorch_linear_ce(h, w, t)

        def f_op(h, w, t):
            return opaque_linear_ce(h, w, t)

        grads_pt = vmap(grad(f_pt, argnums=1), in_dims=(0, None, 0))(
            hidden, weight, labels
        )
        grads_op = vmap(grad(f_op, argnums=1), in_dims=(0, None, 0))(
            hidden, weight, labels
        )

        print(f"\nLinear CE vmap(grad) d_weight (V={vocab_size}):")
        assert_precision(
            grads_op.float(),
            grads_pt.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="per-example weight.grad",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Fused vmap(grad) must save memory vs materialized logits."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def make_pt_fn():
            def f(h, t):
                return pytorch_linear_ce(h, weight, t)

            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        def make_op_fn():
            def f(h, t):
                return opaque_linear_ce(h, weight, t)

            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        pt_stats = measure_time_and_memory(make_pt_fn(), hidden, labels)
        op_stats = measure_time_and_memory(make_op_fn(), hidden, labels)

        assert_perf_benefit(
            pt_stats, op_stats, label=f"Linear CE vmap(grad) (V={vocab_size})"
        )


# ============================================================================
# Peak Memory Tests
# ============================================================================


@pytest.mark.cuda
class TestLinearCEMemory:
    """Loss-only fused linear-CE does not materialize token-vocabulary logits."""

    def test_hidden_backward_stays_logits_free(self, measure_incremental_cuda_memory):
        torch.manual_seed(46)
        hidden = torch.randn(
            MEMORY_TEST_BATCH,
            MEMORY_TEST_SEQ_LEN,
            MEMORY_TEST_HIDDEN_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            MEMORY_TEST_VOCAB,
            MEMORY_TEST_HIDDEN_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        labels = torch.randint(
            0,
            MEMORY_TEST_VOCAB,
            (MEMORY_TEST_BATCH, MEMORY_TEST_SEQ_LEN),
            device="cuda",
        )

        def run_backward():
            loss = opaque_linear_cross_entropy_loss(hidden, weight, labels)
            return torch.autograd.grad(loss, hidden)[0]

        grad_hidden, stats = measure_incremental_cuda_memory(run_backward)
        materialized_logits_bytes = (
            MEMORY_TEST_BATCH
            * (MEMORY_TEST_SEQ_LEN - 1)
            * MEMORY_TEST_VOCAB
            * hidden.element_size()
        )

        assert stats["peak_bytes"] < materialized_logits_bytes, stats
        assert stats["allocated_bytes"] < materialized_logits_bytes, stats
        assert grad_hidden.shape == hidden.shape

    def test_vmap_grad_avoids_shifted_hidden_copy(self):
        """vmap(grad) does not materialize the activation-sized causal shift."""
        torch.manual_seed(47)
        hidden = torch.randn(
            VMAP_COPY_TEST_VMAP_BATCH,
            VMAP_COPY_TEST_BATCH,
            VMAP_COPY_TEST_SEQ_LEN,
            VMAP_COPY_TEST_HIDDEN_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            VMAP_COPY_TEST_VOCAB,
            VMAP_COPY_TEST_HIDDEN_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        labels = torch.randint(
            0,
            VMAP_COPY_TEST_VOCAB,
            (
                VMAP_COPY_TEST_VMAP_BATCH,
                VMAP_COPY_TEST_BATCH,
                VMAP_COPY_TEST_SEQ_LEN,
            ),
            device="cuda",
        )

        def loss_fn(h, t):
            return opaque_linear_ce(h, weight, t)

        vmap_grad = vmap(grad(loss_fn, argnums=0), in_dims=(0, 0))
        vmap_grad(hidden, labels)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
        ) as profiler:
            grad_hidden = vmap_grad(hidden, labels)
            torch.cuda.synchronize()

        shifted_hidden_bytes = (
            VMAP_COPY_TEST_VMAP_BATCH
            * VMAP_COPY_TEST_BATCH
            * (VMAP_COPY_TEST_SEQ_LEN - 1)
            * VMAP_COPY_TEST_HIDDEN_DIM
            * hidden.element_size()
        )
        contiguous_bytes = sum(
            getattr(
                event,
                "device_memory_usage",
                getattr(event, "cuda_memory_usage", 0),
            )
            for event in profiler.key_averages()
            if event.key == "aten::contiguous"
        )

        assert grad_hidden.shape == hidden.shape
        assert contiguous_bytes < shifted_hidden_bytes, contiguous_bytes


# ============================================================================
# Standard Performance Tests
# ============================================================================


class TestLinearCEPerformance:
    """Benchmark forward and backward performance vs PyTorch materialized path."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Forward-only: fused vs materialized logits."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        def pytorch_fn(h, w):
            return pytorch_linear_ce(h, w, labels)

        def opaque_fn(h, w):
            return opaque_linear_ce(h, w, labels)

        pt_stats = measure_time_and_memory(pytorch_fn, hidden, weight)
        op_stats = measure_time_and_memory(opaque_fn, hidden, weight)

        assert_perf_benefit(
            pt_stats, op_stats, label=f"Linear CE forward (V={vocab_size})"
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Forward+backward: fused vs materialized logits."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]

        hidden = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        def pytorch_fn(h, w):
            return pytorch_linear_ce(h, w, labels)

        def opaque_fn(h, w):
            return opaque_linear_ce(h, w, labels)

        pt_stats = measure_time_and_memory(pytorch_fn, hidden, weight)
        op_stats = measure_time_and_memory(opaque_fn, hidden, weight)

        assert_perf_benefit(
            pt_stats, op_stats, label=f"Linear CE backward (V={vocab_size})"
        )


# ============================================================================
# Softcapping and Logit Scaling Tests
# ============================================================================


class TestLinearCESoftcapping:
    """Test logit softcapping (Gemma2) and logit scaling (Granite)."""

    def test_softcapping_forward(self, assert_precision, mellum_config):
        """Softcapping forward matches PyTorch reference at mellum scale."""
        torch.manual_seed(42)
        softcap = 30.0
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(vocab, hidden_dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        out_pt = pytorch_linear_ce(hidden, weight, labels, softcap=softcap)
        out_op = opaque_linear_ce(hidden, weight, labels, softcap=softcap)

        print(f"\nLinear CE softcapping forward (V={vocab}):")
        assert_precision(
            out_op.float().unsqueeze(0),
            out_pt.float().unsqueeze(0),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="loss",
        )

    def test_softcapping_backward(self, assert_precision, mellum_config):
        """Softcapping backward matches PyTorch reference at mellum scale."""
        torch.manual_seed(42)
        softcap = 30.0
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]

        hidden_pt = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(vocab, hidden_dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden_pt, weight, labels, softcap=softcap)
        loss_pt.backward()

        hidden_op = hidden_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden_op, weight, labels, softcap=softcap)
        loss_op.backward()

        print(f"\nLinear CE softcapping backward (V={vocab}):")
        assert_precision(
            hidden_op.grad.float(),
            hidden_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="hidden_states.grad",
        )

    def test_logit_scaling_forward(self, assert_precision, mellum_config):
        """Logit scaling (Granite) forward matches PyTorch reference at mellum scale."""
        torch.manual_seed(42)
        scaling = 8.0
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(vocab, hidden_dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        out_pt = pytorch_linear_ce(hidden, weight, labels, scaling=scaling)
        out_op = opaque_linear_ce(hidden, weight, labels, scaling=scaling)

        print(f"\nLinear CE logit scaling forward (V={vocab}):")
        assert_precision(
            out_op.float().unsqueeze(0),
            out_pt.float().unsqueeze(0),
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="loss",
        )

    def test_logit_scaling_backward(self, assert_precision, mellum_config):
        """Logit scaling backward matches PyTorch reference at mellum scale."""
        torch.manual_seed(42)
        scaling = 8.0
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]

        hidden_pt = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight_pt = torch.randn(
            vocab,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        loss_pt = pytorch_linear_ce(hidden_pt, weight_pt, labels, scaling=scaling)
        loss_pt.backward()

        hidden_op = hidden_pt.detach().clone().requires_grad_(True)
        weight_op = weight_pt.detach().clone().requires_grad_(True)
        loss_op = opaque_linear_ce(hidden_op, weight_op, labels, scaling=scaling)
        loss_op.backward()

        print(f"\nLinear CE logit scaling backward (V={vocab}):")
        assert_precision(
            hidden_op.grad.float(),
            hidden_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="hidden_states.grad",
        )
        assert_precision(
            weight_op.grad.float(),
            weight_pt.grad.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="weight.grad",
        )

    @pytest.mark.parametrize(
        ("softcap", "scaling"),
        [(30.0, 0.0), (0.0, 8.0)],
        ids=["softcap", "logit-scaling"],
    )
    def test_logit_transform_vmap_grad(
        self, assert_precision, mellum_config, softcap, scaling
    ):
        """Logit transforms under vmap(grad) match PyTorch at mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]

        hidden = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(vocab, hidden_dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        def f_pt(h, t):
            return pytorch_linear_ce(h, weight, t, softcap=softcap, scaling=scaling)

        def f_op(h, t):
            return opaque_linear_ce(h, weight, t, softcap=softcap, scaling=scaling)

        grads_pt = vmap(grad(f_pt, argnums=0), in_dims=(0, 0))(hidden, labels)
        grads_op = vmap(grad(f_op, argnums=0), in_dims=(0, 0))(hidden, labels)

        print(f"\nLinear CE transformed-logit vmap(grad) (V={vocab}):")
        assert_precision(
            grads_op.float(),
            grads_pt.float(),
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="per-example gradients",
        )


# ============================================================================
# Convenience wrapper test
# ============================================================================


class TestLinearCEWrapper:
    """Test the convenience wrapper function."""

    def test_wrapper_matches_apply(self, assert_precision, mellum_config):
        """opaque_linear_cross_entropy_loss matches manual .apply() + reduce."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        hidden_dim = mellum_config["hidden_dim"]
        vocab = mellum_config["vocab_size"]

        hidden = torch.randn(
            batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(vocab, hidden_dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # Manual: .apply() returns nll_sum, reduce ourselves
        nll_sum = Opaque_LinearCrossEntropyLoss.apply(
            hidden,
            weight,
            labels,
            -100,
            0,
            0.0,
            False,  # use_token_scaling
            1.0,
        )
        shifted = labels[..., 1:].contiguous().flatten()
        n_valid = (shifted != -100).sum().float().clamp(min=1)
        out_manual = nll_sum / n_valid

        # Wrapper does the same internally
        out_wrapper = opaque_linear_cross_entropy_loss(
            hidden,
            weight,
            labels,
        )

        print(f"\nWrapper vs manual .apply() + reduce (V={vocab}):")
        assert_precision(
            out_wrapper.float().unsqueeze(0),
            out_manual.float().unsqueeze(0),
            rtol=1e-5,
            atol=1e-5,
            label="loss",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
