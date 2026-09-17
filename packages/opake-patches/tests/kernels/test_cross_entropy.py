"""
Cross Entropy Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-4b scale (uses mellum_config from conftest).
Parametrized over vocab sizes: 32768 (single-chunk) and 128256 (Mellum-4b, chunked path).
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opake.api.patches.kernels.cross_entropy import Opake_CrossEntropyLoss

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Cross entropy tolerances (accumulation over large vocab dimension)
RTOL_CE_FORWARD = 1e-4
ATOL_CE_FORWARD = 1e-6
RTOL_CE_BACKWARD = 1e-4
ATOL_CE_BACKWARD = 1e-6

# Vocab sizes: single-chunk (<= 65536) and chunked (> 65536)
VOCAB_SIZES = [32768, 128256]
MEMORY_TEST_VOCAB = 128256
MEMORY_TEST_ROWS = 256
MEMORY_SLACK_BYTES = 16 * 1024**2


# ============================================================================
# Reference Implementations
# ============================================================================


def pytorch_cross_entropy(logits, labels):
    """PyTorch reference: F.cross_entropy with reduction='mean'."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss


def opake_cross_entropy(logits, labels):
    """Opake kernel: Opake_CrossEntropyLoss with masked mean for vmap compatibility."""
    losses, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, 0.0)
    # For vmap compatibility, avoid data-dependent control flow
    mask = (labels != -100).float()
    n_valid = mask.sum()
    masked_losses = losses * mask
    return masked_losses.sum() / torch.clamp(n_valid, min=1.0)


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestCrossEntropyForward:
    """Test forward pass precision."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_matches_pytorch(self, assert_precision, mellum_config, vocab_size):
        """Forward: opake vs pytorch at mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits = torch.randn(
            batch, seq_len, vocab_size, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        out_pytorch = pytorch_cross_entropy(logits, labels)
        out_opake = opake_cross_entropy(logits, labels)

        print(f"\nCE Forward (vocab={vocab_size}):")
        assert_precision(
            out_opake.unsqueeze(0),
            out_pytorch.unsqueeze(0),
            rtol=RTOL_CE_FORWARD,
            atol=ATOL_CE_FORWARD,
            label="loss",
        )

    @pytest.mark.parametrize("vocab_size", [128256])
    def test_forward_other_vocab_sizes(self, assert_precision, vocab_size):
        """Forward with other large vocab sizes (e.g. LLaMA 3 128K)."""
        torch.manual_seed(42)

        logits = torch.randn(2, 64, vocab_size, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (2, 64), device="cuda")

        losses_op, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, 0.0)

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        losses_pt = F.cross_entropy(logits_flat, labels_flat, reduction="none")

        print(f"\nCE Forward (vocab={vocab_size}):")
        assert_precision(
            losses_op.reshape(-1),
            losses_pt,
            rtol=RTOL_CE_FORWARD,
            atol=ATOL_CE_FORWARD,
            label="per-token losses",
        )


# ============================================================================
# Backward Pass Tests
# ============================================================================


class TestCrossEntropyBackward:
    """Test backward pass precision."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_matches_pytorch(
        self, assert_precision, mellum_config, vocab_size
    ):
        """Backward: opake vs pytorch logits.grad at mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        # PyTorch reference
        logits_pt = torch.randn(
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        out_pt = pytorch_cross_entropy(logits_pt, labels)
        out_pt.backward()

        # Opake kernel
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        out_op = opake_cross_entropy(logits_op, labels)
        out_op.backward()

        print(f"\nCE Backward (vocab={vocab_size}):")
        assert_precision(
            logits_op.grad,
            logits_pt.grad,
            rtol=RTOL_CE_BACKWARD,
            atol=ATOL_CE_BACKWARD,
            label="logits.grad",
        )

    def test_backward_does_not_mutate_forward_logits(self, mellum_config):
        """Backward must leave the saved forward logits tensor untouched."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(
            batch,
            seq_len,
            vocab,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        logits_before = logits.detach().clone()
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        out = opake_cross_entropy(logits, labels)
        out.backward()

        assert torch.equal(logits.detach(), logits_before), (
            "Backward mutated the forward logits tensor"
        )
        assert logits.grad.is_contiguous()

    def test_backward_ignores_masked_labels(self, mellum_config):
        """Verify -100 labels produce zero gradient (not softmax probs)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(
            batch,
            seq_len,
            vocab,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        labels[:, -10:] = -100  # Mask last 10 positions

        losses, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, 0.0)
        losses.sum().backward()

        # Gradients at -100 positions must be exactly zero
        masked_grad = logits.grad[:, -10:, :]
        assert masked_grad.abs().max() == 0.0, (
            f"Non-zero grad at -100 positions: max={masked_grad.abs().max():.2e}"
        )


# ============================================================================
# Vmap Tests
# ============================================================================


class TestCrossEntropyVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_forward_precision(self, assert_precision, mellum_config, vocab_size):
        """Batched forward: opake Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        out_pt = vmap(pytorch_cross_entropy, in_dims=(0, 0))(logits, labels)
        out_op = vmap(opake_cross_entropy, in_dims=(0, 0))(logits, labels)

        print(f"\nCE vmap forward (vocab={vocab_size}):")
        assert_precision(
            out_op.unsqueeze(0),
            out_pt.unsqueeze(0),
            rtol=RTOL_CE_FORWARD,
            atol=ATOL_CE_FORWARD,
            label="loss",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Triton vmap forward must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        pt_stats = measure_time_and_memory(
            lambda lgt, t: vmap(pytorch_cross_entropy, in_dims=(0, 0))(lgt, t),
            logits,
            labels,
        )
        op_stats = measure_time_and_memory(
            lambda lgt, t: vmap(opake_cross_entropy, in_dims=(0, 0))(lgt, t),
            logits,
            labels,
        )

        assert_perf_benefit(
            pt_stats, op_stats, label=f"CE vmap forward (V={vocab_size})"
        )


class TestCrossEntropyVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_grad_precision(self, assert_precision, mellum_config, vocab_size):
        """Per-example gradients: opake Triton vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def f_pt(lgt, t):
            return pytorch_cross_entropy(lgt, t)

        def f_op(lgt, t):
            return opake_cross_entropy(lgt, t)

        # grad w.r.t. logits only (argnums=0), labels are not differentiable
        grads_pt = vmap(grad(f_pt, argnums=0), in_dims=(0, 0))(logits, labels)
        grads_op = vmap(grad(f_op, argnums=0), in_dims=(0, 0))(logits, labels)

        print(f"\nCE vmap(grad) (vocab={vocab_size}):")
        assert_precision(
            grads_op,
            grads_pt,
            rtol=RTOL_CE_BACKWARD,
            atol=ATOL_CE_BACKWARD,
            label="per-example gradients",
        )

    def test_vmap_backward_does_not_mutate_forward_logits(self):
        """Vmapped backward reads logits without using them as gradient storage."""
        torch.manual_seed(43)
        logits = torch.randn(2, 2, 8, 32768, device="cuda", dtype=torch.float32)
        logits_before = logits.clone()
        labels = torch.randint(0, logits.shape[-1], logits.shape[:-1], device="cuda")

        grads = vmap(grad(opake_cross_entropy, argnums=0), in_dims=(0, 0))(
            logits, labels
        )

        assert torch.equal(logits, logits_before)
        assert grads.is_contiguous()
        assert grads.data_ptr() != logits.data_ptr()

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Triton vmap(grad) must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch,
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
        )
        labels = torch.randint(
            0, vocab_size, (vmap_batch, batch, seq_len), device="cuda"
        )

        def make_pt_fn():
            def f(lgt, t):
                return pytorch_cross_entropy(lgt, t)

            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        def make_op_fn():
            def f(lgt, t):
                return opake_cross_entropy(lgt, t)

            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        pt_stats = measure_time_and_memory(make_pt_fn(), logits, labels)
        op_stats = measure_time_and_memory(make_op_fn(), logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label=f"CE vmap(grad) (V={vocab_size})")


# ============================================================================
# Peak Memory Tests
# ============================================================================


class TestCrossEntropyMemory:
    """Backward allocates only the required logits-gradient-sized output."""

    @staticmethod
    def _assert_one_gradient_buffer(stats, grad):
        grad_bytes = grad.numel() * grad.element_size()
        assert (
            grad_bytes <= stats["allocated_bytes"] <= (grad_bytes + MEMORY_SLACK_BYTES)
        ), stats
        assert stats["peak_bytes"] <= grad_bytes + MEMORY_SLACK_BYTES, stats

    def test_standalone_backward_peak(self, measure_incremental_cuda_memory):
        """Standalone backward has no second full-vocabulary allocation."""
        torch.manual_seed(44)
        logits = torch.randn(
            MEMORY_TEST_ROWS,
            MEMORY_TEST_VOCAB,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        labels = torch.randint(0, MEMORY_TEST_VOCAB, (MEMORY_TEST_ROWS,), device="cuda")

        def run_backward():
            loss = opake_cross_entropy(logits, labels)
            return torch.autograd.grad(loss, logits)[0]

        grad_logits, stats = measure_incremental_cuda_memory(run_backward)

        self._assert_one_gradient_buffer(stats, grad_logits)
        assert grad_logits.is_contiguous()
        assert grad_logits.data_ptr() != logits.data_ptr()

    def test_vmap_backward_peak(self, measure_incremental_cuda_memory):
        """Vmapped backward has one gradient buffer for all mapped rows."""
        torch.manual_seed(45)
        vmap_batch = 4
        rows_per_sample = MEMORY_TEST_ROWS // vmap_batch
        logits = torch.randn(
            vmap_batch,
            rows_per_sample,
            MEMORY_TEST_VOCAB,
            device="cuda",
            dtype=torch.bfloat16,
        )
        labels = torch.randint(
            0,
            MEMORY_TEST_VOCAB,
            (vmap_batch, rows_per_sample),
            device="cuda",
        )
        mapped_grad = vmap(grad(opake_cross_entropy, argnums=0), in_dims=(0, 0))

        grad_logits, stats = measure_incremental_cuda_memory(
            lambda: mapped_grad(logits, labels)
        )

        self._assert_one_gradient_buffer(stats, grad_logits)
        assert grad_logits.is_contiguous()
        assert grad_logits.data_ptr() != logits.data_ptr()


# ============================================================================
# Performance Tests
# ============================================================================


class TestCrossEntropyPerformance:
    """Benchmark forward and backward performance."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Forward-only: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits = torch.randn(
            batch, seq_len, vocab_size, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        def pytorch_fn(lgt, t):
            return pytorch_cross_entropy(lgt, t)

        def opake_fn(lgt, t):
            return opake_cross_entropy(lgt, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opake_fn, logits, labels)

        assert_perf_benefit(
            pt_stats,
            op_stats,
            label=f"CE forward (V={vocab_size})",
            max_perf_overhead=0.60,
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config, vocab_size
    ):
        """Forward+backward: opake vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits = torch.randn(
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        def pytorch_fn(lgt, t):
            return pytorch_cross_entropy(lgt, t)

        def opake_fn(lgt, t):
            return opake_cross_entropy(lgt, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opake_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label=f"CE backward (V={vocab_size})")


# ============================================================================
# Softcapping and Logit Scaling Tests
# ============================================================================


class TestCrossEntropySoftcapping:
    """Test logit softcapping (Gemma 2) and logit scaling (Cohere)."""

    def test_softcapping_forward(self, assert_precision, mellum_config):
        """Softcapping forward matches PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        softcap = 30.0

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # PyTorch reference: manually apply softcapping then F.cross_entropy
        capped = softcap * torch.tanh(logits / softcap)
        ref = F.cross_entropy(
            capped.reshape(-1, vocab), labels.reshape(-1), reduction="none"
        ).reshape(batch, seq_len)

        # Opake: pass raw logits + softcap param
        losses_op, _ = Opake_CrossEntropyLoss.apply(logits, labels, softcap, 0, 0.0)

        print(f"\nSoftcapping forward (V={vocab}):")
        assert_precision(losses_op, ref, rtol=1e-4, atol=1e-6, label="per-token losses")

    def test_softcapping_backward(self, assert_precision, mellum_config):
        """Softcapping backward matches PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        softcap = 30.0

        logits_pt = torch.randn(
            batch,
            seq_len,
            vocab,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # PyTorch reference
        capped = softcap * torch.tanh(logits_pt / softcap)
        loss_pt = F.cross_entropy(capped.reshape(-1, vocab), labels.reshape(-1))
        loss_pt.backward()

        # Opake
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        losses_op, _ = Opake_CrossEntropyLoss.apply(logits_op, labels, softcap, 0, 0.0)
        mask = (labels != -100).float()
        loss_op = (losses_op * mask).sum() / mask.sum().clamp(min=1)
        loss_op.backward()

        print(f"\nSoftcapping backward (V={vocab}):")
        assert_precision(
            logits_op.grad, logits_pt.grad, rtol=1e-3, atol=1e-5, label="logits.grad"
        )

    def test_logit_scaling_forward(self, assert_precision, mellum_config):
        """Logit scaling forward matches PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        logit_scale = 0.0625

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # PyTorch reference: manually apply scaling then F.cross_entropy
        scaled = logit_scale * logits
        ref = F.cross_entropy(
            scaled.reshape(-1, vocab), labels.reshape(-1), reduction="none"
        ).reshape(batch, seq_len)

        # Opake: pass raw logits + scale param
        losses_op, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, logit_scale, 0.0)

        print(f"\nLogit scaling forward (V={vocab}):")
        assert_precision(losses_op, ref, rtol=1e-4, atol=1e-6, label="per-token losses")

    def test_logit_scaling_backward(self, assert_precision, mellum_config):
        """Logit scaling backward matches PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        logit_scale = 0.0625

        logits_pt = torch.randn(
            batch,
            seq_len,
            vocab,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # PyTorch reference
        scaled = logit_scale * logits_pt
        loss_pt = F.cross_entropy(scaled.reshape(-1, vocab), labels.reshape(-1))
        loss_pt.backward()

        # Opake
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        losses_op, _ = Opake_CrossEntropyLoss.apply(
            logits_op, labels, 0, logit_scale, 0.0
        )
        mask = (labels != -100).float()
        loss_op = (losses_op * mask).sum() / mask.sum().clamp(min=1)
        loss_op.backward()

        print(f"\nLogit scaling backward (V={vocab}):")
        assert_precision(
            logits_op.grad, logits_pt.grad, rtol=1e-3, atol=1e-5, label="logits.grad"
        )

    def test_softcapping_vmap_grad(self, assert_precision, mellum_config):
        """Softcapping with vmap(grad) matches PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]
        softcap = 30.0

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        def f_pt(lgt, t):
            v = lgt.shape[-1]
            capped = softcap * torch.tanh(lgt / softcap)
            return F.cross_entropy(capped.reshape(-1, v), t.reshape(-1))

        def f_op(lgt, t):
            losses, _ = Opake_CrossEntropyLoss.apply(lgt, t, softcap, 0, 0.0)
            mask = (t != -100).float()
            return (losses * mask).sum() / mask.sum().clamp(min=1)

        grads_pt = vmap(grad(f_pt, argnums=0), in_dims=(0, 0))(logits, labels)
        grads_op = vmap(grad(f_op, argnums=0), in_dims=(0, 0))(logits, labels)

        print(f"\nSoftcapping vmap(grad) (V={vocab}):")
        assert_precision(
            grads_op, grads_pt, rtol=1e-3, atol=1e-5, label="per-example gradients"
        )

    def test_combined_softcapping_and_scaling(self, assert_precision, mellum_config):
        """Both softcapping and scaling applied together."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        softcap = 30.0
        logit_scale = 0.0625

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # PyTorch reference: scale then softcap (same order as kernel)
        transformed = softcap * torch.tanh((logit_scale * logits) / softcap)
        ref = F.cross_entropy(
            transformed.reshape(-1, vocab), labels.reshape(-1), reduction="none"
        ).reshape(batch, seq_len)

        # Opake: pass raw logits + both params
        losses_op, _ = Opake_CrossEntropyLoss.apply(
            logits, labels, softcap, logit_scale, 0.0
        )

        print(f"\nCombined softcap+scaling forward (V={vocab}):")
        assert_precision(losses_op, ref, rtol=1e-4, atol=1e-6, label="per-token losses")


# ============================================================================
# Label Smoothing Tests
# ============================================================================


class TestCrossEntropyLabelSmoothing:
    """``label_smoothing`` matches ``F.cross_entropy(..., label_smoothing=ls)``."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("label_smoothing", [0.05, 0.1, 0.2])
    def test_forward_matches_pytorch(
        self, assert_precision, mellum_config, vocab_size, label_smoothing
    ):
        """Forward parity across single-chunk and chunked vocabs."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits = torch.randn(
            batch, seq_len, vocab_size, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        ref = F.cross_entropy(
            logits_flat,
            labels_flat,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        losses_op, _ = Opake_CrossEntropyLoss.apply(
            logits, labels, 0, 0, label_smoothing
        )

        print(f"\nCE smoothed forward (vocab={vocab_size}, ls={label_smoothing}):")
        assert_precision(
            losses_op.reshape(-1),
            ref,
            rtol=RTOL_CE_FORWARD,
            atol=ATOL_CE_FORWARD,
            label="per-token smoothed losses",
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("label_smoothing", [0.05, 0.1, 0.2])
    def test_backward_matches_pytorch(
        self, assert_precision, mellum_config, vocab_size, label_smoothing
    ):
        """Backward parity: smoothed gradient matches ``F.cross_entropy``."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits_pt = torch.randn(
            batch,
            seq_len,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        # PyTorch reference (mean reduction over valid positions)
        ref_loss = F.cross_entropy(
            logits_pt.reshape(-1, vocab_size),
            labels.reshape(-1),
            label_smoothing=label_smoothing,
            reduction="mean",
        )
        ref_loss.backward()

        # Opake kernel — match the mean reduction inside opake_cross_entropy
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        losses, _ = Opake_CrossEntropyLoss.apply(
            logits_op, labels, 0, 0, label_smoothing
        )
        mask = (labels.reshape(-1) != -100).float()
        n_valid = mask.sum()
        op_loss = (losses.reshape(-1) * mask).sum() / torch.clamp(n_valid, min=1.0)
        op_loss.backward()

        print(f"\nCE smoothed backward (vocab={vocab_size}, ls={label_smoothing}):")
        assert_precision(
            logits_op.grad,
            logits_pt.grad,
            rtol=RTOL_CE_BACKWARD,
            atol=ATOL_CE_BACKWARD,
            label="logits.grad (smoothed)",
        )

    def test_ignore_index_keeps_zero_grad_with_smoothing(self, mellum_config):
        """``label == -100`` positions still produce zero gradient under smoothing."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(
            batch,
            seq_len,
            vocab,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        labels[:, -10:] = -100

        losses, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, 0.1)
        losses.sum().backward()

        masked_grad = logits.grad[:, -10:, :]
        assert masked_grad.abs().max() == 0.0, (
            f"Non-zero grad at -100 positions under smoothing: "
            f"max={masked_grad.abs().max():.2e}"
        )

        # Smoothed loss at -100 positions should be zero too.
        losses_only, _ = Opake_CrossEntropyLoss.apply(
            logits.detach(), labels, 0, 0, 0.1
        )
        masked_losses = losses_only[:, -10:]
        assert masked_losses.abs().max() == 0.0, (
            f"Non-zero smoothed loss at -100 positions: "
            f"max={masked_losses.abs().max():.2e}"
        )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_zero_smoothing_matches_standard_ce(
        self, assert_precision, mellum_config, vocab_size
    ):
        """``label_smoothing=0`` must match plain ``F.cross_entropy`` exactly.

        Guards against accidental activation of the smoothing-gated forward
        rewrite when ``DO_LABEL_SMOOTHING`` is False — the smoothing-off
        path must still produce standard CE.
        """
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]

        logits = torch.randn(
            batch, seq_len, vocab_size, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

        losses_op, _ = Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, 0.0)
        ref = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="none",
        )

        assert_precision(
            losses_op.reshape(-1),
            ref,
            rtol=RTOL_CE_FORWARD,
            atol=ATOL_CE_FORWARD,
            label="label_smoothing=0 vs F.cross_entropy",
        )

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, 2.0])
    def test_out_of_range_smoothing_raises(self, mellum_config, bad_value):
        """``label_smoothing`` outside [0.0, 1.0] raises ValueError early."""
        torch.manual_seed(42)
        vocab_size = mellum_config["vocab_size"]
        logits = torch.randn(2, 8, vocab_size, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (2, 8), device="cuda")

        with pytest.raises(ValueError, match="label_smoothing"):
            Opake_CrossEntropyLoss.apply(logits, labels, 0, 0, bad_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
