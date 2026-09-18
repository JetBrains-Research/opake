import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestCrossEntropyPatches:
    """Test that patched cross-entropy loss produces correct outputs."""

    def test_causal_lm_loss_matches_pytorch(self, device):
        """Patched ForCausalLMLoss should match F.cross_entropy reference."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        labels[:, -2:] = -100
        loss = _opake_causal_lm_loss(logits, labels, vocab_size)
        import torch.nn as nn

        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)
        assert torch.allclose(loss, ref, rtol=0.001, atol=0.001), (
            f"Cross-entropy loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )

    def test_backward_through_patched_loss(self, device):
        """Gradients should flow through patched cross-entropy loss."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(
            batch, seq_len, vocab_size, device=device, requires_grad=True
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        loss = _opake_causal_lm_loss(logits, labels, vocab_size)
        loss.backward()
        assert logits.grad is not None, "No gradient through patched loss"
        assert not torch.isnan(logits.grad).any(), "NaN in loss gradients"
        assert not torch.isinf(logits.grad).any(), "Inf in loss gradients"

    def test_label_smoothing_flows_through_kwargs(self, device):
        """``label_smoothing`` propagates from loss_function kwargs to the kernel."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        torch.manual_seed(0)
        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        labels[:, -2:] = -100
        ls = 0.1

        # Opake path: forward smoothing via kwargs (mirrors HF's loss_kwargs).
        loss = _opake_causal_lm_loss(logits, labels, vocab_size, label_smoothing=ls)

        # Reference: HF's shift + masked F.cross_entropy with smoothing.
        import torch.nn as nn

        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(
            logits_flat,
            shift_labels_flat,
            ignore_index=-100,
            label_smoothing=ls,
        )
        assert torch.allclose(loss, ref, rtol=RTOL, atol=ATOL), (
            f"Smoothed CE mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )
