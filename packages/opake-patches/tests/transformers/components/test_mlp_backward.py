import pytest
import torch
from opake_test_support import requires_hf_auth
from transformers import AutoConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestGradients:
    """Test that gradients through patched modules are correct."""

    @requires_hf_auth
    def test_backward_through_patched_mlp(self, device):
        """Gradients should flow correctly through patched MLP."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1
        mlp = LlamaMLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient computed through patched MLP"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"
