import pytest
import torch
import torch.nn.functional as F
from opake_test_support import requires_hf_auth
from transformers import AutoConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestMLPPatches:
    """Test that patched MLP forward produces correct outputs."""

    @requires_hf_auth
    def test_swiglu_mlp_matches(self, device):
        """Patched LlamaMLP should match PyTorch reference."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1
        mlp = LlamaMLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"LlamaMLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    @requires_hf_auth
    def test_geglu_exact_mlp_matches(self, device):
        """Patched GemmaMLP should match PyTorch reference."""
        try:
            from transformers.models.gemma.modeling_gemma import GemmaMLP
        except ImportError:
            pytest.skip("Gemma not available")
        config = AutoConfig.from_pretrained("google/gemma-2b")
        config.num_hidden_layers = 1
        mlp = GemmaMLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate="none") * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"GemmaMLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    @requires_hf_auth
    def test_geglu_approx_mlp_matches(self, device):
        """Patched Gemma2MLP should match PyTorch reference."""
        try:
            from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP
        except ImportError:
            pytest.skip("Gemma2 not available")
        config = AutoConfig.from_pretrained("google/gemma-2-2b")
        config.num_hidden_layers = 1
        mlp = Gemma2MLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate="tanh") * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"Gemma2MLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )
