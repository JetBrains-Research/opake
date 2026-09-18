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
class TestVmapCompatibility:
    """Test that patched modules work under vmap for DP-SGD."""

    @requires_hf_auth
    def test_vmap_patched_mlp(self, device):
        """Patched MLP should produce correct output under vmap."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1
        mlp = LlamaMLP(config).to(device)
        x = torch.randn(4, 2, 16, config.hidden_size, device=device)
        out = torch.vmap(mlp)(x)
        assert not torch.isnan(out).any(), "NaN in vmap MLP output"
        for i in range(x.shape[0]):
            ref = mlp(x[i])
            assert torch.allclose(out[i], ref, rtol=RTOL, atol=ATOL), (
                f"vmap output[{i}] mismatch vs sequential"
            )
