"""Full fine-tuning vmap(grad) parity for the patched RMSNorm kernel (DP-SGD).

Regression for the ``_RMSNormBackward.vmap`` batch-summed dW bug: with
trainable norm weights, every example received the batch-sum as its
"per-example" dW.  Invisible to LoRA runs (norm weights frozen).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from opake.functional import make_functional
from opake.patches import apply_model_patches

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)


@pytest.mark.cuda
class TestRMSNormFullFTVmapGrad:
    def test_full_ft_vmap_grad_matches_eager(self, device):
        """Full-FT per-sample grads via vmap(grad(...)) must match plain autograd."""
        # TF32 rounds fp32 matmuls differently in the vmap path vs the eager
        # loop, which would swamp the tolerance below.
        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            config = Qwen2Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                tie_word_embeddings=False,
            )
            model = AutoModelForCausalLM.from_config(config)
            model = model.to(device)
            apply_model_patches(model, performance=True, compat=True)
            model.eval()

            norm_weights = [
                n for n, _ in model.named_parameters() if "norm" in n.lower()
            ]
            assert norm_weights, "expected trainable RMSNorm weights in full FT"

            fmodel, trainable, frozen = make_functional(
                model, disable_autograd_tracking=True, partition_trainable=True
            )
            assert any("norm" in k.lower() for k in trainable), (
                "RMSNorm weights must be in the trainable partition"
            )

            B, T = 4, 16
            input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
            attention_mask = torch.ones_like(input_ids)

            def per_example_loss(tp, ids, mask):
                merged = {**frozen, **tp}
                out = fmodel(merged, input_ids=ids, attention_mask=mask)
                # Nonlinear reduction so grad_out is non-constant.
                return out.logits.float().square().mean()

            vmap_grads = torch.vmap(
                torch.func.grad(per_example_loss), in_dims=(None, 0, 0)
            )(trainable, input_ids, attention_mask)

            for i in range(B):
                tp = {
                    k: v.detach().clone().requires_grad_(True)
                    for k, v in trainable.items()
                }
                per_example_loss(tp, input_ids[i], attention_mask[i]).backward()
                for k, ref in tp.items():
                    got = vmap_grads[k][i]
                    rel = (got - ref.grad).norm() / ref.grad.norm().clamp(min=1e-12)
                    # The batch-sum bug gave rel error O(1); 1e-2 leaves ample
                    # margin over fp32 kernel-vs-eager numerics.
                    assert rel < 1e-2, (
                        f"vmap per-sample grad mismatch for {k}[{i}]: "
                        f"rel={rel:.3e} max diff {(got - ref.grad).abs().max():.3e}"
                    )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev
