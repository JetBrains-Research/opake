"""Test fused LoRA kernels on parallel-residual architectures (Cohere/Cohere2).

In these models, the same normalized input feeds both the attention-QKV and
MLP branches. Previously, the MLP backward path (both eager and vmap)
overwrote this shared buffer with dX, corrupting the QKV branch's per-sample
LoRA weight gradients when it ran afterward.

Refs: https://github.com/JetBrains-Research/opake/issues/401
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Cohere2Config

from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)


@pytest.mark.cuda
class TestFusedLoRAParallelResidual:
    """Fused LoRA kernels must not mutate the shared activation in parallel-residual layers."""

    def test_cohere2_vmap_grad_matches_eager(self, device):
        """vmap(grad()) per-sample LoRA grads should match per-sample eager loop.

        Regression test for https://github.com/JetBrains-Research/opake/issues/401

        Cohere2 uses a parallel-residual architecture where the same normalized
        input feeds both attention-QKV and MLP branches. If the MLP backward
        path overwrites the shared input buffer before QKV backward reads it,
        the per-sample LoRA weight gradients for attention projections are
        computed against corrupted data (dX values instead of activations).
        """
        from opake.functional import make_functional

        # Disable TF32 so fp32 matmuls are deterministic across bmm/mm paths.
        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            config = Cohere2Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                max_position_embeddings=64,
            )
            model = AutoModelForCausalLM.from_config(config)
            # LoRA on both QKV and MLP projections
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",  # attention-side
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # MLP-side
                ],
            )
            model = get_peft_model(model, lora_config).to(device)
            # Non-zero lora_B so dA grads are informative (PEFT inits B=0).
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.05)
            apply_model_patches(model, performance=True, compat=True, kernels=True)
            model.eval()

            fmodel, trainable, frozen = make_functional(
                model, disable_autograd_tracking=True, partition_trainable=True
            )

            B, T = 4, 16
            input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
            attention_mask = torch.ones_like(input_ids)

            def per_example_loss(tp, ids, mask):
                merged = {**frozen, **tp}
                out = fmodel(merged, input_ids=ids, attention_mask=mask)
                return out.logits.float().square().mean()

            # vmap(grad()) path
            vmap_grads = torch.vmap(
                torch.func.grad(per_example_loss), in_dims=(None, 0, 0)
            )(trainable, input_ids, attention_mask)

            # Per-sample eager loop
            for i in range(B):
                tp = {
                    k: v.detach().clone().requires_grad_(True)
                    for k, v in trainable.items()
                }
                per_example_loss(tp, input_ids[i], attention_mask[i]).backward()
                for k, ref in tp.items():
                    got = vmap_grads[k][i]
                    rel = (got - ref.grad).norm() / ref.grad.norm().clamp(min=1e-12)
                    assert rel < 1e-2, (
                        f"vmap per-sample grad mismatch for {k}[{i}]: "
                        f"rel={rel:.3e} max diff {(got - ref.grad).abs().max():.3e}"
                    )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev

    def test_cohere2_attention_lora_grads_isolated(self, device):
        """Isolate attention-side LoRA gradients to catch QKV X-corruption directly.

        By comparing vmap(grad()) vs eager for attention-only LoRA params,
        we specifically target the bug where MLP backward overwrites the shared
        X buffer before QKV backward reads it for _per_sample_lora_grads().
        """
        from opake.functional import make_functional

        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(42)
            config = Cohere2Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                max_position_embeddings=64,
            )
            model = AutoModelForCausalLM.from_config(config)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",  # attention only
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # MLP also present
                ],
            )
            model = get_peft_model(model, lora_config).to(device)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.05)
            apply_model_patches(model, performance=True, compat=True, kernels=True)
            model.eval()

            fmodel, trainable, frozen = make_functional(
                model, disable_autograd_tracking=True, partition_trainable=True
            )

            # Only check attention-side LoRA params (the ones affected by X corruption)
            attention_keys = [
                k
                for k in trainable
                if any(s in k for s in ("q_proj", "k_proj", "v_proj"))
            ]
            assert attention_keys, "No attention-side LoRA params found"

            B, T = 4, 16
            input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
            attention_mask = torch.ones_like(input_ids)

            def per_example_loss(tp, ids, mask):
                merged = {**frozen, **tp}
                out = fmodel(merged, input_ids=ids, attention_mask=mask)
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
                for k in attention_keys:
                    ref = tp[k]
                    got = vmap_grads[k][i]
                    rel = (got - ref.grad).norm() / ref.grad.norm().clamp(min=1e-12)
                    assert rel < 1e-2, (
                        f"Attention LoRA grad mismatch for {k}[{i}]: "
                        f"rel={rel:.3e} (X corruption suspected)"
                    )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev
