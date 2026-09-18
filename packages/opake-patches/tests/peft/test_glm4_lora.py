"""Test fused LoRA QKV patching for GLM4 (Opake_LoRA_QKV kernel).

Glm4Attention has separate q_proj/k_proj/v_proj (no combined qkv_proj), no
Q/K normalization, and a standard RoPE + attention flow — the same shape the
fused QKV wrapper already supports for Qwen2. GLM4 defaults
``attention_bias=True``, so these tests exercise the frozen-bias fusion path
and confirm trainable biases still fall back to the unfused path.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.models.glm4.modeling_glm4 import Glm4Config

from opake.api.engine.clipping import clipped_grad
from opake.functional import make_functional
from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)


def _tiny_glm4_config(**overrides):
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_bias": True,
        # GLM4's default pad_token_id (151329) exceeds this tiny vocab_size.
        "pad_token_id": 0,
    }
    kwargs.update(overrides)
    return Glm4Config(**kwargs)


@pytest.mark.cuda
class TestFusedLoRAQKVGlm4:
    """GLM4 attention fuses its frozen biased Q/K/V projections."""

    def test_glm4_fuses_biased_qkv_projections(self, device):
        """GLM4 attention fuses Q/K/V when the base-layer bias is frozen."""
        config = _tiny_glm4_config()
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert hasattr(attn, "_opake_fused_qkv"), (
                "GLM4 attention should have _opake_fused_qkv after auto-fuse"
            )
            assert attn.q_proj.base_layer.bias is not None
            assert not attn.q_proj.base_layer.bias.requires_grad

    def test_glm4_skips_qkv_fusion_with_trainable_bias(self, device):
        """QKV fusion preserves gradients for PEFT configurations that train biases."""
        config = _tiny_glm4_config(num_hidden_layers=1)
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="all",
                target_modules=["q_proj", "k_proj", "v_proj"],
            ),
        ).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        attn = model.model.model.layers[0].self_attn
        assert attn.q_proj.base_layer.bias.requires_grad
        assert not hasattr(attn, "_opake_fused_qkv")

    def test_glm4_fused_qkv_model_forward_backward(self, device):
        """Full GLM4 model with fused LoRA QKV should produce valid forward+backward."""
        config = _tiny_glm4_config()
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj"],
            ),
        ).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        for layer in model.model.model.layers:
            assert hasattr(layer.self_attn, "_opake_fused_qkv")

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        assert not torch.isnan(loss), "NaN loss from fused LoRA QKV GLM4 model"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert has_grad, "No gradients computed"

    def test_glm4_biased_qkv_clipped_grad(self, device):
        """Fused biased GLM4 QKV projections support per-example gradients."""
        config = _tiny_glm4_config(num_hidden_layers=1)
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj"],
            ),
        ).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)

        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
        outputs = model(input_ids, labels=input_ids)
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, labels):
            return fmodel(
                {**frozen_params, **trainable_params}, ids, labels=labels
            ).loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, input_ids, state=clip_state)
        assert all(torch.isfinite(grad).all() for grad in grads.pytree.values())

    def test_glm4_vmap_grad_matches_eager(self, device):
        """vmap(grad()) per-sample LoRA grads should match a per-sample eager loop.

        GLM4 uses an interleaved, partial-rotary ``apply_rotary_pos_emb`` (not
        LLaMA's contiguous rotate_half); the fused QKV wrapper resolves that
        function dynamically from the model's own module, so this test also
        guards against any RoPE-shape mismatch that fusion could introduce.
        """
        tf32_prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            config = _tiny_glm4_config(num_hidden_layers=1)
            model = AutoModelForCausalLM.from_config(config)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config).to(device)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.05)
            apply_model_patches(model, performance=False, compat=True, lora=True)
            model.eval()
            assert hasattr(model.model.model.layers[0].self_attn, "_opake_fused_qkv")

            fmodel, trainable, frozen = make_functional(
                model, disable_autograd_tracking=True, partition_trainable=True
            )

            batch, seq = 4, 16
            input_ids = torch.randint(0, config.vocab_size, (batch, seq), device=device)
            attention_mask = torch.ones_like(input_ids)

            def per_example_loss(tp, ids, mask):
                merged = {**frozen, **tp}
                out = fmodel(merged, input_ids=ids, attention_mask=mask)
                return out.logits.float().square().mean()

            vmap_grads = torch.vmap(
                torch.func.grad(per_example_loss), in_dims=(None, 0, 0)
            )(trainable, input_ids, attention_mask)

            for i in range(batch):
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

    def test_glm4_qkv_fusion_matches_unfused_forward(self, device):
        """Fused QKV forward output should match the unfused reference forward."""
        torch.manual_seed(0)
        config = _tiny_glm4_config(num_hidden_layers=1)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )

        ref_model = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        ref_model.eval()

        fused_model = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        fused_model.load_state_dict(ref_model.state_dict())
        apply_model_patches(fused_model, performance=False, compat=True, lora=True)
        fused_model.eval()
        assert hasattr(fused_model.model.model.layers[0].self_attn, "_opake_fused_qkv")

        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
            fused_logits = fused_model(input_ids).logits
        assert torch.allclose(ref_logits, fused_logits, rtol=1e-3, atol=1e-3), (
            f"Fused/unfused GLM4 logits mismatch: "
            f"max diff {(ref_logits - fused_logits).abs().max():.3e}"
        )
