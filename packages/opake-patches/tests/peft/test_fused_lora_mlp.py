import pytest
import torch
import torch.nn.functional as F
from opake_test_support import requires_hf_auth
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestFusedLoRAMLP:
    """Test fused LoRA MLP patching via Opake_LoRA_MLP kernel."""

    def test_biased_up_projection_is_not_fused(self, device):
        """MLPs with an up-projection bias must retain their original forward."""
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            mlp_bias=True,
        )
        model = AutoModelForCausalLM.from_config(config)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
        ).to(device)

        mlp = model.model.model.layers[0].mlp
        assert mlp.up_proj.base_layer.bias is not None
        apply_model_patches(model, performance=False, compat=True, lora=True)
        assert "forward" not in vars(mlp)

    def test_fused_lora_mlp_forward(self, device):
        """Fused LoRA MLP forward should match PyTorch matmul reference."""
        from opake.api.patches.kernels.lora import Opake_LoRA_MLP

        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = (2, 16, 256, 512, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)
        Wg = torch.randn(intermediate, hidden, device=device)
        Wu = torch.randn(intermediate, hidden, device=device)
        Wd = torch.randn(hidden, intermediate, device=device)
        Ag = torch.randn(hidden, rank, device=device)
        Bg = torch.randn(rank, intermediate, device=device)
        Au = torch.randn(hidden, rank, device=device)
        Bu = torch.randn(rank, intermediate, device=device)
        Ad = torch.randn(intermediate, rank, device=device)
        Bd = torch.randn(rank, hidden, device=device)
        out_fused, _, _, _ = Opake_LoRA_MLP.apply(
            x, Wg, Ag, Bg, scaling, Wu, Au, Bu, scaling, Wd, Ad, Bd, scaling, 0
        )
        gate = F.linear(x, Wg) + x @ Ag @ Bg * scaling
        up = F.linear(x, Wu) + x @ Au @ Bu * scaling
        h = F.silu(gate) * up
        ref = F.linear(h, Wd) + h @ Ad @ Bd * scaling
        assert torch.allclose(out_fused, ref, rtol=0.001, atol=0.001), (
            f"Fused LoRA MLP output mismatch: max diff {(out_fused - ref).abs().max():.2e}"
        )

    def test_fused_lora_mlp_backward(self, device):
        """Fused LoRA MLP should produce correct gradients."""
        from opake.api.patches.kernels.lora import Opake_LoRA_MLP

        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = (2, 16, 256, 512, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
        Wg = torch.randn(intermediate, hidden, device=device)
        Wu = torch.randn(intermediate, hidden, device=device)
        Wd = torch.randn(hidden, intermediate, device=device)
        Ag = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bg = torch.randn(rank, intermediate, device=device, requires_grad=True)
        Au = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bu = torch.randn(rank, intermediate, device=device, requires_grad=True)
        Ad = torch.randn(intermediate, rank, device=device, requires_grad=True)
        Bd = torch.randn(rank, hidden, device=device, requires_grad=True)
        out, _, _, _ = Opake_LoRA_MLP.apply(
            x, Wg, Ag, Bg, scaling, Wu, Au, Bu, scaling, Wd, Ad, Bd, scaling, 0
        )
        out.sum().backward()
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"
        assert Ag.grad is not None, "No gradient for gate LoRA A"
        assert Bd.grad is not None, "No gradient for down LoRA B"

    def test_fused_lora_mlp_vmap_grad_matches_eager(self, device):
        """Per-sample LoRA grads via vmap(grad(...)) must match plain autograd.

        Regression for the ``_LoRAMLPBackward.vmap`` buffer-reuse bug: dX was
        written into X's storage (``torch.mm(..., out=X_flat)``) *before* the
        per-sample dA/dB bmm's read ``X_3d`` (a view of the same storage), so
        every gate/up ``lora_B`` gradient was computed against dX values
        instead of the activations — garbage gradients that silently poisoned
        non-DP SFT training while losses stayed exact.

        Mirrors the production DP-SGD path at micro scale: patched PEFT model,
        ``make_functional``, ``vmap(grad(per_example_loss))`` — compared
        against a per-sample plain-autograd loop through the SAME patched
        model (whose eager backward is verified against vanilla HF elsewhere).
        """
        from transformers import Qwen2Config

        from opake.functional import make_functional

        # TF32 (default-on in NVIDIA containers) rounds fp32 matmuls to ~1e-3
        # relative; the vmap path (bmm) and the eager loop (mm) round
        # differently, which would swamp the tolerance below.
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
            )
            model = AutoModelForCausalLM.from_config(config)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config).to(device)
            # Non-zero lora_B so dA grads are informative too (PEFT inits B=0,
            # where dA is identically zero and a corrupted dA would be missed).
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.05)
            apply_model_patches(model, performance=True, compat=True, kernels=False)
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
                    # The clobber bug produced rel ≈ 1.0 with cosine ≈ 0; a
                    # 1e-2 ceiling leaves two orders of magnitude of margin
                    # over fp32 kernel-vs-eager numerics.
                    assert rel < 1e-2, (
                        f"vmap per-sample grad mismatch for {k}[{i}]: "
                        f"rel={rel:.3e} max diff {(got - ref.grad).abs().max():.3e}"
                    )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_prev

    @requires_hf_auth
    def test_apply_model_patches_on_peft_model(self, device):
        """apply_model_patches() should fuse MLP layers on a PEFT-wrapped model."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        layers = model.model.model.layers
        for layer in layers:
            mlp = layer.mlp
            assert "forward" in vars(mlp), "MLP forward should be fused"

    @requires_hf_auth
    def test_fused_lora_mlp_model_forward_backward(self, device):
        """Full model with fused LoRA MLP should produce valid forward+backward."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        assert not torch.isnan(loss), "NaN loss from fused LoRA MLP model"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert has_grad, "No gradients computed"

    @requires_hf_auth
    def test_apply_peft_model_patches_manual(self, device):
        """apply_peft_model_patches() should work for manually loaded PEFT models."""
        from opake.patches.peft import apply_peft_model_patches

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        from peft.mapping_func import get_peft_model as raw_get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model = raw_get_peft_model(model, lora_config).to(device)
        layers = model.model.model.layers
        assert "forward" not in vars(layers[0].mlp), (
            "MLP should not be fused before apply_peft_model_patches()"
        )
        apply_peft_model_patches(model)
        assert "forward" in vars(layers[0].mlp), (
            "MLP should be fused after apply_peft_model_patches()"
        )
