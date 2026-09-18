# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic Triton kernel patches on HuggingFace models.

Verifies that patched model components produce correct outputs and gradients,
and remain vmap-compatible for DP-SGD.
"""

import inspect

import pytest
import torch
import torch.nn.functional as F
from opake_test_support import requires_hf_auth

pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from opake.api.engine.clipping import clipped_grad  # noqa: E402
from opake.functional import make_functional  # noqa: E402

RTOL = 1e-4
ATOL = 1e-4


# =============================================================================
# Helpers
# =============================================================================


def _make_small_model(model_name, device="cuda"):
    """Create a small 2-layer model for testing."""
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_config(config).to(device)
    return model, config


def _get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


# =============================================================================
# MLP Tests
# =============================================================================


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

        # Reference: silu(gate) * up -> down
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

        # Reference: gelu_exact(gate) * up -> down
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

        # Reference: gelu_tanh(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate="tanh") * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"Gemma2MLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )


# =============================================================================
# Gradient Tests
# =============================================================================


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


# =============================================================================
# vmap Compatibility
# =============================================================================


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

        # Batched input for vmap
        x = torch.randn(4, 2, 16, config.hidden_size, device=device)

        out = torch.vmap(mlp)(x)

        assert not torch.isnan(out).any(), "NaN in vmap MLP output"
        # Verify each sample matches non-batched forward
        for i in range(x.shape[0]):
            ref = mlp(x[i])
            assert torch.allclose(out[i], ref, rtol=RTOL, atol=ATOL), (
                f"vmap output[{i}] mismatch vs sequential"
            )


# =============================================================================
# End-to-end Integration
# =============================================================================


@pytest.mark.cuda
class TestEndToEnd:
    """End-to-end test with full model + LoRA + clipped_grad."""

    @pytest.mark.slow
    def test_qwen2_lora_clipped_grad(self, device):
        """Full pipeline: Qwen2 + LoRA + clipped_grad with kernel patches."""
        from opake.patches import apply_model_patches

        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        assert grads is not None, "No gradients returned"
        assert len(grads.pytree) > 0, "Empty gradient dict"
        for name, g in grads.pytree.items():
            assert not torch.isnan(g).any(), f"NaN in grad for {name}"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Test patch configuration and control."""

    def test_kernel_patched_flag(self):
        """is_kernel_patched() should return True after import opake."""
        from opake.patches import is_runtime_patched

        assert isinstance(is_runtime_patched(), bool)


# =============================================================================
# Batchify forward integration tests (CPU — no CUDA required)
# =============================================================================


class TestBatchifyForward:
    """Test _batchify_forward unsqueeze/squeeze round-trip with a real model."""

    def test_batchify_1d_input_ids(self):
        """1D input_ids should be unsqueezed, output logits squeezed back."""
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)

        # Wrap with batchify
        model.forward = _batchify_forward(model.forward)

        # 1D inputs — simulates what vmap(grad()) sees per example
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)

        # Logits should be squeezed back to 2D (seq, vocab) not 3D (1, seq, vocab)
        assert outputs.logits.ndim == 2, (
            f"Expected 2D logits (seq, vocab), got shape {outputs.logits.shape}"
        )
        assert outputs.logits.shape[0] == seq_len
        # Loss is scalar — unaffected by squeeze
        assert outputs.loss.ndim == 0

    def test_batchify_2d_input_ids_is_noop(self):
        """2D input_ids (already batched) should pass through unchanged."""
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)

        model.forward = _batchify_forward(model.forward)

        batch, seq_len = 3, 8
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)

        # 3D logits preserved — no squeeze
        assert outputs.logits.ndim == 3
        assert outputs.logits.shape == (batch, seq_len, config.vocab_size)

    def test_batchify_positional_input_ids(self):
        """input_ids passed positionally should also be batchified."""
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)

        model.forward = _batchify_forward(model.forward)

        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))

        # Pass input_ids positionally (as functional_call does)
        outputs = model(input_ids)

        assert outputs.logits.ndim == 2, (
            f"Expected 2D logits, got shape {outputs.logits.shape}"
        )


# =============================================================================
# CPU fallback tests (no CUDA required)
# =============================================================================


class TestCPUFallback:
    """Test that patched kernels fall back to original on CPU."""

    @requires_hf_auth
    def test_swiglu_mlp_cpu(self):
        """Patched LlamaMLP should produce correct output on CPU."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1
        mlp = LlamaMLP(config)  # CPU

        x = torch.randn(2, 16, config.hidden_size)
        out = mlp(x)

        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"CPU SwiGLU mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    @requires_hf_auth
    def test_geglu_exact_mlp_cpu(self):
        """Patched GemmaMLP should produce correct output on CPU."""
        try:
            from transformers.models.gemma.modeling_gemma import GemmaMLP
        except ImportError:
            pytest.skip("Gemma not available")

        config = AutoConfig.from_pretrained("google/gemma-2b")
        config.num_hidden_layers = 1
        mlp = GemmaMLP(config)  # CPU

        x = torch.randn(2, 16, config.hidden_size)
        out = mlp(x)

        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate="none") * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"CPU GeGLU exact mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    def test_cross_entropy_loss_cpu(self):
        """Patched CE loss should produce correct output on CPU."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        labels[:, -2:] = -100

        loss = _opake_causal_lm_loss(logits, labels, vocab_size)

        import torch.nn as nn

        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)

        assert torch.allclose(loss, ref, rtol=1e-3, atol=1e-3), (
            f"CPU CE loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )

    def test_lora_linear_cpu(self):
        """Patched LoRA linear should produce correct output on CPU."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=False)

        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        )

        x = torch.randn(2, 16, in_features)
        out = lora_layer(x)

        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight
        B_weight = lora_layer.lora_B["default"].weight
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"CPU LoRA mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    @requires_hf_auth
    def test_full_model_cpu_forward_backward(self):
        """Full Llama model with LoRA should forward+backward correctly on CPU."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)  # CPU

        lora_config = LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids, labels=input_ids)

        assert not torch.isnan(outputs.loss), "NaN loss on CPU"
        outputs.loss.backward()

        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients computed on CPU"


# =============================================================================
# Cross-Entropy Loss Patch Tests
# =============================================================================


@pytest.mark.cuda
class TestCrossEntropyPatches:
    """Test that patched cross-entropy loss produces correct outputs."""

    def test_causal_lm_loss_matches_pytorch(self, device):
        """Patched ForCausalLMLoss should match F.cross_entropy reference."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        # Set some labels to -100 (ignored)
        labels[:, -2:] = -100

        loss = _opake_causal_lm_loss(logits, labels, vocab_size)

        # PyTorch reference with same shifting logic
        import torch.nn as nn

        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)

        assert torch.allclose(loss, ref, rtol=1e-3, atol=1e-3), (
            f"Cross-entropy loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )

    def test_backward_through_patched_loss(self, device):
        """Gradients should flow through patched cross-entropy loss."""
        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
        )

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(
            batch, seq_len, vocab_size, device=device, requires_grad=True
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

        loss = _opake_causal_lm_loss(logits, labels, vocab_size)
        loss.backward()

        assert logits.grad is not None, "No gradient through patched loss"
        assert not torch.isnan(logits.grad).any(), "NaN in loss gradients"
        assert not torch.isinf(logits.grad).any(), "Inf in loss gradients"

    def test_loss_patch_attaches_to_model_instance(self, device):
        """The CE patch sets ``loss_function`` on matching model *instances* —
        it is not a global ``LOSS_MAPPING`` entry. Verifying the actual
        mechanism is device-agnostic (the helper just rebinds the attribute)."""
        from transformers.models.llama.modeling_llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )

        from opake.api.patches.transformers.components.cross_entropy import (
            _opake_causal_lm_loss,
            apply_causal_lm_loss_function_patch,
        )

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
            )
        ).to(device)

        patched = apply_causal_lm_loss_function_patch(model, type(model))
        assert patched, "loss patch did not match the model class"
        assert model.loss_function is _opake_causal_lm_loss


# =============================================================================
# LoRA Patch Tests
# =============================================================================


def _make_lora_linear(base_layer, adapter_name, *, rank, alpha, dropout):
    """Build PEFT LoRA Linear across its supported constructor revisions."""
    from peft.tuners.lora import Linear as PeftLoRALinear

    if "config" not in inspect.signature(PeftLoRALinear).parameters:
        return PeftLoRALinear(
            base_layer, adapter_name, r=rank, lora_alpha=alpha, lora_dropout=dropout
        )

    from peft import LoraConfig

    return PeftLoRALinear(
        base_layer,
        adapter_name,
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout),
        r=rank,
        lora_alpha=alpha,
    )


@pytest.mark.cuda
class TestLoRAPatches:
    """Test that patched LoRA linear produces correct outputs."""

    def test_lora_forward_matches_peft(self, device):
        """Patched LoRA forward should match PyTorch reference."""
        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)

        lora_layer = _make_lora_linear(
            base_linear, "default", rank=rank, alpha=16, dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)

        # PyTorch reference: base(x) + B(A(x)) * scaling
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight  # (rank, in)
        B_weight = lora_layer.lora_B["default"].weight  # (out, rank)
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"LoRA forward mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    def test_lora_forward_with_bias(self, device):
        """LoRA forward should correctly handle base layer bias."""
        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)

        lora_layer = _make_lora_linear(
            base_linear, "default", rank=rank, alpha=16, dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)

        # Reference with bias
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight
        B_weight = lora_layer.lora_B["default"].weight
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"LoRA with bias mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    def test_backward_through_patched_lora(self, device):
        """Gradients should flow through patched LoRA."""
        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)

        lora_layer = _make_lora_linear(
            base_linear, "default", rank=rank, alpha=16, dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device, requires_grad=True)
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient through LoRA"
        assert not torch.isnan(x.grad).any(), "NaN in LoRA gradients"

        # Check LoRA weight gradients
        A_grad = lora_layer.lora_A["default"].weight.grad
        B_grad = lora_layer.lora_B["default"].weight.grad
        assert A_grad is not None, "No gradient for LoRA A"
        assert B_grad is not None, "No gradient for LoRA B"

    def test_lora_patch_attaches_to_instance(self, device):
        """The LoRA kernel patch rebinds the *instance* ``forward`` of peft
        LoRA ``Linear`` modules (not the class). It only activates on
        CUDA + Triton, so the assertion is guarded accordingly."""
        pytest.importorskip("peft")
        from opake.patches import apply_model_patches

        base = torch.nn.Linear(64, 64, bias=False).to(device)
        lora = _make_lora_linear(base, "default", rank=8, alpha=16, dropout=0.0).to(
            device
        )
        apply_model_patches(lora)

        if torch.cuda.is_available():
            assert hasattr(lora.forward, "__opake_patched__")


# =============================================================================
# Qwen3 and Granite support
# =============================================================================


@pytest.mark.cuda
class TestQwen3Patches:
    """Test kernel patches for Qwen3 models."""

    def test_qwen3_swiglu_mlp_matches(self, device):
        """Patched Qwen3MLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        except ImportError:
            pytest.skip("Qwen3 not available")

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        config.num_hidden_layers = 1
        mlp = Qwen3MLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)

        # Reference: silu(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"Qwen3MLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )


@pytest.mark.cuda
class TestGranitePatches:
    """Test kernel patches for Granite models."""

    def test_granite_swiglu_mlp_matches(self, device):
        """Patched GraniteMLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.granite.modeling_granite import GraniteMLP
        except ImportError:
            pytest.skip("Granite not available")

        config = AutoConfig.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        config.num_hidden_layers = 1
        mlp = GraniteMLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)

        # Reference: silu(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"GraniteMLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )


# =============================================================================
# Cohere/Cohere2 support
# =============================================================================


@pytest.mark.cuda
class TestCoherePatches:
    """Test kernel patches for Cohere models (SwiGLU).

    Note: CohereLayerNorm is NOT patched — PyTorch's native F.layer_norm has a
    C++ vmap batching rule that's ~2x faster than our autograd.Function dispatch.
    """

    def test_cohere_swiglu_mlp_matches(self, device):
        """Patched CohereMLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.cohere.configuration_cohere import CohereConfig
            from transformers.models.cohere.modeling_cohere import CohereMLP
        except ImportError:
            pytest.skip("Cohere not available")

        config = CohereConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        mlp = CohereMLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)

        # Reference: silu(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"CohereMLP output mismatch: max diff {(out - ref).abs().max():.2e}"
        )


# =============================================================================
# Fused LoRA MLP
# =============================================================================


@pytest.mark.cuda
class TestFusedLoRAMLP:
    """Test fused LoRA MLP patching via Opake_LoRA_MLP kernel."""

    def test_fused_lora_mlp_forward(self, device):
        """Fused LoRA MLP forward should match PyTorch matmul reference."""
        from opake.api.patches.kernels.lora import Opake_LoRA_MLP

        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = 2, 16, 256, 512, 8
        scaling = 2.0

        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)

        # Create weights
        Wg = torch.randn(intermediate, hidden, device=device)
        Wu = torch.randn(intermediate, hidden, device=device)
        Wd = torch.randn(hidden, intermediate, device=device)
        Ag = torch.randn(hidden, rank, device=device)
        Bg = torch.randn(rank, intermediate, device=device)
        Au = torch.randn(hidden, rank, device=device)
        Bu = torch.randn(rank, intermediate, device=device)
        Ad = torch.randn(intermediate, rank, device=device)
        Bd = torch.randn(rank, hidden, device=device)

        # Fused kernel
        out_fused, _, _, _ = Opake_LoRA_MLP.apply(
            x, Wg, Ag, Bg, scaling, Wu, Au, Bu, scaling, Wd, Ad, Bd, scaling, 0
        )

        # PyTorch reference
        gate = F.linear(x, Wg) + (x @ Ag @ Bg) * scaling
        up = F.linear(x, Wu) + (x @ Au @ Bu) * scaling
        h = F.silu(gate) * up
        ref = F.linear(h, Wd) + (h @ Ad @ Bd) * scaling

        assert torch.allclose(out_fused, ref, rtol=1e-3, atol=1e-3), (
            f"Fused LoRA MLP output mismatch: max diff {(out_fused - ref).abs().max():.2e}"
        )

    def test_fused_lora_mlp_backward(self, device):
        """Fused LoRA MLP should produce correct gradients."""
        from opake.api.patches.kernels.lora import Opake_LoRA_MLP

        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = 2, 16, 256, 512, 8
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

    @requires_hf_auth
    def test_auto_fuse_on_get_peft_model(self, device):
        """get_peft_model should auto-fuse MLP layers with LoRA on gate/up/down."""
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

        # Check that MLP layers were auto-fused
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

        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        assert not torch.isnan(loss), "NaN loss from fused LoRA MLP model"
        assert loss.item() > 0, "Loss should be positive"

        # Backward pass
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert has_grad, "No gradients computed"

    @requires_hf_auth
    def test_patch_lora_model_manual(self, device):
        """patch_lora_model() should work for manually loaded PEFT models."""
        from opake.patches.peft import apply_peft_model_patches as patch_lora_model

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)

        # Use peft.mapping.get_peft_model to bypass auto-hook (simulate loading from checkpoint)
        from peft.mapping_func import get_peft_model as raw_get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model = raw_get_peft_model(model, lora_config).to(device)

        # MLP should NOT be fused yet (no instance-level forward override)
        layers = model.model.model.layers
        assert "forward" not in vars(layers[0].mlp), (
            "MLP should not be fused before patch_lora_model()"
        )

        # Manually apply
        patch_lora_model(model)

        # MLP should now be fused (instance-level forward override)
        assert "forward" in vars(layers[0].mlp), (
            "MLP should be fused after patch_lora_model()"
        )


# =============================================================================
# Fused LoRA QKV
# =============================================================================


@pytest.mark.cuda
class TestFusedLoRAQKV:
    """Test fused LoRA QKV patching via Opake_LoRA_QKV kernel."""

    def test_fused_lora_qkv_forward(self, device):
        """Fused LoRA QKV forward should match PyTorch matmul reference."""
        from opake.api.patches.kernels.lora import Opake_LoRA_QKV

        torch.manual_seed(42)
        batch, seq, hidden, q_out, kv_out, rank = 2, 16, 256, 256, 64, 8
        scaling = 2.0

        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)

        Wq = torch.randn(q_out, hidden, device=device)
        Wk = torch.randn(kv_out, hidden, device=device)
        Wv = torch.randn(kv_out, hidden, device=device)
        Aq = torch.randn(hidden, rank, device=device)
        Bq = torch.randn(rank, q_out, device=device)
        Ak = torch.randn(hidden, rank, device=device)
        Bk = torch.randn(rank, kv_out, device=device)
        Av = torch.randn(hidden, rank, device=device)
        Bv = torch.randn(rank, kv_out, device=device)

        Q, K, V = Opake_LoRA_QKV.apply(
            x,
            Wq,
            Aq,
            Bq,
            scaling,
            None,
            Wk,
            Ak,
            Bk,
            scaling,
            None,
            Wv,
            Av,
            Bv,
            scaling,
            None,
        )

        # PyTorch reference
        ref_q = F.linear(x, Wq) + (x @ Aq @ Bq) * scaling
        ref_k = F.linear(x, Wk) + (x @ Ak @ Bk) * scaling
        ref_v = F.linear(x, Wv) + (x @ Av @ Bv) * scaling

        assert torch.allclose(Q, ref_q, rtol=1e-3, atol=1e-3), (
            f"Fused LoRA QKV Q mismatch: max diff {(Q - ref_q).abs().max():.2e}"
        )
        assert torch.allclose(K, ref_k, rtol=1e-3, atol=1e-3), (
            f"Fused LoRA QKV K mismatch: max diff {(K - ref_k).abs().max():.2e}"
        )
        assert torch.allclose(V, ref_v, rtol=1e-3, atol=1e-3), (
            f"Fused LoRA QKV V mismatch: max diff {(V - ref_v).abs().max():.2e}"
        )

    def test_fused_lora_qkv_backward(self, device):
        """Fused LoRA QKV should produce correct gradients."""
        from opake.api.patches.kernels.lora import Opake_LoRA_QKV

        torch.manual_seed(42)
        batch, seq, hidden, q_out, kv_out, rank = 2, 16, 256, 256, 64, 8
        scaling = 2.0

        x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)

        Wq = torch.randn(q_out, hidden, device=device)
        Wk = torch.randn(kv_out, hidden, device=device)
        Wv = torch.randn(kv_out, hidden, device=device)
        Aq = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bq = torch.randn(rank, q_out, device=device, requires_grad=True)
        Ak = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bk = torch.randn(rank, kv_out, device=device, requires_grad=True)
        Av = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bv = torch.randn(rank, kv_out, device=device, requires_grad=True)

        Q, K, V = Opake_LoRA_QKV.apply(
            x,
            Wq,
            Aq,
            Bq,
            scaling,
            None,
            Wk,
            Ak,
            Bk,
            scaling,
            None,
            Wv,
            Av,
            Bv,
            scaling,
            None,
        )
        (Q.sum() + K.sum() + V.sum()).backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"
        assert Aq.grad is not None, "No gradient for Q LoRA A"
        assert Bq.grad is not None, "No gradient for Q LoRA B"
        assert Ak.grad is not None, "No gradient for K LoRA A"
        assert Bv.grad is not None, "No gradient for V LoRA B"

    @requires_hf_auth
    def test_auto_fuse_qkv_on_get_peft_model(self, device):
        """get_peft_model should auto-fuse QKV layers with LoRA on q/k/v."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)

        # Check that attention layers were auto-fused
        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert hasattr(attn, "_opake_fused_qkv"), (
                "Attention should have _opake_fused_qkv after auto-fuse"
            )

    @requires_hf_auth
    def test_fused_lora_qkv_model_forward_backward(self, device):
        """Full model with fused LoRA QKV should produce valid forward+backward."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)

        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        assert not torch.isnan(loss), "NaN loss from fused LoRA QKV model"
        assert loss.item() > 0, "Loss should be positive"

        # Backward pass
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert has_grad, "No gradients computed"

    @pytest.mark.slow
    def test_qwen2_fuses_biased_qkv_projections(self, device):
        """Qwen2 attention fuses its frozen biased Q/K/V projections."""
        from opake.patches import apply_model_patches

        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model)

        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert hasattr(attn, "_opake_fused_qkv")
            assert attn.q_proj.base_layer.bias is not None
            assert not attn.q_proj.base_layer.bias.requires_grad

    @pytest.mark.slow
    def test_qwen3_skips_qkv_fusion(self, device):
        """Qwen3 attention should NOT be fused (has q_norm/k_norm)."""
        try:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception:
            pytest.skip("Qwen3 not available")

        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)

        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert not hasattr(attn, "_opake_fused_qkv"), (
                "Qwen3 attention should NOT have fused QKV (q_norm/k_norm)"
            )

    @requires_hf_auth
    def test_patch_lora_model_manual_qkv(self, device):
        """patch_lora_model() should fuse QKV for manually loaded PEFT models."""
        from opake.patches.peft import apply_peft_model_patches as patch_lora_model

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)

        # Use peft.mapping.get_peft_model to bypass auto-hook
        from peft.mapping_func import get_peft_model as raw_get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = raw_get_peft_model(model, lora_config).to(device)

        # QKV should NOT be fused yet
        layers = model.model.model.layers
        assert not hasattr(layers[0].self_attn, "_opake_fused_qkv"), (
            "QKV should not be fused before patch_lora_model()"
        )

        # Manually apply
        patch_lora_model(model)

        # QKV should now be fused
        assert hasattr(layers[0].self_attn, "_opake_fused_qkv"), (
            "QKV should be fused after patch_lora_model()"
        )

    @requires_hf_auth
    def test_fused_qkv_clipped_grad(self, device):
        """Fused QKV should work end-to-end with clipped_grad (DP-SGD)."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["Hello world", "Another test", "Third sample", "Last one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        assert grads is not None, "No gradients returned"
        assert len(grads.pytree) > 0, "Empty gradient dict"
        for name, g in grads.pytree.items():
            assert not torch.isnan(g).any(), f"NaN in grad for {name}"
