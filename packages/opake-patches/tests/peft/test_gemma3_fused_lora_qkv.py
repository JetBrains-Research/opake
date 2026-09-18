import copy
import types

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Gemma3TextConfig

from opake.api.engine.clipping import clipped_grad
from opake.api.patches.peft.components.qkv import _resolve_fused_qkv_forward_factory
from opake.api.patches.peft.components.qkv_gemma3 import (
    _fused_qkv_gemma3_attention_forward,
    _make_fused_qkv_gemma3_attention_forward,
)
from opake.functional import make_functional
from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

RTOL = 0.001
ATOL = 0.001


def _gemma3_config(**overrides):
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    kwargs.update(overrides)
    return Gemma3TextConfig(**kwargs)


def _lora_config(**overrides):
    kwargs = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    kwargs.update(overrides)
    return LoraConfig(**kwargs)


def _peft_gemma3(device, config=None, lora_config=None):
    config = config if config is not None else _gemma3_config()
    model = AutoModelForCausalLM.from_config(config)
    model = get_peft_model(
        model, lora_config if lora_config is not None else _lora_config()
    )
    return model.to(device).eval(), config


def _decoder_layers(model):
    return model.base_model.model.model.layers


class TestGemma3FusedQKVRouting:
    """Routing gates for Gemma3 fused LoRA QKV (device independent)."""

    def test_gemma3_attention_uses_dedicated_factory(self):
        """Gemma3 attention resolves to the normalization-aware wrapper."""
        model, _ = _peft_gemma3("cpu")
        attn = _decoder_layers(model)[0].self_attn
        assert (
            _resolve_fused_qkv_forward_factory(attn)
            is _make_fused_qkv_gemma3_attention_forward
        )

    def test_gemma3_fuses_qkv_projections(self):
        """All-QKV LoRA adapters without dropout are fused on Gemma3."""
        model, _ = _peft_gemma3("cpu")
        apply_model_patches(model, performance=False, compat=True, lora=True)
        for layer in _decoder_layers(model):
            attn = layer.self_attn
            assert hasattr(attn, "_opake_fused_qkv")
            assert getattr(attn.forward, "__opake_lora_qkv_patched__", False)
            assert attn.q_norm is not None
            assert attn.k_norm is not None

    def test_gemma3_skips_fusion_with_partial_adapters(self):
        """Fusion requires adapters on all three projections."""
        model, _ = _peft_gemma3(
            "cpu", lora_config=_lora_config(target_modules=["q_proj", "v_proj"])
        )
        apply_model_patches(model, performance=False, compat=True, lora=True)
        assert not hasattr(_decoder_layers(model)[0].self_attn, "_opake_fused_qkv")

    def test_gemma3_skips_fusion_with_lora_dropout(self):
        """Fusion is skipped when active LoRA dropout would be bypassed."""
        from opake.patches.peft import apply_peft_model_patches

        model, _ = _peft_gemma3("cpu", lora_config=_lora_config(lora_dropout=0.1))
        apply_peft_model_patches(model)
        assert not hasattr(_decoder_layers(model)[0].self_attn, "_opake_fused_qkv")

    def test_gemma3_skips_fusion_with_trainable_bias(self):
        """Trainable projection biases keep the unfused path."""
        model, _ = _peft_gemma3(
            "cpu",
            config=_gemma3_config(attention_bias=True),
            lora_config=_lora_config(bias="all"),
        )
        apply_model_patches(model, performance=False, compat=True, lora=True)
        attn = _decoder_layers(model)[0].self_attn
        assert attn.q_proj.base_layer.bias.requires_grad
        assert not hasattr(attn, "_opake_fused_qkv")


def _position_embeddings(model, attn, hidden_states):
    """Build Gemma3 rotary embeddings for the attention layer's rope schedule."""
    rotary = model.base_model.model.model.rotary_emb
    seq_len = hidden_states.shape[-2]
    position_ids = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .expand(hidden_states.shape[0], -1)
    )
    return rotary(hidden_states, position_ids, attn.layer_type)


def _install_reference_fused_qkv(attn):
    """Route the fused wrapper through unfused Q/K/V projections.

    Lets the device-independent pipeline (norm placement, RoPE, cache, dispatch)
    be validated without the CUDA-only fused kernel.
    """

    def _reference_qkv(self, hidden_states):
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    attn._opake_fused_qkv = types.MethodType(_reference_qkv, attn)


class TestGemma3FusedWrapperPipeline:
    """The dedicated wrapper reproduces the upstream Gemma3 attention pipeline."""

    def test_wrapper_matches_upstream_attention_output(self):
        """Wrapper output equals the unpatched Gemma3 attention output."""
        torch.manual_seed(0)
        model, config = _peft_gemma3("cpu")
        attn = _decoder_layers(model)[0].self_attn
        _install_reference_fused_qkv(attn)

        hidden_states = torch.randn(2, 6, config.hidden_size)
        position_embeddings = _position_embeddings(model, attn, hidden_states)

        with torch.no_grad():
            expected, _ = attn(hidden_states, position_embeddings)
            actual, _ = _fused_qkv_gemma3_attention_forward(
                attn, hidden_states, position_embeddings
            )
        assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
            f"max diff {(actual - expected).abs().max():.2e}"
        )

    def test_wrapper_applies_qk_norm_after_transpose(self):
        """Skipping q_norm/k_norm would change the output — the wrapper applies them."""
        torch.manual_seed(0)
        model, config = _peft_gemma3("cpu")
        attn = _decoder_layers(model)[0].self_attn
        _install_reference_fused_qkv(attn)
        with torch.no_grad():
            attn.q_norm.weight.add_(0.5)
            attn.k_norm.weight.add_(0.5)

        hidden_states = torch.randn(2, 6, config.hidden_size)
        position_embeddings = _position_embeddings(model, attn, hidden_states)

        with torch.no_grad():
            expected, _ = attn(hidden_states, position_embeddings)
            actual, _ = _fused_qkv_gemma3_attention_forward(
                attn, hidden_states, position_embeddings
            )
        assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_wrapper_matches_upstream_cached_decoding(self):
        """Prefill + single-step decoding through the wrapper matches upstream."""
        torch.manual_seed(0)
        from transformers import DynamicCache

        model, config = _peft_gemma3("cpu")
        attn = _decoder_layers(model)[0].self_attn
        _install_reference_fused_qkv(attn)

        prompt = torch.randn(1, 6, config.hidden_size)
        step = torch.randn(1, 1, config.hidden_size)
        outputs = []
        for forward in (
            attn.__call__,
            _fused_qkv_gemma3_attention_forward.__get__(attn),
        ):
            cache = DynamicCache(config=config)
            with torch.no_grad():
                forward(
                    prompt,
                    _position_embeddings(model, attn, prompt),
                    past_key_values=cache,
                )
                out, _ = forward(
                    step,
                    _position_embeddings(model, attn, step),
                    past_key_values=cache,
                )
            outputs.append(out)
        assert torch.allclose(outputs[1], outputs[0], rtol=RTOL, atol=ATOL)

    def test_wrapper_is_vmap_safe(self):
        """The wrapper's negative-index reshapes survive an extra vmap dimension."""
        torch.manual_seed(0)
        model, config = _peft_gemma3("cpu")
        attn = _decoder_layers(model)[0].self_attn
        _install_reference_fused_qkv(attn)

        hidden_states = torch.randn(3, 1, 6, config.hidden_size)
        position_embeddings = _position_embeddings(model, attn, hidden_states[0])

        def single(x):
            out, _ = _fused_qkv_gemma3_attention_forward(attn, x, position_embeddings)
            return out

        with torch.no_grad():
            batched = torch.func.vmap(single)(hidden_states)
            expected = torch.stack([single(x) for x in hidden_states])
        assert torch.allclose(batched, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused LoRA QKV kernels require CUDA/Triton",
)
class TestGemma3FusedQKVNumerics:
    """Numerical behavior of the Gemma3 fused LoRA QKV wrapper."""

    def test_forward_backward(self, device):
        """Fused Gemma3 attention trains without NaNs."""
        model, config = _peft_gemma3(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()
        trained = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert trained
        assert all(torch.isfinite(p.grad).all() for p in trained)

    def test_fused_matches_unfused_logits(self, device):
        """Fused QKV keeps Gemma3 logits equal to the unfused pipeline."""
        torch.manual_seed(0)
        reference, config = _peft_gemma3(device)
        fused = copy.deepcopy(reference)
        apply_model_patches(fused, performance=False, compat=True, lora=True)
        assert hasattr(_decoder_layers(fused)[0].self_attn, "_opake_fused_qkv")

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        with torch.no_grad():
            expected = reference(input_ids).logits
            actual = fused(input_ids).logits
        assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
            f"max diff {(actual - expected).abs().max():.2e}"
        )

    def test_fused_matches_unfused_cached_decoding(self, device):
        """Cached decoding stays equivalent under fused QKV."""
        torch.manual_seed(0)
        reference, config = _peft_gemma3(device)
        fused = copy.deepcopy(reference)
        apply_model_patches(fused, performance=False, compat=True, lora=True)

        prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
        next_token = torch.randint(0, config.vocab_size, (1, 1), device=device)
        with torch.no_grad():
            ref_prefill = reference(prompt, use_cache=True)
            ref_step = reference(
                next_token,
                past_key_values=ref_prefill.past_key_values,
                use_cache=True,
            ).logits
            fused_prefill = fused(prompt, use_cache=True)
            fused_step = fused(
                next_token,
                past_key_values=fused_prefill.past_key_values,
                use_cache=True,
            ).logits
        assert torch.allclose(fused_step, ref_step, rtol=RTOL, atol=ATOL), (
            f"max diff {(fused_step - ref_step).abs().max():.2e}"
        )

    def test_clipped_grad_per_example_gradients(self, device):
        """Fused Gemma3 QKV supports vmap(grad) per-example gradients."""
        model, config = _peft_gemma3(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)

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
        assert grads.pytree
        assert all(torch.isfinite(grad).all() for grad in grads.pytree.values())
