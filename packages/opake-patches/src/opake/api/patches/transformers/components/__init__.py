# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel patches and vmap compatibility wrappers for HuggingFace Transformers models.

Provides model components like Opake's vmap-compatible Triton kernels and compatibility wrappers.
These components are applied per-instance via explicit patching calls.

**Scope:** targets are **decoder-only text** architectures (``ForCausalLM`` /
shared text modules). Vision-language and other multimodal heads are not part
of the default patch matrix.

Patched components:
- MLP activations: SwiGLU (LLaMA, Mistral, Ministral, Qwen2, Qwen3, SmolLM3, OLMo2, OLMo3, Granite, Cohere, Cohere2, Exaone4), Phi3-style SwiGLU (Phi3, Glm4), and GeGLU (Gemma, Gemma2, Gemma3)
- RMSNorm: Llama-style, Gemma-style, OLMo2-style, GLM4-style, and Gemma3-style RMSNorm modules (incl. q_norm/k_norm rebinds inside Gemma3Attention / Exaone4Attention)
- Fused add + post-attention RMSNorm on decoder layers with Llama-style residual ordering (Llama, Mistral/Ministral, Qwen2/3, SmolLM3, Gemma, Phi-3, Granite). OLMo2/OLMo3/Cohere/Cohere2/Gemma3/Exaone4 keep this OFF because their decoder layers apply ``post_attention_layernorm`` between attention output and the residual add (incompatible with the fused primitive's residual-first contract).
- RoPE: apply_rotary_pos_emb for all supported models (standard half-split rotation)
- Cross-entropy loss: per-model ForCausalLM ``loss_function`` override (fp32 fallback)
- Fused linear + CE: ForCausalLM.forward replaced to skip lm_head materialization (bf16/fp16)
- LoRA: peft.tuners.lora.Linear forward + explicit fused QKV (Opake_LoRA_QKV) and MLP (Opake_LoRA_MLP) when ``apply_model_patches()`` is called on a PEFT-wrapped model
"""

__all__ = []
