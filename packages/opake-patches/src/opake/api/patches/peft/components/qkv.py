# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused vmap-compatible PEFT LoRA QKV projection replacement."""

from __future__ import annotations

import sys

from ._utils import _active_lora_dtype, _extract_lora_params_and_bias
from .qkv_gemma3 import (
    _FUSEABLE_GEMMA3_QKV_ATTENTION_CLASSES,
    _make_fused_qkv_gemma3_attention_forward,
)
from .qkv_qwen3 import (
    _FUSEABLE_QWEN3_QKV_ATTENTION_CLASSES,
    _make_fused_qkv_qwen3_attention_forward,
)

_FUSEABLE_QKV_ATTENTION_CLASSES = {
    "LlamaAttention",
    "MistralAttention",
    "GemmaAttention",
    "Gemma2Attention",
    "GraniteAttention",
    "Cohere2Attention",
    "Qwen2Attention",
    "Glm4Attention",
}


def _resolve_fused_qkv_forward_factory(attn):
    """Return the fused-forward factory for an attention module, if supported.

    Resolves the attention class through its MRO so PEFT-wrapped subclasses are
    recognized. Architectures whose forward differs from the generic pipeline
    (for example Gemma3 and Qwen3, which normalize Q/K after projection) map to
    their own dedicated factory instead of the generic one.
    """
    for cls in type(attn).__mro__:
        if cls.__name__ in _FUSEABLE_GEMMA3_QKV_ATTENTION_CLASSES:
            return _make_fused_qkv_gemma3_attention_forward
        if cls.__name__ in _FUSEABLE_QWEN3_QKV_ATTENTION_CLASSES:
            return _make_fused_qkv_qwen3_attention_forward
        if cls.__name__ in _FUSEABLE_QKV_ATTENTION_CLASSES:
            return _make_fused_qkv_attention_forward
    return None


def _opake_fused_lora_qkv(self, hidden_states):
    """Compute Q, K, V using fused Opake_LoRA_QKV kernel.

    Replaces 3 separate q_proj/k_proj/v_proj LoRA calls with a single
    fused kernel call that shares X computation across all three projections.
    """
    from opake.api.patches.kernels.lora import Opake_LoRA_QKV

    dtype = _active_lora_dtype(hidden_states)

    Wq, Aq, Bq, Sq, bq = _extract_lora_params_and_bias(self.q_proj)
    Wk, Ak, Bk, Sk, bk = _extract_lora_params_and_bias(self.k_proj)
    Wv, Av, Bv, Sv, bv = _extract_lora_params_and_bias(self.v_proj)

    # Keep full weights in their parameter dtype across the autograd boundary.
    # The custom Function casts them transiently in forward and backward.
    hidden_states = hidden_states.to(dtype)

    return Opake_LoRA_QKV.apply(
        hidden_states,
        Wq,
        Aq,
        Bq,
        Sq,
        bq,
        Wk,
        Ak,
        Bk,
        Sk,
        bk,
        Wv,
        Av,
        Bv,
        Sv,
        bv,
    )


def _make_fused_qkv_attention_forward(original_forward):
    """Create attention forward with fused QKV LoRA projection.

    Replaces the standard attention forward when Q/K/V all have LoRA adapters.
    Uses Opake_LoRA_QKV for the projection step, then continues with the
    standard RoPE + attention + o_proj pipeline.

    Uses negative indexing (transpose(-3, -2)) for vmap safety.

    Args:
        original_forward: Bound method of the attention instance.
    """

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Fused QKV projection via Opake_LoRA_QKV
        Q, K, V = self._opake_fused_qkv(hidden_states)
        query_states = Q.view(hidden_shape).transpose(-3, -2)
        key_states = K.view(hidden_shape).transpose(-3, -2)
        value_states = V.view(hidden_shape).transpose(-3, -2)

        # RoPE — resolve from the attention class's own module (already patched)
        model_module = sys.modules[type(self).__module__]
        apply_rotary_pos_emb = model_module.apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # KV cache (training: past_key_values is None)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Attention dispatch — resolve from the class's own module (already patched)
        eager_attention_forward = model_module.eager_attention_forward
        if self.config._attn_implementation != "eager":
            ALL_ATTENTION_FUNCTIONS = model_module.ALL_ATTENTION_FUNCTIONS
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # O projection (still uses individual LoRA_W via patched peft.Linear.forward)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward
