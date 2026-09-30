# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused vmap-compatible PEFT LoRA QKV projection replacement."""

from __future__ import annotations

import sys
from functools import partial

from ._utils import _active_lora_dtype, _extract_lora_params_and_bias
from .qkv_gemma3 import (
    _FUSEABLE_GEMMA3_QKV_ATTENTION_CLASSES,
    _make_fused_qkv_gemma3_attention_forward,
)
from .qkv_qwen3 import (
    _FUSEABLE_QWEN3_QKV_ATTENTION_CLASSES,
    _make_fused_qkv_qwen3_attention_forward,
)


def _standard_policy(attn):
    return True, {}


def _mistral_policy(attn):
    return True, {"sliding_window": getattr(attn.config, "sliding_window", None)}


def _cohere2_policy(attn):
    return attn.sliding_window is not None, {"sliding_window": attn.sliding_window}


def _gemma2_policy(attn):
    return True, {
        "sliding_window": attn.sliding_window,
        "softcap": attn.attn_logit_softcapping,
    }


def _qwen2_policy(attn):
    return True, {"sliding_window": attn.sliding_window}


_GENERIC_QKV_ATTENTION_POLICIES = {
    "LlamaAttention": _standard_policy,
    "MistralAttention": _mistral_policy,
    "GemmaAttention": _standard_policy,
    "Gemma2Attention": _gemma2_policy,
    "GraniteAttention": _standard_policy,
    "Cohere2Attention": _cohere2_policy,
    "Qwen2Attention": _qwen2_policy,
    "Glm4Attention": _standard_policy,
}


def _resolve_fused_qkv_forward_factory(attn):
    """Select a fused forward for a supported attention class in the MRO."""
    for cls in type(attn).__mro__:
        if cls.__name__ in _FUSEABLE_GEMMA3_QKV_ATTENTION_CLASSES:
            return _make_fused_qkv_gemma3_attention_forward
        if cls.__name__ in _FUSEABLE_QWEN3_QKV_ATTENTION_CLASSES:
            return _make_fused_qkv_qwen3_attention_forward
        policy = _GENERIC_QKV_ATTENTION_POLICIES.get(cls.__name__)
        if policy is not None:
            return partial(_make_fused_qkv_attention_forward, policy=policy)
    return None


def _opake_fused_lora_qkv(self, hidden_states):
    """Compute Q, K, and V with one fused LoRA kernel."""
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


def _fused_qkv_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    *,
    policy,
    **kwargs,
):
    """Run the shared attention pipeline with fused Q/K/V projections."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    Q, K, V = self._opake_fused_qkv(hidden_states)
    query_states = Q.view(hidden_shape).transpose(-3, -2)
    key_states = K.view(hidden_shape).transpose(-3, -2)
    value_states = V.view(hidden_shape).transpose(-3, -2)

    model_module = sys.modules[type(self).__module__]
    cos, sin = position_embeddings
    apply_rope, attention_kwargs = policy(self)
    if apply_rope:
        query_states, key_states = model_module.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = model_module.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = model_module.ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **attention_kwargs,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _make_fused_qkv_attention_forward(original_forward, *, policy):
    """Use the fused pipeline on CUDA and the original forward elsewhere."""

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
        return _fused_qkv_attention_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            policy=policy,
            **kwargs,
        )

    return forward
