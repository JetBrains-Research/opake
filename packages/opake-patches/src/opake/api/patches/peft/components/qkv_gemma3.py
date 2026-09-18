# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused vmap-compatible PEFT LoRA QKV replacement for Gemma3 text attention.

Gemma3 attention normalizes query and key states with per-head RMSNorm after
projection, reshape, and transpose, and it dispatches attention with its own
sliding-window argument. The generic fused wrapper does neither, so Gemma3 gets
a dedicated forward that reproduces the upstream computation exactly and only
replaces the three separate Q/K/V projection calls with one fused kernel call.
"""

from __future__ import annotations

import sys

_FUSEABLE_GEMMA3_QKV_ATTENTION_CLASSES = {
    "Gemma3Attention",
}


def _fused_qkv_gemma3_attention_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Gemma3 attention forward whose Q/K/V projections come from ``self._opake_fused_qkv``.

    Preserves the Gemma3 pipeline: projections -> reshape/transpose ->
    ``q_norm``/``k_norm`` -> Gemma3 RoPE -> cache update -> attention dispatch
    (with ``sliding_window``) -> ``o_proj`` -> ``(attn_output, attn_weights)``.

    Uses negative indexing (``transpose(-3, -2)``) for vmap safety.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    Q, K, V = self._opake_fused_qkv(hidden_states)
    query_states = Q.view(hidden_shape).transpose(-3, -2)
    key_states = K.view(hidden_shape).transpose(-3, -2)
    value_states = V.view(hidden_shape).transpose(-3, -2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    model_module = sys.modules[type(self).__module__]
    apply_rotary_pos_emb = model_module.apply_rotary_pos_emb
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": kwargs.get("cache_position"),
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    eager_attention_forward = model_module.eager_attention_forward
    all_attention_functions = model_module.ALL_ATTENTION_FUNCTIONS
    get_interface = getattr(all_attention_functions, "get_interface", None)
    if get_interface is not None:
        attention_interface = get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
    elif self.config._attn_implementation != "eager":
        attention_interface = all_attention_functions[self.config._attn_implementation]
    else:
        attention_interface = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _make_fused_qkv_gemma3_attention_forward(original_forward):
    """Create a Gemma3 attention forward with fused QKV LoRA projection.

    The fused kernel only runs on CUDA; every other device — and any call that
    omits ``position_embeddings`` — falls back to the unmodified Gemma3 forward.

    Args:
        original_forward: Bound method of the attention instance.
    """

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda or position_embeddings is None:
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        return _fused_qkv_gemma3_attention_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    return forward
