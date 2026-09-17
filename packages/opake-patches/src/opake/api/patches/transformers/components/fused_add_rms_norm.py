# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused residual-add and RMSNorm replacements for supported transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opake.api.patches.transformers._compat import IS_TRANSFORMERS_V5

if TYPE_CHECKING:
    import torch


def _post_attn_eps_and_weight(layer) -> tuple[torch.Tensor, float]:
    norm = layer.post_attention_layernorm
    w = norm.weight
    eps = float(getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6))
    return w, eps


def _fused_add_rms_fac_llama(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opake.api.patches.kernels.fused_add_rms_norm import Opake_FusedAddRMSNorm

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual, _ = Opake_FusedAddRMSNorm.apply(
            hidden_states,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    return forward


def _fused_add_rms_fac_gemma(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opake.api.patches.kernels.fused_add_rms_norm import Opake_FusedAddRMSNorm

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual, _ = Opake_FusedAddRMSNorm.apply(
            hidden_states,
            residual,
            weight,
            eps,
            1.0,
            "gemma",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    return forward


def _fused_add_rms_fac_phi3(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opake.api.patches.kernels.fused_add_rms_norm import Opake_FusedAddRMSNorm

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        xa = self.resid_attn_dropout(hidden_states)
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual, _ = Opake_FusedAddRMSNorm.apply(
            xa,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states

    return forward


def _fused_add_rms_fac_granite(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opake.api.patches.kernels.fused_add_rms_norm import Opake_FusedAddRMSNorm

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        xa = hidden_states * self.residual_multiplier
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual, _ = Opake_FusedAddRMSNorm.apply(
            xa,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        # v4 GraniteDecoderLayer returns a tuple; v5 returns a bare tensor.
        if IS_TRANSFORMERS_V5:
            return hidden_states
        outputs = (hidden_states,)
        if kwargs.get("output_attentions"):
            outputs += (None,)
        return outputs

    return forward
