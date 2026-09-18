# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap-compatible PEFT LoRA MLP replacements for supported model families."""

from __future__ import annotations

import torch

from ._utils import (
    _active_lora_dtype,
    _extract_lora_params,
    _lora_projections_use_dtype,
)

_MLP_ACTIVATION_MAP = {
    "LlamaMLP": 0,  # ACTIVATION_SWIGLU
    "MistralMLP": 0,
    "MinistralMLP": 0,
    "Qwen2MLP": 0,
    "Qwen3MLP": 0,
    "SmolLM3MLP": 0,
    "GraniteMLP": 0,
    "CohereMLP": 0,
    "Cohere2MLP": 0,
    "Olmo2MLP": 0,
    "Olmo3MLP": 0,
    "Glm4MLP": 0,
    "Phi3MLP": 0,
    "GemmaMLP": 1,  # ACTIVATION_GEGLU_EXACT
    "Gemma2MLP": 2,  # ACTIVATION_GEGLU_APPROX
    "Gemma3MLP": 2,  # ACTIVATION_GEGLU_APPROX (gelu_pytorch_tanh)
    "Exaone4MLP": 0,  # ACTIVATION_SWIGLU
}


def _is_phi3_style_mlp(mlp):
    """Check if MLP uses combined gate_up_proj (Phi3 style)."""
    return hasattr(mlp, "gate_up_proj") and not hasattr(mlp, "gate_proj")


def _make_fused_lora_mlp_forward(original_forward, activation_type):
    """Create fused LoRA MLP forward using Opake_LoRA_MLP kernel.

    Replaces separate gate_proj + up_proj + activation + down_proj
    with a single fused kernel call.

    Args:
        original_forward: Bound method of the MLP instance.
        activation_type: 0=SwiGLU, 1=GeGLU_exact, 2=GeGLU_approx.
    """

    def forward(self, x):
        if not x.is_cuda:
            return original_forward(x)
        from opake.api.patches.kernels.lora import Opake_LoRA_MLP

        dtype = _active_lora_dtype(x)
        if not torch.is_autocast_enabled("cuda") and not _lora_projections_use_dtype(
            self, ("gate_proj", "up_proj", "down_proj"), dtype
        ):
            return original_forward(x)

        Wg, Ag, Bg, Sg = _extract_lora_params(self.gate_proj)
        Wu, Au, Bu, Su = _extract_lora_params(self.up_proj)
        Wd, Ad, Bd, Sd = _extract_lora_params(self.down_proj)

        # Keep full weights in their parameter dtype across the autograd boundary.
        # The custom Function casts them transiently in forward and backward.
        x = x.to(dtype)

        out, _gate, _up, _h = Opake_LoRA_MLP.apply(
            x,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            activation_type,
        )

        # Add biases if present (most models don't have MLP bias)
        if getattr(self.gate_proj, "base_layer", self.gate_proj).bias is not None:
            return original_forward(x)
        if getattr(self.down_proj, "base_layer", self.down_proj).bias is not None:
            return original_forward(x)

        return out

    return forward
