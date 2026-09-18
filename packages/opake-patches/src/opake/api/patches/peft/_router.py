# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import types

from .components._utils import _bias_is_frozen, _has_lora, _no_bias, _no_lora_dropout
from .components.linear import _make_lora_linear_forward
from .components.mlp import (
    _MLP_ACTIVATION_MAP,
    _is_phi3_style_mlp,
    _make_fused_lora_mlp_forward,
)
from .components.qkv import (
    _opake_fused_lora_qkv,
    _resolve_fused_qkv_forward_factory,
)

logger = logging.getLogger(__name__)


def _lora_patching_allowed() -> bool:
    """Return whether it is safe to install LoRA kernel wrappers."""
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _find_decoder_layers(model):
    """Find decoder layers across different model architectures."""
    for path_parts in [
        ["model", "model", "layers"],
        ["base_model", "model", "model", "layers"],
    ]:
        obj = model
        for attr in path_parts:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__iter__"):
            return list(obj)
    return []


def _auto_fuse_lora(model):
    """Auto-detect and fuse LoRA layers with Opake fused kernels.

    Called from explicit PEFT patching on an already wrapped model. Walks
    decoder layers and:
    1. Fuses Q/K/V projections when all three have LoRA (Opake_LoRA_QKV)
    2. Fuses gate/up/down projections when all three have LoRA (Opake_LoRA_MLP)
    """
    layers = _find_decoder_layers(model)
    if not layers:
        return

    qkv_count = 0
    mlp_count = 0

    for layer in layers:
        # --- QKV fusion ---
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if getattr(attn, "_opake_lora_qkv_patched", False):
                continue

            # Check attention class follows a supported QKV projection pattern.
            fused_forward_factory = _resolve_fused_qkv_forward_factory(attn)

            if (
                fused_forward_factory is not None
                and _has_lora(attn, "q_proj")
                and _has_lora(attn, "k_proj")
                and _has_lora(attn, "v_proj")
                and _bias_is_frozen(attn, "q_proj")
                and _bias_is_frozen(attn, "k_proj")
                and _bias_is_frozen(attn, "v_proj")
                and _no_lora_dropout(attn, "q_proj")
                and _no_lora_dropout(attn, "k_proj")
                and _no_lora_dropout(attn, "v_proj")
            ):
                attn._opake_fused_qkv = types.MethodType(
                    _opake_fused_lora_qkv,
                    attn,
                )
                fused_qkv_fwd = fused_forward_factory(attn.forward)
                fused_qkv_fwd.__opake_lora_qkv_patched__ = True
                attn.forward = types.MethodType(fused_qkv_fwd, attn)
                attn._opake_lora_qkv_patched = True
                qkv_count += 1

        # --- MLP fusion ---
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Skip Phi3-style combined gate_up_proj (not supported by fused kernel)
        if _is_phi3_style_mlp(mlp):
            continue

        if getattr(mlp, "_opake_lora_mlp_patched", False):
            continue

        # Check all three projections have LoRA and no active dropout
        if not (
            _has_lora(mlp, "gate_proj")
            and _has_lora(mlp, "up_proj")
            and _has_lora(mlp, "down_proj")
            and _no_bias(mlp, "gate_proj")
            and _no_bias(mlp, "up_proj")
            and _no_bias(mlp, "down_proj")
            and _no_lora_dropout(mlp, "gate_proj")
            and _no_lora_dropout(mlp, "up_proj")
            and _no_lora_dropout(mlp, "down_proj")
        ):
            continue

        # Determine activation type from MLP class name
        cls_name = type(mlp).__name__
        # The MLP might be wrapped by PEFT, check through base
        if cls_name not in _MLP_ACTIVATION_MAP:
            # Try unwrapping: PEFT wraps the module, check the class hierarchy
            for parent_cls in type(mlp).__mro__:
                if parent_cls.__name__ in _MLP_ACTIVATION_MAP:
                    cls_name = parent_cls.__name__
                    break
            else:
                continue

        activation_type = _MLP_ACTIVATION_MAP[cls_name]

        fused_mlp_fwd = _make_fused_lora_mlp_forward(mlp.forward, activation_type)
        fused_mlp_fwd.__opake_lora_mlp_patched__ = True
        mlp.forward = types.MethodType(fused_mlp_fwd, mlp)
        mlp._opake_lora_mlp_patched = True
        mlp_count += 1

    if qkv_count > 0 or mlp_count > 0:
        logger.debug(
            f"opake: Fused LoRA applied to {qkv_count} QKV + {mlp_count} MLP layers"
        )


def apply_peft_model_patches(
    model, *, performance: bool = True, compat: bool = True, **kwargs
) -> None:
    """Manually apply fused LoRA patching (QKV + MLP + Linear) to a PEFT model.

    Use this on an already wrapped PEFT model, including after
    ``get_peft_model()`` or when loading a PEFT checkpoint.

    Detects and fuses:
    - Base `Linear` projections with LoRA → Opake_LoRA_Linear
    - Q/K/V projections with LoRA → Opake_LoRA_QKV
    - gate/up/down projections with LoRA → Opake_LoRA_MLP

    Args:
        model: A PEFT-wrapped model with LoRA adapters.
        performance: Whether to enable fused LoRA performance patches.
        compat: Reserved compatibility switch accepted by the common patch API.
        **kwargs: Additional patch options. ``lora`` overrides
            ``performance`` when supplied.
    """
    lora = kwargs.get("lora", performance)
    if not lora or not _lora_patching_allowed():
        return

    patched_lora = False
    for module in model.modules():
        cls_name = type(module).__name__
        if (
            cls_name == "Linear"
            and "peft.tuners.lora" in type(module).__module__
            and not hasattr(module.forward, "__opake_patched__")
        ):
            new_fwd = _make_lora_linear_forward(type(module).forward)
            new_fwd.__opake_patched__ = True
            module.forward = types.MethodType(new_fwd, module)
            patched_lora = True

    if patched_lora:
        logger.debug("opake: Applied Triton kernel patches for peft.LoRA.Linear")

    _auto_fuse_lora(model)
