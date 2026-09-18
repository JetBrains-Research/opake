# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Family registry — built-in + user-registered HuggingFace model families.

Dispatch flow (called from
:func:`apply_transformers_model_patches`):

1. ``detect_family(model)`` → short family name from
   ``model.config.model_type``.
2. ``get_family_apply_fn(name)`` → the registered apply function.
3. The apply function is what
   :func:`opake.patches.transformers.make_apply_model_patches` returns.

Both built-in and user-defined families share the same registration path:
``register_family(name, apply_fn)``.  Built-in families register
themselves at import time from
``opake.patches.transformers.models.X``; downstream users register
their own families the same way (typically right after building
``apply_X_patches`` via :func:`make_apply_model_patches`).

User registration:

    from opake.api.patches.transformers import (make_apply_family_patches, make_apply_model_patches, register_family)

    apply_my_fam_family_patches = make_apply_family_patches(
        family="my_fam", module_path="my_pkg.modeling_my_fam",
    )
    apply_my_fam_patches = make_apply_model_patches(
        family="my_fam",
        family_apply=apply_my_fam_family_patches,
        module_path="my_pkg.modeling_my_fam",
        classes={...},
        activation_kind="swiglu",
    )
    register_family("my_fam", apply_my_fam_patches)

Now ``opake.patches.apply_model_patches(my_fam_instance)`` routes to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from opake.api.patches.transformers.types import ModelPatchFn

_FAMILY_REGISTRY: dict[str, ModelPatchFn] = {}


def register_family(name: str, apply_fn: ModelPatchFn) -> None:
    """Register a HuggingFace model family.

    The dispatcher will route models with
    ``model.config.model_type == name`` to ``apply_fn``.  Used both by
    opake's shipped families (each ``opake.patches.transformers.models.X``
    calls this on import) and by downstream users adding their own.

    Args:
        name: The HF ``model_type`` string (matches
            :func:`opake.patches.transformers.family_name`).  Re-registering
            an existing name overwrites the previous registration.
        apply_fn: Callable with signature
            ``apply(model, *, performance=True, compat=True, **kwargs) -> None``,
            typically the return value of
            :func:`opake.patches.transformers.make_apply_model_patches`.
    """
    _FAMILY_REGISTRY[name] = apply_fn


def get_family_apply_fn(name: str) -> ModelPatchFn | None:
    """Return the registered apply function for a family, or ``None``."""
    return _FAMILY_REGISTRY.get(name)


def supported_families() -> list[str]:
    """List currently registered families (built-ins + user-registered)."""
    return sorted(_FAMILY_REGISTRY)


def detect_family(model: nn.Module) -> str | None:
    """Detect the model family from the model config."""
    config = getattr(model, "config", None)
    if config:
        model_type = getattr(config, "model_type", None)
        if model_type == "gemma3_text":
            return "gemma3"
        return model_type
    return None


__all__ = [
    "detect_family",
    "get_family_apply_fn",
    "register_family",
    "supported_families",
]
