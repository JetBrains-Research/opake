# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Public API for the HuggingFace transformers patch layer.

The factories here let downstream users register their own model
families::

    from opaque.api.patches.transformers import (make_apply_family_patches, make_apply_model_patches, register_activation_kind)

    # 1) (Optional) Register a custom kernel variant under a name.
    register_activation_kind("my_glu", my_factory)

    # 2) Build the family-runtime patch fn (mutates module-level globals
    #    in `my_pkg.my_modeling`; once-per-process, idempotent).
    apply_my_family_family_patches = make_apply_family_patches(
        family="my_family",
        module_path="my_pkg.my_modeling",
    )

    # 3) Build the per-model patch fn (mutates classes on the model instance).
    apply_my_family_patches = make_apply_model_patches(
        family="my_family",
        family_apply=apply_my_family_family_patches,
        module_path="my_pkg.my_modeling",
        classes={"mlp": "MyMLP", "rms_norm": "MyRMSNorm",
                 "decoder_layer": "MyDecoder", "causal_lm": "MyForCausalLM"},
        activation_kind="my_glu",   # registered above; or pass the callable directly
        rms_norm_kind="llama",      # any built-in or callable
        fused_add_rms_kind=None,
    )

    # 4) Use it.
    apply_my_family_patches(my_model)

For most callers the convenience wrapper
``opaque.patches.apply_model_patches(model)`` is enough — it discovers
the family from the model and dispatches to the right
``apply_X_patches``.  This module is for users who:

- ship their own architecture and want it to participate in opaque's
  vmap-safety + Triton kernel patches, or
- need power-user access to the family-runtime layer (e.g. enable
  vmap-safe RoPE on a class without owning a model instance yet).
"""

# Eagerly import built-in family modules so each one's import-time
# ``register_family(...)`` call lands in the registry.  Each file is
# tiny (just two factory invocations and the register call); the actual
# ``transformers.models.X`` modeling module is imported lazily inside
# the patch function on first use.
from opaque.api.patches.transformers import models  # noqa: F401
from opaque.api.patches.transformers._factory import (
    make_apply_model_patches,
    register_activation_kind,
    register_fused_add_rms_kind,
    register_moe_kind,
    register_rms_norm_kind,
)
from opaque.api.patches.transformers._family import (
    family_name,
    make_apply_family_patches,
)
from opaque.api.patches.transformers._registry import (
    register_family,
    supported_families,
)
from opaque.api.patches.transformers._router import apply_transformers_model_patches
from opaque.api.patches.transformers.components.router import (
    MoeGeometry,
    moe_geometry,
)
from opaque.api.patches.transformers.types import (
    FamilyPatchFn,
    ForwardFactory,
    ForwardFn,
    ModelPatchFn,
    ModulePatcher,
)

__all__ = [
    "apply_transformers_model_patches",
    "family_name",
    "FamilyPatchFn",
    "ForwardFactory",
    "ForwardFn",
    "make_apply_family_patches",
    "make_apply_model_patches",
    "ModelPatchFn",
    "ModulePatcher",
    "MoeGeometry",
    "moe_geometry",
    "register_activation_kind",
    "register_family",
    "register_fused_add_rms_kind",
    "register_moe_kind",
    "register_rms_norm_kind",
    "supported_families",
]
