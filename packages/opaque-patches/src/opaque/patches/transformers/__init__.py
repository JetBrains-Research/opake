"""HF Transformers compat patches — vmap-safe attention, KV cache, custom-family registry."""

from opaque.api.patches.transformers import (
    FamilyPatchFn,
    ForwardFactory,
    ForwardFn,
    ModelPatchFn,
    ModulePatcher,
    MoeGeometry,
    apply_transformers_model_patches,
    family_name,
    make_apply_family_patches,
    make_apply_model_patches,
    moe_geometry,
    register_activation_kind,
    register_family,
    register_fused_add_rms_kind,
    register_rms_norm_kind,
    supported_families,
)

__all__ = [
    "FamilyPatchFn",
    "ForwardFactory",
    "ForwardFn",
    "ModelPatchFn",
    "ModulePatcher",
    "MoeGeometry",
    "apply_transformers_model_patches",
    "family_name",
    "make_apply_family_patches",
    "make_apply_model_patches",
    "moe_geometry",
    "register_activation_kind",
    "register_family",
    "register_fused_add_rms_kind",
    "register_rms_norm_kind",
    "supported_families",
]
