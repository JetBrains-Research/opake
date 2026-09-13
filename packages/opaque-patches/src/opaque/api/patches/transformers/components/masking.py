"""Model-local causal-mask replacement wiring."""


def apply_module_masking_patch(mod) -> bool:
    """Rebind causal-mask helpers in a model module's local namespace.

    Args:
        mod: Imported Hugging Face model module to inspect and patch.

    Returns:
        ``True`` when at least one causal-mask helper was replaced.
    """
    from opaque.api.patches.transformers.runtime.masking import (
        vmap_create_causal_mask,
        vmap_create_recurrent_attention_mask,
        vmap_create_sliding_window_causal_mask,
    )

    patched = False
    if hasattr(mod, "create_causal_mask"):
        mod.create_causal_mask = vmap_create_causal_mask
        patched = True
    if hasattr(mod, "create_sliding_window_causal_mask"):
        mod.create_sliding_window_causal_mask = vmap_create_sliding_window_causal_mask
        patched = True
    if hasattr(mod, "create_recurrent_attention_mask"):
        mod.create_recurrent_attention_mask = vmap_create_recurrent_attention_mask
        patched = True
    return patched


def apply_compact_sdpa_sliding_window_masking_patch(mod) -> bool:
    """Install compact no-padding SDPA masking for a compatible model module."""
    from opaque.api.patches.transformers.runtime.masking import (
        vmap_create_compact_sdpa_sliding_window_causal_mask,
    )

    patched = apply_module_masking_patch(mod)
    if hasattr(mod, "create_sliding_window_causal_mask"):
        mod.create_sliding_window_causal_mask = (
            vmap_create_compact_sdpa_sliding_window_causal_mask
        )
        patched = True
    return patched
