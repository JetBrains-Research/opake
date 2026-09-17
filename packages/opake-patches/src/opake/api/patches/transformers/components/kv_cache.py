# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Key-value cache handling patches for functional per-example gradients."""


def _disable_kv_cache(forward_fn):
    """Wrap forward to disable KV cache when no existing cache is passed.

    HuggingFace models default to ``config.use_cache = True``, which
    allocates a ``DynamicCache`` on every forward pass — the cache is
    built but never consumed during training. Under ``vmap`` the same
    cache objects also leak via circular references (cache → tensors →
    autograd metadata → cache) that defeat refcounting, so the patch
    addresses both memory paths.

    This wrapper forces ``use_cache=False`` during training when
    ``past_key_values`` is ``None`` or an empty cache. Generation must retain
    its initial empty cache so it can populate it after the first decoding
    step.
    """
    import functools

    if getattr(forward_fn, "_opake_cache_disabled", False):
        return forward_fn

    @functools.wraps(forward_fn)
    def wrapper(*args, **kwargs):
        import torch

        model = args[0] if args else None
        past = kwargs.get("past_key_values")
        if past is None:
            has_cached_data = False
        elif hasattr(past, "get_seq_length"):
            has_cached_data = past.get_seq_length() > 0
        else:
            has_cached_data = len(past) > 0
        if (
            torch.is_grad_enabled()
            and getattr(model, "training", False)
            and not has_cached_data
        ):
            kwargs["use_cache"] = False
        return forward_fn(*args, **kwargs)

    wrapper._opake_cache_disabled = True
    return wrapper


def apply_kv_cache_patch(target_cls: type | None, model=None) -> None:
    """Apply the KV cache disabler wrapper to a specific model class and instance."""
    if target_cls is None:
        return

    import types

    # Global class-level patch
    if hasattr(target_cls, "forward"):
        fwd = target_cls.forward
        if not getattr(fwd, "_opake_cache_disabled", False):
            new_fwd = _disable_kv_cache(fwd)
            target_cls.forward = new_fwd

    # Instance-level patch
    if model is not None:
        for module in model.modules():
            if type(module) is target_cls:
                fwd = module.forward
                unbound = getattr(fwd, "__func__", fwd)
                if not getattr(unbound, "_opake_cache_disabled", False):
                    new_unbound = _disable_kv_cache(unbound)
                    module.forward = types.MethodType(new_unbound, module)
