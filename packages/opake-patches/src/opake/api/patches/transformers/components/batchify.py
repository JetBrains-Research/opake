# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Batch-dimension compatibility patches for Hugging Face model forwards."""


def _batchify_forward(original_forward):
    """Wrap a model ``forward`` to handle batchless inputs under vmap.

    Under ``vmap(grad(...))``, per-example inputs lack the batch dimension:
    ``input_ids`` is 1-D ``(seq,)`` instead of 2-D ``(batch, seq)``.
    HuggingFace models universally assume batched inputs, so this wrapper
    adds the batch dimension on entry and strips it on exit.

    This is analogous to PyTorch's *batchify* pattern in ``attention.cpp``
    for unbatched SDPA inputs.

    The wrapper is a no-op for normal (already-batched) inputs.

    Delegates to :func:`opake.functional.with_batch_dim`.
    """
    from opake.functional import with_batch_dim

    return with_batch_dim(
        original_forward,
        batch_argnums=(1,),  # self is 0, input_ids is 1
        batch_kwargs={
            "input_ids": 2,
            "attention_mask": 2,
            "labels": 2,
            "position_ids": 2,
            "inputs_embeds": 3,
        },
        min_ndim=2,
    )


def apply_batchify_patch(target_cls: type | None, model=None) -> None:
    """Apply the batchify wrapper to a specific model class and instance."""
    if target_cls is None:
        return

    import types

    # Global class-level patch
    if hasattr(target_cls, "forward"):
        fwd = target_cls.forward
        if not getattr(fwd, "_opake_batchified", False):
            new_fwd = _batchify_forward(fwd)
            new_fwd._opake_batchified = True
            target_cls.forward = new_fwd

    # Instance-level patch
    if model is not None:
        for module in model.modules():
            if type(module) is target_cls:
                fwd = module.forward
                unbound = getattr(fwd, "__func__", fwd)
                if not getattr(unbound, "_opake_batchified", False):
                    new_unbound = _batchify_forward(unbound)
                    new_unbound._opake_batchified = True
                    module.forward = types.MethodType(new_unbound, module)
