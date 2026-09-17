# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace glue for gradient checkpointing under vmap(grad(...)).

Not candidates for upstreaming and with no torch-version dependency, so applied
on every regime (whenever ``transformers`` imports), independent of the
torch-core patches.
"""

from __future__ import annotations

import logging

from opake.api.patches.torch.functorch_transform import under_functorch_transform

logger = logging.getLogger(__name__)


def apply() -> None:
    """Install Hugging Face checkpointing compatibility patches when available."""
    try:
        import transformers
    except ImportError:
        logger.info(
            "opake: transformers not importable; HuggingFace checkpoint glue "
            "skipped. The torch-side checkpoint+vmap composition does not depend "
            "on it."
        )
        return

    _force_non_reentrant(transformers)
    _make_input_require_grads_vmap_safe(transformers)


def _force_non_reentrant(transformers) -> None:
    """Force ``use_reentrant=False`` in ``gradient_checkpointing_enable``.

    HF defaults to the reentrant path, which is incompatible with functorch
    transforms. Also flips ``.training`` on the checkpoint-carrying modules (PEFT
    keeps base layers in eval, which otherwise makes checkpoint a no-op).
    """
    orig_enable = transformers.PreTrainedModel.gradient_checkpointing_enable

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        if gradient_checkpointing_kwargs.get("use_reentrant", False):
            logger.warning(
                "opake: overriding use_reentrant=True to False; the reentrant "
                "checkpoint path is incompatible with vmap(grad(...))."
            )
        gradient_checkpointing_kwargs["use_reentrant"] = False
        orig_enable(self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        for m in self.modules():
            if getattr(m, "gradient_checkpointing", False):
                m.training = True

    transformers.PreTrainedModel.gradient_checkpointing_enable = (
        gradient_checkpointing_enable
    )


def _make_input_require_grads_vmap_safe(transformers) -> None:
    """Make ``enable_input_require_grads``' forward hook a no-op under a transform.

    The hook calls ``output.requires_grad_(True)`` on input embeddings; under
    vmap(grad(...)) that both raises and is redundant (the transform already
    tracks differentiability). Handles 4.x (``_require_grads_hook``) and 5.x
    (``_require_grads_hooks``).
    """
    orig_enable = transformers.PreTrainedModel.enable_input_require_grads

    def _vmap_safe(hook):
        def wrapped(module, args, output):
            if under_functorch_transform():
                return output
            return hook(module, args, output)

        return wrapped

    def enable_input_require_grads(self):
        orig_enable(self)
        handles = getattr(self, "_require_grads_hooks", None)
        if not handles:
            single = getattr(self, "_require_grads_hook", None)
            handles = [single] if single is not None else []
        for handle in handles:
            # Rewrap in place so handle.remove() keeps working.
            hooks_dict = handle.hooks_dict_ref()
            if hooks_dict is None or handle.id not in hooks_dict:
                continue
            hooks_dict[handle.id] = _vmap_safe(hooks_dict[handle.id])

    transformers.PreTrainedModel.enable_input_require_grads = enable_input_require_grads
