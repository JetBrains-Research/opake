# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Data collator patches for Poisson sampling compatibility.

Poisson sampling (used for privacy amplification in DP-SGD) can yield empty
batches.  HuggingFace data collators crash on empty input lists because they
index ``examples[0]`` unconditionally.  These patches wrap the collator with
:func:`empty_collate` which learns the output structure from the first
non-empty call.

Controlled by:
    OPAKE_SKIP_TRANSFORMERS_DATA_PATCHES: "all" or "collator"
"""

import functools

from opake.functional import empty_collate

_WRAPPER_ATTR = "_opake_collate"


def apply_collator_patches(*, empty_batches: bool = True) -> None:
    """Patch HuggingFace data collators to handle empty example lists.

    Currently patches:

    - ``DataCollatorForLanguageModeling.torch_call``
    """
    if empty_batches is False:
        return

    try:
        from transformers import DataCollatorForLanguageModeling
    except ImportError:
        return

    if getattr(DataCollatorForLanguageModeling.torch_call, "__opake_patched__", False):
        return

    _original_torch_call = DataCollatorForLanguageModeling.torch_call

    def _patched_torch_call(self, examples):
        wrapper = getattr(self, _WRAPPER_ATTR, None)
        if wrapper is None:
            wrapper = empty_collate(functools.partial(_original_torch_call, self))
            setattr(self, _WRAPPER_ATTR, wrapper)
        return wrapper(examples)

    _patched_torch_call.__opake_patched__ = True
    DataCollatorForLanguageModeling.torch_call = _patched_torch_call
