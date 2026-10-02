# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Length-ordered microbatches with per-microbatch padding trim.

EXPERIMENTAL, NOT YET DP-REVIEWED. Enabled by
``TrainingArguments.trim_microbatch_padding``.

The collator pads the whole logical batch to its longest example, and the
clipping engine slices microbatches along the example dimension only, so every
example is computed at the logical batch's longest length. These helpers:

1. order the logical batch by real length (a permutation of examples), and
2. drop each microbatch's trailing columns in which no row has an attended
   token (``attention_mask != 0``) or a valid label (``labels != -100``).

Why per-example outputs should be unchanged (the argument the DP review must
confirm): the clipped sum is invariant to the order of examples; in a causal
LM, positions never attend to later positions, so dropping trailing columns
does not change real-token outputs; HF's shifted loss pairs ``logits[t]`` with
``labels[t + 1]``, and every dropped pair has ``labels[t + 1] == -100``.
Sampling, noise, and accounting are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

_IGNORE_INDEX = -100
#: ``[batch, sequence]`` rank of ``attention_mask`` and of the trimmed tensors.
_SEQUENCE_NDIM = 2


def _required_positions(batch_keys: tuple[str, ...]) -> tuple[int, int | None]:
    """Positions of ``attention_mask`` and ``labels`` in ``(params, *batch_args)``."""
    if "attention_mask" not in batch_keys:
        raise ConfigurationError(
            *(
                "trim_microbatch_padding=True needs an 'attention_mask' batch "
                f"key; the collator emits {list(batch_keys)!r}.",
            )
        )
    mask_pos = 1 + batch_keys.index("attention_mask")
    labels_pos = 1 + batch_keys.index("labels") if "labels" in batch_keys else None
    return mask_pos, labels_pos


def _real_lengths(attention_mask: Tensor, labels: Tensor | None) -> Tensor:
    """Per-row length up to and including the last attended token or valid label."""
    keep = attention_mask != 0
    if labels is not None:
        keep = keep | (labels != _IGNORE_INDEX)
    positions = torch.arange(1, keep.shape[1] + 1, device=keep.device)
    return torch.where(keep, positions, 0).amax(dim=1)


def sort_batch_by_length(
    batch_args: tuple[Any, ...], batch_keys: tuple[str, ...]
) -> tuple[Any, ...]:
    """Reorder every per-example tensor longest-first (a stable permutation)."""
    mask_pos, labels_pos = _required_positions(batch_keys)
    mask = batch_args[mask_pos - 1]
    if mask.ndim != _SEQUENCE_NDIM or mask.shape[0] == 0:
        return batch_args
    labels = batch_args[labels_pos - 1] if labels_pos is not None else None
    lengths = _real_lengths(mask, labels)
    order = torch.argsort(lengths, descending=True, stable=True)
    n = mask.shape[0]
    return tuple(
        t.index_select(0, order)
        if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == n
        else t
        for t in batch_args
    )


def microbatch_padding_trimmer(
    batch_keys: tuple[str, ...],
) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    """Return the ``_microbatch_transform`` that trims trailing padding columns.

    Every batch tensor whose second dimension equals the padded sequence length
    is cut to the microbatch's longest real length. One host sync per
    microbatch reads that length.
    """
    mask_pos, labels_pos = _required_positions(batch_keys)
    batch_positions = range(1, 1 + len(batch_keys))

    def trim(args: tuple[Any, ...]) -> tuple[Any, ...]:
        mask = args[mask_pos]
        if mask.ndim != _SEQUENCE_NDIM or mask.shape[0] == 0:
            return args
        padded_len = mask.shape[1]
        labels = args[labels_pos] if labels_pos is not None else None
        keep_len = max(int(_real_lengths(mask, labels).max()), 1)
        if keep_len >= padded_len:
            return args
        out = list(args)
        for pos in batch_positions:
            t = out[pos]
            if (
                isinstance(t, torch.Tensor)
                and t.ndim >= _SEQUENCE_NDIM
                and t.shape[1] == padded_len
            ):
                out[pos] = t[:, :keep_len]
        return tuple(out)

    return trim


__all__ = ["microbatch_padding_trimmer", "sort_batch_by_length"]
