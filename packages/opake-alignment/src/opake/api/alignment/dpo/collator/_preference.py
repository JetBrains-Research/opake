"""DPO preference collator — ``(B, ...)`` per-side layout.

.. note:: Layout choice

   TRL uses a ``(2B, L)`` concatenated layout; opake-alignment uses a
   ``(B, ...)`` layout because per-example DP-SGD clips per preference pair —
   the chosen/rejected forward passes share one per-example gradient.

Factory function :func:`preference_collator` returns a callable that
batch-collates DPO preference examples into six mandatory tensors (chosen and
rejected input-ids, attention-masks, completion-masks — all ``(B, L)`` with
*independent* ``L`` for each side) plus optional ``ref_*_logps`` scalars
``(B,)`` when the input dicts carry them.

Sequences are right-padded: pad tokens are appended after the content, so the
real tokens are left-aligned and padding fills the tail.  Sequences longer than
``max_length`` are truncated from the right (keep-start).  ``pad_to_multiple_of``
rounds each side's sequence length up to the nearest multiple (independently),
never down.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.api.alignment.dpo.collator.types import (
        PreferenceBatch,
        PreferenceExample,
    )

__all__ = ["preference_collator"]


def _pad_side(
    sequences: list[list[int]],
    pad_token_id: int,
    max_length: int,
    pad_to_multiple_of: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad + truncate a list of token-id sequences.

    Args:
        sequences: Variable-length lists of integer token ids.
        pad_token_id: Token id used for padding positions.
        max_length: Hard truncation length (keep-start).
        pad_to_multiple_of: When given, the padded length is rounded up to
            the next multiple.  Applied *after* computing the per-batch
            maximum, before allocating tensors.

    Returns:
        A pair ``(input_ids, attention_mask)`` each of shape ``(B, L)``.
    """
    # Truncate first so the per-batch max reflects the effective lengths.
    truncated: list[list[int]] = [s[:max_length] for s in sequences]
    batch_max = max(len(s) for s in truncated) if truncated else 0

    if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
        batch_max = math.ceil(batch_max / pad_to_multiple_of) * pad_to_multiple_of

    b = len(truncated)
    ids = torch.full((b, batch_max), pad_token_id, dtype=torch.long)
    mask = torch.zeros(b, batch_max, dtype=torch.long)

    for i, seq in enumerate(truncated):
        length = len(seq)
        if length > 0:
            ids[i, :length] = torch.tensor(seq, dtype=torch.long)
            mask[i, :length] = 1

    return ids, mask


def _pad_completion_mask(
    masks: list[list[int]],
    target_length: int,
) -> torch.Tensor:
    """Right-pad completion masks with 0 to ``target_length``.

    Args:
        masks: Variable-length lists of 0/1 integers (completion spans).
        target_length: Target padded length (must be >= each mask's length
            after truncation, which the caller guarantees).

    Returns:
        Long tensor of shape ``(B, target_length)``.
    """
    b = len(masks)
    out = torch.zeros(b, target_length, dtype=torch.long)
    for i, m in enumerate(masks):
        length = len(m)
        if length > 0:
            out[i, :length] = torch.tensor(m, dtype=torch.long)
    return out


def preference_collator(
    pad_token_id: int,
    max_length: int,
    *,
    pad_to_multiple_of: int | None = None,
) -> Callable[[list[PreferenceExample]], PreferenceBatch]:
    """DPO preference collator — factory returning a callable.

    .. note:: Layout choice

       TRL uses a ``(2B, L)`` concatenated layout; opake-alignment uses a
       ``(B, ...)`` layout because per-example DP-SGD clips per preference
       pair — the chosen/rejected forward passes share one per-example
       gradient.

    Args:
        pad_token_id: Token id used for right-padding.
        max_length: Maximum sequence length.  Sequences longer than this are
            truncated from the right (keep-start).
        pad_to_multiple_of: When given, each side's padded sequence length is
            independently rounded up to the nearest multiple.

    Returns:
        A callable ``collate(batch: list[PreferenceExample]) -> PreferenceBatch``
        that accepts a list of example dicts and returns:

        **Mandatory keys** (always present):

        - ``chosen_input_ids``: ``(B, Lc)`` long tensor, right-padded with ``pad_token_id``.
        - ``chosen_attention_mask``: ``(B, Lc)`` long tensor; 1 for real tokens, 0 for pad.
        - ``chosen_completion_mask``: ``(B, Lc)`` long tensor; 1 in the completion span, 0 else.
        - ``rejected_input_ids``: ``(B, Lr)`` long tensor.
        - ``rejected_attention_mask``: ``(B, Lr)`` long tensor.
        - ``rejected_completion_mask``: ``(B, Lr)`` long tensor.

        ``Lc`` and ``Lr`` are the per-side batch maxima (may differ from each
        other) after truncation, rounded up to ``pad_to_multiple_of`` if given.

        **Optional keys** (present iff *all* examples in the batch carry them):

        - ``ref_chosen_logps``: ``(B,)`` float tensor.
        - ``ref_rejected_logps``: ``(B,)`` float tensor.

    Each input example dict must have:

    - ``chosen_input_ids``: ``list[int]``
    - ``rejected_input_ids``: ``list[int]``
    - ``chosen_completion_mask``: ``list[int]``
    - ``rejected_completion_mask``: ``list[int]``

    And may optionally carry ``ref_chosen_logps: float`` and
    ``ref_rejected_logps: float``.
    """

    def collate(batch: list[PreferenceExample]) -> PreferenceBatch:
        # ------------------------------------------------------------------ #
        # Extract per-side sequences.
        # ------------------------------------------------------------------ #
        chosen_ids_raw: list[list[int]] = [ex["chosen_input_ids"] for ex in batch]
        rejected_ids_raw: list[list[int]] = [ex["rejected_input_ids"] for ex in batch]
        chosen_cmask_raw: list[list[int]] = [
            ex["chosen_completion_mask"] for ex in batch
        ]
        rejected_cmask_raw: list[list[int]] = [
            ex["rejected_completion_mask"] for ex in batch
        ]

        # Truncate completion masks in sync with input ids (keep-start).
        chosen_cmask_trunc: list[list[int]] = [m[:max_length] for m in chosen_cmask_raw]
        rejected_cmask_trunc: list[list[int]] = [
            m[:max_length] for m in rejected_cmask_raw
        ]

        # ------------------------------------------------------------------ #
        # Pad chosen side.
        # ------------------------------------------------------------------ #
        chosen_input_ids, chosen_attention_mask = _pad_side(
            chosen_ids_raw,
            pad_token_id=pad_token_id,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        chosen_completion_mask = _pad_completion_mask(
            chosen_cmask_trunc,
            target_length=chosen_input_ids.shape[1],
        )

        # ------------------------------------------------------------------ #
        # Pad rejected side (independently — Lc may differ from Lr).
        # ------------------------------------------------------------------ #
        rejected_input_ids, rejected_attention_mask = _pad_side(
            rejected_ids_raw,
            pad_token_id=pad_token_id,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        rejected_completion_mask = _pad_completion_mask(
            rejected_cmask_trunc,
            target_length=rejected_input_ids.shape[1],
        )

        # ------------------------------------------------------------------ #
        # Assemble output dict.
        # ------------------------------------------------------------------ #
        out: PreferenceBatch = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_completion_mask": chosen_completion_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_completion_mask": rejected_completion_mask,
        }

        # Optional ref logps — included only if ALL examples carry them.
        if all("ref_chosen_logps" in ex for ex in batch):
            out["ref_chosen_logps"] = torch.tensor(
                [ex["ref_chosen_logps"] for ex in batch],
                dtype=torch.float32,
            )
        if all("ref_rejected_logps" in ex for ex in batch):
            out["ref_rejected_logps"] = torch.tensor(
                [ex["ref_rejected_logps"] for ex in batch],
                dtype=torch.float32,
            )

        return out

    return collate
