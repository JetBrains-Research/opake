"""Language-modeling collator — factory function.

Produces a stateless callable that transforms a list of per-example dicts into
a :class:`~opake.api.alignment.sft.collator.types.LMBatch` ready for a causal
language model forward pass.

The public API is the factory function :func:`language_modeling_collator`; a
private :class:`_LMCollator` class inside this module handles the
implementation. Callers must never instantiate :class:`_LMCollator` directly;
public builders in this repo are exposed as factory functions rather than
user-instantiated classes.

Design notes
------------
* **Truncation** — keep-start: examples longer than ``max_length`` are
  truncated from the right (tokens beyond position ``max_length - 1`` are
  dropped).  This matches the most common causal-LM training convention and is
  deterministic.
* **Padding** — right-pad with ``pad_token_id`` to the length of the longest
  (post-truncation) example in the batch, subject to ``pad_to_multiple_of``.
* **Labels** — copy of ``input_ids`` with pad positions set to ``-100``.  When
  ``completion_only_loss=True`` and a ``completion_mask`` is present,
  non-completion positions are also set to ``-100``. Examples without a mask
  retain full-sequence labels, matching TRL's language-modeling collator.
* **completion_mask output key** — included only when at least one example in
  the batch supplies ``"completion_mask"``.  This preserves backward
  compatibility with simple SFT datasets that do not carry the field.
* **Determinism** — collation is purely functional given the input list;
  identical inputs produce identical output tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.api.alignment.sft.collator.types import LMBatch, LMExample

__all__ = ["language_modeling_collator"]

_IGNORE_INDEX: int = -100


class _LMCollator:
    """Private implementation of the language-modeling collator.

    Do NOT instantiate this class directly outside of :func:`language_modeling_collator`.

    Args:
        pad_token_id: Token id used for right-padding.
        max_length: Maximum sequence length (keep-start truncation applied
            before padding).
        completion_only_loss: When ``True``, non-completion positions in
            ``labels`` are masked to ``-100`` in addition to pad positions.
        pad_to_multiple_of: When set, the padded length ``L`` is rounded up to
            the nearest multiple of this value.  Useful for tensor-core
            alignment on modern hardware.
    """

    __slots__ = (
        "_completion_only_loss",
        "_max_length",
        "_pad_to_multiple_of",
        "_pad_token_id",
    )

    def __init__(
        self,
        pad_token_id: int,
        max_length: int,
        *,
        completion_only_loss: bool,
        pad_to_multiple_of: int | None,
    ) -> None:
        self._pad_token_id = pad_token_id
        self._max_length = max_length
        self._completion_only_loss = completion_only_loss
        self._pad_to_multiple_of = pad_to_multiple_of

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, examples: list[LMExample]) -> LMBatch:
        """Collate *examples* into a padded batch.

        Args:
            examples: List of per-example dicts.  Each dict must contain
                ``"input_ids": list[int]``.  Optionally contains
                ``"completion_mask": list[int]`` (``1`` on completion tokens,
                ``0`` elsewhere).

        Returns:
            A :class:`~opake.api.alignment.sft.collator.types.LMBatch` dict with
            keys ``input_ids``, ``attention_mask``, ``labels``, and optionally
            ``completion_mask``.
        """
        if not examples:
            return self._empty_batch()

        # ---- Truncate each example to max_length (keep-start) --------
        input_ids_list: list[list[int]] = []
        completion_mask_list: list[list[int] | None] = []
        has_completion_mask = False

        for ex in examples:
            ids: list[int] = ex["input_ids"][: self._max_length]
            input_ids_list.append(ids)

            cm_raw: list[int] | None = ex.get("completion_mask")
            if cm_raw is not None:
                has_completion_mask = True
                completion_mask_list.append(cm_raw[: self._max_length])
            else:
                completion_mask_list.append(None)

        # ---- Compute padded length L ---------------------------------
        max_len = max(len(ids) for ids in input_ids_list)
        # Cap at max_length (truncation already applied, so max_len <= max_length).
        L = min(max_len, self._max_length)
        if self._pad_to_multiple_of is not None and self._pad_to_multiple_of > 1:
            remainder = L % self._pad_to_multiple_of
            if remainder != 0:
                L = L + (self._pad_to_multiple_of - remainder)

        # ---- Build tensors ------------------------------------------
        B = len(examples)
        input_ids_t = torch.full((B, L), self._pad_token_id, dtype=torch.long)
        attention_mask_t = torch.zeros((B, L), dtype=torch.long)
        labels_t = torch.full((B, L), _IGNORE_INDEX, dtype=torch.long)
        if has_completion_mask:
            completion_mask_t = torch.zeros((B, L), dtype=torch.long)

        for i, (ids, cm) in enumerate(
            zip(input_ids_list, completion_mask_list, strict=False)
        ):
            seq_len = len(ids)
            input_ids_t[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask_t[i, :seq_len] = 1
            # Labels: copy input_ids; pad positions already -100.
            labels_t[i, :seq_len] = torch.tensor(ids, dtype=torch.long)

            if has_completion_mask and cm is not None:
                cm_tensor = torch.tensor(cm, dtype=torch.long)
                # Pad completion_mask with 0s at padding positions (already 0).
                completion_mask_t[i, :seq_len] = cm_tensor  # type: ignore[possibly-undefined]
                if self._completion_only_loss:
                    # Mask non-completion real tokens in labels.
                    non_completion = cm_tensor == 0
                    # Build a position index for the real tokens.
                    positions = torch.arange(seq_len, dtype=torch.long)
                    masked_positions = positions[non_completion]
                    labels_t[i].scatter_(0, masked_positions, _IGNORE_INDEX)
            elif has_completion_mask:
                # No completion subset was supplied, so the fully supervised
                # real-token span is also the completion span for this row.
                completion_mask_t[i, :seq_len] = 1  # type: ignore[possibly-undefined]
        # ---- Assemble output ----------------------------------------
        out: LMBatch = {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
        }
        if has_completion_mask:
            out["completion_mask"] = completion_mask_t  # type: ignore[possibly-undefined]
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_batch(self) -> LMBatch:
        """Return a well-typed empty batch (B=0) for the zero-example edge case."""
        empty = torch.zeros((0, 0), dtype=torch.long)
        return {
            "input_ids": empty,
            "attention_mask": empty,
            "labels": empty,
        }


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def language_modeling_collator(
    pad_token_id: int,
    max_length: int,
    *,
    completion_only_loss: bool = False,
    pad_to_multiple_of: int | None = None,
) -> Callable[[list[LMExample]], LMBatch]:
    """Return a callable that collates per-example dicts into a padded :class:`LMBatch`.

    This is a factory function: it returns a
    plain callable, not a user-instantiated class.  The returned callable is
    stateless and deterministic — the same input list always produces identical
    output tensors.

    Args:
        pad_token_id: Token id used to right-pad sequences to the batch
            length.
        max_length: Maximum sequence length.  Examples longer than this are
            truncated from the right (keep-start) before padding.
        completion_only_loss: When ``True``, positions where
            ``completion_mask == 0`` are set to ``-100`` in ``labels`` so that
            prompt tokens do not contribute to the language-modelling loss.
            Defaults to ``False`` (standard next-token prediction over the
            full sequence).
        pad_to_multiple_of: When set, the padded length ``L`` is rounded up to
            the nearest multiple of this value before the batch tensors are
            allocated.  Useful for tensor-core alignment (e.g. ``8`` or
            ``64``).  ``None`` disables rounding.

    Returns:
        A ``collate(examples: list[LMExample]) -> LMBatch``
        callable.  See :class:`~opake.api.alignment.sft.collator.types.LMBatch`
        for the output schema.

    Example::

        collate = language_modeling_collator(pad_token_id=0, max_length=16)
        batch = collate([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}])
        # batch["input_ids"].shape == (2, 3)
    """
    return _LMCollator(
        pad_token_id=pad_token_id,
        max_length=max_length,
        completion_only_loss=completion_only_loss,
        pad_to_multiple_of=pad_to_multiple_of,
    )
