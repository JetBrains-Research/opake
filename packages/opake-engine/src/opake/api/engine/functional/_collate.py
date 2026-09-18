# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Collate utilities for variable-size / empty batches.

The :func:`empty_collate` wrapper handles the empty-batch case that arises
from any Poisson-style sampler (``opake.dpsgd.sampling.PoissonSampler``,
``opake.dpftrl.sampling.CyclicPoissonSampler``, ...) without knowing which
mechanism produced the empty list.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


def _empty_like(template: T) -> T:
    """Create an empty-batch replica of a collate output.

    Recursively walks the structure and replaces every tensor with a
    zero-batch-dim copy (``tensor[:0]``), preserving dtype, device,
    and all non-batch dimensions.  Non-tensor leaves are passed through
    unchanged.  Supports dicts, ``Mapping`` subclasses (e.g. HuggingFace
    ``BatchEncoding``), lists, and tuples.

    Args:
        template: A representative collate output (first non-empty batch).

    Returns:
        Structure identical to *template* but with every tensor having
        ``shape[0] == 0``.
    """
    if isinstance(template, torch.Tensor):
        return template[:0]
    if isinstance(template, Mapping):
        return {k: _empty_like(v) for k, v in template.items()}
    if isinstance(template, tuple):
        return tuple(_empty_like(v) for v in template)
    if isinstance(template, list):
        return [_empty_like(v) for v in template]
    return template


class _EmptyCollator(Generic[T]):
    """Pickle-safe stateful wrapper used by :func:`empty_collate`.

    ``DataLoader`` serializes its collator when workers use the ``spawn`` or
    ``forkserver`` multiprocessing start methods.  Keeping this callable at
    module scope makes the wrapper serializable whenever the wrapped collator
    and learned template are serializable too.
    """

    def __init__(self, collate_fn: Callable[..., T]) -> None:
        self._collate_fn = collate_fn
        self._template: T | None = None
        self._update_wrapper_metadata()

    def _update_wrapper_metadata(self) -> None:
        # updated=() skips the default `__dict__` merge, and restricting
        # assigned to plain string metadata skips `__annotations__` /
        # `__type_params__`: both default `update_wrapper` behaviors would
        # otherwise copy arbitrary (possibly non-pickleable) objects from
        # collate_fn onto this instance's __dict__, and __getstate__ below
        # only protects the two attributes it explicitly names.
        functools.update_wrapper(
            self,
            self._collate_fn,
            assigned=("__module__", "__name__", "__qualname__", "__doc__"),
            updated=(),
        )

    def __getstate__(self) -> dict[str, object]:
        # Pickle only the state needed to reconstruct behavior. This avoids
        # ever depending on what update_wrapper happens to have copied onto
        # the instance, so a collate_fn with a non-pickleable annotation or
        # type parameter still leaves the wrapper itself pickleable.
        return {"_collate_fn": self._collate_fn, "_template": self._template}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._collate_fn = state["_collate_fn"]
        self._template = state["_template"]
        self._update_wrapper_metadata()

    def __call__(self, examples):
        if not examples:
            if self._template is None:
                return self._collate_fn(examples)
            return _empty_like(self._template)

        result = self._collate_fn(examples)

        if self._template is None:
            self._template = copy.deepcopy(result)

        return result


def empty_collate(collate_fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a collate function so empty example lists yield zero-batch outputs.

    Poisson-style samplers can yield empty index lists, producing batches with
    zero examples. Standard collators (e.g., HuggingFace
    ``DataCollatorForLanguageModeling``) crash on empty input because they
    unconditionally index ``examples[0]``.

    This wrapper learns the output structure from the first non-empty batch
    and returns a properly-shaped empty batch (batch dim = 0) when the
    example list is empty::

        collate = empty_collate(my_collate_fn)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

        for batch in loader:
            input_ids = batch["input_ids"]   # shape (0, seq_len) when empty
            if len(input_ids) == 0:
                continue
            ...

    .. note::

        If the very first batch is empty (before any structure has been
        learned), the call falls through to *collate_fn* which will
        typically crash — this is the correct signal that the sample rate
        is misconfigured.

    Args:
        collate_fn: Original collate function.

    Returns:
        Wrapped function that returns empty-batch-dim outputs for empty
        example lists.
    """
    return _EmptyCollator(collate_fn)
