"""Loss-based membership scoring for privacy auditing.

Computes per-example membership scores using the same ``torch.func.vmap``
and ``batch_argnums`` pattern as :func:`opake.clipped_grad`. Higher scores
indicate higher likelihood of being a training member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opake.api.auditing.attacks._helpers import (
    _bind_scores,
    _extract_batch_tensors,
    _merge_args,
    _scoring_loader,
    _validate_batch_argnums,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.api.auditing._coin_flip import CanaryScores, CoinFlip

__all__ = ["loss_scores"]


def loss_scores(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    coin_flip: CoinFlip,
    dataset: Any,
    reference_scores: CanaryScores | None = None,
    batch_size: int = 32,
    collate_fn: Callable | None = None,
) -> CanaryScores:
    """Compute membership scores as negative per-example loss.

    Higher score = lower loss = more likely a training member.
    Uses ``torch.func.vmap`` for per-example loss computation, following
    the same ``batch_argnums`` convention as :func:`~opake.clipped_grad`.

    The scorer builds its own loader over the partition's canaries and
    pairs every score with the dataset index of the example that produced
    it, returning :class:`~opake.auditing.types.CanaryScores` — the form
    :func:`~opake.auditing.one_run` requires.  The pairing is joined by
    identifier, so no iteration order over the batches can misalign it.
    ``collate_fn`` receives the raw canary examples and must return one
    row per example in the order it received them; identifiers are
    attached per batch before collation, so a ``collate_fn`` that reorders
    within a batch does misalign the pairing and cannot be detected.

    To audit scores computed by some other pipeline, attest their
    identifiers with :func:`~opake.auditing.canary_scores` instead.

    When ``reference_scores`` are provided, the returned scores are
    ``current - reference``, which equals ``loss(w0) - loss(wt)``
    (the loss reduction from initial to current model). This matches
    Algorithm 3 of Steinke et al. (2023).  The reference is aligned by
    identifier before subtraction.

    Args:
        loss_fn: Per-example loss function, same as used with
            :func:`~opake.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches. Must be sorted, unique, non-negative.
        coin_flip: The audit partition to score against.
        dataset: The full dataset the partition was created from; canaries
            are selected internally by identifier.
        reference_scores: Baseline scores from an untrained model, over the
            same partition. When provided, returned scores are the loss
            reduction ``scores - reference_scores``. Typically obtained by
            calling ``loss_scores`` on the untrained model before training.
        batch_size: Batch size of the internal loader. Defaults to 32.
        collate_fn: Collation for the internal loader. Defaults to
            ``torch.utils.data.default_collate``.

    Returns:
        A :class:`~opake.auditing.types.CanaryScores`, shape ``(n,)``:
        negated losses (higher = more likely member), optionally adjusted
        by reference scores, each carrying its canary's dataset index.

    Raises:
        ValueError: If ``batch_argnums`` is malformed, ``batch_size`` is
            not positive, or ``collate_fn`` changes the batch row count.
        TypeError: If ``batch_size`` is not an int, or ``reference_scores``
            does not carry identifiers.

    Example (HuggingFace pattern)::

        def canary_collate(examples):
            batch = data_collator(examples)
            return (batch["input_ids"].to(device),)

        ref = auditing.loss_scores(
            loss_fn, initial_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset,
            batch_size=32, collate_fn=canary_collate,
        )
        scores = auditing.loss_scores(
            loss_fn, trained_params,
            batch_argnums=(1,),
            coin_flip=cf, dataset=dataset,
            batch_size=32, collate_fn=canary_collate,
            reference_scores=ref,
        )
        estimate = auditing.one_run(scores, coin_flip=cf)
    """
    import torch

    _validate_batch_argnums(batch_argnums, len(args))
    loader = _scoring_loader(
        dataset=dataset,
        coin_flip=coin_flip,
        batch_size=batch_size,
        collate_fn=collate_fn,
        reference_scores=reference_scores,
    )

    # Build in_dims for vmap: None for non-batch args, 0 for batch args
    n_args = len(args) + len(batch_argnums)
    in_dims = tuple(0 if i in batch_argnums else None for i in range(n_args))
    per_example_fn = torch.func.vmap(loss_fn, in_dims=in_dims)

    all_positions: list[int] = []
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for positions, batch in loader:
            all_positions.extend(int(position) for position in positions)
            batch_tensors = _extract_batch_tensors(batch, batch_argnums)

            full_args = _merge_args(args, batch_tensors, batch_argnums)

            losses = per_example_fn(*full_args)
            all_scores.append(-losses.detach().cpu().numpy())

    scores = np.concatenate(all_scores) if all_scores else np.empty(0)

    return _bind_scores(
        scores,
        all_positions,
        coin_flip=coin_flip,
        reference_scores=reference_scores,
    )
