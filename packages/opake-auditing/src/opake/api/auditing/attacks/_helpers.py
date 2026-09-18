"""Shared helpers for membership inference scoring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opake.api.auditing._coin_flip import CanaryScores
from opake.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.api.auditing._coin_flip import CoinFlip


def _validate_batch_argnums(batch_argnums: tuple[int, ...], n_non_batch: int) -> None:
    """Validate batch_argnums constraints."""
    if not batch_argnums:
        raise ConfigurationError(
            *(f"batch_argnums must be non-empty, got {batch_argnums}",)
        )
    if any(a < 0 for a in batch_argnums):
        raise ConfigurationError(
            *(f"batch_argnums must be non-negative, got {batch_argnums}",)
        )
    if len(set(batch_argnums)) != len(batch_argnums):
        raise ConfigurationError(
            *(f"batch_argnums must be unique, got {batch_argnums}",)
        )
    if tuple(sorted(batch_argnums)) != batch_argnums:
        raise ConfigurationError(
            *(f"batch_argnums must be sorted, got {batch_argnums}",)
        )
    n_total = n_non_batch + len(batch_argnums)
    if max(batch_argnums) >= n_total:
        raise ConfigurationError(
            *(
                f"batch_argnums index {max(batch_argnums)} out of range for "
                f"{n_total} total arguments ({n_non_batch} non-batched + "
                f"{len(batch_argnums)} batched), got {batch_argnums}",
            )
        )


def _merge_args(
    args: tuple[Any, ...],
    batch_tensors: tuple[Any, ...],
    batch_argnums: tuple[int, ...],
) -> list[Any]:
    """Merge non-batched args and batch tensors into a single arg list."""
    n_total = len(args) + len(batch_argnums)
    result: list[Any] = [None] * n_total

    for pos, tensor in zip(batch_argnums, batch_tensors, strict=False):
        result[pos] = tensor

    arg_iter = iter(args)
    for i in range(n_total):
        if i not in batch_argnums:
            result[i] = next(arg_iter)

    return result


def _extract_batch_tensors(
    batch: Any,
    batch_argnums: tuple[int, ...],
) -> tuple[Any, ...]:
    """Extract tensors from a collated batch.

    Returns a tuple of tensors matching the length of ``batch_argnums``.
    """
    if isinstance(batch, (list, tuple)):
        return tuple(batch[i] for i in range(len(batch_argnums)))
    return (batch,)


class _IndexedCanaries:
    """Map-style dataset yielding ``(position, example)`` canary pairs."""

    def __init__(self, dataset: Any, canary_indices: np.ndarray):
        self._dataset = dataset
        self._canary_indices = canary_indices

    def __len__(self) -> int:
        return len(self._canary_indices)

    def __getitem__(self, position: int) -> tuple[int, Any]:
        return position, self._dataset[int(self._canary_indices[position])]


def _collated_length(collated: Any) -> int | None:
    """Leading dimension of a collated batch, or None if undeterminable."""
    probe = (
        collated[0] if isinstance(collated, (list, tuple)) and collated else collated
    )
    shape = getattr(probe, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def _canary_loader(
    dataset: Any,
    coin_flip: CoinFlip,
    batch_size: int,
    collate_fn: Callable | None,
) -> Any:
    """Build the internal DataLoader for verified canary scoring.

    Each batch arrives as ``(positions, collated_examples)``: the canary
    positions ride alongside the examples through collation, so the
    pairing never relies on the loader's iteration order.  Positions are
    captured before collation, so ``collate_fn`` must emit one row per
    input example in the order it received them — a collate that drops or
    reorders rows breaks the pairing.  Dropped rows are caught here;
    reordering within a batch cannot be detected and is a caller
    obligation.
    """
    import torch.utils.data as tud

    example_collate = tud.default_collate if collate_fn is None else collate_fn

    def indexed_collate(batch: list[tuple[int, Any]]) -> tuple[list[int], Any]:
        positions = [position for position, _ in batch]
        examples = [example for _, example in batch]
        collated = example_collate(examples)
        n_collated = _collated_length(collated)
        if n_collated is not None and n_collated != len(positions):
            raise ConfigurationError(
                *(
                    f"collate_fn returned {n_collated} rows for a batch of "
                    f"{len(positions)} canaries; it must emit exactly one row "
                    "per example, in the order received, or scores lose the "
                    "identifiers they are paired with",
                )
            )
        return positions, collated

    return tud.DataLoader(
        _IndexedCanaries(dataset, coin_flip.canary_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=indexed_collate,
    )


def _scoring_loader(
    *,
    dataset: Any,
    coin_flip: CoinFlip,
    batch_size: int,
    collate_fn: Callable | None,
    reference_scores: CanaryScores | None,
) -> Any:
    """Validate scoring arguments and build the canary loader."""
    dataset_size = len(dataset)
    if coin_flip.dataset_size is not None and dataset_size != coin_flip.dataset_size:
        raise ConfigurationError(
            *(
                f"dataset length ({dataset_size}) does not match the dataset size "
                f"({coin_flip.dataset_size}) the CoinFlip was constructed from; "
                "scoring requires the full concatenated dataset, not a training "
                "subset or split",
            )
        )
    if np.any(coin_flip.canary_indices < 0) or np.any(
        coin_flip.canary_indices >= dataset_size
    ):
        raise ConfigurationError(
            *(
                f"canary indices must be within dataset bounds [0, {dataset_size}); "
                "scoring requires the full concatenated dataset used to construct "
                "the CoinFlip",
            )
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise InputTypeError(
            *(f"batch_size must be an int, got {type(batch_size).__name__}",)
        )
    if batch_size < 1:
        raise ConfigurationError(*(f"batch_size must be positive, got {batch_size}",))
    if reference_scores is not None and not isinstance(reference_scores, CanaryScores):
        raise InputTypeError(
            *(
                "reference_scores must carry canary identifiers; compute it "
                "with the same coin_flip= and dataset=, or attest identifiers "
                "with canary_scores(values, canary_indices=...)",
            )
        )
    return _canary_loader(dataset, coin_flip, batch_size, collate_fn)


def _aligned_reference(ids: np.ndarray, reference_scores: CanaryScores) -> np.ndarray:
    """Return reference score values aligned to ``ids`` by identifier."""
    ref_ids = reference_scores.canary_indices
    if ref_ids.size == 0:
        if ids.size:
            raise ConfigurationError(
                *(
                    "reference_scores are empty but scores are not; compute "
                    "the reference over the same coin_flip and dataset",
                )
            )
        return np.empty(0, dtype=float)
    sorter = np.argsort(ref_ids, kind="stable")
    pos = np.searchsorted(ref_ids[sorter], ids)
    pos = np.minimum(pos, ref_ids.size - 1)
    if not np.all(ref_ids[sorter][pos] == ids):
        raise ConfigurationError(
            *(
                "reference_scores do not cover the scored canaries; compute "
                "the reference over the same coin_flip and dataset",
            )
        )
    return reference_scores.scores[sorter[pos]]


def _bind_scores(
    scores: np.ndarray,
    positions: list[int],
    *,
    coin_flip: CoinFlip,
    reference_scores: CanaryScores | None,
) -> CanaryScores:
    """Apply reference calibration and attach canary identifiers."""
    ids = coin_flip.canary_indices[np.asarray(positions, dtype=int)]
    if reference_scores is not None:
        scores = np.subtract(scores, _aligned_reference(ids, reference_scores))
    return CanaryScores(scores, canary_indices=ids)
