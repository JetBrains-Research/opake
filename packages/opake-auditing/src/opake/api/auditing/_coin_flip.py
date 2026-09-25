"""Coin-flip partitioning for canary-based privacy auditing.

Shared infrastructure used by all auditing approaches (OneRun, Nasr, etc.).
Each canary is independently included or excluded from training with
probability 0.5 (a fair coin flip).

Membership scores enter the estimator as :class:`CanaryScores`: each score
carries the dataset index of the canary it was computed for, and
:meth:`CoinFlip.split_scores` joins scores to coin-flip labels by that
identifier.  The order scores arrive in therefore cannot misalign them
against the labels — wrong, missing, or duplicated identifiers raise
instead.  Whether each score carries the *right* identifier is settled
earlier, when the score is produced; see :func:`canary_scores`.

Reference:
    Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training
    Run." NeurIPS 2023. https://arxiv.org/abs/2305.08846
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from opake.exceptions import ConfigurationError, InputTypeError
from opake.random import fold_in

if TYPE_CHECKING:
    from opake.random.types import RngKey

__all__ = ["CanaryScores", "CoinFlip", "canary_scores", "coin_flip"]

_CANARY_SELECTION_DOMAIN = "opake.auditing.canary_selection"
_COIN_FLIP_DOMAIN = "opake.auditing.coin_flip"
_INCLUSION_PROBABILITY = 0.5
_DATASET_ATTESTATION_VERSION = "v1"


def _source_fingerprint(dataset: Any) -> str | None:
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is None:
        return None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ConfigurationError(
            *(
                "dataset `_fingerprint` must be a non-empty string, got "
                f"{type(fingerprint).__name__}",
            )
        )
    return fingerprint


def _frame(digest: Any, tag: bytes, payload: bytes) -> None:
    """Append one unambiguous field to an attestation digest."""
    digest.update(len(tag).to_bytes(2, byteorder="big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, byteorder="big"))
    digest.update(payload)


def _update_shape(digest: Any, shape: tuple[int, ...]) -> None:
    _frame(digest, b"rank", len(shape).to_bytes(8, byteorder="big"))
    for dimension in shape:
        _frame(digest, b"dimension", int(dimension).to_bytes(8, byteorder="big"))


def _update_tensor(digest: Any, value: torch.Tensor) -> None:
    if (
        value.layout != torch.strided
        or value.is_quantized
        or value.device.type == "meta"
    ):
        raise ConfigurationError(
            *(
                "dataset attestation supports only dense, strided, non-quantized "
                f"tensors with materialized values; got {value.layout} on {value.device}",
            )
        )
    canonical = (
        value.detach()
        .resolve_conj()
        .resolve_neg()
        .cpu()
        .contiguous()
        .clone(memory_format=torch.contiguous_format)
    )
    _frame(digest, b"torch-dtype", str(canonical.dtype).encode())
    _update_shape(digest, tuple(canonical.shape))
    _frame(digest, b"torch-values", bytes(canonical.untyped_storage()))


def _update_array(digest: Any, value: np.ndarray) -> None:
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise ConfigurationError(
            *(
                "dataset attestation does not support object or structured NumPy "
                f"dtypes; got {value.dtype}",
            )
        )
    canonical = np.ascontiguousarray(value)
    _frame(digest, b"numpy-dtype", canonical.dtype.str.encode())
    _update_shape(digest, canonical.shape)
    _frame(digest, b"numpy-values", canonical.tobytes(order="C"))


def _update_value(digest: Any, value: Any) -> None:
    """Append a canonical structural value to an attestation digest."""
    if value is None:
        _frame(digest, b"none", b"")
    elif isinstance(value, bool):
        _frame(digest, b"bool", bytes([value]))
    elif isinstance(value, int):
        _frame(digest, b"int", str(value).encode())
    elif isinstance(value, float):
        _frame(digest, b"float64", struct.pack(">d", value))
    elif isinstance(value, str):
        _frame(digest, b"str", value.encode(errors="surrogatepass"))
    elif isinstance(value, bytes):
        _frame(digest, b"bytes", value)
    elif isinstance(value, bytearray):
        _frame(digest, b"bytearray", bytes(value))
    elif isinstance(value, torch.Tensor):
        _frame(digest, b"torch", b"")
        _update_tensor(digest, value)
    elif isinstance(value, np.ndarray):
        _frame(digest, b"numpy", b"")
        _update_array(digest, value)
    elif isinstance(value, np.generic):
        _frame(digest, b"numpy-scalar", b"")
        _update_array(digest, np.asarray(value))
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ConfigurationError(
                *("dataset attestation requires mappings with string keys",)
            )
        _frame(digest, b"mapping", len(value).to_bytes(8, byteorder="big"))
        for key in sorted(value):
            _update_value(digest, key)
            _update_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        tag = b"list" if isinstance(value, list) else b"tuple"
        _frame(digest, tag, len(value).to_bytes(8, byteorder="big"))
        for item in value:
            _update_value(digest, item)
    else:
        raise ConfigurationError(
            *(
                "dataset attestation cannot encode row values of type "
                f"{type(value).__module__}.{type(value).__qualname__}",
            )
        )


def _update_dataset_row(digest: Any, dataset: Any, index: int) -> None:
    """Append one row without invoking arbitrary dataset transforms."""
    source_fingerprint = _source_fingerprint(dataset)
    if source_fingerprint is not None:
        _frame(digest, b"source", source_fingerprint.encode(errors="surrogatepass"))
        _frame(digest, b"source-index", index.to_bytes(8, "big", signed=True))
        return
    if isinstance(dataset, Subset):
        source_index = int(dataset.indices[index])
        _frame(digest, b"subset-index", source_index.to_bytes(8, "big", signed=True))
        _update_dataset_row(digest, dataset.dataset, source_index)
        return
    if isinstance(dataset, TensorDataset):
        _frame(digest, b"tensor-dataset", len(dataset.tensors).to_bytes(8, "big"))
        _update_value(digest, tuple(tensor[index] for tensor in dataset.tensors))
        return
    if isinstance(dataset, torch.Tensor):
        _frame(digest, b"tensor-dataset-row", b"")
        _update_value(digest, dataset[index])
        return
    if isinstance(dataset, np.ndarray):
        _frame(digest, b"array-dataset-row", b"")
        _update_value(digest, dataset[index])
        return
    if isinstance(dataset, (list, tuple)):
        tag = b"list-dataset-row" if isinstance(dataset, list) else b"tuple-dataset-row"
        _frame(digest, tag, b"")
        _update_value(digest, dataset[index])
        return
    raise ConfigurationError(
        *(
            "dataset attestation requires a deterministic `_fingerprint` or a "
            "supported in-memory dataset (list, tuple, torch.Tensor, "
            "numpy.ndarray, TensorDataset, or Subset); got "
            f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        )
    )


def _dataset_fingerprint(dataset: Any, canary_indices: np.ndarray) -> str:
    """Attest a stable source identity or every selected canary row."""
    digest = hashlib.blake2b(digest_size=32)
    digest.update(b"opake.auditing.dataset-attestation\0")
    digest.update(_DATASET_ATTESTATION_VERSION.encode())
    source_fingerprint = _source_fingerprint(dataset)
    if source_fingerprint is not None:
        _frame(digest, b"source", source_fingerprint.encode(errors="surrogatepass"))
        return f"{_DATASET_ATTESTATION_VERSION}:source:{digest.hexdigest()}"

    for raw_index in np.sort(canary_indices):
        index = int(raw_index)
        _frame(digest, b"canary-index", index.to_bytes(8, "big", signed=True))
        _update_dataset_row(digest, dataset, index)
    return f"{_DATASET_ATTESTATION_VERSION}:rows:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True)
class CanaryScores:
    """Membership scores paired with stable canary identifiers.

    Each ``scores[k]`` carries ``canary_indices[k]`` — the dataset index
    of the canary it was computed for.  :meth:`CoinFlip.split_scores`
    joins scores to coin-flip labels by these identifiers, so the scoring
    order does not matter and cannot silently misalign the pairing.

    Produced by :func:`~opake.auditing.loss_scores` and
    :func:`~opake.auditing.gradient_scores` when scoring in verified
    mode (``coin_flip=`` + ``dataset=``).  Use the :func:`canary_scores`
    factory to attest identifiers for scores computed outside those
    helpers.

    Both arrays are defensively copied and marked read-only.  This guards
    against honest mistakes (post-hoc sorting, in-place edits), not
    adversarial callers.

    Attributes:
        scores: Membership scores, shape ``(num_canaries,)``, float.
        canary_indices: Dataset index of the canary behind each score.
    """

    scores: np.ndarray
    canary_indices: np.ndarray

    def __post_init__(self) -> None:
        scores = np.array(self.scores, dtype=float)
        indices = np.array(self.canary_indices)
        if scores.ndim != 1:
            raise ConfigurationError(
                *(f"scores must be 1-D, got shape {scores.shape}",)
            )
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ConfigurationError(
                *(
                    "canary_indices must be a 1-D integer array, got "
                    f"shape {indices.shape}, dtype {indices.dtype}",
                )
            )
        if scores.shape != indices.shape:
            raise ConfigurationError(
                *(
                    f"scores and canary_indices must have equal length, got "
                    f"{scores.shape[0]} scores for {indices.shape[0]} indices",
                )
            )
        if np.unique(indices).size != indices.size:
            raise ConfigurationError(
                *(
                    "canary_indices must be unique; duplicate identifiers make "
                    "the score join ambiguous",
                )
            )
        scores.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "canary_indices", indices)

    def __len__(self) -> int:
        return self.scores.shape[0]

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy:
            return np.array(self.scores, dtype=dtype)
        return np.asarray(self.scores, dtype=dtype)

    def __repr__(self) -> str:
        return f"CanaryScores(num_canaries={len(self)})"


def canary_scores(scores: Any, *, canary_indices: Any) -> CanaryScores:
    """Attest which canary each externally computed score belongs to.

    :func:`~opake.auditing.loss_scores` and
    :func:`~opake.auditing.gradient_scores` already return
    :class:`CanaryScores` in verified mode; use this factory for scores
    computed by some other pipeline.  Pass the identifiers in whatever
    order the scores were computed — :meth:`CoinFlip.split_scores` joins
    on them rather than assuming a position.

    Args:
        scores: Membership scores, shape ``(num_canaries,)``, float.
        canary_indices: Dataset index of the canary behind each score, in
            the same order as ``scores``.

    Returns:
        A :class:`CanaryScores` pairing each score with its identifier.

    Raises:
        ValueError: If either array is not 1-D, the identifiers are not
            integers, the lengths disagree, or an identifier repeats.

    Example::

        scores = auditing.canary_scores(values, canary_indices=ids)
        estimate = auditing.one_run(scores, coin_flip=cf)
    """
    return CanaryScores(scores, canary_indices=canary_indices)


@dataclasses.dataclass(frozen=True)
class CoinFlip:
    """Coin-flip partitioning for canary-based privacy auditing.

    Each canary is independently included or excluded from training
    with probability 0.5 (a fair coin flip). This class only handles
    the partition — it does not know about scoring or epsilon estimation.

    Use the :func:`coin_flip` factory to create instances from a dataset.

    Attributes:
        num_canaries: Total number of canary examples.
        canary_indices: All canary dataset indices.
        in_indices: Canary indices included in training (coin = heads).
        out_indices: Canary indices excluded from training (coin = tails).
        dataset_size: Size of the dataset used to create this partition, if known.
    """

    num_canaries: int
    canary_indices: np.ndarray
    _in_mask: np.ndarray
    in_indices: np.ndarray
    out_indices: np.ndarray
    dataset_size: int | None = None
    _dataset_fingerprint: str | None = dataclasses.field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"CoinFlip(num_canaries={self.num_canaries}, "
            f"n_in={len(self.in_indices)}, n_out={len(self.out_indices)})"
        )

    def _matches_dataset(self, dataset: Any) -> bool:
        """Return whether a dataset matches the recorded source guard."""
        return self._dataset_fingerprint is None or self._dataset_fingerprint == (
            _dataset_fingerprint(dataset, self.canary_indices)
        )

    def train_indices(self, dataset_size: int | None = None) -> list[int]:
        """Dataset indices to use for training.

        Returns all indices in ``range(dataset_size)`` except the excluded
        canaries (coin = tails).

        Args:
            dataset_size: Total number of examples in the full dataset. Defaults
                to the size recorded when this partition was created.

        Returns:
            Sorted list of training indices.

        Raises:
            ConfigurationError: If an explicit size conflicts with the size
                recorded by this partition, or neither is available.
        """
        if (
            dataset_size is not None
            and self.dataset_size is not None
            and dataset_size != self.dataset_size
        ):
            raise ConfigurationError(
                *(
                    f"dataset_size ({dataset_size}) does not match the dataset size "
                    f"({self.dataset_size}) recorded by this CoinFlip",
                )
            )
        size = self.dataset_size if dataset_size is None else dataset_size
        if size is None:
            raise ConfigurationError(
                *("dataset_size must be provided when not recorded on CoinFlip",)
            )
        excluded = set(self.out_indices.tolist())
        return [i for i in range(size) if i not in excluded]

    def train_subset(self, dataset: Any) -> Subset:
        """Return a ``Subset`` containing all training examples.

        Includes all non-canary examples plus included canaries (coin = heads).
        Excludes held-out canaries (coin = tails).

        Args:
            dataset: The full dataset.

        Returns:
            ``torch.utils.data.Subset`` over training indices.

        Raises:
            ConfigurationError: If the dataset size differs from the size
                recorded by this partition.
        """
        return Subset(dataset, self.train_indices(len(dataset)))

    def split_scores(self, scores: CanaryScores) -> tuple[np.ndarray, np.ndarray]:
        """Split per-canary scores into in-group and out-group.

        Joins ``scores`` to the partition by canary identifier, so any
        scoring order is accepted; identifiers that are not canaries of
        this partition, appear twice, or are missing raise instead of
        silently pairing scores with the wrong coin-flip labels.

        Args:
            scores: Membership scores carrying canary identifiers, as
                returned by the scoring functions in verified mode (or
                constructed explicitly to attest identifiers).

        Returns:
            ``(in_scores, out_scores)`` tuple, in ``canary_indices``
            order within each group.

        Raises:
            TypeError: If ``scores`` is a bare array without identifiers.
            ValueError: If the identifiers do not join one-to-one onto
                this partition's canaries.
        """
        if not isinstance(scores, CanaryScores):
            raise InputTypeError(
                *(
                    f"split_scores() requires CanaryScores, got "
                    f"{type(scores).__name__}. Bare score arrays cannot prove "
                    "score-to-membership pairing (a shuffled scoring loader "
                    "silently misaligns scores with coin flips). Score with "
                    "loss_scores(..., coin_flip=cf, dataset=dataset) / "
                    "gradient_scores(..., coin_flip=cf, dataset=dataset), or "
                    "attest identifiers explicitly with canary_scores(values, "
                    "canary_indices=...).",
                )
            )
        canonical = self._join_scores(scores)
        return canonical[self._in_mask], canonical[~self._in_mask]

    def _join_scores(self, scores: CanaryScores) -> np.ndarray:
        """Realign ``scores`` to ``canary_indices`` order by identifier."""
        want = self.canary_indices
        have = scores.canary_indices

        if np.unique(want).size != want.size:
            raise ConfigurationError(
                *(
                    "canary_indices of this partition contain duplicates; "
                    "the score join is ambiguous",
                )
            )
        if want.size == 0:
            if have.size:
                raise ConfigurationError(
                    *(f"got {have.size} scores for a partition with no canaries",)
                )
            return np.empty(0, dtype=float)

        sorter = np.argsort(want, kind="stable")
        sorted_want = want[sorter]
        pos = np.searchsorted(sorted_want, have)
        pos = np.minimum(pos, want.size - 1)
        matched = sorted_want[pos] == have
        if not np.all(matched):
            unexpected = have[~matched]
            raise ConfigurationError(
                *(
                    f"{unexpected.size} score identifier(s) are not canaries of "
                    f"this partition (e.g. {unexpected[:5].tolist()}); the "
                    "scores were computed for different examples or a different "
                    "CoinFlip.",
                )
            )

        slots = sorter[pos]
        filled = np.bincount(slots, minlength=want.size)
        duplicated = want[filled > 1]
        missing = want[filled == 0]
        if duplicated.size or missing.size:
            raise ConfigurationError(
                *(
                    f"score identifiers do not cover the partition's canaries "
                    f"one-to-one: {duplicated.size} duplicated "
                    f"(e.g. {duplicated[:5].tolist()}), {missing.size} missing "
                    f"(e.g. {missing[:5].tolist()}). Check for drop_last=True, "
                    "distributed samplers, or scoring a wrong subset.",
                )
            )

        canonical = np.empty(want.size, dtype=float)
        canonical[slots] = scores.scores
        return canonical


def _candidate_pool(candidate_indices: Any, dataset_size: int) -> np.ndarray:
    """Validate and canonicalize eligible canary indices."""
    pool = np.asarray(candidate_indices)
    if pool.ndim != 1:
        raise ConfigurationError(
            *(f"candidate_indices must be 1-D, got shape {pool.shape}",)
        )
    if pool.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(pool.dtype, np.integer):
        raise ConfigurationError(
            *(f"candidate_indices must contain integers, got dtype {pool.dtype}",)
        )

    unique = np.unique(pool)
    if unique.size != pool.size:
        raise ConfigurationError(*("candidate_indices must be unique",))

    invalid = unique[(unique < 0) | (unique >= dataset_size)]
    if invalid.size:
        raise ConfigurationError(
            *(
                "candidate_indices must be within range(len(dataset)); "
                f"got {invalid[:5].tolist()}",
            )
        )
    return unique


def coin_flip(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
    candidate_indices: Any | None = None,
) -> CoinFlip:
    """Create a coin-flip partition for canary-based auditing.

    Selects ``num_canaries`` examples uniformly without replacement and
    flips a fair coin for each. Selection uses the whole dataset unless
    ``candidate_indices`` supplies an eligible pool. Fix that pool before
    target training, independently of the target run's coins and outputs.

    Args:
        dataset: A dataset with a deterministic ``_fingerprint``, or a
            supported in-memory dataset whose canary rows can be encoded
            without invoking transforms.
        num_canaries: Number of canary examples to designate.
        key: RNG key for reproducible canary selection and coin flips.
        candidate_indices: Optional 1-D collection of unique, in-range
            integer indices eligible to become canaries. Passing exactly
            ``num_canaries`` indices designates them all.

    Returns:
        A :class:`CoinFlip` with the canary partition.

    Raises:
        ValueError: If the candidate pool is invalid or too small, or the
            dataset cannot be deterministically attested.

    Example::

        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        train_data = dataset.select(cf.train_indices())
    """
    dataset_size = len(dataset)
    if num_canaries < 0:
        raise ConfigurationError(
            *(f"num_canaries must be non-negative, got {num_canaries}",)
        )
    if candidate_indices is None:
        population: int | np.ndarray = dataset_size
        pool_size = dataset_size
        pool_name = "dataset size"
    else:
        population = _candidate_pool(candidate_indices, dataset_size)
        pool_size = population.size
        pool_name = "candidate pool size"
    if num_canaries > pool_size:
        raise ConfigurationError(
            *(f"num_canaries ({num_canaries}) exceeds {pool_name} ({pool_size})",)
        )

    rng = np.random.default_rng(fold_in(key, _CANARY_SELECTION_DOMAIN).seed)
    canary_indices = rng.choice(population, size=num_canaries, replace=False)
    coin_rng = np.random.default_rng(fold_in(key, _COIN_FLIP_DOMAIN).seed)
    in_mask = np.asarray(
        coin_rng.random(num_canaries) < _INCLUSION_PROBABILITY, dtype=bool
    )

    return CoinFlip(
        num_canaries=num_canaries,
        canary_indices=canary_indices,
        _in_mask=in_mask,
        in_indices=canary_indices[in_mask],
        out_indices=canary_indices[~in_mask],
        dataset_size=dataset_size,
        _dataset_fingerprint=_dataset_fingerprint(dataset, canary_indices),
    )
