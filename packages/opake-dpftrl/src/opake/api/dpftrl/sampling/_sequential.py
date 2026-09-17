"""Sequential batch sampler for fixed-order training.

This sampler iterates through a dataset sequentially in fixed-size
batches, dropping the last incomplete batch.  It is used by the BLT
(Buffered Linear Toeplitz) mechanism, which requires a deterministic
batch order with fixed ``min_sep`` between participations.

Unlike Poisson or BnB sampling there is no randomness — the dataset
should be pre-shuffled once before constructing the sampler.
"""

from collections.abc import Iterator, Mapping, Sized
from typing import Any

from torch.utils.data import Sampler

from opake.exceptions import ConfigurationError


class SequentialBatchSampler(Sampler):
    """Deterministic fixed-size sequential batching.

    Yields contiguous, non-overlapping chunks of ``batch_size`` indices
    ``[0, ..., B-1], [B, ..., 2B-1], ...``.  The last chunk is dropped
    when smaller than ``batch_size``.

    With ``n_steps`` set, the fixed batch order cycles: pass 2 repeats
    the identical batches in the identical order, so across the run
    every example participates exactly ``n_steps // num_batches`` times
    with exactly ``num_batches`` steps between participations — the
    ``min_sep`` / ``max_participations`` contract the BLT accounting
    assumes, enforced by one sampler over the whole stream.

    This sampler has no RNG key — it is fully deterministic.
    Call ``dataset.shuffle(seed=...)`` once beforehand to randomise
    which examples land in which batch.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        batch_size: Exact number of examples per batch (must be ≥ 1).
        n_steps: Total number of batches to yield across the run.  Must
            be a positive multiple of the per-pass batch count
            (``len(data_source) // batch_size``), so participation
            counts stay uniform.  ``None`` yields a single pass.

    Example:
        >>> sampler = SequentialBatchSampler(dataset, batch_size=256)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        n_steps: int | None = None,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ConfigurationError(*("data_source must not be empty",))
        if batch_size < 1:
            raise ConfigurationError(*(f"batch_size must be >= 1, got {batch_size}",))

        self.data_source: Sized = data_source
        self._num_samples = len(data_source)
        self._batch_size = batch_size
        self._num_batches = self._num_samples // self._batch_size

        if n_steps is not None:
            if n_steps < 1:
                raise ConfigurationError(
                    *(f"n_steps must be >= 1 or None, got {n_steps}",)
                )
            if self._num_batches == 0:
                raise ConfigurationError(
                    *(
                        f"batch_size ({batch_size}) exceeds dataset size "
                        f"({self._num_samples}); no complete batch to cycle.",
                    )
                )
            if n_steps % self._num_batches != 0:
                raise ConfigurationError(
                    *(
                        f"n_steps ({n_steps}) must be a positive multiple of the "
                        f"per-pass batch count ({self._num_batches}); a partial "
                        "final pass would make participation counts non-uniform.",
                    )
                )
        self.n_steps = n_steps
        self._consumed = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yield fixed-size batches of contiguous indices.

        The last incomplete batch (if any) is dropped.  With ``n_steps``
        set, the fixed order cycles until ``n_steps`` batches have been
        yielded.  Iteration resumes from ``self._consumed`` so a loaded
        sampler continues at its saved cursor.
        """
        total = self._num_batches if self.n_steps is None else self.n_steps
        for i in range(self._consumed, total):
            start = (i % self._num_batches) * self._batch_size
            # Increment before yield so a snapshot taken mid-iter
            # reports the count of batches actually emitted so far.
            self._consumed = i + 1
            yield list(range(start, start + self._batch_size))

    def __len__(self) -> int:
        """Declared batches remaining.

        After a partial run, reflects what ``__iter__`` will yield —
        so ``len(DataLoader(...))`` matches the resumed iteration count.
        """
        total = self._num_batches if self.n_steps is None else self.n_steps
        return total - self._consumed

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Batch size (exact, not statistical)."""
        return float(self._batch_size)


def _state_dict_sequential(s: SequentialBatchSampler) -> dict[str, Any]:
    """Serialise ``SequentialBatchSampler`` state.

    Iteration is fully deterministic with no RNG and no Markov state,
    so the cursor alone fixes the resume point.  ``num_samples`` is
    persisted so the loader can validate the template dataset length —
    a mismatched length would change ``_num_batches`` and thereby what
    indices each cursor position maps to.
    """
    return {
        "consumed": int(s._consumed),
        "num_samples": int(s._num_samples),
        "batch_size": int(s._batch_size),
        "n_steps": s.n_steps,
    }


def _from_state_dict_sequential(
    template: SequentialBatchSampler, sd: Mapping[str, Any]
) -> SequentialBatchSampler:
    """Rebuild ``SequentialBatchSampler`` at the saved cursor.

    ``n_steps`` comes from the template — the caller may extend or
    shorten the run on resume; the cursor fixes the resume position
    within the cycling order.

    Raises ``ValueError`` if the template dataset length differs from
    the snapshot — ``_num_batches = num_samples // batch_size`` drives
    the index ranges each yield emits.
    """
    saved_n = int(sd["num_samples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ConfigurationError(
            *(
                f"SequentialBatchSampler.from_state_dict: template dataset "
                f"length {template_n} does not match snapshot "
                f"num_samples={saved_n}.  Restoring with a differently-sized "
                "dataset would silently expose / skip different indices after "
                "resume.",
            )
        )
    sampler = SequentialBatchSampler(
        template.data_source,
        batch_size=int(sd["batch_size"]),
        n_steps=template.n_steps,
    )
    sampler._consumed = int(sd["consumed"])
    return sampler


def _register_sequential_batch_sampler_serializer() -> None:
    from opake.serialization import register_serializer

    register_serializer(
        SequentialBatchSampler,
        _state_dict_sequential,
        _from_state_dict_sequential,
    )


_register_sequential_batch_sampler_serializer()
