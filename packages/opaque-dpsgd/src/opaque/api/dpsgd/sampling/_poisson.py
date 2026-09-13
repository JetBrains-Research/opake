"""Poisson subsampling for DP-SGD training.

Poisson subsampling: each example is independently included with
probability ``sample_rate``.  Optional ``truncated_batch_size`` caps the
realised batch for stable sizes and memory; that is **weaker** for privacy than
plain Poisson at the same ``sample_rate``—pair with
:func:`opaque.dpsgd.accounting.poisson` passing both
``truncated_batch_size`` and the exact pre-sampling ``dataset_size``.
Truncation selects a uniformly random subset, as required by Theorem 3.1 of
https://arxiv.org/abs/2508.15089.

For distributed training, shard the dataset **before** creating the
sampler using ``local_shard()`` and derive a per-rank key with
``fold_in(key, rank)``.

Resume contract: register the sampler with
:mod:`opaque.serialization` (done at module-import time below) so
``state_dict(sampler)`` / ``from_state_dict(template, sd)`` round-trip
the cursor + RNG.  ``__iter__`` continues from ``self._consumed`` and
``__len__`` reports the remaining batches — see
``tests/sampling/test_poisson_state_dict.py``.
"""

from collections.abc import Iterator, Mapping, Sized
from typing import Any, Self

import numpy as np
from torch.utils.data import Sampler

from opaque.exceptions import ConfigurationError, InputTypeError
from opaque.random import fold_in
from opaque.random.types import RngKey

from ._helpers import _plain_poisson_step_indices

POISSON_STREAM_FOLD = "opaque.dpsgd.poisson"


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification (DP-SGD).

    Each example is independently included with probability
    ``sample_rate``.  When ``truncated_batch_size`` is set, batches are
    capped at that size (uniform without replacement from the Poisson
    sample).

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``::

        from opaque.distributed import local_shard

        shard = local_shard(dataset, rank=rank, world_size=world_size)
        sampler = PoissonSampler(
            shard, sample_rate=0.01, key=fold_in(key(42), rank)
        )

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Probability of including each example ``∈ (0, 1]``.
            At ``1.0`` every example is in every batch — no amplification;
            account the step with :func:`opaque.dpsgd.accounting.gaussian`.
        n_steps: Number of batches to yield. ``None`` yields indefinitely.
        truncated_batch_size: Optional cap on per-step batch size (truncated
            Poisson; use matching accounting—privacy is weaker than uncapped
            Poisson at the same ``sample_rate``).
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = PoissonSampler(
            dataset, sample_rate=0.01, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property).
        - Expected batch size: ``len(data_source) * sample_rate`` (uncapped).
        - Use with ``DataLoader``'s ``batch_sampler`` parameter.
        - Resume via :func:`opaque.serialization.state_dict` /
          :func:`opaque.serialization.from_state_dict`: state captures the
          domain-separated stream key and consumed-step count.
    """

    def __init__(
        self,
        data_source: Sized,
        sample_rate: float,
        n_steps: int | None = None,
        truncated_batch_size: int | None = None,
        *,
        key: RngKey,
    ):
        self._initialize(
            data_source,
            sample_rate,
            n_steps,
            truncated_batch_size,
            stream_key=fold_in(key, POISSON_STREAM_FOLD),
        )

    @classmethod
    def _from_stream_key(
        cls,
        data_source: Sized,
        sample_rate: float,
        n_steps: int | None = None,
        truncated_batch_size: int | None = None,
        *,
        stream_key: RngKey,
    ) -> Self:
        """Construct from an already domain-separated stream key."""
        sampler = object.__new__(cls)
        sampler._initialize(
            data_source,
            sample_rate,
            n_steps,
            truncated_batch_size,
            stream_key=stream_key,
        )
        return sampler

    def _initialize(
        self,
        data_source: Sized,
        sample_rate: float,
        n_steps: int | None,
        truncated_batch_size: int | None,
        *,
        stream_key: RngKey,
    ) -> None:
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ConfigurationError(
                *(f"sample_rate must be in (0, 1], got {sample_rate}",)
            )
        if n_steps is not None and n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1 or None, got {n_steps}",))
        if truncated_batch_size is not None:
            if type(truncated_batch_size) is not int:
                raise InputTypeError(
                    *(
                        "truncated_batch_size must be an int, got "
                        f"{type(truncated_batch_size).__name__}",
                    )
                )
            if truncated_batch_size < 1:
                raise ConfigurationError(
                    *(f"truncated_batch_size must be >= 1, got {truncated_batch_size}",)
                )

        self.data_source: Sized = data_source
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.truncated_batch_size = truncated_batch_size

        self._num_samples = len(data_source)
        self._stream_key = stream_key
        self.generator = np.random.default_rng(stream_key.seed)
        self._consumed = 0

    def _sample_step(self) -> list[int]:
        return _plain_poisson_step_indices(
            self.generator,
            self._num_samples,
            self.sample_rate,
            self.truncated_batch_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each iteration samples the entire dataset using Poisson subsampling,
        optionally truncated.  ``self._consumed`` increments after each
        yielded batch so state-dict snapshots reflect the live position.
        """
        if self.n_steps is None:
            while True:
                batch = self._sample_step()
                self._consumed += 1
                yield batch
        else:
            for _ in range(self._consumed, self.n_steps):
                batch = self._sample_step()
                self._consumed += 1
                yield batch

    def __len__(self) -> int:
        """Return the number of batches the iterator will yield.

        After a partial run, this is ``n_steps - consumed`` so
        ``len(DataLoader(batch_sampler=sampler))`` reflects the batches
        actually remaining (not the original total).  ``__iter__``
        starts at ``self._consumed``; ``__len__`` matches.

        Raises:
            TypeError: If n_steps is None (infinite iteration).
        """
        if self.n_steps is None:
            raise InputTypeError(*("len() of unsized object (n_steps=None)",))
        return self.n_steps - self._consumed

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate (before truncation)."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling (before truncation)."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)


def _state_dict_poisson(s: PoissonSampler) -> dict[str, Any]:
    """Serialise ``PoissonSampler`` state.

    Saves the domain-separated NumPy stream key, ``num_samples`` (so load
    validates the template dataset length), and a ``consumed`` counter (so the
    loader can replay the generator to the resume position). The dataset itself
    is not serialised; it comes from the ``template`` argument on load.
    """
    return {
        "key_seed": int(s._stream_key.seed),
        "key_impl": str(s._stream_key.impl),
        "consumed": int(s._consumed),
        "num_samples": int(s._num_samples),
        "sample_rate": float(s.sample_rate),
        "n_steps": s.n_steps,
        "truncated_batch_size": s.truncated_batch_size,
    }


def _from_state_dict_poisson(
    template: PoissonSampler, sd: Mapping[str, Any]
) -> PoissonSampler:
    """Rebuild ``PoissonSampler`` at the saved cursor.

    The dataset and ``n_steps`` come from ``template`` so a caller may extend
    or shorten the run. The domain-separated stream key, ``sample_rate``, and
    ``truncated_batch_size`` come from ``sd`` and must reproduce the saved
    sampling math. Replay advances NumPy by ``consumed`` discarded
    ``_sample_step`` calls so the next batch matches a continuous run.

    Raises ``ConfigurationError`` if the template dataset length differs from
    the snapshot because per-step Poisson sampling covers the full dataset.
    """
    saved_n = int(sd["num_samples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ConfigurationError(
            *(
                f"PoissonSampler.from_state_dict: template dataset length "
                f"{template_n} does not match snapshot num_samples={saved_n}.  "
                "Restoring with a differently-sized dataset would silently emit "
                "a different Poisson stream.",
            )
        )
    sampler = PoissonSampler._from_stream_key(
        template.data_source,
        sample_rate=float(sd["sample_rate"]),
        # Use ``template.n_steps`` so callers may extend or shorten the run on
        # restore; the saved cursor fixes the resume position.
        n_steps=template.n_steps,
        truncated_batch_size=sd.get("truncated_batch_size"),
        stream_key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    consumed = int(sd["consumed"])
    for _ in range(consumed):
        sampler._sample_step()
    sampler._consumed = consumed
    return sampler


def _register_poisson_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(PoissonSampler, _state_dict_poisson, _from_state_dict_poisson)


_register_poisson_sampler_serializer()
