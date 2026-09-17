"""Warm-start b-min-sep subsampling for BandMF (Dong & Ganesh, arXiv:2602.09338).

Algorithm 2 assigns each example an initial state ``s_j`` so the expected batch
size is approximately constant across iterations, then Poisson-samples from
eligible examples (those not in any of the previous ``b-1`` batches).

The per-iteration inclusion probability ``p`` matches the paper (not the
per-example rate ``p_0`` used in cyclic Poisson accounting). For target
``p_0 = E[|B|]/|D|``, use ``p = p_0 / (1 - p_0 * (b - 1))`` when ``b > 1``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from torch.utils.data import Sampler

from opake.exceptions import ConfigurationError
from opake.random import fold_in
from opake.random.types import RngKey

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sized

B_MIN_SEP_STREAM_FOLD = "opake.dpftrl.b_min_sep"


class BMinSepSampler(Sampler):
    """Warm-start b-min-sep sampler for BandMF privacy amplification.

    Each iteration draws a batch by including each *eligible* example
    independently with probability ``p``. Eligibility excludes any example
    that appeared in one of the previous ``bands - 1`` batches (Algorithm 2).

    Args:
        data_source: Dataset with ``__len__``.
        bands: Minimum-separation parameter ``b`` (same as BandMF bandwidth).
        sampling_prob: Paper's ``p`` in ``(0, 1]`` (per-iteration inclusion
            probability for each eligible example).
        n_steps: Number of batches to yield.
        key: RNG key for reproducibility.

    Note:
        Use ``batch_sampler=...`` in ``DataLoader``. Batch sizes are random
        (Poisson). For the same expected batch size as Poisson with
        per-example rate ``p_0`` and ``b`` bands, set
        ``sampling_prob = p_0 / (1 - p_0 * (bands - 1))`` (``bands==1`` → ``p_0``).
    """

    def __init__(
        self,
        data_source: Sized,
        bands: int,
        sampling_prob: float,
        n_steps: int,
        *,
        key: RngKey,
    ):
        self._initialize(
            data_source,
            bands,
            sampling_prob,
            n_steps,
            stream_key=fold_in(key, B_MIN_SEP_STREAM_FOLD),
        )

    @classmethod
    def _from_stream_key(
        cls,
        data_source: Sized,
        bands: int,
        sampling_prob: float,
        n_steps: int,
        *,
        stream_key: RngKey,
    ) -> Self:
        """Construct from an already domain-separated stream key."""
        sampler = object.__new__(cls)
        sampler._initialize(
            data_source,
            bands,
            sampling_prob,
            n_steps,
            stream_key=stream_key,
        )
        return sampler

    def _initialize(
        self,
        data_source: Sized,
        bands: int,
        sampling_prob: float,
        n_steps: int,
        *,
        stream_key: RngKey,
    ) -> None:
        super().__init__()
        if len(data_source) == 0:
            raise ConfigurationError(*("data_source must not be empty",))
        if bands < 1:
            raise ConfigurationError(*(f"bands must be >= 1, got {bands}",))
        if not 0 < sampling_prob <= 1:
            raise ConfigurationError(
                *(f"sampling_prob must be in (0, 1], got {sampling_prob}",)
            )
        if n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {n_steps}",))

        self.num_examples = len(data_source)
        self.data_source: Sized = data_source
        self.bands = bands
        self.sampling_prob = sampling_prob
        self.n_steps = n_steps
        self._stream_key = stream_key
        self.generator = np.random.default_rng(stream_key.seed)

        # Warm-start initial cooldown (Algorithm 2, lines 1–2).
        self._bars_left = np.zeros(self.num_examples, dtype=np.int32)
        if bands > 1:
            p = sampling_prob
            denom = 1.0 + (bands - 1) * p
            for j in range(self.num_examples):
                if self.generator.random() * denom < 1.0:
                    self._bars_left[j] = 0
                else:
                    # Uniform on {1, …, b-1}
                    self._bars_left[j] = int(self.generator.integers(1, bands))

        self._recent: deque[set[int]] = deque(maxlen=max(0, bands - 1))
        self._consumed = 0

    def _sample_step(self) -> list[int]:
        """Advance the Markov chain by one step and return the next batch.

        Mutates ``self.generator``, ``self._bars_left``, and
        ``self._recent``.  State-dict replay calls this helper to
        fast-forward without yielding.
        """
        exclude: set[int] = set()
        for batch in self._recent:
            exclude.update(batch)

        eligible = [
            idx for idx in range(self.num_examples) if self._bars_left[idx] == 0
        ]
        eligible = [i for i in eligible if i not in exclude]

        n_elig = len(eligible)
        if n_elig == 0:
            batch_indices: list[int] = []
        else:
            sample_size = self.generator.binomial(n=n_elig, p=self.sampling_prob)
            if sample_size > 0:
                pos = self.generator.choice(
                    n_elig, size=sample_size, replace=False, shuffle=False
                )
                batch_indices = [eligible[int(k)] for k in pos]
            else:
                batch_indices = []

        batch_set = set(batch_indices)

        # Advance cooldown: one step closer to eligibility.
        mask = self._bars_left > 0
        self._bars_left[mask] -= 1

        # Selected examples cannot appear in the next (bands - 1) iterations.
        for idx in batch_indices:
            self._bars_left[idx] = max(self._bars_left[idx], self.bands - 1)

        self._recent.append(batch_set)
        return batch_indices

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self._consumed, self.n_steps):
            batch = self._sample_step()
            self._consumed += 1
            yield batch

    def __len__(self) -> int:
        """Batches remaining (``n_steps - consumed``).

        After a partial run, reflects what ``__iter__`` will yield —
        so ``len(DataLoader(...))`` matches the resumed iteration count.
        """
        return self.n_steps - self._consumed

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Approximate expected batch size once the chain has mixed."""
        p0 = self.sampling_prob / (1.0 + self.sampling_prob * max(0, self.bands - 1))
        return self.num_examples * p0

    @property
    def batch_size_variance(self) -> float:
        """Approximate variance (Poisson on eligible pool ~ full size)."""
        p = self.sampling_prob
        n = float(self.num_examples)
        return n * p * (1.0 - p)


def _state_dict_b_min_sep(s: BMinSepSampler) -> dict[str, Any]:
    """Serialise ``BMinSepSampler`` state.

    The Markov state (``_bars_left`` cooldown array and ``_recent`` deque) is
    path-dependent, so the loader reconstructs it from the domain-separated
    stream
    key by replaying ``consumed`` steps. ``num_examples`` is persisted so load
    can validate the template length before reconstructing the warm start.
    """
    return {
        "key_seed": int(s._stream_key.seed),
        "key_impl": str(s._stream_key.impl),
        "consumed": int(s._consumed),
        "num_examples": int(s.num_examples),
        "bands": int(s.bands),
        "sampling_prob": float(s.sampling_prob),
        "n_steps": int(s.n_steps),
    }


def _from_state_dict_b_min_sep(
    template: BMinSepSampler, sd: Mapping[str, Any]
) -> BMinSepSampler:
    """Rebuild ``BMinSepSampler`` at the saved cursor.

    The dataset comes from ``template``. The saved domain-separated stream key
    starts the original Markov chain; ``consumed`` calls to ``_sample_step``
    reconstruct the cooldown array and recent-batch deque.

    Raises ``ConfigurationError`` if the template dataset length differs from
    the snapshot because the warm start and eligibility scan depend on it.
    """
    saved_n = int(sd["num_examples"])
    template_n = len(template.data_source)
    if saved_n != template_n:
        raise ConfigurationError(
            *(
                f"BMinSepSampler.from_state_dict: template dataset length "
                f"{template_n} does not match snapshot num_examples={saved_n}.  "
                "Restoring with a differently-sized dataset would silently "
                "reconstruct a different Markov chain.",
            )
        )
    sampler = BMinSepSampler._from_stream_key(
        template.data_source,
        bands=int(sd["bands"]),
        sampling_prob=float(sd["sampling_prob"]),
        # Use ``template.n_steps`` so callers may extend or shorten the run on
        # restore; the saved cursor and replay fix the resume position.
        n_steps=int(template.n_steps),
        stream_key=RngKey(seed=int(sd["key_seed"]), impl=str(sd["key_impl"])),
    )
    consumed = int(sd["consumed"])
    for _ in range(consumed):
        sampler._sample_step()
    sampler._consumed = consumed
    return sampler


def _register_b_min_sep_sampler_serializer() -> None:
    from opake.serialization import register_serializer

    register_serializer(
        BMinSepSampler, _state_dict_b_min_sep, _from_state_dict_b_min_sep
    )


_register_b_min_sep_sampler_serializer()
