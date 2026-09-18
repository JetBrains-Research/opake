# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Block allocation structure adapted in part from the ICML 2026 reference
# implementation for "Efficient privacy loss accounting for subsampling and
# random allocation" (Apache-2.0), then reworked for Opake's k-out-of-t API.
# See ../../../../../NOTICE in this package for the full attribution.
"""K-out-of-t allocation sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from torch.utils.data import Sampler

from opake.exceptions import ConfigurationError
from opake.random import fold_in
from opake.random.types import RngKey

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sized


_Allocation = Literal["block", "total"]
K_OUT_OF_T_STREAM_FOLD = "opake.dpsgd.k_out_of_t"


class KOutOfTSampler(Sampler):
    """Allocate every example to ``k`` of ``t`` training steps.

    ``allocation="block"`` partitions the horizon into ``k`` contiguous,
    nearly equal blocks. Every example is assigned to one batch in each block,
    independently redrawing the assignment at block boundaries.

    ``allocation="total"`` chooses exactly ``k`` distinct steps uniformly from
    the complete ``t``-step horizon for every example. Its current accountant
    uses the block scheme as a conservative upper bound.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        k: int,
        t: int,
        allocation: _Allocation,
        key: RngKey,
    ):
        self._initialize(
            data_source,
            k=k,
            t=t,
            allocation=allocation,
            stream_key=fold_in(key, K_OUT_OF_T_STREAM_FOLD, allocation),
        )

    @classmethod
    def _from_stream_key(
        cls,
        data_source: Sized,
        *,
        k: int,
        t: int,
        allocation: _Allocation,
        stream_key: RngKey,
    ) -> Self:
        """Construct from an already domain-separated stream key."""
        sampler = object.__new__(cls)
        sampler._initialize(
            data_source,
            k=k,
            t=t,
            allocation=allocation,
            stream_key=stream_key,
        )
        return sampler

    def _initialize(
        self,
        data_source: Sized,
        *,
        k: int,
        t: int,
        allocation: _Allocation,
        stream_key: RngKey,
    ) -> None:
        super().__init__()
        if len(data_source) == 0:
            raise ConfigurationError(*("data_source must not be empty",))
        if t < 1:
            raise ConfigurationError(*(f"t must be >= 1, got {t}",))
        if not 1 <= k <= t:
            raise ConfigurationError(*(f"k must be in [1, t={t}], got {k}",))
        if allocation not in ("block", "total"):
            raise ConfigurationError(
                *(f"allocation must be 'block' or 'total', got {allocation!r}",)
            )

        self.data_source: Sized = data_source
        self.k = int(k)
        self.t = int(t)
        self.allocation = allocation
        self._num_samples = len(data_source)
        self._stream_key = stream_key
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """Number of batches yielded so far (resume cursor)."""
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        """Horizon-average expected number of examples per batch."""
        return self._num_samples * self.k / self.t

    @property
    def block_sizes(self) -> tuple[int, ...] | None:
        """Contiguous block sizes, or ``None`` for total allocation."""
        if self.allocation == "total":
            return None
        floor = self.t // self.k
        num_ceil = self.t - floor * self.k
        return (floor,) * (self.k - num_ceil) + (floor + 1,) * num_ceil

    def _block_bins(self, block: int, size: int) -> list[list[int]]:
        """Return the independently drawn partition for one allocation block."""
        rng = np.random.default_rng(fold_in(self._stream_key, block).seed)
        assignment = rng.integers(0, size, size=self._num_samples)
        bins: list[list[int]] = [[] for _ in range(size)]
        for idx, bin_index in enumerate(assignment):
            bins[bin_index].append(idx)
        return bins

    def _total_batches(self) -> Iterator[list[int]]:
        """Yield the uniform global ``k``-out-of-``t`` allocation."""
        rng = np.random.default_rng(self._stream_key.seed)
        remaining = np.full(self._num_samples, self.k, dtype=np.int64)
        for step in range(self.t):
            probabilities = remaining / (self.t - step)
            mask = rng.random(self._num_samples) < probabilities
            yield np.flatnonzero(mask).tolist()
            remaining -= mask

    def __iter__(self) -> Iterator[list[int]]:
        if self.allocation == "block":
            floor = self.t // self.k
            num_ceil = self.t - floor * self.k
            num_floor = self.k - num_ceil
            floor_span = num_floor * floor
            bins: list[list[int]] | None = None
            current_block: int | None = None
            for step in range(self._consumed, self.t):
                if step < floor_span:
                    block, slot = divmod(step, floor)
                    size = floor
                else:
                    block_offset, slot = divmod(step - floor_span, floor + 1)
                    block = num_floor + block_offset
                    size = floor + 1
                if bins is None or block != current_block:
                    bins = self._block_bins(block, size)
                    current_block = block
                self._consumed = step + 1
                yield bins[slot]
            return

        for step, batch in enumerate(self._total_batches()):
            if step < self._consumed:
                continue
            self._consumed = step + 1
            yield batch

    def __len__(self) -> int:
        return self.t - self._consumed


def _state_dict_k_out_of_t(sampler: KOutOfTSampler) -> dict[str, Any]:
    return {
        "key_seed": int(sampler._stream_key.seed),
        "key_impl": str(sampler._stream_key.impl),
        "consumed": sampler.consumed,
        "num_samples": sampler._num_samples,
        "k": sampler.k,
        "t": sampler.t,
        "allocation": sampler.allocation,
    }


def _from_state_dict_k_out_of_t(
    template: KOutOfTSampler,
    state: Mapping[str, Any],
) -> KOutOfTSampler:
    """Restore the saved stream on an unchanged allocation schedule."""
    if len(template.data_source) != int(state["num_samples"]):
        raise ConfigurationError(
            *(
                "KOutOfTSampler.from_state_dict: template dataset length "
                f"{len(template.data_source)} does not match snapshot "
                f"num_samples={state['num_samples']}",
            )
        )
    for name in ("k", "t", "allocation"):
        saved = state[name]
        expected = getattr(template, name)
        if type(saved) is not type(expected) or saved != expected:
            message = (
                f"KOutOfTSampler.from_state_dict: {name} mismatch: "
                f"snapshot={saved!r}, template={expected!r}"
            )
            raise ConfigurationError(message)
    restored = KOutOfTSampler._from_stream_key(
        template.data_source,
        k=template.k,
        t=template.t,
        allocation=template.allocation,
        stream_key=RngKey(seed=int(state["key_seed"]), impl=str(state["key_impl"])),
    )
    restored._consumed = int(state["consumed"])
    return restored


def _register_k_out_of_t_sampler_serializer() -> None:
    from opake.serialization import register_serializer

    register_serializer(
        KOutOfTSampler,
        _state_dict_k_out_of_t,
        _from_state_dict_k_out_of_t,
    )


_register_k_out_of_t_sampler_serializer()

__all__ = ["KOutOfTSampler"]
