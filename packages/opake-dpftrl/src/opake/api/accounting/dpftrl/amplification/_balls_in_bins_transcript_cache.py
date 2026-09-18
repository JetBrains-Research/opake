"""Rust-backed projected draws for Balls-in-Bins calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from opake.api.accounting.core import _native
from opake.api.accounting.core._native_cache import native_cache

if TYPE_CHECKING:
    from collections.abc import Callable

_cache = native_cache(
    name="balls_in_bins_transcripts",
    max_bytes_env="OPAKE_BNB_TRANSCRIPT_CACHE_MAX_BYTES",
    default_max_bytes=4 * 1024 * 1024 * 1024,
    max_entries=2,
    nbytes_estimate=lambda key: 2 * key[2] * key[1] * 8 + key[2] * 8 + len(key[0]) * 8,
    destructor=_native.drop_bnb_transcript_corpus,
)

_T = TypeVar("_T")


def with_handle(
    gram: tuple[float, ...],
    num_bins: int,
    num_mc_samples: int,
    mc_seed: int,
    use_handle: Callable[[int], _T],
    *,
    native=_native,
) -> _T | None:
    """Use one projected-draw corpus, or return ``None`` when it cannot fit."""
    key = (gram, num_bins, num_mc_samples, mc_seed)

    def factory() -> int:
        return native.register_bnb_transcript_corpus(
            list(gram), num_bins, num_mc_samples, mc_seed
        )

    try:
        return _cache.with_handle(key, factory, use_handle)
    except ValueError:
        return None
