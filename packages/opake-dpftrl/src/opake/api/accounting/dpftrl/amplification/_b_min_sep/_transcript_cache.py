"""Rust-backed transcript corpora for b-min-sep MC (compact memory, reuse across σ)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from opake.api.accounting.core import _native
from opake.api.accounting.core._native_cache import native_cache

if TYPE_CHECKING:
    from collections.abc import Callable

_cache = native_cache(
    name="b_min_sep_transcripts",
    max_bytes_env="OPAKE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES",
    default_max_bytes=4 * 1024 * 1024 * 1024,
    max_entries=4,
    nbytes_estimate=lambda k: 3 * k[3] * k[1] * 8,  # 3 * num_mc_samples * n_steps * 8
    destructor=_native.drop_b_min_sep_transcript_corpus,
)

_T = TypeVar("_T")


def _key_and_factory(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> tuple[tuple, Callable[[], int]]:
    key = (strategy_coef, n_steps, p, num_mc_samples, mc_seed)

    def factory() -> int:
        return _native.register_b_min_sep_transcript_corpus(
            list(strategy_coef),
            n_steps,
            p,
            num_mc_samples,
            mc_seed,
        )

    return key, factory


def get_handle_or_none(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> int | None:
    """Return a cached Rust corpus handle (creating it once if absent).

    .. warning::

        The returned handle is racy under concurrent
        :func:`opake.api.accounting.core._native_cache._clear_all_native_caches`.
        Production call sites that run alongside
        :func:`opake.accounting.calibration.calibrate` should use
        :func:`with_handle` instead, which holds the cache lock around
        both the lookup and the use of the handle.
    """
    key, factory = _key_and_factory(strategy_coef, n_steps, p, num_mc_samples, mc_seed)
    try:
        return _cache.get_or_create(key, factory)
    except ValueError:
        return None


def with_handle(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
    use_handle: Callable[[int], _T],
) -> _T | None:
    """Atomically resolve a Rust corpus handle and run ``use_handle(handle)``.

    Returns ``use_handle(handle)`` when the cache is enabled and the
    requested corpus fits in the byte budget; returns ``None`` when the
    cache cannot serve a handle (caller must fall back to a one-shot MC
    that doesn't take a handle). Holds the cache lock for the entire
    use, so a concurrent
    :func:`opake.api.accounting.core._native_cache._clear_all_native_caches`
    (e.g. from :func:`opake.accounting.calibration.calibrate`'s
    ``finally`` block) cannot destruct the handle mid-use.
    """
    key, factory = _key_and_factory(strategy_coef, n_steps, p, num_mc_samples, mc_seed)
    try:
        return _cache.with_handle(key, factory, use_handle)
    except ValueError:
        return None
