"""Balls-in-Bins calibration reuses sigma-independent Monte Carlo work."""

from __future__ import annotations

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core import _native
from opaque.api.accounting.core._native_cache import _clear_all_native_caches
from opaque.api.accounting.dpftrl.amplification import (
    _balls_in_bins_transcript_cache as transcript_cache,
)
from opaque.api.accounting.dpftrl.amplification._balls_in_bins import BallsInBins
from opaque.dpftrl.noise import lambda_cgd_strategy

_CONFIG = {
    "discretization": 1e-3,
    "mc_resolution": 1e-2,
    "mc_failure_probability": 1e-2,
}


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    monkeypatch.delenv("OPAQUE_BNB_TRANSCRIPT_CACHE_MAX_BYTES", raising=False)
    monkeypatch.setattr(
        transcript_cache._cache, "_default_max_bytes", 4 * 1024 * 1024 * 1024
    )
    _clear_all_native_caches()
    BallsInBins.pld.cache_clear()
    yield
    _clear_all_native_caches()
    BallsInBins.pld.cache_clear()


def _process(noise_multiplier: float) -> BallsInBins:
    return ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(
            noise_multiplier,
            lambda_cgd_strategy(lambda_=0.7, normalized=True),
        ),
        num_bins=8,
        n_steps=32,
    )


def test_distinct_noise_probes_prepare_one_transcript_corpus(monkeypatch):
    """Repeated sigma probes pay RNG and Cholesky preparation only once."""
    calls = {"prepare": 0, "reuse": 0, "one_shot": 0}
    original_prepare = _native.register_bnb_transcript_corpus
    original_reuse = _native.bnb_pld_from_transcript_handle
    original_one_shot = _native.bnb_mc_pld

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def reuse(*args, **kwargs):
        calls["reuse"] += 1
        return original_reuse(*args, **kwargs)

    def one_shot(*args, **kwargs):
        calls["one_shot"] += 1
        return original_one_shot(*args, **kwargs)

    monkeypatch.setattr(_native, "register_bnb_transcript_corpus", prepare)
    monkeypatch.setattr(_native, "bnb_pld_from_transcript_handle", reuse)
    monkeypatch.setattr(_native, "bnb_mc_pld", one_shot)

    low = _process(1.0).pld(**_CONFIG).epsilon_at(2e-2)
    high = _process(2.0).pld(**_CONFIG).epsilon_at(2e-2)

    assert low > high
    assert calls == {"prepare": 1, "reuse": 2, "one_shot": 0}


def test_oversized_corpus_preserves_one_shot_fallback(monkeypatch):
    monkeypatch.setenv("OPAQUE_BNB_TRANSCRIPT_CACHE_MAX_BYTES", "0")
    calls = 0
    original_one_shot = _native.bnb_mc_pld

    def one_shot(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_one_shot(*args, **kwargs)

    monkeypatch.setattr(_native, "bnb_mc_pld", one_shot)

    assert _process(1.0).pld(**_CONFIG).epsilon_at(2e-2) > 0
    assert calls == 1
