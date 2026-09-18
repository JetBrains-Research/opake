"""Tests for b_min_sep BandMF amplification."""

from dataclasses import asdict

import pytest

import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.dpftrl.amplification._b_min_sep import (
    participation_p_from_per_example_rate,
)
from opake.dpftrl.noise import band_mf_strategy


def test_p_conversion():
    p0 = 0.05
    b = 8
    p = participation_p_from_per_example_rate(p0, b)
    assert p > p0
    assert abs(1.0 / p0 - (1.0 / p + (b - 1))) < 1e-9


def test_p_conversion_bands_one_is_identity():
    assert participation_p_from_per_example_rate(0.07, 1) == 0.07


def test_p_conversion_rejects_infeasible_p0():
    with pytest.raises(ValueError, match="infeasible"):
        participation_p_from_per_example_rate(p0=0.5, bands=4)


def test_sampling_prob_property_matches_helper():
    strategy = band_mf_strategy(bands=4)
    inner = ftrl_acc.mf_gaussian(1.0, strategy)
    p0 = 0.02
    proc = ftrl_acc.b_min_sep(inner, n_steps=40, p0=p0)
    assert proc.sampling_prob == participation_p_from_per_example_rate(p0, 4)


def test_sampling_prob_degenerates_to_p0_for_bands_one():
    strategy = band_mf_strategy(bands=1)
    inner = ftrl_acc.mf_gaussian(1.0, strategy)
    proc = ftrl_acc.b_min_sep(inner, n_steps=40, p0=0.05)
    assert proc.sampling_prob == 0.05


def test_b_min_sep_smoke_pld():
    strategy = band_mf_strategy(bands=4)
    inner = ftrl_acc.mf_gaussian(1.0, strategy)
    proc = ftrl_acc.b_min_sep(
        inner,
        n_steps=40,
        p0=0.02,
    )
    pld = proc.pld(
        seed=123,
        mc_resolution=5e-3,
        mc_failure_probability=1e-2,
    )
    eps = pld.epsilon_at(1e-2)
    assert eps > 0.0
    assert eps < 500.0
    assert pld.mc_failure_probability == pytest.approx(1e-2)
    assert pld.mc_confidence == pytest.approx(0.99)
    assert 0.0 < pld.mc_resolution <= 5e-3
    assert pld.epsilon_at(pld.infinity_mass) == float("inf")
    assert pld.beta_at(0.1) == 0.0
    assert pld.risk_at(0.5) == 0.0


def test_transcript_cache_reuses_same_handle():
    """Repeated cache lookup returns the same Rust corpus handle."""
    from opake.api.accounting.dpftrl.amplification._b_min_sep._transcript_cache import (
        get_handle_or_none,
    )

    h1 = get_handle_or_none((1.0, 0.0), 12, 0.08, 80, 7)
    h2 = get_handle_or_none((1.0, 0.0), 12, 0.08, 80, 7)
    assert h1 is not None
    assert h2 is not None
    assert h1 == h2


def _drain_cache(tc) -> None:
    """Drop any leftover native handles so tests start from a clean slate."""
    tc._cache.clear()


def test_transcript_cache_evicts_lru(monkeypatch):
    """Cache caps entries and drops LRU native handle *before* registering new."""
    from opake.api.accounting.dpftrl.amplification._b_min_sep import (
        _transcript_cache as tc,
    )

    monkeypatch.delenv("OPAKE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES", raising=False)
    monkeypatch.setattr(tc._cache, "_max_entries", 2)
    # Force a healthy byte budget regardless of what the test process
    # started with; ``_max_bytes`` was frozen at module-load time and
    # ``_refresh_max_bytes`` honours the env var we just cleared, so
    # both the env override and a direct attribute clamp are safe.
    monkeypatch.setattr(tc._cache, "_default_max_bytes", 4 * 1024 * 1024 * 1024)
    monkeypatch.setattr(tc._cache, "_max_bytes", 4 * 1024 * 1024 * 1024)
    _drain_cache(tc)

    calls: list[tuple[str, int]] = []
    orig_register = tc._native.register_b_min_sep_transcript_corpus
    orig_destructor = tc._cache._destructor

    def tracking_destructor(handle: int) -> None:
        calls.append(("drop", handle))
        orig_destructor(handle)

    def tracking_register(*args, **kwargs) -> int:
        hid = orig_register(*args, **kwargs)
        calls.append(("register", hid))
        return hid

    monkeypatch.setattr(tc._cache, "_destructor", tracking_destructor)
    monkeypatch.setattr(
        tc._native, "register_b_min_sep_transcript_corpus", tracking_register
    )

    h1 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 1)
    h2 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 2)
    assert h1 is not None
    assert h2 is not None
    assert len(tc._cache) == 2
    assert calls == [("register", h1), ("register", h2)]

    # The third distinct entry must drop the LRU BEFORE registering the new
    # corpus — that ordering is the regression this PR fixes.
    calls.clear()
    h3 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 3)
    assert h3 is not None
    assert len(tc._cache) == 2
    assert calls == [("drop", h1), ("register", h3)], (
        f"eviction must precede registration; got {calls}"
    )

    # Touching h2 promotes it to MRU; next insert evicts h3, not h2.
    assert tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 2) == h2
    calls.clear()
    h4 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 4)
    assert h4 is not None
    assert calls == [("drop", h3), ("register", h4)]
    assert len(tc._cache) == 2

    _drain_cache(tc)


def test_transcript_cache_evicts_for_byte_cap(monkeypatch):
    """Byte budget forces eviction even when entry count is below the cap."""
    from opake.api.accounting.dpftrl.amplification._b_min_sep import (
        _transcript_cache as tc,
    )

    monkeypatch.setattr(tc._cache, "_max_entries", 16)
    # ``_refresh_max_bytes`` re-reads the env var on every cache hit,
    # so set the env var rather than monkeypatching the attribute (which
    # the next refresh would clobber).  3 * 64 * 10 * 8 = 15 360 bytes
    # fits exactly one entry.
    monkeypatch.setenv(
        "OPAKE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES", str(3 * 64 * 10 * 8)
    )
    _drain_cache(tc)

    calls: list[tuple[str, int]] = []
    orig_register = tc._native.register_b_min_sep_transcript_corpus
    orig_destructor = tc._cache._destructor

    def tracking_destructor(handle: int) -> None:
        calls.append(("drop", handle))
        orig_destructor(handle)

    def tracking_register(*args, **kwargs) -> int:
        hid = orig_register(*args, **kwargs)
        calls.append(("register", hid))
        return hid

    monkeypatch.setattr(tc._cache, "_destructor", tracking_destructor)
    monkeypatch.setattr(
        tc._native, "register_b_min_sep_transcript_corpus", tracking_register
    )

    h1 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 10)
    assert h1 is not None
    calls.clear()
    h2 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 11)
    assert h2 is not None
    # Drop-before-register also holds on the byte-cap path.
    assert calls == [("drop", h1), ("register", h2)]
    assert len(tc._cache) == 1

    _drain_cache(tc)


def test_b_min_sep_stricter_than_mf_only():
    """Subsampling should lower ε at fixed σ vs unamplified BandMF PLD.

    Uses a low ``p0`` so subsampling amplification dominates over the
    multi-group composition cost of unamplified MF.
    """
    from opake.api.accounting.core import _native as native
    from opake.api.accounting.core.discretization import get_discretization

    strategy = band_mf_strategy(bands=5)
    # Use a low-noise / low-sample-rate regime where b-min-sep amplification
    # strictly beats unamplified composition.
    inner = ftrl_acc.mf_gaussian(1.0, strategy)
    bms = ftrl_acc.b_min_sep(
        inner,
        n_steps=50,
        p0=0.01,
    )
    cfg = get_discretization()
    pld_mf = native.mf_gaussian_pld(
        1.0,
        strategy.sensitivity(n_steps=50, min_sep=50, max_participations=1),
        cfg.to_native(),
    )
    eps_mf = pld_mf.epsilon_at(1e-2)
    eps_bms = bms.pld(
        seed=1,
        mc_resolution=5e-3,
        mc_failure_probability=1e-2,
    ).epsilon_at(1e-2)
    assert eps_bms < eps_mf


def test_calibration_reports_one_overall_mc_confidence_budget():
    import opake.accounting as acc

    prev = asdict(acc.get_discretization())
    acc.set_discretization(
        mc_resolution=5e-2,
        mc_failure_probability=1e-2,
        seed=3,
    )
    try:

        def process(nm):
            return ftrl_acc.b_min_sep(
                ftrl_acc.mf_gaussian(nm, band_mf_strategy(bands=2)),
                n_steps=8,
                p0=0.02,
            )

        result = acc.calibrate(
            acc.epsilon_budget(0.1, delta=0.1),
            process,
            0.5,
            5.0,
            tolerance=0.5,
            max_iterations=8,
        )
    finally:
        acc.set_discretization(**prev)

    assert result.mc_failure_probability == pytest.approx(1e-2)
    assert result.mc_confidence == pytest.approx(0.99)


def test_composed_mc_pld_uses_one_overall_confidence_and_resolution_budget():
    left = ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(0.8, band_mf_strategy(bands=2)),
        n_steps=8,
        p0=0.02,
    )
    right = ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(1.2, band_mf_strategy(bands=2)),
        n_steps=8,
        p0=0.02,
    )

    pld = (left | right).pld(
        mc_resolution=0.1,
        mc_failure_probability=0.02,
        seed=5,
    )

    assert pld.mc_failure_probability <= 0.02 + 1e-12
    assert pld.mc_resolution <= 0.1 + 1e-12


def test_repeated_mc_process_preserves_overall_resolution():
    step = ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
        n_steps=8,
        p0=0.02,
    )
    pld = (step * 2).pld(
        mc_resolution=0.1,
        mc_failure_probability=0.01,
        seed=2,
    )

    assert pld.mc_resolution <= 0.1 + 1e-12
    assert pld.mc_failure_probability == pytest.approx(0.01)
