"""BandMF participation accounting across the three sensitivity consumers.

The un-amplified ``mf_gaussian`` PLD is a *single* Gaussian at
``noise_multiplier / sensitivity`` over the whole horizon, so its
sensitivity must be the participation-aware one.  Both amplified routes
price the repeat participations themselves -- cyclic Poisson by
``self_compose(num_groups)`` (https://arxiv.org/abs/2306.08153, Theorem 4),
b-min-sep by the warm-start recursion
(https://arxiv.org/abs/2602.09338, Section 5) -- so they must keep the
single-participation column norm.  These tests pin both halves.
"""

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.noise import band_mf_strategy

_DELTA = 1e-6


@pytest.mark.parametrize(
    ("n_steps", "bands", "noise_multiplier"),
    [(1024, 64, 2.0), (128, 8, 2.0)],
)
def test_unamplified_pld_matches_k_fold_gaussian_composition(
    n_steps, bands, noise_multiplier
):
    """Bare BandMF over k participations == k-fold plain-Gaussian composition.

    ``bands`` divides ``n_steps`` here, so every participating column has
    full norm 1 and the schema sensitivity is exactly ``sqrt(k)``.  A single
    Gaussian at ``sigma / sqrt(k)`` is then the same mechanism as ``k``
    compositions at ``sigma`` -- which is what cyclic Poisson degenerates to
    at ``sample_rate=1.0``, an independent route through the library.
    """
    strategy = band_mf_strategy(bands=bands, momentum=0.95)

    bare = ftrl_acc.mf_gaussian(
        noise_multiplier, strategy, n_steps=n_steps, min_sep=bands
    )
    k_fold = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(noise_multiplier, strategy),
        sample_rate=1.0,
        n_steps=n_steps,
    )

    assert bare.pld().epsilon_at(_DELTA) == pytest.approx(
        k_fold.pld().epsilon_at(_DELTA), rel=1e-7
    )


def test_unamplified_pld_tracks_the_declared_participation_schema():
    """Bare epsilon must respond to min_sep / max_participations."""
    strategy = band_mf_strategy(bands=8, momentum=0.95)

    def eps(**schema):
        return (
            ftrl_acc.mf_gaussian(2.0, strategy, n_steps=128, **schema)
            .pld()
            .epsilon_at(_DELTA)
        )

    single = eps(min_sep=128, max_participations=1)
    two = eps(min_sep=8, max_participations=2)
    sixteen = eps(min_sep=8, max_participations=None)

    assert single < two < sixteen


# --- Amplified routes: pinned to the values produced before BandMF's
# --- sensitivity became participation-aware.  These must not move.


# Tolerance for the "this fix did not move the amplified routes" pins. The
# excluded sensitivity regression is a factor of sqrt(k') -- 4x at these
# configurations -- so these limits remain many orders of magnitude tighter.
_PIN_REL = 1e-9
# This long-horizon convolution has a slightly larger platform-specific drift.
_PRODUCTION_PIN_REL = 2e-9


def test_cyclic_poisson_epsilon_is_unchanged():
    strategy = band_mf_strategy(bands=8, momentum=0.95)
    inner = ftrl_acc.mf_gaussian(2.0, strategy)

    unsampled = ftrl_acc.poisson(inner, sample_rate=1.0, n_steps=128)
    sampled = ftrl_acc.poisson(inner, sample_rate=0.05, n_steps=128)

    assert unsampled.pld().epsilon_at(_DELTA) == pytest.approx(
        10.997151210060439, rel=_PIN_REL
    )
    assert sampled.pld().epsilon_at(_DELTA) == pytest.approx(
        0.5662867696910785, rel=_PIN_REL
    )


def test_b_min_sep_epsilon_is_unchanged():
    strategy = band_mf_strategy(bands=4)
    proc = ftrl_acc.b_min_sep(ftrl_acc.mf_gaussian(1.0, strategy), n_steps=40, p0=0.02)

    pld = proc.pld(seed=123, mc_resolution=5e-3, mc_failure_probability=1e-2)

    assert pld.epsilon_at(1e-2) == pytest.approx(0.8493924366281685, rel=_PIN_REL)


@pytest.mark.slow
def test_cyclic_poisson_production_anchor_is_unchanged():
    """The documented band-MF calibration anchor, pinned end to end."""
    proc = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.141283, band_mf_strategy(bands=64, momentum=0.95)),
        sample_rate=0.032768,
        n_steps=15625,
    )

    assert proc.pld().epsilon_at(_DELTA) == pytest.approx(
        2.9986033025122927, rel=_PRODUCTION_PIN_REL
    )
