"""BandMF sensitivity under a min-separation participation schema.

``BandMfStrategy.sensitivity`` must honour ``min_sep`` /
``max_participations``: for a ``bands``-banded lower-triangular ``C`` the
columns of two participations at least ``bands`` rows apart are orthogonal,
so the schema sensitivity ``max_π ‖Σ_{j∈π} C[:,j]‖`` grows with the
participation count (https://arxiv.org/abs/2306.08153, Theorem 2).  These
tests pin that value against the library's own exact banded routine.
"""

import pytest
import torch

from opake.api.dpftrl.noise._sensitivity import (
    get_sensitivity_banded,
    minsep_true_max_participations,
)
from opake.dpftrl.noise import band_mf_strategy

# ``get_sensitivity_banded`` rounds through a float32 tensor, so compare at
# float32 precision rather than float64.
_F32_RTOL = 1e-6


def _dense_lower_triangular_toeplitz(coefs: torch.Tensor, n: int) -> torch.Tensor:
    """Materialize the dense n x n banded lower-triangular Toeplitz C."""
    padded = torch.zeros(n, dtype=torch.float64)
    padded[: len(coefs)] = coefs
    rows = torch.arange(n).unsqueeze(1)
    cols = torch.arange(n).unsqueeze(0)
    lag = rows - cols
    return torch.where(lag >= 0, padded[lag.clamp(min=0)], torch.zeros(()))


@pytest.mark.parametrize(("n_steps", "bands"), [(512, 64), (128, 8), (256, 32)])
def test_matches_exact_banded_sensitivity_on_dense_c(n_steps, bands):
    """Participation-aware value == the exact routine on the dense C."""
    strategy = band_mf_strategy(bands=bands, momentum=0.95)
    coefs = strategy.coefficients(n_steps=n_steps)
    dense = _dense_lower_triangular_toeplitz(coefs, n_steps)

    # min_sep >= bands, so X = C.T @ C is min_sep-banded and the exact
    # routine applies.
    expected = get_sensitivity_banded(dense, min_sep=bands, max_participations=None)
    reported = strategy.sensitivity(
        n_steps=n_steps, min_sep=bands, max_participations=None
    )

    assert reported == pytest.approx(expected, rel=_F32_RTOL)


@pytest.mark.parametrize(("n_steps", "bands"), [(512, 64), (128, 8)])
def test_grows_as_sqrt_of_participation_count(n_steps, bands):
    """Column norms are 1, so the schema sensitivity is exactly sqrt(k)."""
    strategy = band_mf_strategy(bands=bands, momentum=0.95)
    for max_participations in (1, 2, 4, 8):
        k = minsep_true_max_participations(
            n=n_steps, min_sep=bands, max_participations=max_participations
        )
        reported = strategy.sensitivity(
            n_steps=n_steps,
            min_sep=bands,
            max_participations=max_participations,
        )
        assert reported == pytest.approx(k**0.5, rel=1e-9)


@pytest.mark.parametrize(("n_steps", "bands"), [(512, 64), (128, 8), (100, 7)])
def test_single_participation_returns_the_column_norm(n_steps, bands):
    """The documented single-participation idiom still yields kappa."""
    strategy = band_mf_strategy(bands=bands, momentum=0.95)
    kappa = float(strategy.coefficients(n_steps=n_steps).norm())

    reported = strategy.sensitivity(
        n_steps=n_steps, min_sep=n_steps, max_participations=1
    )

    assert reported == pytest.approx(kappa, rel=1e-12)
    # The BandMF optimizer normalizes its coefficients to unit L2 norm.
    assert kappa == pytest.approx(1.0, rel=1e-9)


def test_ragged_horizon_is_not_the_naive_sqrt_k_bound():
    """Theorem 2's kappa*sqrt(k') is only an upper bound when min_sep ∤ n."""
    n_steps, bands = 100, 7
    strategy = band_mf_strategy(bands=bands, momentum=0.9)
    dense = _dense_lower_triangular_toeplitz(
        strategy.coefficients(n_steps=n_steps), n_steps
    )
    k = minsep_true_max_participations(n=n_steps, min_sep=bands)

    reported = strategy.sensitivity(n_steps=n_steps, min_sep=bands)

    assert reported == pytest.approx(
        get_sensitivity_banded(dense, min_sep=bands), rel=_F32_RTOL
    )
    # Trailing columns are truncated, so the exact value is strictly below
    # the sqrt(k) bound -- a fix that hardcoded sqrt(k) would over-noise.
    assert reported < k**0.5


def test_accepts_recipes_whose_coefficients_leave_the_theorem_domain():
    """momentum=0.0 emits a few sub-zero ulp; the bound must still resolve."""
    n_steps, bands = 512, 64
    strategy = band_mf_strategy(bands=bands, momentum=0.0)
    with pytest.warns(UserWarning, match="momentum=0.0"):
        coefs = strategy.coefficients(n_steps=n_steps)
    assert float(coefs.min()) < 0.0  # strict closed form would reject these

    reported = strategy.sensitivity(n_steps=n_steps, min_sep=bands)

    dense = _dense_lower_triangular_toeplitz(coefs, n_steps)
    assert reported == pytest.approx(
        get_sensitivity_banded(dense, min_sep=bands), rel=_F32_RTOL
    )
