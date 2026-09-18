"""Cross-validation: random allocation vs the paper's reference implementation.

``random_allocation`` (MIT, github.com/moshenfeld/random_allocation) is the
reference implementation published with Feldman & Shenfeld.  It ships the
*analytic* bounds — RDP-derived (``direct``) and decomposition-based — not
the PLD transform Opake implements, so this is a **one-sided** check: our
bound must sit at or below theirs at every setting.  Equality is not
expected and would in fact be a bad sign, since the whole reason to compute
the PLD is that it is tighter than the analytic bounds.

The direction is what matters. If our ε ever exceeded an independently
implemented valid bound, the transform would be unsound; that is the failure
this catches, and it catches it against code we did not write.

Requires optional deps — install with ``uv sync --group dev``.
The ``pytest.importorskip`` below gates the whole module.

Run with::

    uv run pytest packages/opake-dpsgd/tests/accounting/test_cross_validation_allocation.py -v
"""

from __future__ import annotations

from functools import lru_cache

import pytest

pytest.importorskip("random_allocation")

from random_allocation import (
    allocation_epsilon_decomposition,
    allocation_epsilon_direct,
)
from random_allocation.comparisons.structs import (
    PrivacyParams,
    SchemeConfig,
)

import opake.dpsgd.accounting as dpsgd_acc

pytestmark = pytest.mark.slow

_DELTA = 1e-8

#: Rényi orders the reference needs for its ``direct`` bound; it raises
#: without them rather than picking a default.
_ALPHA_ORDERS = list(range(2, 61))

#: ``(sigma, num_bins)`` — the same grid as the Rust golden-value test.
_CASES = [(1.0, 8), (1.0, 64), (2.0, 64), (1.0, 128)]


@lru_cache
def _reference(fn, sigma: float, t: int) -> float:
    params = PrivacyParams(
        sigma=sigma, num_steps=t, num_selected=1, num_epochs=1, delta=_DELTA
    )
    return fn(params, SchemeConfig(allocation_direct_alpha_orders=_ALPHA_ORDERS))


@lru_cache
def _ours(sigma: float, t: int) -> float:
    return dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(sigma), k=1, t=t, allocation="block"
    ).epsilon_at(_DELTA)


@pytest.mark.parametrize(("sigma", "t"), _CASES)
@pytest.mark.parametrize(
    "reference",
    [allocation_epsilon_direct, allocation_epsilon_decomposition],
    ids=["direct", "decomposition"],
)
def test_at_or_below_reference_bound(reference, sigma: float, t: int):
    ours = _ours(sigma, t)
    theirs = _reference(reference, sigma, t)
    assert ours <= theirs + 1e-9, (
        f"σ={sigma} t={t}: ours {ours} exceeds the reference bound {theirs}"
    )


@pytest.mark.parametrize(("sigma", "t"), _CASES)
def test_strictly_tighter_than_the_analytic_bounds(sigma: float, t: int):
    """Not merely sound but strictly better — the reason for the PLD route.

    Measured gaps on this grid run 8%–39%; asserted loosely at >2% so the
    test tracks the claim rather than today's discretisation.
    """
    ours = _ours(sigma, t)
    best = min(
        _reference(allocation_epsilon_direct, sigma, t),
        _reference(allocation_epsilon_decomposition, sigma, t),
    )
    assert ours < 0.98 * best, f"σ={sigma} t={t}: ours {ours} vs best analytic {best}"
