"""Public type definitions for :mod:`opake.dpftrl.accounting.amplification`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from opake.api.accounting.dpftrl.amplification._b_min_sep import BMinSep
from opake.api.accounting.dpftrl.amplification._balls_in_bins import BallsInBins
from opake.api.accounting.dpftrl.amplification._poisson import CyclicPoisson


@runtime_checkable
class MfAmplification(Protocol):
    """Participation-structure contract every MF amplifier exposes.

    The three values define the user's worst-case participation pattern
    and feed both accounting (gram/sensitivity at the right horizon) and
    noise (streaming matrix sized to the same window).  The amplifier is
    the single source of truth; strategies do not carry these knobs.

    Implementations:

    ===============  =======================  ==========================
    Amplification    ``min_sep``              ``max_participations``
    ===============  =======================  ==========================
    BallsInBins      ``num_bins``             ``n_steps // num_bins``
    BMinSep          ``inner.strategy.bands`` ``n_steps // bands``
    CyclicPoisson    ``1`` (no guarantee)     ``n_steps`` (worst case)
    ===============  =======================  ==========================
    """

    n_steps: int

    @property
    def min_sep(self) -> int: ...

    @property
    def max_participations(self) -> int: ...


__all__ = ["BMinSep", "BallsInBins", "CyclicPoisson", "MfAmplification"]
