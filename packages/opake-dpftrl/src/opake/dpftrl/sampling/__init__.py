"""Matrix-factorization participation samplers façade for DP-FTRL.

These samplers generate the participation patterns required by MF
mechanisms (band-MF, BLT, BSR, lambda-CGD) — b-min-separation, Poisson
``CyclicPoissonSampler`` (``bands=1`` identity baseline vs ``bands>1``
cyclic BandMF), balls-in-bins, and sequential batches.
"""

from opake.api.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)

__all__ = [
    "BMinSepSampler",
    "BallsInBinsSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
