"""Opake DP-FTRL: correlated (matrix-factorization) noise mechanisms.

Strategies (BLT, BSR, BiSR, band-MF, λ-CGD, identity) + DP-FTRL-specific
participation samplers (b-min-sep, Poisson, balls-in-bins, sequential).

Compatible clipping rules live in :mod:`opake.dpftrl.clipping` — the MF privacy
proof requires a constant per-step record sensitivity, which both
:func:`~opake.dpftrl.clipping.clipped_grad` (fixed threshold) and
:func:`~opake.dpftrl.clipping.auto_clipped_grad` (AUTO-S smooth scaling, Bu
et al., `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_, NeurIPS 2023)
provide by construction. The DP-SGD-specific
:func:`~opake.dpsgd.clipping.adaptive_clipped_grad` is *not* compatible:
its threshold drifts across steps based on the noisy clipping rate, so
the per-step sensitivity varies and the standard MF analysis breaks.
Functional optimizers (including the universal ``adamw`` that consumes
private ``noisy_squared_grads`` streams) live in :mod:`opake.optimizers`.

Strategy and noise-state dataclasses (``BandMfStrategy``, ``BltStrategy``,
``BisrStrategy``, ``BsrStrategy``, ``IdentityStrategy``,
``LambdaCgdStrategy``, ``MfStrategy``, ``MFNoiseState``,
``SecondMomentMFNoiseState``) live in :mod:`opake.dpftrl.noise.types`.
The cross-cutting ``SecondMomentNoiseOutput`` lives in :mod:`opake.types`.

The :mod:`opake.dpftrl.accounting` subpackage (DP-FTRL-specific privacy
accounting factories, requires ``opake-accounting``) is **lazy-imported**:
``import opake.dpftrl; opake.dpftrl.accounting.band_mf(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute
access — so callers that only need noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

from opake.dpftrl import clipping, noise, sampling
from opake.dpftrl.clipping import auto_clipped_grad, clipped_grad, per_group
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opake.dpftrl import accounting as accounting

try:
    __version__ = _pkg_version("opake-dpftrl")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opake.dpftrl.accounting``.

    Defers loading ``opake.accounting`` (and its native Rust extension)
    until ``opake.dpftrl.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opake.dpftrl.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opake.dpftrl' has no attribute {name!r}")  # noqa: TRY003 - preserve standard Python error contract


__all__ = [
    "__version__",
    # Subpackages
    "accounting",
    "clipping",
    "noise",
    "sampling",
    # MF-safe clipping (re-exported for one-stop imports)
    "clipped_grad",
    "auto_clipped_grad",
    "per_group",
    # Dispatchers
    "mf_gaussian_noise",
    # Strategy factories
    "band_mf_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    # Samplers
    "BallsInBinsSampler",
    "BMinSepSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
