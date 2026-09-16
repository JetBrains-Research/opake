"""Opaque DP-FTRL: correlated (matrix-factorization) noise mechanisms.

Strategies (BLT, BSR, BiSR, band-MF, λ-CGD, identity) + DP-FTRL-specific
participation samplers (b-min-sep, Poisson, balls-in-bins, sequential).

Compatible clipping rules live in :mod:`opaque.dpftrl.clipping` — the MF privacy
proof requires a constant per-step record sensitivity, which both
:func:`~opaque.dpftrl.clipping.clipped_grad` (fixed threshold) and
:func:`~opaque.dpftrl.clipping.auto_clipped_grad` (AUTO-S smooth scaling, Bu
et al., `Automatic Clipping <https://arxiv.org/abs/2206.07136>`_, NeurIPS 2023)
provide by construction. The DP-SGD-specific
:func:`~opaque.dpsgd.clipping.adaptive_clipped_grad` is *not* compatible:
its threshold drifts across steps based on the noisy clipping rate, so
the per-step sensitivity varies and the standard MF analysis breaks.
Functional optimizers (including the universal ``adamw`` that consumes
private ``noisy_squared_grads`` streams) live in :mod:`opaque.optimizers`.

Strategy and noise-state dataclasses (``BandMfStrategy``, ``BltStrategy``,
``BisrStrategy``, ``BsrStrategy``, ``IdentityStrategy``,
``LambdaCgdStrategy``, ``MfStrategy``, ``MFNoiseState``) live in
:mod:`opaque.dpftrl.noise.types`.

The :mod:`opaque.dpftrl.accounting` subpackage (DP-FTRL-specific privacy
accounting factories, requires ``opaque-accounting``) is **lazy-imported**:
``import opaque.dpftrl; opaque.dpftrl.accounting.band_mf(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute
access — so callers that only need noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

from opaque.dpftrl import clipping, noise, sampling
from opaque.dpftrl.clipping import auto_clipped_grad, clipped_grad, per_group
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opaque.dpftrl import accounting as accounting

try:
    __version__ = _pkg_version("opaque-dpftrl")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opaque.dpftrl.accounting``.

    Defers loading ``opaque.accounting`` (and its native Rust extension)
    until ``opaque.dpftrl.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opaque.dpftrl.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opaque.dpftrl' has no attribute {name!r}")  # noqa: TRY003 - preserve standard Python error contract


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
