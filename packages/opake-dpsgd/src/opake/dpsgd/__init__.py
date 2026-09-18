"""Opake DP-SGD: Differentially Private SGD mechanisms.

Gaussian noise, adaptive and AUTO-S clipping, and the standard + truncated
Poisson samplers. Fixed and AUTO-S clipping live in
:mod:`opake.dpsgd.clipping` (AUTO-S keeps a constant per-record sensitivity
bound).  Functional optimizers (including the universal ``adamw`` with
DP bias-correction and private second-moment paths) live in
:mod:`opake.optimizers`.

Clipping state and aux types (including ``AdaptiveClipState``,
``AdaptiveClippedGradAux``, and AUTO-S types) live in
:mod:`opake.dpsgd.clipping.types`.  ``GaussianNoiseState`` lives in
:mod:`opake.dpsgd.noise.types`.

The :mod:`opake.dpsgd.accounting` subpackage (DP-SGD-specific privacy
accounting factories, requires ``opake-accounting``) is **lazy-imported**:
``import opake.dpsgd; opake.dpsgd.accounting.gaussian(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute access
— so callers that only need clipping / noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

from opake.dpsgd import clipping, noise, sampling
from opake.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
    per_group,
)
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opake.dpsgd import accounting as accounting

try:
    __version__ = _pkg_version("opake-dpsgd")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opake.dpsgd.accounting``.

    Defers loading ``opake.accounting`` (and its native Rust extension)
    until ``opake.dpsgd.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opake.dpsgd.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opake.dpsgd' has no attribute {name!r}")  # noqa: TRY003 - preserve standard Python error contract


__all__ = [
    "__version__",
    # Subpackages
    "accounting",
    "clipping",
    "noise",
    "sampling",
    # Clipping
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
    # Noise mechanisms
    "gaussian_noise",
    # Sampling
    "PoissonSampler",
]
