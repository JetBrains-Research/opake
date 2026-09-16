"""Opaque DP-SGD: Differentially Private SGD mechanisms.

Gaussian noise (optionally bounded — Chen and Hale,
`The Bounded Gaussian Mechanism for Differential Privacy
<https://arxiv.org/abs/2211.17230>`_, 2024), adaptive and AUTO-S clipping,
and the standard + truncated Poisson samplers. Fixed and AUTO-S clipping live in
:mod:`opaque.dpsgd.clipping` (AUTO-S keeps a constant per-record sensitivity
bound).  Functional optimizers (including the universal ``adamw`` with
DP bias-correction and JME second-moment routing) live in
:mod:`opaque.optimizers`.

Clipping state and aux types (including ``AdaptiveClipState``,
``AdaptiveClippedGradAux``, and AUTO-S types) live in
:mod:`opaque.dpsgd.clipping.types`. ``GaussianNoiseState``, ``JmeNoiseState``,
and ``JmeAllocation`` live in :mod:`opaque.dpsgd.noise.types`; the projected
JME factory lives in :mod:`opaque.dpsgd.noise`.

The :mod:`opaque.dpsgd.accounting` subpackage (DP-SGD-specific privacy
accounting factories, requires ``opaque-accounting``) is **lazy-imported**:
``import opaque.dpsgd; opaque.dpsgd.accounting.gaussian(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute access
— so callers that only need clipping / noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

from opaque.dpsgd import clipping, noise, sampling
from opaque.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
    per_group,
)
from opaque.dpsgd.noise import gaussian_noise, jme_noise
from opaque.dpsgd.sampling import PoissonSampler

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opaque.dpsgd import accounting as accounting

try:
    __version__ = _pkg_version("opaque-dpsgd")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opaque.dpsgd.accounting``.

    Defers loading ``opaque.accounting`` (and its native Rust extension)
    until ``opaque.dpsgd.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opaque.dpsgd.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opaque.dpsgd' has no attribute {name!r}")  # noqa: TRY003 - preserve standard Python error contract


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
    "jme_noise",
    # Sampling
    "PoissonSampler",
]
