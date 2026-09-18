"""Public type definitions for :mod:`opake.accounting`.

One-stop import for every dataclass / state / config type in the
accounting package, including the interactive :class:`Accountant`
container.  The functional surface (``gaussian()``, ``poisson()``,
``calibrate()``, etc.) lives in the package init.

For narrower namespaces, types are also re-exported from per-subpackage
``types`` modules:

- :mod:`opake.accounting.mechanisms.types` — mechanism dataclasses
- :mod:`opake.accounting.amplification.types` — subsampling dataclasses
- :mod:`opake.accounting.composition.types` — composition-node types

DP-SGD-specific dataclasses (``Gaussian``, ``Poisson``, ``ParallelPoisson``,
``AdaClip``) are re-exported from :mod:`opake.dpsgd.accounting.types`
(requires the ``opake-dpsgd`` install); DP-FTRL-specific dataclasses
(``BandMf``, ``Blt``, ``LambdaCgd``, ``Bisr``, ``Bsr``, ``MfGaussian``,
``IdentityMf``, ``CyclicPoisson``, ``BMinSep``, ``BallsInBins``) from
:mod:`opake.dpftrl.accounting.types` (requires ``opake-dpftrl``).  This
module only re-exports the cross-cutting types that live in
``opake-accounting`` itself.
"""

from __future__ import annotations

from opake.api.accounting.core._accountant import Accountant
from opake.api.accounting.core._base import DpProcess
from opake.api.accounting.core._budgets import (
    AdvantageBudget,
    BetaBudget,
    Budget,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
)
from opake.api.accounting.core._horizon import DpHorizonProcess
from opake.api.accounting.core.calibration import CalibrateResult
from opake.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)
from opake.api.accounting.core.discretization import DiscretizationConfig
from opake.api.accounting.core.mechanisms.types import (
    EpsDelta,
    Identity,
    NonPrivate,
)

__all__ = [
    # Interactive container
    "Accountant",
    # Algebra base
    "DpProcess",
    "DpHorizonProcess",
    # Budgets
    "Budget",
    "EpsilonBudget",
    "DeltaBudget",
    "AdvantageBudget",
    "BetaBudget",
    "RiskBudget",
    # Calibration / discretization
    "CalibrateResult",
    "DiscretizationConfig",
    # Generic mechanisms
    "EpsDelta",
    "Identity",
    "NonPrivate",
    # Composition
    "Composed",
    "Repeated",
    "CachedProcess",
]
