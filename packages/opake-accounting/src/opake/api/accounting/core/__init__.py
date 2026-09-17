"""Core PLD accounting — native extension, algebra primitives, and process types.

The opake-accounting wheel ships its impl here; users access the same
surface through the ``opake.accounting`` façade.

Native PyO3 extension lands at
``opake.api.accounting.core.opake_accounting`` (matches the maturin
``module-name`` setting); aliased to ``_native`` so submodules can use
a short private name.
"""

# Native PyO3 extension — compiled artifact lives at
# ``opake/api/accounting/core/opake_accounting.abi3.so`` (named after
# the Rust crate). Aliased to ``_native`` for use across submodules.
try:
    from . import opake_accounting as _native  # noqa: F401
except ImportError as e:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "opake.api.accounting.core native extension not found. "
        "Build with: uv run maturin develop --release "
        "-m packages/opake-accounting/Cargo.toml"
    ) from e

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("opake-accounting")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Side-effect import: registers Accountant + DpProcess subclasses with
# the unified ``opake.api.base.serialization`` registry.
import opake.api.accounting.core._serialization  # noqa: F401
from opake.api.accounting.core._accountant import Accountant
from opake.api.accounting.core._budgets import register_budget_serializer
from opake.api.accounting.core.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)
from opake.api.accounting.core.composition import cached, compose, repeat
from opake.api.accounting.core.discretization import (
    get_discretization,
    set_discretization,
)
from opake.api.accounting.core.mechanisms import eps_delta, identity, nonprivate

from . import (
    amplification,
    calibration,
    composition,
    discretization,
    mechanisms,
)

__all__ = [
    "__version__",
    # Submodules
    "amplification",
    "calibration",
    "composition",
    "discretization",
    "mechanisms",
    # Accountant
    "Accountant",
    # Discretization
    "set_discretization",
    "get_discretization",
    # Generic mechanisms
    "eps_delta",
    "identity",
    "nonprivate",
    # Composition
    "repeat",
    "compose",
    "cached",
    # Calibration / budgets
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    "register_budget_serializer",
    "calibrate",
]
