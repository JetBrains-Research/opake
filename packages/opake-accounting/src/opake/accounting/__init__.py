"""Differential privacy accounting using Privacy Loss Distributions (PLD).

Cross-cutting accounting surface — composition, calibration, generic
mechanisms (``identity``, ``nonprivate``, ``eps_delta``).

Algorithm-specific factories live in their respective packages
(``opake-dpsgd`` / ``opake-dpftrl``):

- :mod:`opake.dpsgd.accounting` — ``gaussian``, ``adaclip``, ``poisson``
  (including its truncated form), ``parallel_poisson``, ``k_out_of_t``.
- :mod:`opake.dpftrl.accounting` — ``mf_gaussian``, ``poisson``,
  ``b_min_sep``, ``balls_in_bins``.

Implementation uses Google's PLD accounting via the ``opake-accounting``
Rust crate (PyO3 bindings).

Example (requires ``opake-dpsgd`` in the environment):

>>> import opake.dpsgd.accounting as dpsgd_acc
>>> step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
>>> training = step * 1000
>>> epsilon = training.epsilon_at(1e-5)
"""

from opake.api.accounting.core import (
    Accountant,
    __version__,
    advantage_budget,
    amplification,
    beta_budget,
    cached,
    calibrate,
    calibration,
    compose,
    composition,
    delta_budget,
    discretization,
    eps_delta,
    epsilon_budget,
    get_discretization,
    identity,
    mechanisms,
    nonprivate,
    register_budget_serializer,
    repeat,
    risk_budget,
    set_discretization,
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
