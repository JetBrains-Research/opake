"""Composition operators for combining DP processes.

Most users should prefer the operator syntax: ``step * 1000`` or ``a | b``.

The convenience functions :func:`repeat`, :func:`compose`, and
:func:`cached` are equivalent to those operators / a ``CachedProcess``
wrapper.

The composition-node dataclasses (``Composed``, ``Repeated``,
``CachedProcess``) live in :mod:`opake.accounting.composition.types`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opake.api.accounting.core.composition._cached import cached
from opake.api.accounting.core.composition._composed import (
    Composed,  # noqa: F401  -- register in _PROCESS_REGISTRY
)
from opake.api.accounting.core.composition._repeated import (
    Repeated,  # noqa: F401  -- register in _PROCESS_REGISTRY
)

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess


def repeat(process: DpProcess, count: int) -> DpProcess:
    """Homogeneous k-fold composition (repeat a process ``count`` times).

    Equivalent to ``process * count``.

    Args:
        process: The process to repeat.
        count: Number of repetitions.

    Returns:
        Composed process.

    Example::

        step = acc.poisson(acc.gaussian(1.1), 0.01)
        training = acc.repeat(step, 1000)  # same as: step * 1000
        eps = training.epsilon_at(1e-5)
    """
    return process * count


def compose(left: DpProcess, right: DpProcess) -> DpProcess:
    """Heterogeneous composition of two processes.

    Equivalent to ``left | right``.

    Args:
        left: First process.
        right: Second process.

    Returns:
        Composed process.

    Example::

        # Multi-phase training with different noise
        phase1 = acc.poisson(acc.gaussian(0.9), 0.01) * 500
        phase2 = acc.poisson(acc.gaussian(0.7), 0.01) * 500
        total = acc.compose(phase1, phase2)  # same as: phase1 | phase2
        eps = total.epsilon_at(1e-5)
    """
    return left | right


__all__ = ["cached", "compose", "repeat"]
