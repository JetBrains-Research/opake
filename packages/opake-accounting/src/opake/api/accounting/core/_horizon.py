"""Whole-horizon DP process marker."""

from __future__ import annotations

from opake.api.accounting.core._base import DpProcess

__all__ = ["DpHorizonProcess"]


# noinspection PyAbstractClass
class DpHorizonProcess(DpProcess):
    """A complete fixed-horizon process that cannot be accounted by prefixes.

    Subclasses implement the ordinary :meth:`DpProcess.pld` contract for their
    full declared ``n_steps`` and are lifecycle markers for trainer integrations.
    """

    n_steps: int
