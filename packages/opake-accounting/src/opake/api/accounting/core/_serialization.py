"""Serialization registration hook for :mod:`opake.accounting`.

Import this module (as :mod:`opake.accounting` does) so
:class:`~opake.accounting._accountant.Accountant` is registered with
:mod:`opake.serialization`.

:class:`~opake.accounting._base.DpProcess` subclasses register in
``__init_subclass__`` when their defining module loads.
"""

from __future__ import annotations

from opake.api.accounting.core._accountant import _register_accountant_serialization

_register_accountant_serialization()

__all__: list[str] = []
