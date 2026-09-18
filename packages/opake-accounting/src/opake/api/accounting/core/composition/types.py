"""Public type definitions for :mod:`opake.accounting.composition`.

Re-exports the composition-node dataclasses for type annotations and
pattern matching. The factory functions (``compose()``, ``repeat()``,
``cached()``) and the ``*`` / ``|`` operator overloads on
:class:`opake.accounting.types.DpProcess` are the recommended user
surface.
"""

from __future__ import annotations

from opake.api.accounting.core.composition._cached import CachedProcess
from opake.api.accounting.core.composition._composed import Composed
from opake.api.accounting.core.composition._repeated import Repeated

__all__ = ["CachedProcess", "Composed", "Repeated"]
