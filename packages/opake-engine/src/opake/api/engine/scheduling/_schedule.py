"""Step-indexed scalar schedule type alias.

Foundational typing primitive shared by every schedule recipe and every
consumer that holds a schedule reference.  Lives in its own leaf module
so impl files and the ``types`` re-export façade can both depend on it
without forming a cycle (recipes depend on ``Schedule``; the façade
depends on the recipes).
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["Schedule"]

Schedule = Callable[[int], float]
