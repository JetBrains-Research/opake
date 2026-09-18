"""Public type definitions for :mod:`opake.random`.

Re-exports :class:`RngKey` for type annotations. The functional surface
(``key``, ``split``, ``fold_in``, …) lives in the package init.
"""

from __future__ import annotations

from opake.api.engine.random._engine import RngKey

__all__ = ["RngKey"]
