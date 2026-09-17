"""Iterative equality for arbitrarily-nested ``DpProcess`` composition trees.

Mirrors :mod:`._iter_hash`: the composition wrappers (``Composed``,
``Repeated``, ``CachedProcess``) delegate ``__eq__`` here so structural
comparison of deep chains is bounded by heap rather than
``sys.getrecursionlimit()``.  Non-wrapper leaves compare via their own
``__eq__``, which is safe because leaves hold no unbounded ``DpProcess``
spine.

Semantics are identical to the dataclass-generated ``__eq__`` each wrapper
previously carried: same-class requirement, field-by-field comparison.  The
hash/eq contract with :func:`._iter_hash.iter_hash` is preserved — both walk
the same structure with the same wrapper tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess


def iter_eq(a: DpProcess, b: DpProcess) -> bool:
    """Iterative structural equality of two ``DpProcess`` trees."""
    # Lazy imports break the cycle: each wrapper module imports this
    # helper for its own ``__eq__``.
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    stack: list[tuple[DpProcess, DpProcess]] = [(a, b)]
    while stack:
        x, y = stack.pop()
        if x is y:
            continue
        if type(x) is not type(y):
            return False
        if isinstance(x, Composed):
            stack.append((x.right, y.right))
            stack.append((x.left, y.left))
        elif isinstance(x, Repeated):
            if x.count != y.count:
                return False
            stack.append((x.inner, y.inner))
        elif isinstance(x, CachedProcess):
            stack.append((x.inner, y.inner))
        elif x != y:  # terminal leaf: its own __eq__ is depth-bounded
            return False
    return True
