"""Iterative repr for arbitrarily-nested ``DpProcess`` composition trees.

Mirrors :mod:`._iter_hash` / :mod:`._iter_eq`: the composition wrappers
delegate ``__repr__`` here so deep chains — left- or right-skewed
``Composed`` spines and the ``cached(acct)``-per-round alternation the
:class:`._cached.CachedProcess` docstring recommends — render without
overflowing the interpreter stack.  Output is string-identical to the
dataclass-generated reprs.

Repr tokens stream through an explicit stack into one chunk buffer, so
live memory is proportional to the final string rather than quadratic in
spine depth.  A child shared by two parents is re-walked per occurrence;
the walk stays time-proportional to the output, which is itself what
grows under sharing.  Non-wrapper leaves render via their own ``repr``,
which is safe because leaves hold no unbounded ``DpProcess`` spine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess


def iter_repr(process: DpProcess) -> str:
    """Iterative pre-order repr of a ``DpProcess`` tree."""
    # Lazy imports break the cycle: each wrapper module imports this
    # helper for its own ``__repr__``.
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    chunks: list[str] = []
    # Nodes still to render plus literal tokens to emit, pushed in
    # reverse emission order.  Tokens are exact ``str`` instances, so
    # the ``type`` check cannot swallow an exotic str-subclass leaf.
    stack: list[DpProcess | str] = [process]
    while stack:
        item = stack.pop()
        if type(item) is str:
            chunks.append(item)
        elif isinstance(item, Composed):
            stack += (")", item.right, ", right=", item.left, "Composed(left=")
        elif isinstance(item, Repeated):
            stack += (f", count={item.count!r})", item.inner, "Repeated(inner=")
        elif isinstance(item, CachedProcess):
            stack += (")", item.inner, "CachedProcess(inner=")
        else:
            chunks.append(repr(item))
    return "".join(chunks)
