"""Iterative hash for arbitrarily-nested ``DpProcess`` composition trees.

Walks the entire tree with an explicit stack so hash depth is bounded
by available heap rather than ``sys.getrecursionlimit()``. The
composition wrappers (``Composed``, ``Repeated``, ``CachedProcess``)
delegate ``__hash__`` here; non-wrapper leaves (``GaussianMechanism``,
``Identity``, ``EpsDelta``, ``NonPrivate``, …) are hashed via their
own ``__hash__``, which is safe because leaves hold no ``DpProcess``
field.

Hashes are consistent with the dataclass ``__eq__`` of each wrapper:
structurally-equal trees produce equal hashes; structurally-distinct
trees produce distinct hashes (modulo standard hash collisions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess

# Sentinel tag strings folded into the hash to distinguish wrapper kinds
# (so e.g. ``Repeated(x, 3)`` doesn't collide with ``Composed(x, x, x)``).
_COMPOSED_TAG = "Composed"
_REPEATED_TAG = "Repeated"
_CACHED_TAG = "Cached"


def iter_hash(process: DpProcess) -> int:
    """Iterative structural hash of a ``DpProcess`` tree.

    Walks the entire tree with an explicit stack — no Python recursion,
    so depth is bounded by available heap rather than
    ``sys.getrecursionlimit()``.
    """
    # Lazy imports break the cycle: each wrapper module imports this
    # helper for its own ``__hash__``.
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    h = 0
    # Explicit DFS stack. Each entry is a node to visit.
    stack: list[DpProcess] = [process]
    while stack:
        node = stack.pop()
        if isinstance(node, Composed):
            # Fold a structural tag, then schedule the children so the
            # left subtree is processed first (LIFO stack semantics).
            h = hash((h, _COMPOSED_TAG))
            stack.append(node.right)
            stack.append(node.left)
            continue
        if isinstance(node, Repeated):
            h = hash((h, _REPEATED_TAG, node.count))
            stack.append(node.inner)
            continue
        if isinstance(node, CachedProcess):
            h = hash((h, _CACHED_TAG))
            stack.append(node.inner)
            continue
        # Terminal leaf. Its own ``__hash__`` is safe — leaves hold no
        # ``DpProcess`` field, so no further recursion is possible.
        h = hash((h, hash(node)))
    return h
