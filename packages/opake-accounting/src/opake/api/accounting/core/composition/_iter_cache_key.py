"""Iterative PLD cache key construction for composition trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess


def iter_cache_key(process: DpProcess) -> tuple[object, ...]:
    """Return a flat, unambiguous cache key without recursive calls."""
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    parts: list[object] = []
    stack: list[tuple[DpProcess, int | None]] = [(process, None)]
    while stack:
        node, repeat_count = stack.pop()
        if isinstance(node, Composed):
            parts.append("Composed")
            # Repeating a composition self-composes its complete PLD; it does
            # not repeat each child independently.
            stack.append((node.right, None))
            stack.append((node.left, None))
            continue
        if isinstance(node, Repeated):
            parts.extend(("Repeated", node.count))
            stack.append((node.inner, node.count))
            continue
        if isinstance(node, CachedProcess):
            parts.append("CachedProcess")
            stack.append((node.inner, repeat_count))
            continue
        if repeat_count is not None:
            parts.append(node._repeated_pld_cache_key(repeat_count))
        else:
            parts.append(node._pld_cache_key())
    return tuple(parts)
