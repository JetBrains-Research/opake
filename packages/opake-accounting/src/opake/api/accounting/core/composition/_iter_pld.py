"""Iterative PLD evaluation for composition-wrapper trees."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from opake.api.accounting.core.discretization import get_discretization

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess, Pld


def iter_pld(process: DpProcess, kwargs: dict[str, Any]) -> Pld:
    """Evaluate composition wrappers without consuming the Python call stack."""
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    tasks: list[tuple[str, object, dict[str, Any] | None]] = [("eval", process, kwargs)]
    results: list[Pld] = []

    while tasks:
        action, value, current_kwargs = tasks.pop()
        if action == "compose":
            count = int(value)
            children = results[-count:]
            del results[-count:]
            result = children[0]
            for child in children[1:]:
                result = result.compose(child)
            results.append(result)
            continue
        if action == "repeat":
            results.append(results.pop().self_compose(int(value)))
            continue

        node = value
        assert current_kwargs is not None
        if isinstance(node, CachedProcess):
            tasks.append(("eval", node.inner, current_kwargs))
            continue
        if isinstance(node, Composed):
            rights: list[DpProcess] = []
            left: DpProcess = node
            while isinstance(left, Composed):
                rights.append(left.right)
                left = left.left

            resolved = get_discretization(
                mc_resolution=current_kwargs["mc_resolution"],
                mc_failure_probability=current_kwargs["mc_failure_probability"],
            )
            group_count = len(rights) + 1
            child_kwargs = dict(current_kwargs)
            child_kwargs["mc_failure_probability"] = (
                resolved.mc_failure_probability / group_count
            )
            child_kwargs["mc_resolution"] = -math.expm1(
                math.log1p(-resolved.mc_resolution) / group_count
            )
            children = [left, *reversed(rights)]
            tasks.append(("compose", group_count, None))
            tasks.extend(("eval", child, child_kwargs) for child in reversed(children))
            continue
        if isinstance(node, Repeated):
            inner = node.inner
            while isinstance(inner, CachedProcess):
                inner = inner.inner
            if isinstance(inner, (Composed, Repeated)):
                resolved = get_discretization(
                    mc_resolution=current_kwargs["mc_resolution"]
                )
                child_kwargs = dict(current_kwargs)
                child_kwargs["mc_resolution"] = -math.expm1(
                    math.log1p(-resolved.mc_resolution) / node.count
                )
                tasks.append(("repeat", node.count, None))
                tasks.append(("eval", inner, child_kwargs))
            else:
                results.append(inner.repeated_pld(node.count, **current_kwargs))
            continue

        results.append(node.pld(**current_kwargs))

    assert len(results) == 1
    return results[0]
