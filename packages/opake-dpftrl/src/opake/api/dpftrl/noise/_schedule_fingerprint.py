"""Materialized learning-rate schedules and complete strategy cache keys."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opake.exceptions import InputTypeError

if TYPE_CHECKING:
    from opake.api.engine.scheduling.types import Schedule


def materialize_schedule(
    lr_schedule: Schedule | None, n_steps: int
) -> tuple[float, ...] | None:
    """Return the schedule values that affect an ``n_steps`` strategy query."""
    if lr_schedule is None:
        return None
    return tuple(float(lr_schedule(step)) for step in range(n_steps))


def strategy_cache_key(
    strategy: Any, n_steps: int
) -> tuple[type[Any], tuple[object, ...]]:
    """Return every strategy recipe input that affects an ``n_steps`` query."""
    if not dataclasses.is_dataclass(strategy):
        raise InputTypeError(
            *(
                "strategy_cache_key() requires a dataclass strategy, got "
                f"{type(strategy).__name__}.",
            )
        )
    return (
        type(strategy),
        tuple(
            (
                field.name,
                (
                    materialize_schedule(getattr(strategy, field.name), n_steps)
                    if field.name == "lr_schedule"
                    else getattr(strategy, field.name)
                ),
            )
            for field in dataclasses.fields(strategy)
        ),
    )
