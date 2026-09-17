"""``OneMinusSqrtSchedule`` + factory."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["OneMinusSqrtSchedule", "one_minus_sqrt_schedule"]


@dataclass(frozen=True, slots=True)
class OneMinusSqrtSchedule:
    """Decay following ``factor = 1 - sqrt(progress)`` from
    ``init_value`` at ``transition_begin`` to ``end_value`` at
    ``transition_begin + transition_steps``.

    Concave decreasing — the value drops faster early than late.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        progress = min(1.0, max(0, step - self.transition_begin) / span)
        factor = 1.0 - math.sqrt(progress)
        return self.end_value + (self.init_value - self.end_value) * factor


def one_minus_sqrt_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> OneMinusSqrtSchedule:
    """Decay following ``factor = 1 - sqrt(progress)``."""
    return OneMinusSqrtSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )
