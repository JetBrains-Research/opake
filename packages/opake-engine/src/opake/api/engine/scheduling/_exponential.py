"""``ExponentialSchedule`` + factory."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ExponentialSchedule", "exponential_schedule"]


@dataclass(frozen=True, slots=True)
class ExponentialSchedule:
    """``init * decay_rate^((step - transition_begin) / transition_steps)``.

    Direction is set by ``decay_rate``: ``< 1`` decay, ``> 1`` growth,
    ``== 1`` constant.  Steps before ``transition_begin`` hold at
    ``init_value``.  When ``staircase`` is true the exponent is floored
    to an integer.  ``end_value`` optionally clamps the result (lower
    bound for decay, upper bound for growth).
    """

    init_value: float
    decay_rate: float
    transition_begin: int = 0
    transition_steps: int = 1
    staircase: bool = False
    end_value: float | None = None

    def __call__(self, step: int) -> float:
        if self.transition_steps <= 0 or self.decay_rate == 0:
            return float(self.init_value)
        decreased = step - self.transition_begin
        if decreased <= 0:
            return float(self.init_value)
        p = decreased / self.transition_steps
        if self.staircase:
            p = math.floor(p)
        decayed = self.init_value * (self.decay_rate**p)
        if self.end_value is not None:
            clip = max if self.decay_rate < 1.0 else min
            return clip(decayed, self.end_value)
        return decayed


def exponential_schedule(
    init_value: float,
    decay_rate: float,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: float | None = None,
) -> ExponentialSchedule:
    """Exponential schedule: ``init * decay_rate^((step - transition_begin) / transition_steps)``."""
    return ExponentialSchedule(
        init_value=float(init_value),
        decay_rate=float(decay_rate),
        transition_begin=int(transition_begin),
        transition_steps=int(transition_steps),
        staircase=bool(staircase),
        end_value=None if end_value is None else float(end_value),
    )
