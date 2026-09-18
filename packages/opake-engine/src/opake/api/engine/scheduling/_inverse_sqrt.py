"""``InverseSqrtSchedule`` + factory."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["InverseSqrtSchedule", "inverse_sqrt_schedule"]


@dataclass(frozen=True, slots=True)
class InverseSqrtSchedule:
    """``init_value * sqrt(T / (s + T))`` where
    ``T = transition_steps`` and ``s = max(0, step - transition_begin)``.

    At ``s=0`` returns ``init_value``; at ``s=T`` returns
    ``init_value / sqrt(2)``.
    """

    init_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        s = max(0, step - self.transition_begin)
        return self.init_value * math.sqrt(span / (s + span))


def inverse_sqrt_schedule(
    init_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> InverseSqrtSchedule:
    """Inverse-square-root decay: ``init_value * sqrt(T / (s + T))``."""
    return InverseSqrtSchedule(
        init_value=float(init_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )
