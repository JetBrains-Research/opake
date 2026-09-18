"""``CosineSchedule`` + factory."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CosineSchedule", "cosine_schedule"]


@dataclass(frozen=True, slots=True)
class CosineSchedule:
    """Cosine annealing from ``init_value`` to ``end_value``.

    ``progress = (step - transition_begin) / transition_steps`` grows
    past 1 and the cosine continues to oscillate — the ``max(0, cos)``
    clip keeps the schedule non-negative.  With ``num_cycles=0.5``
    (default) this is a single half-cosine bottoming out at
    ``end_value`` when ``progress == 1``; larger values produce
    additional oscillations.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0
    num_cycles: float = 0.5

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        progress = max(0, step - self.transition_begin) / span
        cos = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
        return self.end_value + (self.init_value - self.end_value) * max(0.0, cos)


def cosine_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
    num_cycles: float = 0.5,
) -> CosineSchedule:
    """Cosine annealing from ``init_value`` to ``end_value`` over ``transition_steps``."""
    return CosineSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
        num_cycles=float(num_cycles),
    )
