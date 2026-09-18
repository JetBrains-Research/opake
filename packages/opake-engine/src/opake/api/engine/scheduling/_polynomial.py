"""``PolynomialSchedule`` + factory."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PolynomialSchedule", "polynomial_schedule"]


@dataclass(frozen=True, slots=True)
class PolynomialSchedule:
    """Polynomial transition from ``init_value`` to ``end_value``.

    ``end + (init - end) * (1 - count/T)^power`` where
    ``count = clamp(step - transition_begin, 0, T)``.

    Steps before ``transition_begin`` hold at ``init_value``; steps after
    ``transition_begin + transition_steps`` hold at ``end_value``.
    """

    init_value: float
    end_value: float
    power: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        count = max(0, min(span, step - self.transition_begin))
        frac = 1.0 - count / span
        return (self.init_value - self.end_value) * (frac**self.power) + self.end_value


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> PolynomialSchedule:
    """Polynomial transition from ``init_value`` to ``end_value`` over
    ``transition_steps``: ``end + (init - end) * (1 - count/T)^power``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at ``end_value``.
    """
    return PolynomialSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        power=float(power),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )
