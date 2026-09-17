"""``LinearSchedule`` + factory.

A degenerate :class:`~._polynomial.PolynomialSchedule` with ``power=1.0``
kept as its own type so call sites read naturally and round-trip data
preserves the distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LinearSchedule", "linear_schedule"]


@dataclass(frozen=True, slots=True)
class LinearSchedule:
    """Linear interpolation from ``init_value`` to ``end_value`` over
    ``transition_steps``.  Equivalent to
    :class:`~._polynomial.PolynomialSchedule` with ``power=1.0``.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        # Hot path: arithmetic inlined.  Equivalent to
        # ``PolynomialSchedule(power=1.0).__call__`` with the ``power``
        # factor dropped.
        span = max(1, self.transition_steps)
        count = max(0, min(span, step - self.transition_begin))
        frac = 1.0 - count / span
        return (self.init_value - self.end_value) * frac + self.end_value


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> LinearSchedule:
    """Linear interpolation from ``init_value`` to ``end_value`` over
    ``transition_steps``, starting at ``transition_begin``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at ``end_value``.
    """
    return LinearSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )
