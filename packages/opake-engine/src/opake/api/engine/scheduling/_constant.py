"""``ConstantSchedule`` + factory."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConstantSchedule", "constant_schedule"]


@dataclass(frozen=True, slots=True)
class ConstantSchedule:
    """``schedule(step) == value`` for every step."""

    value: float

    def __call__(self, _step: int) -> float:
        return self.value


def constant_schedule(value: float) -> ConstantSchedule:
    """Return a schedule that yields ``value`` at every step."""
    return ConstantSchedule(value=float(value))
