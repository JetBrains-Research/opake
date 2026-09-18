"""``WithRestarts`` composition wrapper + factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from opake.api.engine.scheduling._schedule import Schedule

__all__ = ["WithRestarts", "with_restarts"]


@dataclass(frozen=True, slots=True)
class WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.

    Each cycle has (real-valued) length ``transition_steps /
    num_cycles``; within a cycle, ``schedule`` is evaluated at the
    cycle-local position ``relative_step mod cycle_length``.  When
    ``num_cycles`` divides ``transition_steps`` the cycle length is an
    integer and the positions are exactly ``relative_step %
    cycle_length``; otherwise the fractional cycle length places restart
    boundaries at ``k * transition_steps / num_cycles`` (matching
    HuggingFace's ``cosine_with_hard_restarts`` fractional-progress
    formula, which does not require divisibility).

    Before ``transition_begin`` returns ``schedule(0)``; after the
    final cycle, returns ``schedule(cycle_length)``.
    """

    schedule: Schedule
    transition_steps: int
    num_cycles: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        cycle_length = self.transition_steps / self.num_cycles
        relative = step - self.transition_begin
        if relative < 0:
            return self.schedule(0)
        if relative >= self.transition_steps:
            return self.schedule(cycle_length)
        # Cycle-local position.  For integer ``cycle_length`` this equals
        # ``relative % cycle_length``; the float form generalises to a
        # non-divisible (transition_steps, num_cycles) pair.
        within_cycle = relative - (relative // cycle_length) * cycle_length
        return self.schedule(within_cycle)


def with_restarts(
    schedule: Schedule,
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.
    """
    if num_cycles <= 0:
        raise ConfigurationError(
            *(f"with_restarts requires num_cycles > 0; got {num_cycles}.",)
        )
    if transition_steps <= 0:
        raise ConfigurationError(
            *(f"with_restarts requires transition_steps > 0; got {transition_steps}.",)
        )
    # ``num_cycles`` need not divide ``transition_steps``: ``WithRestarts``
    # uses a real-valued cycle length so restart boundaries land at
    # ``k * transition_steps / num_cycles`` (HF parity).  Divisible inputs
    # still yield exactly integer-aligned cycles.
    return WithRestarts(
        schedule=schedule,
        transition_steps=int(transition_steps),
        num_cycles=int(num_cycles),
        transition_begin=int(transition_begin),
    )
