"""``WithWarmup`` composition wrapper + factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opake.api.engine.scheduling._constant import ConstantSchedule
from opake.api.engine.scheduling._ramp import resolve_ramp
from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.api.engine.scheduling._schedule import Schedule

__all__ = ["WithWarmup", "with_warmup"]


@dataclass(frozen=True, slots=True)
class WithWarmup:
    """Multiplicative ``init_value → 1`` ramp over the first
    ``transition_steps`` steps; afterwards return ``schedule(step)``
    unchanged.

    ``schedule`` may be a recipe (any :data:`Schedule`), a raw callable
    (won't round-trip through serialization), or a scalar ``float``
    (treated as :class:`~._constant.ConstantSchedule`).  The factory
    function :func:`with_warmup` normalises scalars at construction so
    the dataclass always stores a callable.

    ``ramp`` is one of ``"linear"``, ``"cosine"``, ``"1-sqrt"``, or a
    callable ``f(progress) -> factor in [0, 1]`` (non-serializable).
    """

    schedule: Schedule
    transition_steps: int
    ramp: str | Callable[[float], float] = "linear"
    init_value: float = 0.0

    def __call__(self, step: int) -> float:
        ramp_fn = resolve_ramp(self.ramp)
        inner_value = self.schedule(step)
        if step >= self.transition_steps:
            return inner_value
        progress = step / self.transition_steps
        if self.init_value == 0.0:
            factor = ramp_fn(progress)
        else:
            factor = self.init_value + (1.0 - self.init_value) * ramp_fn(progress)
        return factor * inner_value


def with_warmup(
    schedule: Schedule | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
    init_value: float = 0.0,
) -> WithWarmup:
    """Multiply ``schedule`` by an ``init_value → 1`` ramp over the
    first ``transition_steps`` steps.

    For "warmup, then decay" shapes, configure the decay with
    ``transition_begin = transition_steps``.

    A scalar ``float`` for ``schedule`` is treated as
    :class:`~._constant.ConstantSchedule`.

    ``ramp`` controls the warmup curve: ``"linear"`` (default),
    ``"cosine"``, ``"1-sqrt"``, or a callable.
    """
    if transition_steps <= 0:
        raise ConfigurationError(
            *(f"with_warmup requires transition_steps > 0; got {transition_steps}.",)
        )
    if not 0.0 <= init_value <= 1.0:
        raise ConfigurationError(
            *(f"with_warmup requires init_value in [0, 1]; got {init_value}.",)
        )
    # Fail-fast on bad string ramps (round-trip the resolver to validate).
    resolve_ramp(ramp)
    inner: Schedule = (
        schedule if callable(schedule) else ConstantSchedule(float(schedule))
    )
    return WithWarmup(
        schedule=inner,
        transition_steps=int(transition_steps),
        ramp=ramp,
        init_value=float(init_value),
    )
