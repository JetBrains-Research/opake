"""``WarmupStableDecay`` composition wrapper + factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opake.api.engine.scheduling._decay import resolve_decay
from opake.api.engine.scheduling._ramp import resolve_ramp
from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["WarmupStableDecay", "warmup_stable_decay"]


@dataclass(frozen=True, slots=True)
class WarmupStableDecay:
    """Three-phase schedule: warmup → constant → decay.

    The schedule follows Hägele et al., `Scaling Laws and Compute-Optimal
    Training Beyond Fixed Training Durations
    <https://arxiv.org/abs/2405.18392>`_ (2024). Phases over the
    ``num_warmup_steps + num_stable_steps + num_decay_steps`` total steps:

    1. ``[0, num_warmup_steps)`` — ramp from ``0`` to ``init_value``
       under ``warmup_ramp``.
    2. ``[num_warmup_steps, stable_end)`` — constant at ``init_value``.
    3. ``[stable_end, total)`` — decay from ``init_value`` down to
       ``end_value`` under ``decay_shape``.

    Beyond ``total`` returns ``end_value``.
    """

    init_value: float
    end_value: float
    num_warmup_steps: int
    num_stable_steps: int
    num_decay_steps: int
    warmup_ramp: str | Callable[[float], float] = "linear"
    decay_shape: str | Callable[[float], float] = "1-sqrt"

    def __call__(self, step: int) -> float:
        warmup_fn = resolve_ramp(self.warmup_ramp, field="warmup_ramp")
        decay_fn = resolve_decay(self.decay_shape, field="decay_shape")
        stable_end = self.num_warmup_steps + self.num_stable_steps
        total = stable_end + self.num_decay_steps
        if step < self.num_warmup_steps:
            return warmup_fn(step / self.num_warmup_steps) * self.init_value
        if step < stable_end:
            return self.init_value
        if step < total:
            p = (step - stable_end) / self.num_decay_steps
            return self.end_value + (self.init_value - self.end_value) * decay_fn(p)
        return self.end_value


def warmup_stable_decay(
    init_value: float,
    end_value: float = 0.0,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_ramp: str | Callable[[float], float] = "linear",
    decay_shape: str | Callable[[float], float] = "1-sqrt",
) -> WarmupStableDecay:
    """Three-phase schedule: warmup → constant → decay.

    See :class:`WarmupStableDecay` for the source and phase details.
    """
    if num_warmup_steps <= 0:
        raise ConfigurationError(
            *(
                f"warmup_stable_decay requires num_warmup_steps > 0; "
                f"got {num_warmup_steps}.",
            )
        )
    if num_stable_steps < 0:
        raise ConfigurationError(
            *(
                f"warmup_stable_decay requires num_stable_steps >= 0; "
                f"got {num_stable_steps}.",
            )
        )
    if num_decay_steps <= 0:
        raise ConfigurationError(
            *(
                f"warmup_stable_decay requires num_decay_steps > 0; got {num_decay_steps}.",
            )
        )
    # Validate named ramps/decays at construction.
    resolve_ramp(warmup_ramp, field="warmup_ramp")
    resolve_decay(decay_shape, field="decay_shape")
    return WarmupStableDecay(
        init_value=float(init_value),
        end_value=float(end_value),
        num_warmup_steps=int(num_warmup_steps),
        num_stable_steps=int(num_stable_steps),
        num_decay_steps=int(num_decay_steps),
        warmup_ramp=warmup_ramp,
        decay_shape=decay_shape,
    )
