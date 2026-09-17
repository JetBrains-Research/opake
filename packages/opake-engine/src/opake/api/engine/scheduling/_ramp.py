"""Named ramp shapes — used by :class:`~._with_warmup.WithWarmup` and
the warmup phase of :class:`~._warmup_stable_decay.WarmupStableDecay`.

A ramp is a function ``progress ∈ [0, 1] → factor ∈ [0, 1]`` that goes
from ``0`` at the start of the warmup to ``1`` at the end.  The lookup
table maps a string discriminator (``"linear"``, ``"cosine"``,
``"1-sqrt"``) to the matching shape function so composition recipes can
store the discriminator and stay serializable.

Callable overrides are accepted by :func:`resolve_ramp` as an escape
hatch for power-user shapes; the resulting recipe is a valid Schedule
at runtime but won't round-trip through the universal serializer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["NAMED_RAMPS", "resolve_ramp"]


# ``progress in [0, 1] -> factor in [0, 1]``.
NAMED_RAMPS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1.0 - math.cos(math.pi * p)),
    "1-sqrt": lambda p: 1.0 - math.sqrt(1.0 - p),
}


def resolve_ramp(
    ramp: str | Callable[[float], float],
    *,
    field: str = "ramp",
) -> Callable[[float], float]:
    """Resolve ``ramp`` to a callable.

    Strings are looked up in :data:`NAMED_RAMPS`; unknown strings raise.
    Callables are returned as-is (caller takes responsibility for the
    non-serializability of recipes carrying them).
    """
    if callable(ramp):
        return ramp
    if ramp not in NAMED_RAMPS:
        raise ConfigurationError(
            *(
                f"Unknown {field}={ramp!r}; expected one of {sorted(NAMED_RAMPS)} "
                f"or a callable.",
            )
        )
    return NAMED_RAMPS[ramp]
