"""Named decay shapes — used by the decay phase of
:class:`~._warmup_stable_decay.WarmupStableDecay`.

A decay shape is a function ``progress ∈ [0, 1] → factor ∈ [0, 1]``
that multiplies ``(init_value - end_value)`` — ``1.0`` at the start of
decay (value = init), ``0.0`` at the end (value = end).  The lookup
table maps a string discriminator (``"linear"``, ``"cosine"``,
``"1-sqrt"``) to the matching shape function so composition recipes can
store the discriminator and stay serializable.

Callable overrides are accepted by :func:`resolve_decay` as an escape
hatch for power-user shapes; the resulting recipe is a valid Schedule
at runtime but won't round-trip through the universal serializer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["NAMED_DECAYS", "resolve_decay"]


# ``progress in [0, 1] -> factor in [0, 1]``; factor multiplies
# ``(init_value - end_value)``.
NAMED_DECAYS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: 1.0 - p,
    "cosine": lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    # Hägele et al. 2024 — concave-down, fast initial drop.
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
}


def resolve_decay(
    decay: str | Callable[[float], float],
    *,
    field: str = "decay_shape",
) -> Callable[[float], float]:
    """Resolve ``decay`` to a callable.  See :func:`~._ramp.resolve_ramp`
    for the strings-vs-callable semantics.
    """
    if callable(decay):
        return decay
    if decay not in NAMED_DECAYS:
        raise ConfigurationError(
            *(
                f"Unknown {field}={decay!r}; expected one of "
                f"{sorted(NAMED_DECAYS)} or a callable.",
            )
        )
    return NAMED_DECAYS[decay]
