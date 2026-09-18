"""Step-indexed schedules + warmup / restart / WSD composition.

Each schedule is a frozen, callable recipe dataclass; the factory
functions construct the matching recipe.  Call-site usage is unchanged
from the previous closure-based API::

    from opake.scheduling import cosine_schedule, with_warmup

    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    schedule(step)   # → float, scheduled value at integer step

The recipe classes themselves live in :mod:`opake.scheduling.types`
for ``isinstance`` checks and type annotations — matching how
:mod:`opake.dpftrl.noise.types` exposes its strategy classes.
"""

from opake.api.engine.scheduling import (
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    warmup_stable_decay,
    with_restarts,
    with_warmup,
)

__all__ = [
    "constant_schedule",
    "cosine_schedule",
    "exponential_schedule",
    "inverse_sqrt_schedule",
    "linear_schedule",
    "one_minus_sqrt_schedule",
    "polynomial_schedule",
    "warmup_stable_decay",
    "with_restarts",
    "with_warmup",
]
