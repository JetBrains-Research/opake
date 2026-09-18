"""Step-indexed scalar schedules — factory functions.

Curves and composition wrappers are frozen, callable recipe dataclasses,
each defined in its own ``_*.py`` impl module.  Factory functions here
construct the matching recipe — call-site usage is unchanged from the
previous closure-based API::

    from opake.api.engine.scheduling import cosine_schedule, with_warmup

    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    schedule(step)   # → float, scheduled value at integer step

The recipe classes themselves live in
:mod:`opake.api.engine.scheduling.types` for ``isinstance`` checks and
type annotations.
"""

from opake.api.engine.scheduling._constant import constant_schedule
from opake.api.engine.scheduling._cosine import cosine_schedule
from opake.api.engine.scheduling._exponential import exponential_schedule
from opake.api.engine.scheduling._inverse_sqrt import inverse_sqrt_schedule
from opake.api.engine.scheduling._linear import linear_schedule
from opake.api.engine.scheduling._one_minus_sqrt import one_minus_sqrt_schedule
from opake.api.engine.scheduling._polynomial import polynomial_schedule
from opake.api.engine.scheduling._warmup_stable_decay import warmup_stable_decay
from opake.api.engine.scheduling._with_restarts import with_restarts
from opake.api.engine.scheduling._with_warmup import with_warmup

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
