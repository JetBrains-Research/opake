"""Public type definitions for :mod:`opake.api.engine.scheduling`.

Pure re-export façade — the :data:`Schedule` callable type alias and
the recipe dataclasses that implement it both live in their
implementation modules; this module just gathers them for ``isinstance``
checks and type annotations, matching how
:mod:`opake.dpftrl.noise.types`, :mod:`opake.dpsgd.noise.types`, etc.
are structured.
"""

from __future__ import annotations

from opake.api.engine.scheduling._constant import ConstantSchedule
from opake.api.engine.scheduling._cosine import CosineSchedule
from opake.api.engine.scheduling._exponential import ExponentialSchedule
from opake.api.engine.scheduling._inverse_sqrt import InverseSqrtSchedule
from opake.api.engine.scheduling._linear import LinearSchedule
from opake.api.engine.scheduling._one_minus_sqrt import OneMinusSqrtSchedule
from opake.api.engine.scheduling._polynomial import PolynomialSchedule
from opake.api.engine.scheduling._schedule import Schedule
from opake.api.engine.scheduling._warmup_stable_decay import WarmupStableDecay
from opake.api.engine.scheduling._with_restarts import WithRestarts
from opake.api.engine.scheduling._with_warmup import WithWarmup

__all__ = [
    "ConstantSchedule",
    "CosineSchedule",
    "ExponentialSchedule",
    "InverseSqrtSchedule",
    "LinearSchedule",
    "OneMinusSqrtSchedule",
    "PolynomialSchedule",
    "Schedule",
    "WarmupStableDecay",
    "WithRestarts",
    "WithWarmup",
]
