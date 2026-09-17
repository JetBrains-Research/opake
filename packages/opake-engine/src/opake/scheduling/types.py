"""Scheduling type façade — :data:`Schedule` and recipe dataclasses.

Re-exports both the ``Schedule`` callable type alias and the recipe
classes that implement it.  Surface mirrors
:mod:`opake.dpftrl.noise.types`: factory functions live in the package
root (``opake.scheduling``), types and recipe classes live here for
``isinstance`` checks and type annotations.
"""

from opake.api.engine.scheduling.types import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    OneMinusSqrtSchedule,
    PolynomialSchedule,
    Schedule,
    WarmupStableDecay,
    WithRestarts,
    WithWarmup,
)

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
